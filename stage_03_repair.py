"""
STAGE 03 — Repair Pass
========================
Reads the raw events from stage 02 and fixes structural issues:
invalid entity references, unknown arc IDs, timestamp ordering,
and event-specific latent fact updates.

The key principle is minimal-edit: only fix what is provably wrong
using the world state as ground truth. Never rewrite event text
for style. Messiness in the text is a feature, not a bug.

Input:  data/world_state.json
        data/events_raw.json
Output: data/events_repaired.json

Pipeline position:
  stage_02_event_timeline -> [stage_03_repair] -> stage_04_note_generation -> ...

To run:
  python stage_03_repair.py

Requirements:
  Ollama running locally with your chosen model pulled.
  Start Ollama: ollama serve
  Run stages 01 and 02 first.
"""

import json
import re
import requests
from pathlib import Path


OLLAMA_URL = "http://localhost:11434/api/generate"
MODEL = "qwen2.5"

WORLD_STATE_FILE  = Path("data") / "world_state.json"
INPUT_FILE        = Path("data") / "events_raw.json"
OUTPUT_FILE       = Path("data") / "events_repaired.json"


REPAIR_PROMPT = """You are validating and repairing a synthetic memory timeline for a RAG dataset.

Fix the input JSON to ensure full consistency with the world state and structural rules.

Fixing Objectives

Correct the following issues:

1. Story arc validity
- Ensure every story_arc_id exists in the world state
- If invalid, map to closest valid arc or correct identifier format

2. Latent fact correctness (critical)
Fix latent_fact_updates so that:
- They are event-specific (not static repeated facts)
- They reflect:
  - reinforcement
  - contradiction
  - activation
  - decay
  - escalation
- Remove generic or unrelated latent fact restatements
- Ensure each update is causally triggered by the event

3. Temporal consistency
- Ensure timestamps are strictly increasing
- Ensure no backward causality (future event influencing past event)
- Preserve original ordering as much as possible

4. Entity validity
- Remove any entity not present in world state
- Ensure correct spelling and consistency of entity names

5. Event type normalization
- Normalize overly specific or inconsistent event types into a consistent taxonomy
- Merge near-duplicates

6. Narrative coherence (light-touch only)
- Do NOT rewrite content for style
- Only fix contradictions, invalid references, and structural issues

Output Requirements
Return ONLY corrected JSON:

{
  "events": [...]
}

No commentary. No explanation."""


def rule_based_repair(events: list[dict], world_state: dict) -> tuple[list[dict], list[str]]:
    """
    Fix issues that can be caught deterministically — no LLM needed.
    Returns (repaired_events, repair_log).

    This runs BEFORE the LLM repair so the LLM gets cleaner input
    and doesn't need to guess at entity/arc IDs.
    """
    valid_entity_ids = {e["id"] for e in world_state.get("entities", [])}
    valid_arc_ids    = {a["arc_id"] for a in world_state.get("story_arcs", [])}

    repair_log = []
    repaired   = []
    prev_ts    = ""

    for event in events:
        e = event.copy()
        eid = e.get("event_id", "?")

        #Remove unknown entities
        original_entities = e.get("involved_entities", [])
        valid_entities = [x for x in original_entities if x in valid_entity_ids]
        removed = set(original_entities) - set(valid_entities)
        if removed:
            repair_log.append(f"{eid}: removed unknown entities {removed}")
        e["involved_entities"] = valid_entities

        #Null out unknown arc IDs (LLM will remap in its pass)
        arc = e.get("story_arc_id")
        if arc and arc not in valid_arc_ids:
            repair_log.append(f"{eid}: nulled unknown arc '{arc}'")
            e["story_arc_id"] = None

        #Clamp importance to 1-5
        imp = e.get("importance")
        if imp is not None:
            try:
                clamped = max(1, min(5, int(imp)))
                if clamped != imp:
                    repair_log.append(f"{eid}: clamped importance {imp} → {clamped}")
                e["importance"] = clamped
            except (ValueError, TypeError):
                repair_log.append(f"{eid}: removed invalid importance value '{imp}'")
                e["importance"] = 3 

        #Flag timestamp ordering 
        ts = e.get("timestamp", "")
        if ts and prev_ts and ts < prev_ts:
            repair_log.append(f"{eid}: timestamp out of order ({ts} after {prev_ts}) — flagged for LLM repair")
        if ts:
            prev_ts = ts

        e["_repair_log"] = repair_log[-1:] if repair_log else []
        repaired.append(e)

    return repaired, repair_log


def call_ollama(prompt: str) -> str:
    """Send a prompt to the local Ollama instance and return the response text."""
    payload = {
        "model": MODEL,
        "prompt": prompt,
        "stream": False,
        "format": "json",
        "options": {
            "num_ctx": 8192,
            "temperature": 0.3,  
            "num_predict": -1,
        }
    }
    try:
        response = requests.post(OLLAMA_URL, json=payload, timeout=300)
        response.raise_for_status()
        return response.json().get("response", "").strip()
    except requests.exceptions.ConnectionError:
        raise ConnectionError(
            f"Could not connect to Ollama at {OLLAMA_URL}. "
            "Is Ollama running? Start it with: ollama serve"
        )
    except requests.exceptions.Timeout:
        raise TimeoutError("Ollama took too long to respond (>300s). Try a smaller model.")


def llm_repair(events: list[dict], world_state: dict) -> list[dict]:
    """
    Pass the pre-repaired events through the LLM for semantic fixes:
    arc remapping, latent fact quality, and timestamp ordering.
    """
    print(f"Running LLM repair via Ollama ({MODEL})...")

    world_state_json = json.dumps(world_state, indent=2, ensure_ascii=False)
    events_json      = json.dumps({"events": events}, indent=2, ensure_ascii=False)

    prompt = (
        f"World state:\n{world_state_json}\n\n"
        f"Raw events to repair:\n{events_json}\n\n"
        f"{REPAIR_PROMPT}"
    )

    raw_text = call_ollama(prompt)
    raw_text = re.sub(r"^```(?:json)?\s*", "", raw_text)
    raw_text = re.sub(r"\s*```$", "", raw_text)

    repaired_data = json.loads(raw_text)
    return repaired_data.get("events", events)  


def strip_repair_logs(events: list[dict]) -> list[dict]:
    """Remove internal _repair_log keys before saving."""
    return [{k: v for k, v in e.items() if k != "_repair_log"} for e in events]


def validate_events(events: list[dict], world_state: dict) -> list[str]:
    warnings = []
    valid_entity_ids = {e["id"] for e in world_state.get("entities", [])}
    valid_arc_ids    = {a["arc_id"] for a in world_state.get("story_arcs", [])}
    seen_ids = set()
    prev_ts  = ""

    for event in events:
        eid = event.get("event_id", "?")

        if eid in seen_ids:
            warnings.append(f"Duplicate event_id: '{eid}'")
        seen_ids.add(eid)

        ts = event.get("timestamp", "")
        if ts and prev_ts and ts < prev_ts:
            warnings.append(f"Event '{eid}' still out of order after repair")
        if ts:
            prev_ts = ts

        for entity in event.get("involved_entities", []):
            if entity not in valid_entity_ids:
                warnings.append(f"Event '{eid}' still has unknown entity: '{entity}'")

        arc = event.get("story_arc_id")
        if arc and arc not in valid_arc_ids:
            warnings.append(f"Event '{eid}' still has unknown arc: '{arc}'")

    return warnings

def load_json(path: Path, label: str) -> dict:
    if not path.exists():
        raise FileNotFoundError(f"{label} not found at '{path}'. Run the previous stage first.")
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def save_events(events_data: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(events_data, f, indent=2, ensure_ascii=False)
    print(f"Saved → {path}")


def print_summary(repair_log: list[str], events_before: int, events_after: int) -> None:
    print("\n── Repair Summary ───────────────────────────────────")
    print(f"  Events before : {events_before}")
    print(f"  Events after  : {events_after}")
    print(f"  Rule-based fixes: {len(repair_log)}")
    if repair_log:
        for entry in repair_log[:10]:
            print(f"    · {entry}")
        if len(repair_log) > 10:
            print(f"    ... and {len(repair_log) - 10} more")
    print("─────────────────────────────────────────────────────\n")

def main():
    world_state  = load_json(WORLD_STATE_FILE, "World state")
    events_data  = load_json(INPUT_FILE, "Raw events")
    raw_events   = events_data.get("events", [])

    print(f"Loaded {len(raw_events)} raw events.")

    print("Pass 1: rule-based repair...")
    pre_repaired, repair_log = rule_based_repair(raw_events, world_state)
    print(f"  {len(repair_log)} rule-based fixes applied.")

    print("Pass 2: LLM semantic repair...")
    llm_repaired = llm_repair(pre_repaired, world_state)

    final_events = strip_repair_logs(llm_repaired)

    warnings = validate_events(final_events, world_state)
    if warnings:
        print(f"\n⚠️  Remaining issues after repair ({len(warnings)}):")
        for w in warnings:
            print(f"   - {w}")
    else:
        print("✓  Validation passed — no issues remaining")

    save_events({"events": final_events}, OUTPUT_FILE)
    print_summary(repair_log, len(raw_events), len(final_events))
    print("Stage 03 complete. Next: run stage_04_note_generation.py")


if __name__ == "__main__":
    main()
