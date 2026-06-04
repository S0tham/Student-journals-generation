"""
STAGE 04 — Note Generation
============================
Reads the repaired events from stage 03 and converts each event into
a realistic, human-like personal note. The LLM only handles surface
realization — the structure comes from the event, not the model.

Input:  data/events_repaired.json
Output: data/notes.json

Pipeline position:
  stage_03_repair -> [stage_04_note_generation] -> stage_05_qa_generation -> ...

To run:
  python stage_04_note_generation.py

Requirements:
  Ollama running locally with your chosen model pulled.
  Start Ollama: ollama serve
  Run stages 01, 02, and 03 first.
"""

import json
import re
import requests
from pathlib import Path


OLLAMA_URL = "http://localhost:11434/api/generate"
MODEL = "llama3.1"

INPUT_FILE  = Path("data") / "events_repaired.json"
OUTPUT_FILE = Path("data") / "notes.json"
WORLD_STATE_FILE = Path("data") / "world_state.json"


NOTE_GENERATION_PROMPT = """Convert the structured event into a realistic personal note.

Requirements:
- Natural language
- Imperfect human writing style
- Concise
- Sometimes partial/incomplete
- Sometimes references prior context implicitly
- Maintain consistency with entities and timeline
- Do not make the note sound polished or fully resolved
- Prefer a lived-in, messy memory fragment style

Return JSON only.

Schema:
{
  "note_id": "",
  "timestamp": "",
  "note_type": "",
  "text": "",
  "entities": [],
  "tags": [],
  "latent_facts": [],
  "story_arc_id": "",
  "importance": ""
}"""


def load_json(path: Path, label: str) -> dict:
    if not path.exists():
        raise FileNotFoundError(
            f"{label} not found at '{path}'. Run the previous stage first."
        )
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def call_ollama(prompt: str) -> str:
    """Send a prompt to the local Ollama instance and return the response text."""
    payload = {
        "model": MODEL,
        "prompt": prompt,
        "stream": False,
        "format": "json",
        "options": {
            "num_ctx": 8192,
            "temperature": 0.8,  
            "num_predict": -1,
        }
    }
    try:
        response = requests.post(OLLAMA_URL, json=payload, timeout=120)
        response.raise_for_status()
        return response.json().get("response", "").strip()
    except requests.exceptions.ConnectionError:
        raise ConnectionError(
            f"Could not connect to Ollama at {OLLAMA_URL}. "
            "Is Ollama running? Start it with: ollama serve"
        )
    except requests.exceptions.Timeout:
        raise TimeoutError("Ollama took too long to respond (>120s).")


def build_note_prompt(event: dict, world_state: dict) -> str:
    """Inject a single event as context for the note generator."""
    event_json = json.dumps(event, indent=2, ensure_ascii=False)
    world_state_json = json.dumps(world_state, indent=2, ensure_ascii=False)
    
    return f"World State (Context):\n{world_state_json}\n\nEvent:\n{event_json}\n\n{NOTE_GENERATION_PROMPT}"


def generate_note(event: dict, world_state: dict, index: int) -> dict | None:
    """
    Generate one note from one event.
    Returns the parsed note dict, or None if generation/parsing fails.
    """
    prompt   = build_note_prompt(event, world_state)
    raw_text = call_ollama(prompt)

    raw_text = re.sub(r"^```(?:json)?\s*", "", raw_text)
    raw_text = re.sub(r"\s*```$", "", raw_text)

    note = json.loads(raw_text)

    if not note.get("note_id"):
        note["note_id"] = f"n_{index:04d}"

    if not note.get("timestamp"):
        note["timestamp"] = event.get("timestamp", "")
    if not note.get("story_arc_id"):
        note["story_arc_id"] = event.get("story_arc_id", "")
    if not note.get("importance"):
        imp = event.get("importance", 3)
        note["importance"] = _importance_label(imp)

    note["source_event_id"] = event.get("event_id", "")

    return note


def _importance_label(value) -> str:
    """Convert numeric importance (1-5) to the low/medium/high label the schema expects."""
    try:
        v = int(value)
    except (ValueError, TypeError):
        return "medium"
    if v <= 2:
        return "low"
    if v == 3:
        return "medium"
    return "high"


def validate_notes(notes: list[dict]) -> list[str]:
    """
    Basic structural validation.
    Returns a list of warning strings (empty = all good).
    """
    warnings  = []
    seen_ids  = set()
    required  = {"note_id", "timestamp", "note_type", "text", "entities",
                 "tags", "latent_facts", "story_arc_id", "importance"}

    for note in notes:
        nid = note.get("note_id", "?")

        if nid in seen_ids:
            warnings.append(f"Duplicate note_id: '{nid}'")
        seen_ids.add(nid)

        missing = required - note.keys()
        if missing:
            warnings.append(f"Note '{nid}' missing keys: {missing}")

        if not note.get("text", "").strip():
            warnings.append(f"Note '{nid}' has empty text")

        imp = note.get("importance", "")
        if imp not in ("low", "medium", "high"):
            warnings.append(f"Note '{nid}' has unexpected importance value: '{imp}'")

    return warnings


def save_notes(notes: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump({"notes": notes}, f, indent=2, ensure_ascii=False)
    print(f"Saved → {path}")


def print_summary(notes: list[dict], failed: int) -> None:
    if not notes:
        print("No notes generated.")
        return

    type_counts: dict[str, int] = {}
    imp_counts:  dict[str, int] = {}
    for n in notes:
        t = n.get("note_type", "unknown")
        i = n.get("importance", "unknown")
        type_counts[t] = type_counts.get(t, 0) + 1
        imp_counts[i]  = imp_counts.get(i, 0) + 1

    print("\n── Note Generation Summary ──────────────────────────")
    print(f"  Notes generated : {len(notes)}")
    print(f"  Failed / skipped: {failed}")
    print(f"\n  By note type:")
    for t, count in sorted(type_counts.items(), key=lambda x: -x[1]):
        print(f"    {count:3d}x  {t}")
    print(f"\n  By importance:")
    for i in ("high", "medium", "low", "unknown"):
        if i in imp_counts:
            print(f"    {imp_counts[i]:3d}x  {i}")
    print(f"\n  Sample note:")
    sample = next((n for n in notes if n.get("text")), None)
    if sample:
        print(f"    [{sample.get('note_id')}] {sample.get('timestamp', '')[:10]}")
        print(f"    \"{sample.get('text', '')[:120]}...\"")
    print("─────────────────────────────────────────────────────\n")


def main():
    events_data = load_json(INPUT_FILE, "Repaired events")
    events      = events_data.get("events", [])
    
    # Voeg deze regel toe om de context in te laden
    world_state_data = load_json(WORLD_STATE_FILE, "World state")
    
    print(f"Loaded {len(events)} repaired events. Generating notes...")

    notes  = []
    failed = 0

    for i, event in enumerate(events):
        eid = event.get("event_id", f"index {i}")
        print(f"  [{i+1:3d}/{len(events)}] {eid}...", end=" ", flush=True)

        try:
            note = generate_note(event, world_state_data, i)
            if note:
                notes.append(note)
                print("✓")
            else:
                print("✗ (empty result)")
                failed += 1
        except (json.JSONDecodeError, Exception) as e:
            print(f"✗ ({type(e).__name__}: {e})")
            failed += 1

    warnings = validate_notes(notes)
    if warnings:
        print(f"\n⚠️  Validation warnings ({len(warnings)}):")
        for w in warnings:
            print(f"   - {w}")
    else:
        print("✓  Validation passed")

    save_notes(notes, OUTPUT_FILE)
    print_summary(notes, failed)
    print("Stage 04 complete. Next: run stage_05_qa_generation.py")


if __name__ == "__main__":
    main()
