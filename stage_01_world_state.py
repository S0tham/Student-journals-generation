"""
STAGE 01 — World State Generator
=================================
Generates the hidden canonical truth for one synthetic user.
This is the foundation of the entire pipeline — all later stages
(events, notes, QA) must stay consistent with this world state.

Output: data/world_state.json

Pipeline position:
  [stage_01_world_state] -> stage_02_event_timeline -> stage_03_repair -> ...

To run:
  python stage_01_world_state.py

Requirements:
  Ollama running locally with llama3.2:3b pulled.
  Start Ollama: ollama serve
  Pull model:   ollama pull llama3.2:3b
"""

import json
import re
import requests
from pathlib import Path

# ---------------------------------------------------------------------------
# Config — swap these to change model or endpoint for this stage only
# ---------------------------------------------------------------------------

OLLAMA_URL = "http://localhost:11434/api/generate"
MODEL = "qwen2.5"

OUTPUT_DIR = Path("data")
OUTPUT_FILE = OUTPUT_DIR / "world_state.json"

# ---------------------------------------------------------------------------
# Prompt — Iman's original, unmodified
# ---------------------------------------------------------------------------

WORLD_STATE_PROMPT = """You are generating the hidden world state for a synthetic personal knowledge dataset used for retrieval-augmented generation (RAG) evaluation.

Generate a realistic long-term life simulation for ONE primary user.

The output should resemble the messy continuity of a real person's life rather than a neatly constructed fictional character profile.

GOALS

The dataset should support:
- longitudinal memory retrieval
- conflicting memory resolution
- temporal reasoning
- entity disambiguation
- incomplete task tracking
- evolving relationships
- mundane recurring details
- emotionally realistic inconsistencies

The simulation should feel partially unfinished, uneven, and organically evolving.

REQUIREMENTS

Core structure:
- One primary user only
- 10–20 recurring entities
- Mixture of:
  - work
  - relationships
  - family
  - logistics
  - finances
  - health
  - hobbies
  - routines
  - travel
  - digital life
  - home maintenance
- Some entities should recur across multiple domains
- Some entities should overlap semantically to create retrieval ambiguity
- Include both major life events and mundane repetitive behaviors

Realism requirements:
- Not every entity should be equally important
- Include dormant or low-activity periods
- Include abandoned habits, stale plans, and forgotten intentions
- Include recurring annoyances and low-stakes routines
- Include unfinished admin tasks
- Include minor repeated failures
- Include emotionally irrational behavior occasionally
- Include inconsistencies between stated intentions and actual behavior
- Include things the user avoids thinking about
- Include changing priorities over time
- Include periods where little changes

Temporal requirements:
- Include explicit dates or relative timelines
- Include temporal dependencies between events
- Some facts should later be corrected, revised, or contradicted
- Some plans should quietly disappear without resolution
- Include future events causing anticipation, anxiety, or preparation
- Include recurring weekly patterns with occasional disruptions
- Ensure not all story arcs peak simultaneously

Social coherence requirements:
- Entities should know each other in overlapping ways
- Relationships should influence unrelated domains
- Include mild interpersonal tensions, obligations, favors, or asymmetries
- Some entities should appear disproportionately often
- Some relationships should evolve gradually over time

Health and behavioral realism:
- Health issues should affect scheduling, mood, work, spending, or routines
- Habits should fluctuate instead of remaining perfectly consistent
- Include coping mechanisms, avoidance behaviors, or self-tracking systems
- Include intentions that repeatedly fail

Financial/logistical realism:
- Financial anxiety should have concrete causes
- Include subscriptions, repairs, appointments, insurance, taxes, forms, shipping delays, or scheduling friction
- Include realistic tradeoffs between time, money, energy, and social obligations

RAG DIFFICULTY REQUIREMENTS

The world state should contain:
- overlapping contexts
- recurring entities in multiple topics
- ambiguous references
- evolving facts
- partially outdated beliefs
- reminders that become stale
- repeated mentions of the same object/project over time
- realistic memory fragmentation

Avoid:
- perfectly clean timelines
- overly dramatic lives
- every event being important
- isolated entities with no social overlap
- excessive symmetry across categories
- profiles where every detail is narratively meaningful

Return ONLY structured JSON.

Return ONLY valid JSON. No markdown, no explanation, no code fences.

Schema:
{
  "user_profile": {
    "user_id": "u_001",
    "name": "...",
    "age": 0,
    "occupation": "...",
    "location": "...",
    "baseline_traits": ["..."]
  },
  "entities": [
    {
      "id": "...",
      "name": "...",
      "type": "person | place | project | object | organization | habit",
      "domains": ["..."],
      "salience": "low | medium | high",
      "notes": "..."
    }
  ],
  "story_arcs": [
    {
      "arc_id": "...",
      "title": "...",
      "status": "active | stalled | resolved | abandoned",
      "involved_entities": ["..."],
      "summary": "..."
    }
  ],
  "projects": [
    {
      "project_id": "...",
      "title": "...",
      "status": "active | paused | abandoned | completed",
      "related_entities": ["..."],
      "notes": "..."
    }
  ],
  "latent_facts": [
    {
      "fact_id": "...",
      "subject": "...",
      "relation": "...",
      "object": "...",
      "arc_id": "...",
      "confidence": "assumed | confirmed | contradicted"
    }
  ]
}"""


# ---------------------------------------------------------------------------
# Core functions
# ---------------------------------------------------------------------------

def call_ollama(prompt: str) -> str:
    """Send a prompt to the local Ollama instance and return the response text."""
    payload = {

        "model": MODEL,
        "prompt": prompt,
        "stream": False,
        "format": "json",
        "options": {
            "num_ctx": 8192,
            "temperature": 0.7,
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


def generate_world_state() -> dict:
    """Call Ollama and return the parsed world state dict."""
    print(f"Generating world state via Ollama ({MODEL})...")
    raw_text = call_ollama(WORLD_STATE_PROMPT)

    # Strip markdown code fences if the model added them anyway
    raw_text = re.sub(r"^```(?:json)?\s*", "", raw_text)
    raw_text = re.sub(r"\s*```$", "", raw_text)

    world_state = json.loads(raw_text)
    return world_state


def validate_world_state(world_state: dict) -> list[str]:
    """
    Basic structural validation.
    Returns a list of warning strings (empty = all good).
    """
    warnings = []
    required_keys = {"user_profile", "entities", "story_arcs", "projects", "latent_facts"}

    for key in required_keys:
        if key not in world_state:
            warnings.append(f"Missing top-level key: '{key}'")

    entities = world_state.get("entities", [])
    if len(entities) < 10:
        warnings.append(f"Only {len(entities)} entities — aim for 10–20")

    entity_ids = {e.get("id") for e in entities}
    for arc in world_state.get("story_arcs", []):
        for eid in arc.get("involved_entities", []):
            if eid not in entity_ids:
                warnings.append(f"Arc '{arc.get('arc_id')}' references unknown entity '{eid}'")

    return warnings


def save_world_state(world_state: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(world_state, f, indent=2, ensure_ascii=False)
    print(f"Saved → {path}")


def print_summary(world_state: dict) -> None:
    """Print a human-readable summary to the console."""
    profile = world_state.get("user_profile", {})
    print("\n── World State Summary ──────────────────────────────")
    print(f"  User        : {profile.get('name')} (age {profile.get('age')})")
    print(f"  Occupation  : {profile.get('occupation')}")
    print(f"  Location    : {profile.get('location')}")
    print(f"  Entities    : {len(world_state.get('entities', []))}")
    print(f"  Story arcs  : {len(world_state.get('story_arcs', []))}")
    print(f"  Projects    : {len(world_state.get('projects', []))}")
    print(f"  Latent facts: {len(world_state.get('latent_facts', []))}")
    print("\n  Arcs:")
    for arc in world_state.get("story_arcs", []):
        print(f"    [{arc.get('status', '?'):10s}] {arc.get('arc_id')} — {arc.get('title')}")
    print("─────────────────────────────────────────────────────\n")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    world_state = generate_world_state()

    warnings = validate_world_state(world_state)
    if warnings:
        print("\n⚠️  Validation warnings:")
        for w in warnings:
            print(f"   - {w}")
    else:
        print("✓  Validation passed")

    save_world_state(world_state, OUTPUT_FILE)
    print_summary(world_state)
    print("Stage 01 complete. Next: run stage_02_event_timeline.py")


if __name__ == "__main__":
    main()