"""
STAGE 05 — QA Generation
==========================
Reads notes.json and generates question-answer pairs grounded in the
note corpus. Uses evidence-first generation: the supporting notes are
chosen before the question is written, so gold evidence is always correct.

Five QA types with increasing difficulty:
  single_hop         — one note is sufficient
  multi_hop          — two or more notes required
  temporal_reasoning — answer depends on event ordering
  conflict_resolution — two notes contradict each other
  unanswerable       — answer is not present in the notes

Input:  data/notes.json
        data/world_state.json
Output: data/qa_pairs.json
"""

import json
import re
import random
import requests
from enum import Enum
from pathlib import Path


OLLAMA_URL = "http://localhost:11434/api/generate"
MODEL      = "qwen2.5"

INPUT_NOTES      = Path("data") / "notes.json"
INPUT_WORLD      = Path("data") / "world_state.json"
OUTPUT_FILE      = Path("data") / "qa_pairs.json"

QA_COUNT = {
    "single_hop":          3,
    "multi_hop":           3,
    "temporal_reasoning":  3,
    "conflict_resolution": 3,
    "unanswerable":        3,
}


class QAType(Enum):
    SINGLE_HOP     = "single_hop"
    MULTI_HOP      = "multi_hop"
    TEMPORAL       = "temporal_reasoning"
    CONFLICT       = "conflict_resolution"
    UNANSWERABLE   = "unanswerable"


# ── Evidence finders ──────────────────────────────────────────────────────────

def find_single_hop_seeds(notes: list[dict]) -> list[list[str]]:
    """One note that contains a clear, answerable fact."""
    return [
        [n["note_id"]]
        for n in notes
        if n.get("text", "").strip() and n.get("importance") in ("medium", "high")
    ]


def find_multi_hop_seeds(notes: list[dict]) -> list[list[str]]:
    """
    Pairs or triples of notes that share an entity but are not adjacent
    in time — the answer requires combining both.
    """
    from collections import defaultdict
    by_entity: dict[str, list[dict]] = defaultdict(list)
    for note in sorted(notes, key=lambda n: n.get("timestamp", "")):
        for entity in note.get("entities", []):
            entity_key = entity.get("id") if isinstance(entity, dict) else str(entity)
            by_entity[entity_key].append(note)

    pairs = []
    for entity, entity_notes in by_entity.items():
        if len(entity_notes) < 2:
            continue
        for i in range(len(entity_notes) - 2):
            pairs.append([entity_notes[i]["note_id"], entity_notes[i + 2]["note_id"]])
    return pairs


def find_temporal_seeds(notes: list[dict]) -> list[list[str]]:
    """
    Three consecutive notes for the same arc — answer requires
    reconstructing the timeline.
    """
    from collections import defaultdict
    by_arc: dict[str, list[dict]] = defaultdict(list)
    for note in sorted(notes, key=lambda n: n.get("timestamp", "")):
        arc = note.get("story_arc_id")
        if arc:
            by_arc[arc].append(note)

    triples = []
    for arc, arc_notes in by_arc.items():
        if len(arc_notes) >= 3:
            triples.append([n["note_id"] for n in arc_notes[:3]])
    return triples


def find_conflict_seeds(notes: list[dict]) -> list[list[str]]:
    """
    Notes that share an entity and have different latent_fact values —
    a direct contradiction to resolve.
    """
    from collections import defaultdict
    entity_facts: dict[tuple, dict] = {}
    pairs = []

    for note in sorted(notes, key=lambda n: n.get("timestamp", "")):
        for entity in note.get("entities", []):
            entity_key = entity.get("id") if isinstance(entity, dict) else str(entity)
            
            for fact in note.get("latent_facts", []):
                key = (entity_key, str(fact)[:40])
                if key in entity_facts:
                    prev = entity_facts[key]
                    if prev["note_id"] != note["note_id"]:
                        pairs.append([prev["note_id"], note["note_id"]])
                else:
                    entity_facts[key] = {"note_id": note["note_id"], "fact": fact}
    return pairs


def find_unanswerable_seeds(notes: list[dict]) -> list[list[str]]:
    """
    Single notes that hint at something but don't answer it —
    the question should be unanswerable from the corpus.
    """
    return [
        [n["note_id"]]
        for n in notes
        if any(word in n.get("text", "").lower()
               for word in ["think", "maybe", "not sure", "might", "forgot", "can't remember"])
    ]


# ── QA generation ─────────────────────────────────────────────────────────────

QA_PROMPT = """You are generating a question-answer pair for a RAG benchmark dataset.

You are given a set of personal notes (the retrieval corpus).
The question must be answerable ONLY from these notes — no outside knowledge.

QA type: {qa_type}

Type-specific instructions:
- single_hop:         Question answerable from exactly one note. Keep it concrete.
- multi_hop:          Question that REQUIRES combining information from all notes given.
                      The answer cannot be found in any single note alone.
- temporal_reasoning: Question whose answer depends on the ORDER of events.
                      e.g. "What changed between the first and last mention of X?"
- conflict_resolution: Two notes contradict each other. The question must ask
                      the reader to identify or resolve the contradiction.
                      e.g. "The notes give two different statuses for X — which is more recent?"
- unanswerable:       The notes hint at something but do not contain the answer.
                      The correct answer is "cannot be determined from the notes."

Notes (these are the ONLY source of truth):
{notes_text}

Rules:
- Answer must be literally supported by the notes — no inference beyond what is stated
- Do not invent facts not present in the notes
- Keep the question specific and unambiguous
- The answer should be 1-3 sentences maximum

Return ONLY valid JSON:
{{
  "question": "...",
  "answer": "...",
  "reasoning_type": "{qa_type}",
  "supporting_notes": {note_ids},
  "required_hops": {hop_count},
  "difficulty": "easy | medium | hard"
}}"""


def call_ollama(prompt: str) -> str:
    payload = {
        "model": MODEL,
        "prompt": prompt,
        "stream": False,
        "format": "json",
        "options": {"num_ctx": 8192, "temperature": 0.5, "num_predict": -1},
    }
    response = requests.post(OLLAMA_URL, json=payload, timeout=300)
    response.raise_for_status()
    return response.json().get("response", "").strip()


def generate_qa_pair(
    note_ids: list[str],
    notes_by_id: dict,
    qa_type: QAType,
    index: int,
) -> dict | None:
    notes_text = "\n\n".join(
        f"[{nid}] {notes_by_id[nid]['text']}"
        for nid in note_ids
        if nid in notes_by_id
    )
    if not notes_text.strip():
        return None

    prompt = QA_PROMPT.format(
        qa_type=qa_type.value,
        notes_text=notes_text,
        note_ids=json.dumps(note_ids),
        hop_count=len(note_ids),
    )

    raw = call_ollama(prompt)
    raw = re.sub(r"^```(?:json)?\s*", "", raw)
    raw = re.sub(r"\s*```$", "", raw)

    qa = json.loads(raw)
    qa["question_id"]       = f"q_{index:04d}"
    qa["supporting_notes"]  = note_ids   
    qa["required_hops"]     = len(note_ids)
    return qa


# ── Main ──────────────────────────────────────────────────────────────────────

def load_json(path: Path, label: str) -> dict:
    if not path.exists():
        raise FileNotFoundError(f"{label} not found at '{path}'.")
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def print_summary(qa_pairs: list[dict], failed: int) -> None:
    from collections import Counter
    types  = Counter(q.get("reasoning_type") for q in qa_pairs)
    diffs  = Counter(q.get("difficulty") for q in qa_pairs)
    hops   = Counter(q.get("required_hops") for q in qa_pairs)

    print("\n── QA Generation Summary ────────────────────────────")
    print(f"  Generated : {len(qa_pairs)}   Failed: {failed}")
    print(f"\n  By type:")
    for t, c in sorted(types.items()):
        print(f"    {c:3d}x  {t}")
    print(f"\n  By difficulty:")
    for d in ("easy", "medium", "hard"):
        print(f"    {diffs.get(d, 0):3d}x  {d}")
    print(f"\n  By required hops:")
    for h, c in sorted(hops.items()):
        print(f"    {c:3d}x  {h} hop(s)")
    print("─────────────────────────────────────────────────────\n")


def main():
    notes_data  = load_json(INPUT_NOTES, "Notes")
    notes       = notes_data.get("notes", [])
    notes_by_id = {n["note_id"]: n for n in notes}

    print(f"Loaded {len(notes)} notes.")

    seed_finders = {
        QAType.SINGLE_HOP: find_single_hop_seeds,
        QAType.MULTI_HOP:  find_multi_hop_seeds,
        QAType.TEMPORAL:   find_temporal_seeds,
        QAType.CONFLICT:   find_conflict_seeds,
        QAType.UNANSWERABLE: find_unanswerable_seeds,
    }

    qa_pairs = []
    failed   = 0
    index    = 0

    for qa_type, finder in seed_finders.items():
        seeds  = finder(notes)
        count  = QA_COUNT.get(qa_type.value, 5)
        sample = random.sample(seeds, min(count, len(seeds)))

        print(f"\n  [{qa_type.value}] {len(seeds)} seeds found, generating {len(sample)}...")

        for note_ids in sample:
            print(f"    [{index+1:3d}] {note_ids}...", end=" ", flush=True)
            try:
                qa = generate_qa_pair(note_ids, notes_by_id, qa_type, index)
                if qa:
                    qa_pairs.append(qa)
                    print("✓")
                else:
                    print("✗ (empty)")
                    failed += 1
            except Exception as e:
                print(f"✗ ({type(e).__name__}: {e})")
                failed += 1
            index += 1

    output_path = OUTPUT_FILE
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump({"qa_pairs": qa_pairs}, f, indent=2, ensure_ascii=False)
    print(f"\nSaved → {output_path}")

    print_summary(qa_pairs, failed)
    print("Stage 05 complete.")


if __name__ == "__main__":
    main()