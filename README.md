# Synthetic RAG Memory Dataset Pipeline

5-stage pipeline for generating synthetic RAG evaluation benchmarks grounded in a single user's life.
Supports two backends: the **Anthropic API** (paid, highest quality) and **Ollama** (local, free).

---

## Architecture

```
stage_01_world_state.py    →  data/world_state.json
         ↓
knowledge_graph.py         →  (in-memory graph, imported by stages 02 and 03)
         ↓
stage_02_event_timeline.py →  data/events_raw.json
         ↓
stage_03_repair.py         →  data/events_repaired.json
         ↓
stage_04_note_generation.py → data/notes.json
         ↓
stage_05_qa_generation.py  →  data/qa_pairs.json
```

`knowledge_graph.py` is not a stage — it is a shared module imported by stages 02 and 03.
It does two things: builds a directed graph of the world state for deterministic validation,
and runs a biased random walk to generate event skeletons before the LLM writes any text.

---

## Setup

### Option A — Anthropic API

Requires a paid API account at [console.anthropic.com](https://console.anthropic.com).

```bash
pip install anthropic networkx
export ANTHROPIC_API_KEY=sk-...
```

### Option B — Ollama (local, free)

```bash
pip install requests networkx
ollama serve          # start Ollama
ollama pull qwen2.5   # or whichever model you configure
```

#### Recommended Ollama models

| Model | RAM needed | JSON reliability | Notes |
|-------|-----------|-----------------|-------|
| `qwen2.5:72b` | ~45 GB | Best | Top choice for production runs |
| `qwen2.5:32b` | ~22 GB | Very good | Best balance of quality and size |
| `llama3.1:70b` | ~40 GB | Good | Strong alternative |
| `qwen2.5` (default) | ~5 GB | Decent | Prototyping only |
| `llama3.1:8b` | ~5 GB | Unreliable | Struggles with complex JSON schemas |

Use at least a 32B model for production runs. The default `qwen2.5` (no size suffix) is the small variant — suitable for testing the pipeline but not for generating high-quality benchmark data.

---

## Configuration

Each stage file currently has its model and paths set at the top of the file:

```python
OLLAMA_URL = "http://localhost:11434/api/generate"
MODEL      = "qwen2.5"
INPUT_FILE = Path("data") / "world_state.json"
```

To change the model, update `MODEL` in each stage file. All output files are written to `data/`.

> A central `config.py` is planned for a future refactor so all parameters can be set in one place.

---

## Usage

Run the stages in order. Each stage reads the output of the previous one.

```bash
# Make sure Ollama is running first
ollama serve

# Then run each stage
python stage_01_world_state.py
python stage_02_event_timeline.py
python stage_03_repair.py
python stage_04_note_generation.py
python stage_05_qa_generation.py
```

To inspect the knowledge graph against an existing world state:

```bash
python knowledge_graph.py
```

---

## Output files

| File | Contents |
|------|----------|
| `data/world_state.json` | Hidden canonical truth: user profile, entities, arcs, projects, latent facts |
| `data/events_raw.json` | Raw chronological event stream with skeletons filled by the LLM |
| `data/events_repaired.json` | Event stream after graph-based structural repair |
| `data/notes.json` | Human-like notes derived from events (the retrieval corpus) |
| `data/qa_pairs.json` | QA pairs with gold supporting note IDs and difficulty labels |

---

## What the validator checks

**Stage 01 — World state**
Schema conformance, ≥10 entities, arc-to-entity references valid.

**Stage 02 — Event timeline**
Unique event IDs, strictly monotonic ISO timestamps, all `story_arc_id` and `involved_entities` references exist in world state, importance in range 1–5.

**Stage 03 — Repair**
Unknown entities stripped, invalid arc IDs nulled, importance clamped, timestamps re-sorted. All checks are deterministic — no LLM needed.

**Stage 04 — Note corpus**
Unique note IDs, all `story_arc_id` references valid, no empty note text, importance values are `low / medium / high`.

**Stage 05 — QA set**
All `supporting_notes` reference real note IDs, unanswerable questions have no supporting notes, at least one of each QA type present.

---

## Files

```
project/
├── stage_01_world_state.py      # Generates hidden ground truth
├── stage_02_event_timeline.py   # Generates event stream via skeleton + LLM
├── stage_03_repair.py           # Graph-based structural repair
├── stage_04_note_generation.py  # Converts events to human-like notes
├── stage_05_qa_generation.py    # Evidence-first QA generation
├── knowledge_graph.py           # Shared graph module (imported by 02 and 03)
└── data/                        # Generated at runtime
    ├── world_state.json
    ├── events_raw.json
    ├── events_repaired.json
    ├── notes.json
    └── qa_pairs.json
```

---

## Design decisions

**Knowledge graph as structural backbone.**
`knowledge_graph.py` converts `world_state.json` into a `networkx` directed graph. Entities, story arcs, projects, and latent facts become nodes and edges. This lets stages 02 and 03 perform structural checks in code rather than asking an LLM to guess at valid IDs.

**Event skeletons.**
Stage 02 pre-fills `story_arc_id`, `involved_entities`, and `timestamp` via a biased random walk over the graph before calling the LLM. The LLM only writes `text` and `latent_fact_updates`. This eliminates the most common source of structural errors.

**Biased random walk.**
Entities are scored per iteration based on arc connectivity, recency, and graph degree. Entities connected to unresolved arcs and dormant entities resurface naturally without instructing the LLM to track this.

**Contradiction injection.**
Stage 02 adds explicit contradiction skeletons: for each unresolved arc, a later event is injected that must revise or contradict a fact from the first. This ensures `conflict_resolution` QA pairs are possible.

**No LLM repair pass.**
Structural errors in the event stream are caught and fixed deterministically in stage 03 using the knowledge graph. An LLM repair pass was removed because it costs tokens, is slow, and risks silent content rewrites that contaminate the benchmark.

**Evidence-first QA generation.**
Stage 05 selects supporting notes before writing a question, not after. Gold evidence is locked at generation time. Five QA types cover a difficulty ladder from `single_hop` to `unanswerable`.

**MESSINESS_RULES in Stage 04.**
The note generation prompt applies one to three randomised rules per note (OMISSION, ABRUPT_END, IMPLICIT_REFERENCE, UNCERTAINTY, LATENT_FACT_AS_TEXTURE). These prevent latent facts from being stated directly in notes, which would make retrieval unrealistically easy.
