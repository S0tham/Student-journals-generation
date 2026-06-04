# Synthetic RAG Memory Dataset Pipeline

A modular, stage-by-stage pipeline for generating a synthetic personal memory dataset
suitable for RAG (Retrieval-Augmented Generation) evaluation.

## Design principle: the train model

Each stage is a **standalone Python file** with one job. You can:
- Run a single stage in isolation
- Swap out the prompt in one stage without touching any other
- Replace a stage entirely (e.g. switch from LLM-based to rule-based event generation)
- Hand off individual stages to different team members

```
stage_01_world_state.py       → generates hidden canonical truth (entities, arcs, facts)
stage_02_event_timeline.py    → generates 90-day event sequence  [TODO]
stage_03_repair.py            → validates + minimally repairs events  [TODO]
stage_04_note_generation.py   → converts events to human-like notes  [TODO]
stage_05_qa_generation.py     → generates questions + answers from notes  [TODO]
stage_06_qa_audit.py          → checks QA answers are grounded in notes  [TODO]
```

Data flows through the `data/` folder:
```
data/
  world_state.json        ← output of stage 01
  events_raw.json         ← output of stage 02
  events_repaired.json    ← output of stage 03
  notes.json              ← output of stage 04
  qa_pairs.json           ← output of stage 05
  qa_audited.json         ← output of stage 06
```



## Running

Run stages in order:

```bash
python stage_01_world_state.py
```

Each script reads the output of the previous stage from `data/` and writes its own output there.

## Modifying a stage

- **Change the model**: edit the `MODEL` constant at the top of any stage file
- **Change the prompt**: edit the `*_PROMPT` constant — Iman's original prompts are preserved as-is
- **Change validation rules**: edit the `validate_*` function in that stage
- **Change output format**: edit the `save_*` function and update the schema comment

## References

- `docs/synthetic_rag_memory_pipeline.docx` — Iman's original pipeline design
- `docs/synthetic_memory_insights.docx` — critique & improvement notes  
- `docs/rag-pipeline-guide.html` — technical implementation guide
