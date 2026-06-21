"""
config.py — Central configuration
===================================
All tunable parameters live here. Edit this file to change the model,
backend, data directory, or pipeline behaviour. Individual stage files
import from here and should not need to be edited for routine runs.
"""

from pathlib import Path

# ── Backend ───────────────────────────────────────────────────────────────────
# "ollama"     — local inference via Ollama (free, no internet required)
# "anthropic"  — Anthropic API (paid, highest quality)

BACKEND = "ollama"

# ── Model ─────────────────────────────────────────────────────────────────────
# Ollama model name (ignored when BACKEND = "anthropic").
# Recommended: qwen2.5:32b or qwen2.5:72b for production runs.
# The default "qwen2.5" (no size suffix) is the small variant — use for
# prototyping only. JSON reliability degrades significantly below 32B.

OLLAMA_MODEL    = "qwen2.5"
OLLAMA_BASE_URL = "http://localhost:11434/api/generate"

# Anthropic model (ignored when BACKEND = "ollama").
ANTHROPIC_MODEL = "claude-opus-4-6"

# ── Generation parameters ─────────────────────────────────────────────────────

MAX_TOKENS  = 8192   # context window passed to the model
MAX_RETRIES = 3      # number of retry attempts on JSON parse failure

# Per-stage temperature overrides.
# Higher = more creative output; lower = more structural reliability.
TEMPERATURE = {
    "stage_01": 0.7,   # world state — needs variety
    "stage_02": 0.7,   # event timeline — needs variety
    "stage_03": 0.3,   # repair — needs precision
    "stage_04": 0.8,   # note generation — needs creative realism
    "stage_05": 0.5,   # QA generation — balanced
}

# ── Pipeline parameters ───────────────────────────────────────────────────────

DURATION_DAYS   = 10    # number of days to simulate in the event timeline
EVENTS_PER_DAY  = 0.8   # average events per day (actual count varies)
START_DATE      = "2024-01-01"  # ISO date string, parsed in stage_02

MIN_ENTITIES    = 10    # minimum entities enforced by the validator
MAX_ENTITIES    = 20    # maximum entities instructed to the model

# Number of QA pairs to generate per type in stage_05.
QA_COUNT = {
    "single_hop":          3,
    "multi_hop":           3,
    "temporal_reasoning":  3,
    "conflict_resolution": 3,
    "unanswerable":        3,
}

# ── Paths ─────────────────────────────────────────────────────────────────────

DATA_DIR = Path("data")

PATHS = {
    "world_state":       DATA_DIR / "world_state.json",
    "events_raw":        DATA_DIR / "events_raw.json",
    "events_repaired":   DATA_DIR / "events_repaired.json",
    "notes":             DATA_DIR / "notes.json",
    "qa_pairs":          DATA_DIR / "qa_pairs.json",
}
