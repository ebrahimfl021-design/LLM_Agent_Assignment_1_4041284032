"""Central configuration. Everything can be overridden with environment variables."""
import os


def _get(name, default, cast):
    try:
        return cast(os.environ.get(name, default))
    except (TypeError, ValueError):
        return default


# Model used for ALL inference (local Ollama only).
MODEL = os.environ.get("OLLAMA_MODEL", "qwen3:8b")

# Per-problem budgets.
MAX_CALLS = _get("AGENT_MAX_CALLS", 10, int)           # max LLM calls per problem
TIME_BUDGET = _get("AGENT_TIME_BUDGET", 240.0, float)  # seconds per problem
CALL_TIMEOUT = _get("AGENT_CALL_TIMEOUT", 150.0, float)  # seconds per LLM call
CODE_TIMEOUT = _get("AGENT_CODE_TIMEOUT", 10.0, float)   # seconds per generated program

NUM_CTX = _get("AGENT_NUM_CTX", 8192, int)
CODE_THINK = os.environ.get("AGENT_CODE_THINK", "0") == "1"

# Persian problems: the analyst also returns a faithful English translation that the
# solvers receive next to the original text (the original stays authoritative).
TRANSLATE = os.environ.get("AGENT_TRANSLATE", "1") == "1"
# Max LLM calls spent choosing among the answers of an ambiguous enumeration.
DISAMBIG_CALLS = _get("AGENT_DISAMBIG_CALLS", 2, int)
