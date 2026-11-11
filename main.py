"""Entry point:  python main.py --input input.json --output output.json

Input : {"id": "...", "problem": "..."}  or a list of such objects.
Output: {"id": "...", "answer": "..."}   or a list in the same order.
A valid JSON file is written in every situation (bad input, missing model, crash).
"""
import argparse
import json
import logging
import os
import sys
import tempfile

log = logging.getLogger("reasoning_agent")


def _setup_logging(verbose):
    handlers = []
    try:
        handlers.append(logging.FileHandler("agent.log", mode="a", encoding="utf-8"))
    except OSError:
        pass
    if verbose:
        handlers.append(logging.StreamHandler(sys.stderr))
    logging.basicConfig(level=logging.INFO, handlers=handlers or [logging.NullHandler()],
                        format="%(asctime)s %(levelname)s %(message)s")


def _validate_item(item):
    """(id, problem_text, error) for one input record."""
    if not isinstance(item, dict):
        return None, "", f"record is {type(item).__name__}, expected an object"
    pid = item.get("id")
    if pid is not None and not isinstance(pid, (str, int, float)):
        pid = str(pid)
    problem = item.get("problem")
    if problem is None:
        return pid, "", "missing 'problem' field"
    if not isinstance(problem, str):
        problem = json.dumps(problem, ensure_ascii=False) if isinstance(problem, (dict, list)) else str(problem)
    if not problem.strip():
        return pid, "", "empty 'problem' field"
    return pid, problem, ""


def _solve_item(agent, item):
    pid, problem, error = _validate_item(item)
    if error:
        log.warning("id=%r: %s", pid, error)
        return {"id": pid, "answer": ""}
    log.info("===== solving id=%r", pid)
    try:
        answer = agent.solve(problem)
    except Exception:
        log.exception("solve failed id=%r", pid)
        answer = ""
    answer = " ".join(str(answer if answer is not None else "").split())  # one clean line
    return {"id": pid, "answer": answer}


def _write(path, obj):
    """Atomic write: a reader never sees a half-written output.json."""
    folder = os.path.dirname(os.path.abspath(path))
    os.makedirs(folder, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=".output-", suffix=".json", dir=folder)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            json.dump(obj, fh, ensure_ascii=False, indent=2)
        os.replace(tmp, path)
    except Exception:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        with open(path, "w", encoding="utf-8") as fh:  # last resort, still valid JSON
            json.dump(obj, fh, ensure_ascii=False, indent=2)


class _Dead:
    """Stand-in when the agent cannot start: every answer is empty, output stays valid."""

    def solve(self, _):
        return ""


def main():
    parser = argparse.ArgumentParser(description="Local LLM reasoning agent (Ollama)")
    parser.add_argument("--input", required=True, help="path of input.json")
    parser.add_argument("--output", required=True, help="path of output.json")
    parser.add_argument("--model", default=None, help="Ollama model (default: qwen3:8b)")
    parser.add_argument("--verbose", action="store_true", help="print the log to stderr")
    args = parser.parse_args()
    _setup_logging(args.verbose)

    try:
        with open(args.input, "r", encoding="utf-8-sig") as fh:
            data = json.load(fh)
    except Exception:
        log.exception("cannot read input")
        _write(args.output, {"id": None, "answer": ""})
        return 0

    try:
        from reasoning_agent import Agent
        agent = Agent(model=args.model)
    except Exception:
        log.exception("cannot start agent")
        agent = _Dead()

    if isinstance(data, list):
        # placeholders first, then fill in one by one: an interrupted run still leaves valid JSON
        result = [{"id": it.get("id") if isinstance(it, dict) else None, "answer": ""} for it in data]
        _write(args.output, result)
        for i, it in enumerate(data):
            result[i] = _solve_item(agent, it)
            _write(args.output, result)
    else:
        _write(args.output, _solve_item(agent, data))
    return 0


if __name__ == "__main__":
    sys.exit(main())
