"""Runs model-written Python programs in a separate, time-limited process."""
import ast
import json
import os
import re
import subprocess
import sys
import tempfile
from dataclasses import dataclass

_BLOCKED = re.compile(
    r"\b(subprocess|socket|shutil|requests|urllib|http\.client|ctypes|multiprocessing|"
    r"__import__|eval\s*\(|exec\s*\(|open\s*\(|"
    r"os\s*\.\s*(system|popen|remove|unlink|rmdir|kill|fork|exec\w*|spawn\w*))"
)


@dataclass
class RunResult:
    ok: bool
    stdout: str
    error: str


def extract_code(text):
    blocks = re.findall(r"```(?:python|py)?\s*\n(.*?)```", text, re.S | re.I)
    if blocks:
        return max(blocks, key=len).strip()
    m = re.search(r"```(?:python|py)?\s*\n(.*)", text, re.S | re.I)  # unterminated fence
    if m:
        return m.group(1).strip()
    return text.strip()


def run_python(code, timeout):
    if not code.strip():
        return RunResult(False, "", "empty program")
    if _BLOCKED.search(code):
        return RunResult(False, "", "forbidden statement used (file/network/process access)")

    def _limits():  # POSIX only
        try:
            import resource
            resource.setrlimit(resource.RLIMIT_AS, (2 << 30, 2 << 30))
            resource.setrlimit(resource.RLIMIT_CPU, (int(timeout) + 2, int(timeout) + 2))
        except Exception:
            pass

    kwargs = {"preexec_fn": _limits} if os.name == "posix" else {}
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "solution.py")
        with open(path, "w", encoding="utf-8") as fh:
            fh.write(code)
        try:
            proc = subprocess.run(
                [sys.executable, "-I", "-X", "utf8", path],
                capture_output=True, text=True, encoding="utf-8", errors="replace",
                timeout=timeout, cwd=tmp, stdin=subprocess.DEVNULL, **kwargs,
            )
        except subprocess.TimeoutExpired:
            return RunResult(False, "", f"timeout after {timeout}s")
        except Exception as exc:
            return RunResult(False, "", f"could not run: {exc}")
    out = (proc.stdout or "")[-3000:]
    if proc.returncode != 0:
        return RunResult(False, out, (proc.stderr or "")[-1500:])
    return RunResult(True, out, "")


def parse_stdout(stdout):
    """Prefer the last 'ANSWER: x' line, otherwise the last non-SOLUTIONS line."""
    lines = [ln.strip() for ln in (stdout or "").splitlines() if ln.strip()]
    for ln in reversed(lines):
        m = re.match(r"(?i)^answer\s*:\s*(.*)$", ln)
        if m:
            return m.group(1).strip()
    lines = [ln for ln in lines if not re.match(r"(?i)^solutions\s*:", ln)]
    return lines[-1] if lines else ""


def _as_answer(item):
    if isinstance(item, (list, tuple)):
        if item and all(isinstance(x, (str, int)) and len(str(x)) == 1 for x in item):
            return "".join(str(x) for x in item)  # (4, 1, 3) -> "413"
        return ", ".join(str(x) for x in item)
    if isinstance(item, bool):
        return str(item)
    if isinstance(item, float) and item.is_integer():
        return str(int(item))
    return str(item).strip()


def parse_solutions(stdout):
    """Values listed on the last 'SOLUTIONS: [...]' line (None when the program printed none).

    The program is asked to enumerate every valid solution, so this list tells the agent
    whether the constraints admit one answer, several (ambiguous) or none (mis-modelled).
    """
    for ln in reversed([ln.strip() for ln in (stdout or "").splitlines()]):
        m = re.match(r"(?i)^solutions\s*:\s*(.*)$", ln)
        if not m:
            continue
        payload = m.group(1).strip()
        val = None
        try:
            val = json.loads(payload)
        except Exception:
            try:
                val = ast.literal_eval(payload)
            except Exception:
                return None
        if isinstance(val, (set, frozenset)):
            val = sorted(val, key=str)
        if not isinstance(val, (list, tuple)):
            val = [val]
        out = []
        for item in val:
            a = _as_answer(item)
            if a and a not in out:
                out.append(a)
        return out
    return None
