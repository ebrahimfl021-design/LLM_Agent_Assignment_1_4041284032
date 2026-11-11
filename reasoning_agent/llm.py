"""Thin wrapper over the local Ollama server with call/time budgets."""
import re
import time

try:  # the package is listed in requirements.txt
    import ollama
except ImportError:  # pragma: no cover
    ollama = None

from . import config

_THINK_RE = re.compile(r"<think>.*?</think>", re.S)


class LLMError(Exception):
    pass


class BudgetExceeded(LLMError):
    pass


def strip_think(text):
    text = _THINK_RE.sub("", text or "")
    if "</think>" in text:
        text = text.split("</think>")[-1]
    return text.strip()


class LLM:
    def __init__(self, model, max_calls, time_budget):
        self.model = model
        self.max_calls = max_calls
        self.deadline = time.time() + time_budget
        self.calls = 0

    def calls_left(self):
        return self.max_calls - self.calls

    def time_left(self):
        return self.deadline - time.time()

    def chat(self, system, user, think=False, temperature=0.3,
             num_predict=1500, json_mode=False, seed=None):
        if ollama is None:
            raise LLMError("the 'ollama' python package is not installed")
        if self.calls >= self.max_calls or self.time_left() < 5:
            raise BudgetExceeded("budget exhausted")
        self.calls += 1
        timeout = max(5.0, min(config.CALL_TIMEOUT, self.time_left()))
        options = {
            "temperature": temperature,
            "num_predict": num_predict,
            "num_ctx": config.NUM_CTX,
        }
        if seed is not None:
            options["seed"] = seed
        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ]
        kwargs = {"model": self.model, "messages": messages, "options": options}
        if json_mode:
            kwargs["format"] = "json"
        try:
            client = ollama.Client(timeout=timeout)
            try:
                resp = client.chat(think=think, **kwargs)
            except TypeError:  # old client without `think`
                if not think:
                    messages[-1]["content"] += "\n\n/no_think"
                resp = client.chat(**kwargs)
        except Exception as exc:  # connection, timeout, model errors...
            raise LLMError(str(exc)) from exc
        try:
            content = resp["message"]["content"] or ""
        except Exception as exc:
            raise LLMError("malformed response") from exc
        return strip_think(content)
