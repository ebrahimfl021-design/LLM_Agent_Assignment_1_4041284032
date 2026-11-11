"""Multi-stage agent:

Analyst (format + task type + translation, merged with deterministic rules)
  -> Task router (strategy for the ten reasoning families)
  -> program-aided Solvers + CoT Solver with Reflection/Repair
  -> Self-Consistency voting (early exit on agreement)
  -> Disambiguator (when exhaustive programs agree the constraints admit several answers)
  -> Judge/Verifier (when solvers disagree)
  -> Normalizer/Validator.
"""
import json
import logging
import re
from collections import OrderedDict
from dataclasses import dataclass

from . import config, prompts
from .answer_format import (Spec, ascii_digits, build_spec, clean_answer, default_spec,
                            describe_spec, detect_format, key_of, loose_clean)
from .llm import LLM, BudgetExceeded, LLMError
from .sandbox import extract_code, parse_solutions, parse_stdout, run_python
from .task_types import classify, glossary, strategy_text

log = logging.getLogger("reasoning_agent")

# Attempt schedule: (kind, temperature). Early exit as soon as two attempts agree.
PLAN = [("code", 0.3), ("code", 0.7), ("cot", 0.6), ("code", 0.8), ("cot", 0.7), ("code", 0.9)]
WEIGHT = {"code": 1.2, "cot": 1.0, "judge": 1.5, "disambig": 1.5, "direct": 0.5}
_PERSIAN = re.compile(r"[؀-ۿ]")


@dataclass
class Candidate:
    answer: str
    key: str
    source: str
    weight: float


@dataclass
class Context:
    """Everything the solvers need about one problem."""
    problem: str
    spec: Spec
    fmt: str
    block: str
    strategy: str
    task_types: list


def _parse_json(text):
    try:
        return json.loads(text)
    except Exception:
        m = re.search(r"\{.*\}", text or "", re.S)
        if m:
            try:
                return json.loads(m.group(0))
            except Exception:
                pass
    return None


class Agent:
    def __init__(self, model=None):
        self.model = model or config.MODEL
        self.cands, self.loose, self.ambig = [], [], []
        self.disambig_tried = False
        self.last_trace = {}

    # ------------------------------------------------------------ public API
    def solve(self, problem_text):
        problem = ascii_digits(str(problem_text or "")).strip()
        if not problem:
            log.warning("empty problem")
            return ""
        llm = LLM(self.model, config.MAX_CALLS, config.TIME_BUDGET)
        self.cands, self.loose, self.ambig = [], [], []
        self.disambig_tried = False
        ctx = self._context(problem, default_spec(problem), {}, "")
        answer = None
        try:
            ctx = self._analyze(llm, problem)
            answer = self._run(llm, ctx)
        except BudgetExceeded:
            log.warning("budget exhausted")
            answer = self._best() or self._ambiguous_fallback()
        except Exception:
            log.exception("unexpected agent error")
            answer = self._best() or self._ambiguous_fallback()
        if answer is None:
            answer = self._best_loose()
        self.last_trace = {"task_types": ctx.task_types, "answer_type": ctx.spec.answer_type,
                           "calls": llm.calls, "candidates": [(c.answer, c.source) for c in self.cands],
                           "ambiguous": self.ambig}
        log.info("FINAL=%r calls=%d types=%s", answer, llm.calls, ctx.task_types)
        return answer or ""

    # ------------------------------------------------------------- pipeline
    def _context(self, problem, spec, data, english):
        types = classify(problem, (data or {}).get("task_type"))
        block = prompts.problem_block(problem, english, glossary(problem))
        return Context(problem, spec, describe_spec(spec), block, strategy_text(types), types)

    def _analyze(self, llm, problem):
        rules = detect_format(problem)
        data = None
        try:
            raw = llm.chat(prompts.ANALYST_SYSTEM, prompts.analyst_user(problem),
                           think=False, temperature=0.0, json_mode=True,
                           num_predict=1800 if config.TRANSLATE else 600)
            data = _parse_json(raw)
        except BudgetExceeded:
            raise
        except LLMError as exc:
            log.warning("analyst failed: %s", exc)
        spec = build_spec(data, problem, rules)
        english = ""
        if config.TRANSLATE and isinstance(data, dict) and _PERSIAN.search(problem):
            english = ascii_digits(str(data.get("english") or "")).strip()
            if _PERSIAN.search(english) or len(english) < 0.3 * len(problem):
                english = ""  # not a usable translation
        ctx = self._context(problem, spec, data, english)
        log.info("SPEC type=%s choices=%s template=%r code_len=%s instr=%r rules=%s",
                 spec.answer_type, spec.choices, spec.template, spec.code_length,
                 spec.instruction, rules)
        log.info("TASK types=%s translated=%s", ctx.task_types, bool(english))
        return ctx

    def _run(self, llm, ctx):
        for i, (kind, temp) in enumerate(PLAN):
            if llm.calls_left() < 2 or llm.time_left() < 15:  # keep one call for the judge
                break
            try:
                if kind == "code":
                    self._code_attempt(llm, ctx, temp, seed=i + 1)
                else:
                    self._cot_attempt(llm, ctx, temp, seed=i + 1)
            except BudgetExceeded:
                raise
            except LLMError as exc:
                log.warning("%s attempt failed: %s", kind, exc)
            winner = self._consensus()
            if winner:
                log.info("consensus after attempt %d: %r", i + 1, winner)
                return winner
            confirmed = self._confirmed_ambiguity()
            if confirmed:
                picked = self._disambiguate(llm, ctx, confirmed)
                if picked:
                    return picked

        groups = self._groups()
        if len(groups) == 1 and not self.ambig:
            return next(iter(groups.values()))[0].answer
        if self.ambig and not groups:
            picked = self._disambiguate(llm, ctx, self._ambiguity_pool())
            return picked or self._ambiguous_fallback()
        if groups and (len(groups) >= 2 or self.ambig) and llm.calls_left() >= 1 and llm.time_left() >= 10:
            picked = self._judge(llm, ctx, groups)
            if picked:
                return picked
            return self._best()
        if not groups and llm.calls_left() >= 1 and llm.time_left() >= 10:
            self._direct(llm, ctx)
        return self._best()

    # ------------------------------------------------------------- attempts
    def _add(self, raw, spec, source):
        ans = clean_answer(raw, spec)
        log.info("%s -> raw=%r clean=%r", source, (raw or "")[-120:], ans)
        if ans is None:
            loose = loose_clean(raw)
            if loose and len(loose) < 200:
                self.loose.append(loose)
            return False
        self.cands.append(Candidate(ans, key_of(ans), source, WEIGHT.get(source, 1.0)))
        return True

    def _clean_solutions(self, values, spec):
        out, keys = [], set()
        for v in values or []:
            a = clean_answer(v, spec)
            if a is not None and key_of(a) not in keys:
                keys.add(key_of(a))
                out.append(a)
        return out

    def _evaluate(self, res, spec):
        """(answers, feedback): the cleaned solution set of a run, or why it is unusable."""
        if not res.ok:
            return None, res.error or "the program crashed"
        listed = parse_solutions(res.stdout)
        sols = self._clean_solutions(listed, spec)
        if listed is not None and not listed:
            return None, ("The program found NO solution, so a constraint is modelled wrongly "
                          "(re-check directions such as before/after, bounds and distinctness).")
        if sols:
            return sols, ""
        final = clean_answer(parse_stdout(res.stdout), spec)
        if final is not None:
            return [final], ""
        return None, (f"Output was: {res.stdout[-300:]!r}. It does not match the required answer "
                      "format.")

    def _code_attempt(self, llm, ctx, temp, seed):
        raw = llm.chat(prompts.CODE_SYSTEM, prompts.code_user(ctx.block, ctx.fmt, ctx.strategy),
                       think=config.CODE_THINK, temperature=temp,
                       num_predict=6000 if config.CODE_THINK else 1800, seed=seed)
        code = extract_code(raw)
        res = run_python(code, config.CODE_TIMEOUT)
        sols, feedback = self._evaluate(res, ctx.spec)
        if sols is None and llm.calls_left() >= 2 and llm.time_left() >= 20:  # one reflection/repair round
            log.info("repairing code: %s", feedback[:200])
            raw = llm.chat(prompts.CODE_SYSTEM,
                           prompts.repair_user(ctx.block, ctx.fmt, ctx.strategy, code, feedback),
                           think=False, temperature=0.2, num_predict=1800, seed=seed)
            code = extract_code(raw)
            res = run_python(code, config.CODE_TIMEOUT)
            sols, feedback = self._evaluate(res, ctx.spec)
        if sols is None:
            log.info("code unusable: %s", feedback[:200])
            return
        if len(sols) == 1:
            self._add(sols[0], ctx.spec, "code")
        else:
            log.info("code found %d valid answers: %s", len(sols), sols)
            self.ambig.append(sols)

    def _cot_attempt(self, llm, ctx, temp, seed):
        raw = llm.chat(prompts.COT_SYSTEM, prompts.cot_user(ctx.block, ctx.fmt, ctx.strategy),
                       think=True, temperature=temp, num_predict=6000, seed=seed)
        self._add(raw, ctx.spec, "cot")

    def _judge(self, llm, ctx, groups):
        cand_list = [(g[0].answer, "+".join(sorted({c.source for c in g}))) for g in groups.values()]
        pool = self._ambiguity_pool()
        known = {key_of(a) for a, _ in cand_list}
        cand_list += [(a, "enumeration") for a in pool if key_of(a) not in known]
        try:
            raw = llm.chat(prompts.JUDGE_SYSTEM,
                           prompts.judge_user(ctx.block, ctx.fmt, cand_list, pool),
                           think=True, temperature=0.3, num_predict=6000, seed=99)
        except BudgetExceeded:
            raise
        except LLMError as exc:
            log.warning("judge failed: %s", exc)
            return None
        ans = clean_answer(raw, ctx.spec)
        log.info("judge -> %r", ans)
        return ans

    def _disambiguate(self, llm, ctx, options):
        """Choose one answer from a solution set the exhaustive programs agree on."""
        if not options or self.disambig_tried:
            return None
        self.disambig_tried = True
        allowed = {key_of(o): o for o in options}
        cot_votes = [c.answer for c in self.cands if c.key in allowed]
        for n in range(max(0, config.DISAMBIG_CALLS)):
            if llm.calls_left() < 1 or llm.time_left() < 10:
                break
            try:
                raw = llm.chat(prompts.DISAMBIG_SYSTEM,
                               prompts.disambig_user(ctx.block, ctx.fmt, options),
                               think=True, temperature=0.3 + 0.3 * n, num_predict=6000, seed=200 + n)
            except BudgetExceeded:
                break
            except LLMError as exc:
                log.warning("disambiguation failed: %s", exc)
                continue
            ans = clean_answer(raw, ctx.spec)
            log.info("disambiguation %d among %s -> %r", n + 1, options, ans)
            if ans is not None and key_of(ans) in allowed:
                self.cands.append(Candidate(allowed[key_of(ans)], key_of(ans), "disambig",
                                            WEIGHT["disambig"]))
                return allowed[key_of(ans)]
        if cot_votes:  # an independent solver already picked one of the options
            return max(set(cot_votes), key=cot_votes.count)
        return None

    def _direct(self, llm, ctx):
        try:
            raw = llm.chat(prompts.DIRECT_SYSTEM, f"{ctx.block}\n\n{ctx.fmt}",
                           think=False, temperature=0.2, num_predict=300)
        except LLMError:
            return
        self._add(raw, ctx.spec, "direct")

    # --------------------------------------------------------------- voting
    def _groups(self):
        groups = OrderedDict()
        for c in self.cands:
            groups.setdefault(c.key, []).append(c)
        return groups

    def _consensus(self):
        groups = self._groups()
        scored = sorted(((sum(c.weight for c in g), len(g), g) for g in groups.values()),
                        key=lambda t: (t[0]), reverse=True)
        if not scored or scored[0][1] < 2:
            return None
        if len(scored) > 1 and scored[1][1] >= 2 and scored[1][0] >= scored[0][0]:
            return None  # contested, keep sampling
        return scored[0][2][0].answer

    def _confirmed_ambiguity(self):
        """A solution set reported by two independent programs (and no unique answer agreed)."""
        seen = {}
        for sols in self.ambig:
            k = tuple(sorted(key_of(s) for s in sols))
            seen.setdefault(k, []).append(sols)
            if len(seen[k]) >= 2:
                return seen[k][0]
        return None

    def _ambiguity_pool(self):
        """All answers found by ambiguous enumerations, most frequently reported first."""
        counts, first = {}, {}
        for sols in self.ambig:
            for s in sols:
                k = key_of(s)
                counts[k] = counts.get(k, 0) + 1
                first.setdefault(k, s)
        return [first[k] for k in sorted(counts, key=lambda k: -counts[k])]

    def _ambiguous_fallback(self):
        pool = self._ambiguity_pool()
        return pool[0] if pool else None

    def _best(self):
        groups = self._groups()
        if not groups:
            return None
        best = max(groups.values(), key=lambda g: sum(c.weight for c in g))
        return best[0].answer

    def _best_loose(self):
        if not self.loose:
            return None
        counts = {}
        for s in self.loose:
            counts[s] = counts.get(s, 0) + 1
        return max(counts, key=counts.get)
