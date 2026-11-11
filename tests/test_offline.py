"""Offline tests (no Ollama needed): python -m unittest discover -s tests -v

They check the deterministic parts (format detection, normalisation, task routing,
SOLUTIONS parsing) and the whole pipeline with a scripted fake model. They do not
measure the accuracy of the real model.
"""
import json
import os
import sys
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from reasoning_agent import llm as llm_mod                      # noqa: E402
from reasoning_agent import Agent                                # noqa: E402
from reasoning_agent.answer_format import build_spec, clean_answer, detect_format  # noqa: E402
from reasoning_agent.sandbox import parse_solutions             # noqa: E402
from reasoning_agent.task_types import classify, days_to_english, glossary  # noqa: E402

with open(os.path.join(ROOT, "examples", "input.json"), encoding="utf-8") as fh:
    PUBLIC = {it["id"]: it["problem"] for it in json.load(fh)}


def spec_for(pid, analyst=None):
    p = PUBLIC[pid]
    return build_spec(analyst or {}, p, detect_format(p))


class FormatTests(unittest.TestCase):
    def test_answer_types_of_public_set(self):
        expected = {"p1_numeric": "template", "p2_scheduling": "letter", "p4_temporal": "integer",
                    "p5_sets": "integer", "p6_quant": "integer", "p7_assignment": "choice",
                    "p8_liar": "choice", "p9_grid": "integer", "p10_code": "code"}
        for pid, t in expected.items():
            self.assertEqual(spec_for(pid).answer_type, t, pid)
        self.assertEqual(spec_for("p10_code").code_length, 3)
        self.assertEqual(spec_for("p1_numeric").template, "<Nima=<integer>, Sara=<integer>, Reza=<integer>>")
        self.assertEqual(spec_for("p8_liar").choices, ["Ali=Truth, Bita=Truth", "Ali=Truth, Bita=Lie",
                                                        "Ali=Lie, Bita=Truth", "Ali=Lie, Bita=Lie"])
        self.assertEqual(spec_for("p7_assignment").choices, ["Monday", "Tuesday", "Wednesday", "Thursday"])
        # statements "A: ..." are not an option list
        self.assertNotEqual(spec_for("p3_logic").answer_type, "choice")

    def test_analyst_cannot_override_grounded_template(self):
        bad = {"answer_type": "template", "template": "Nima=<integer>, Sara=<integer>, Reza=<integer>"}
        self.assertEqual(spec_for("p1_numeric", bad).template, "<Nima=<integer>, Sara=<integer>, Reza=<integer>>")

    def test_normalisation(self):
        t = spec_for("p1_numeric")
        self.assertEqual(clean_answer("ANSWER: Nima=30, Sara=27, Reza=15", t), "<Nima=30, Sara=27, Reza=15>")
        self.assertEqual(clean_answer("<Nima=30,Sara=27,Reza=15>", t), "<Nima=30, Sara=27, Reza=15>")
        code = spec_for("p10_code")
        for raw in ("413", "(4, 1, 3)", "[4,1,3]", "ANSWER: 4 1 3", "**413**"):
            self.assertEqual(clean_answer(raw, code), "413", raw)
        self.assertIsNone(clean_answer("41", code))
        num = spec_for("p6_quant")
        self.assertEqual(clean_answer("599.99999999", num), "600")
        self.assertEqual(clean_answer("600.0", num), "600")
        self.assertEqual(clean_answer("[8]", spec_for("p5_sets")), "8")
        self.assertEqual(clean_answer("['C']", spec_for("p2_scheduling")), "C")
        day = spec_for("p7_assignment")
        self.assertEqual(clean_answer("ANSWER: سه‌شنبه", day), "Tuesday")
        self.assertEqual(clean_answer("monday", day), "Monday")
        liar = spec_for("p8_liar")
        self.assertEqual(clean_answer("Ali = Lie , Bita = Truth", liar), "Ali=Lie, Bita=Truth")

    def test_solutions_line(self):
        self.assertEqual(parse_solutions('x\nSOLUTIONS: ["413", "431"]\nANSWER: 413'), ["413", "431"])
        self.assertEqual(parse_solutions("SOLUTIONS: [(4, 1, 3), (4, 3, 1)]"), ["413", "431"])
        self.assertEqual(parse_solutions("SOLUTIONS: []"), [])
        self.assertIsNone(parse_solutions("ANSWER: 3"))


class RouterTests(unittest.TestCase):
    def test_each_public_problem_gets_its_family(self):
        expected = {"p1_numeric": "algebra", "p2_scheduling": "ordering", "p3_logic": "logic",
                    "p4_temporal": "temporal", "p5_sets": "sets", "p6_quant": "quant",
                    "p7_assignment": "assignment", "p8_liar": "knights", "p9_grid": "grid",
                    "p10_code": "code"}
        for pid, fam in expected.items():
            self.assertIn(fam, classify(PUBLIC[pid]), pid)

    def test_glossary(self):
        g = glossary(PUBLIC["p7_assignment"])
        self.assertIn("Monday, Tuesday, Wednesday, Thursday", g)
        self.assertIn("4 consecutive days", g)
        self.assertEqual(days_to_english("پنج‌شنبه"), "Thursday")
        self.assertEqual(days_to_english("شنبه"), "Saturday")


# ------------------------------------------------------------- fake model
class FakeOllama:
    """Replaces the `ollama` package: answers by the role found in the system prompt."""

    def __init__(self, script):
        self.script = script
        self.calls = []

    def Client(self, timeout=None):  # noqa: N802 (mimics ollama.Client)
        return self

    def chat(self, model=None, messages=None, options=None, think=None, format=None):
        system = messages[0]["content"]
        role = ("analyst" if "analyse a reasoning problem" in system else
                "code" if "writing a program" in system else
                "disambig" if "reviewing an exhaustive search" in system else
                "judge" if "strict verifier" in system else "cot")
        self.calls.append(role)
        reply = self.script[role]
        if callable(reply):
            reply = reply(messages[1]["content"])
        return {"message": {"content": reply}}


def program(solutions_expr, answer_expr):
    return ("```python\nimport json\n"
            f"sols = {solutions_expr}\n"
            "print('SOLUTIONS: ' + json.dumps(sols))\n"
            f"print('ANSWER: ' + str({answer_expr}))\n```")


class PipelineTests(unittest.TestCase):
    def run_agent(self, pid, script):
        fake = FakeOllama(script)
        saved = llm_mod.ollama
        llm_mod.ollama = fake
        try:
            agent = Agent(model="fake")
            return agent.solve(PUBLIC[pid]), fake.calls, agent
        finally:
            llm_mod.ollama = saved

    def test_template_answer_and_early_exit(self):
        brute = ("[f'<Nima={2*r}, Sara={r+12}, Reza={r}>' for r in range(73) if 2*r + r+12 + r == 72]")
        ans, calls, _ = self.run_agent("p1_numeric", {
            "analyst": json.dumps({"answer_type": "template", "task_type": "algebra",
                                   "template": "Nima=<integer>, Sara=<integer>, Reza=<integer>",
                                   "english": "Nima, Sara and Reza have 72 tokens in total ..."}),
            "code": program(brute, "sols[0]"), "cot": "ANSWER: <Nima=30, Sara=27, Reza=15>"})
        self.assertEqual(ans, "<Nima=30, Sara=27, Reza=15>")
        self.assertEqual(calls, ["analyst", "code", "code"])  # two programs agree -> stop

    def test_ambiguous_enumeration_goes_to_disambiguator(self):
        brute = ("[f'{a}{b}{a-b}' for a in range(1,5) for b in range(1,5) "
                 "if a > b and len({a, b, a-b}) == 3 and 0 < a-b < 5 and a + b + (a-b) == 8]")
        ans, calls, agent = self.run_agent("p10_code", {
            "analyst": json.dumps({"answer_type": "code", "task_type": "code", "english": "A three-digit code ..."}),
            "code": program(brute, "sols[0]"), "disambig": "Both fit; choosing.\nANSWER: 431",
            "cot": "ANSWER: 413"})
        self.assertEqual(agent.ambig[0], ["413", "431"])
        self.assertEqual(calls, ["analyst", "code", "code", "disambig"])
        self.assertEqual(ans, "431")  # whatever the disambiguator picks, as long as it is allowed

    def test_choice_with_disagreement_uses_judge(self):
        replies = iter(["ANSWER: Ali=Truth, Bita=Lie", "ANSWER: Ali=Lie, Bita=Truth"] * 5)
        ans, calls, _ = self.run_agent("p8_liar", {
            "analyst": "not json at all",
            "code": "```python\nprint(1/0)\n```",            # crashes -> repair -> crashes
            "cot": lambda _u: next(replies), "judge": "checked\nANSWER: Ali=Lie, Bita=Truth"})
        self.assertEqual(ans, "Ali=Lie, Bita=Truth")
        self.assertIn("judge", calls)
        self.assertLessEqual(len(calls), 10)

    def test_dead_model_still_returns_string(self):
        saved = llm_mod.ollama
        llm_mod.ollama = None
        try:
            self.assertEqual(Agent(model="x").solve(PUBLIC["p5_sets"]), "")
        finally:
            llm_mod.ollama = saved


if __name__ == "__main__":
    unittest.main()
