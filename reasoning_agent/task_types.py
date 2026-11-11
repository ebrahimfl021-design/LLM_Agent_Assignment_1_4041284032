"""Task-type router.

Recognises which reasoning skill a problem needs (ten families: algebra, ordering,
logical inference, temporal/critical path, sets, multi-step quantities, assignment,
truth-teller/liar, grid shortest path, multi-constraint codes) and supplies a solving
METHOD for it.  Classification is deterministic keyword scoring (zero LLM calls); the
analyst's own guess only breaks ties.  Strategies describe methods, never answers, so
they generalise to unseen problems of the same family.
"""
import re
from dataclasses import dataclass


@dataclass(frozen=True)
class TaskType:
    name: str
    title: str
    keywords: tuple
    strategy: str


TASK_TYPES = (
    TaskType(
        "algebra", "numeric reasoning / algebra",
        (r"در مجموع", r"مجموعا", r"روی هم", r"دو برابر", r"سه برابر", r"نصف", r"بیشتر از",
         r"کمتر از", r"هرکدام چند", r"هر کدام چند", r"in total", r"altogether", r"twice",
         r"times as (many|much)", r"more than", r"fewer than", r"less than", r"half of",
         r"how many does each"),
        "Introduce one unknown per quantity and write EVERY sentence as an equation. Solve exactly: "
        "either enumerate integer values with itertools.product over a range bounded by the stated "
        "total, or solve the linear system with fractions.Fraction. Check every equation with the "
        "values found. Fill an answer template in the order in which the template lists the names."),
    TaskType(
        "ordering", "ordering / sequencing constraints",
        (r"جایگاه", r"ترتیب", r"قبل از", r"بعد از", r"بلافاصله", r"اولین", r"آخرین", r"position",
         r"\border\b", r"before", r"after", r"immediately", r"\bfirst\b", r"\blast\b", r"sequence",
         r"\bslot"),
        "Positions are 1..n. Use itertools.permutations over all items and test every constraint "
        "literally: 'X before Y' -> pos[X] < pos[Y]; 'A immediately after C' -> pos[A] == pos[C] + 1; "
        "'X is not first or last' -> 1 < pos[X] < n. Collect EVERY valid order, derive the asked "
        "value from each valid order and list all distinct values in SOLUTIONS."),
    TaskType(
        "logic", "logical inference over statements",
        (r"گزاره", r"درست است", r"نادرست", r"دقیقا یکی از", r"دقیقا دو", r"statement",
         r"exactly one of", r"exactly two of", r"\btrue\b", r"\bfalse\b", r"if and only if"),
        "Enumerate every possible world (for example each box the object can be in). In each world "
        "evaluate every statement as a Python boolean, count the true ones and keep only the worlds "
        "that match the stated count. Report the asked value of every surviving world."),
    TaskType(
        "temporal", "temporal reasoning / scheduling with durations",
        (r"روز زمان", r"زمان می ?برد", r"طول می ?کشد", r"پس از پایان", r"بعد از پایان", r"هم ?زمان",
         r"زودترین", r"دیرترین", r"duration", r"takes \d+", r"after .{0,30}finish", r"in parallel",
         r"earliest", r"latest", r"critical path", r"deadline", r"\bhours?\b", r"\bdays?\b"),
        "Model the activities as a DAG (activity -> duration, prerequisites). earliest_start = max "
        "of the earliest_finish of all prerequisites (0 when it has none); earliest_finish = "
        "earliest_start + duration. Activities without a dependency run in parallel. The project "
        "finish is the max earliest_finish unless another moment is asked for."),
    TaskType(
        "sets", "sets and counting",
        (r"هر دو", r"هردو", r"هیچ ?کدام", r"هیچکدام", r"حداقل یکی", r"فقط یکی از", r"\bboth\b",
         r"neither", r"none of", r"at least one", r"only one of", r"venn", r"\bunion\b",
         r"intersection"),
        "Use inclusion-exclusion: |A or B| = |A| + |B| - |A and B| (add the triple term for three "
        "sets); 'neither' = total - |A or B|; 'only A' = |A| - |A and B|. Compute with integers "
        "and assert that every region is non-negative."),
    TaskType(
        "quant", "multi-step quantitative reasoning",
        (r"%", r"٪", r"درصد", r"تخفیف", r"افزایش", r"کاهش", r"سود", r"زیان", r"قیمت", r"نسبت",
         r"percent", r"discount", r"increase", r"decrease", r"price", r"profit", r"interest",
         r"\bratio\b", r"\brate\b"),
        "Apply each change to the CURRENT value, in the stated order: successive percentages "
        "multiply (1 + p/100) or (1 - p/100) and never simply add. To recover an original value "
        "divide the final value by the product of the factors. Use fractions.Fraction and print an "
        "integer when the result is integral."),
    TaskType(
        "assignment", "assignment with constraints",
        (r"هرکدام در یکی از", r"هر کدام در یکی از", r"هیچ دو نفری", r"روزهای", r"شنبه", r"جمعه",
         r"monday", r"tuesday", r"wednesday", r"thursday", r"friday", r"no two", r"each .{0,40}one of",
         r"\bassign", r"زودتر از", r"دیرتر از", r"earlier than", r"later than"),
        "Map the people to the slots with itertools.permutations (every person gets a distinct "
        "slot) using EXACTLY the slot list of the problem (see the glossary for Persian day ranges). "
        "'earlier' means a smaller slot index, 'exactly one day after' means index + 1. Collect all "
        "valid assignments and report the asked value for each."),
    TaskType(
        "knights", "truth-tellers and liars",
        (r"راستگو", r"دروغگو", r"راست می ?گوید", r"دروغ می ?گوید", r"truth-?teller", r"\bliars?\b",
         r"knights?", r"knaves?", r"always lies", r"always tells the truth"),
        "Enumerate every Truth/Lie assignment with itertools.product. A person is consistent when "
        "the truth value of their statement == (they are a truth-teller). Self-referential "
        "statements ('I and X are of different types') are evaluated on the assignment being "
        "tested. Keep only the assignments where every person is consistent."),
    TaskType(
        "grid", "grid planning / shortest path",
        (r"شبکه", r"مختصات", r"ربات", r"مسدود", r"حرکت", r"\bgrid\b", r"robot", r"coordinates?",
         r"blocked", r"obstacle", r"shortest path", r"\bmoves?\b", r"\bsteps?\b"),
        "Run BFS over cells (x, y) from the start; one move = one step up/down/left/right (unless "
        "other moves are allowed). Skip blocked cells and cells outside the bounds; when a lower "
        "bound is not stated, coordinates are not negative. The answer is the number of moves on "
        "the shortest path (the start cell counts as 0 moves)."),
    TaskType(
        "code", "combination of several constraints (codes, digits)",
        (r"رقم", r"رقمی", r"\bکد\b", r"رمز", r"\bdigits?\b", r"\bcode\b", r"combination",
         r"password", r"\bpin\b"),
        "Enumerate candidate codes with itertools.permutations (distinct digits) or "
        "itertools.product (repetition allowed) over the allowed digit set. Digit positions are "
        "1-based from the left. Apply EVERY constraint literally, collect all codes that satisfy "
        "all of them and print each code as its digits concatenated without separators."),
)

BY_NAME = {t.name: t for t in TASK_TYPES}
GENERIC_STRATEGY = ("Translate every sentence into an exact condition, prefer exhaustive search or "
                    "direct computation, and verify the result against every condition.")


def normalize_text(text):
    """Lower-case, unify Arabic/Persian letter variants and drop ZWNJ / diacritics."""
    t = (text or "").replace("‌", " ").replace("‏", " ").replace("‎", " ")
    t = t.replace("ي", "ی").replace("ك", "ک").replace("ة", "ه").replace("أ", "ا").replace("إ", "ا")
    t = t.replace("ً", "").replace("ٌ", "").replace("ٍ", "").replace("ّ", "")
    return re.sub(r"\s+", " ", t).strip().lower()


def score(problem):
    """Keyword score per task type (count of distinct matching keyword patterns)."""
    text = normalize_text(problem)
    out = {}
    for t in TASK_TYPES:
        hits = sum(1 for kw in t.keywords if re.search(kw, text))
        if hits:
            out[t.name] = hits
    return out


def classify(problem, analyst_guess=None, top=2):
    """Return up to `top` task-type names, best first.

    A type is kept when it scores at least half of the best score; the analyst's guess is
    promoted to the front when it is a known type that the keywords also support, and it is
    used alone when the keywords find nothing.
    """
    scores = score(problem)
    guess = analyst_guess if analyst_guess in BY_NAME else None
    if not scores:
        return [guess] if guess else []
    best = max(scores.values())
    ranked = sorted(scores, key=lambda n: (-scores[n], [t.name for t in TASK_TYPES].index(n)))
    ranked = [n for n in ranked if scores[n] * 2 >= best]
    if guess:
        if guess in ranked:
            ranked.remove(guess)
        ranked.insert(0, guess)
    return ranked[:top]


def strategy_text(names):
    """Strategy hints for the given task types, ready to paste into a prompt."""
    lines = []
    for n in names:
        t = BY_NAME.get(n)
        if t:
            lines.append(f"- [{t.title}] {t.strategy}")
    if not lines:
        lines.append(f"- {GENERIC_STRATEGY}")
    return "Solving strategy for this kind of problem:\n" + "\n".join(lines)


# ------------------------------------------------------------------ glossary
_WEEK = (  # Persian week order, Saturday first
    (r"شنبه", "Saturday"), (r"یک ?شنبه", "Sunday"), (r"دو ?شنبه", "Monday"),
    (r"سه ?شنبه", "Tuesday"), (r"چهار ?شنبه", "Wednesday"), (r"پنج ?شنبه", "Thursday"),
    (r"جمعه", "Friday"),
)
_PREFIX = r"(?<![؀-ۿ])"   # not preceded by another Persian letter


def _day_regex(i):
    pat = _WEEK[i][0]
    if i == 0:  # bare "شنبه" must not be the tail of "دوشنبه", "پنج شنبه", ...
        return _PREFIX + r"(?<!یک )(?<!دو )(?<!سه )(?<!چهار )(?<!پنج )" + pat
    return _PREFIX + pat


def persian_days(text):
    """Indices (0=Saturday) of the Persian weekday names mentioned, in order of appearance."""
    t = normalize_text(text)
    found = []
    for i in range(len(_WEEK)):
        for m in re.finditer(_day_regex(i), t):
            found.append((m.start(), i))
    seen, out = set(), []
    for _, i in sorted(found):
        if i not in seen:
            seen.add(i)
            out.append(i)
    return out


def days_to_english(text):
    """Replace Persian weekday names by their English names (longest names first)."""
    t = normalize_text(text)
    for i in sorted(range(len(_WEEK)), key=lambda k: -len(_WEEK[k][0])):
        t = re.sub(_day_regex(i), _WEEK[i][1], t)
    return t


def glossary(problem):
    """Exact meaning of Persian weekday names and 'X تا Y' day ranges found in the problem."""
    days = persian_days(problem)
    if not days:
        return ""
    t = normalize_text(problem)
    lines = ["Glossary (Persian week order starts on Saturday): " +
             ", ".join(f"{_WEEK[i][0].replace(' ?', '')} = {_WEEK[i][1]}" for i in days)]
    for a in range(len(_WEEK)):
        for b in range(len(_WEEK)):
            if a == b:
                continue
            if re.search(_day_regex(a) + r"\s*(?:تا|الی)\s*" + _day_regex(b)[len(_PREFIX):], t):
                span = [_WEEK[k][1] for k in (range(a, b + 1) if a < b else
                                              list(range(a, 7)) + list(range(0, b + 1)))]
                lines.append(f"'{_WEEK[a][1]} to {_WEEK[b][1]}' means exactly these {len(span)} "
                             f"consecutive days: {', '.join(span)}")
    return "\n".join(lines)
