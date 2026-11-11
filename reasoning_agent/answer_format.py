"""Answer-format specification, normalisation and validation.

The final `answer` must contain ONLY the requested value, so every candidate
goes through `clean_answer` which strips decoration and validates it against
the format that the problem asks for (integer / number / letter / code /
closed choice list / pattern with placeholders / free text).

The format comes from two sources that are merged in `build_spec`:
  * `detect_format`  - deterministic rules over the problem text (exact copies of
                       templates and option lists, "only the integer", "3-digit code" ...);
  * the LLM analyst  - JSON guess, accepted only where it is grounded in the problem.
"""
import ast
import re
from dataclasses import dataclass, field

from .task_types import days_to_english, normalize_text

TYPES = {"integer", "number", "letter", "code", "choice", "template", "text"}
_DIGITS = str.maketrans("۰۱۲۳۴۵۶۷۸۹٠١٢٣٤٥٦٧٨٩٫٬", "01234567890123456789.,")
_PLACEHOLDER = re.compile(r"<[^<>]*>")


def ascii_digits(s):
    return (s or "").translate(_DIGITS)


def squash(s):
    return re.sub(r"\s+", " ", s or "").strip()


def key_of(s):
    return re.sub(r"\s+", "", s or "").casefold()


@dataclass
class Spec:
    answer_type: str = "text"
    choices: list = field(default_factory=list)
    template: str = ""
    instruction: str = ""
    problem: str = ""
    code_length: int = 0


# ---------------------------------------------------------------- templates
def sanitize_template(t):
    t = (t or "").strip().strip("`\"'")
    if t.rfind("<") > t.rfind(">"):
        t += ">"
    if t.startswith("<") and t.count("<") > t.count(">"):
        t = t[1:]
    return t


def _split_template(t):
    parts, pos = [], 0
    for m in _PLACEHOLDER.finditer(t):
        if m.start() > pos:
            parts.append(("lit", t[pos:m.start()]))
        parts.append(("ph", m.group(0)[1:-1].strip().lower()))
        pos = m.end()
    if pos < len(t):
        parts.append(("lit", t[pos:]))
    return parts


def _ph_regex(name):
    if "int" in name:
        return r"[-+]?\d+"
    if "num" in name or "float" in name or "decimal" in name:
        return r"[-+]?\d+(?:\.\d+)?"
    if "letter" in name or "char" in name:
        return r"[A-Za-z]"
    if "digit" in name:
        return r"\d+"
    return r".+?"


def _lit_regex(val):
    toks = val.split()
    return r"\s*" + r"\s*".join(re.escape(t) for t in toks) + r"\s*" if toks else r"\s*"


def _template_regex(parts, optional_edges=False):
    """Regex for a template; with optional_edges the leading/trailing punctuation of the
    pattern (e.g. the outer '<' and '>') may be missing in the candidate."""
    rx = []
    last = len(parts) - 1
    for i, (kind, val) in enumerate(parts):
        if kind != "lit":
            rx.append("(" + _ph_regex(val) + ")")
            continue
        if optional_edges and i == 0:
            m = re.match(r"^(\s*[^\w\s]+)(.*)$", val, re.S)
            if m:
                rx.append(r"(?:" + _lit_regex(m.group(1)) + r")?" + _lit_regex(m.group(2)))
                continue
        if optional_edges and i == last:
            m = re.match(r"^(.*?)([^\w\s]+\s*)$", val, re.S)
            if m:
                rx.append(_lit_regex(m.group(1)) + r"(?:" + _lit_regex(m.group(2)) + r")?")
                continue
        rx.append(_lit_regex(val))
    return re.compile("".join(rx), re.I)


def _match_template(s, template):
    """Match `s` against the template and rebuild it with the template's exact literals.

    A second, lenient pass tolerates a missing wrapper such as the outer '<' ... '>' of
    '<Nima=<integer>, Sara=<integer>>', so 'Nima=3, Sara=4' is restored to '<Nima=3, Sara=4>'.
    """
    parts = _split_template(template)
    for lenient in (False, True):
        rx = _template_regex(parts, optional_edges=lenient)
        m = rx.fullmatch(s.strip()) or rx.search(s)
        if m:
            groups = list(m.groups())
            out = []
            for kind, val in parts:
                out.append(val if kind == "lit" else groups.pop(0).strip())
            return "".join(out)
    return None


# ------------------------------------------------- deterministic format rules
_NESTED_TMPL = re.compile(r"<[^<>]*?(?:<[A-Za-z_ ]{1,20}>[^<>]*?)+>")
_FLAT_TMPL = re.compile(
    r"[A-Za-z_][\w\-]*\s*[=:]\s*<[A-Za-z_ ]{1,20}>"
    r"(?:\s*[,;]\s*[A-Za-z_][\w\-]*\s*[=:]\s*<[A-Za-z_ ]{1,20}>)*")
_CHOICE_CUE = re.compile(  # an answer verb must sit next to the list, so "one of the
    # statements below is true: A: ..." is not mistaken for an option list
    r"(?:یکی از[^:：\n]{0,60}?(?:برگردانید|بنویسید|انتخاب کنید|بازگردانید)"
    r"|(?:return|answer|output|reply with|choose|select|pick)[^:：\n]{0,40}?(?:one of|from)[^:：\n]{0,30})"
    r"\s*[:：]\s*(.+)",
    re.I | re.S)
_TAIL_NOISE = re.compile(r"\s*(?:را\s*)?(?:برگردانید|بنویسید|انتخاب کنید)\s*\.?\s*$")
_NUM_WORDS = {"یک": 1, "دو": 2, "سه": 3, "چهار": 4, "پنج": 5, "شش": 6, "هفت": 7, "هشت": 8,
              "نه": 9, "ده": 10, "one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6,
              "seven": 7, "eight": 8, "nine": 9, "ten": 10}


def _split_options(seg):
    seg = _TAIL_NOISE.sub("", seg.strip())
    seg = re.split(r"\n\s*\n", seg)[0].strip()      # the list ends at a blank line
    for sep in (r"\s/\s|/", r"\|", r"\n"):
        parts = [p.strip(" \t-•*.") for p in re.split(sep, seg)]
        parts = [p for p in parts if p]
        if len(parts) >= 2:
            return parts
    parts = [p.strip(" \t-•*.") for p in re.split(r"\s*[,،]\s*|\s+(?:یا|or)\s+", seg)]
    return [p for p in parts if p]


def detect_format(problem):
    """Rule-based answer format, as a dict shaped like the analyst's JSON (may be partial)."""
    text = ascii_digits(problem or "")
    out = {}
    m = _NESTED_TMPL.search(text) or _FLAT_TMPL.search(text)
    if m:
        out["template"] = m.group(0).strip()
        out["answer_type"] = "template"
        return out

    m = _CHOICE_CUE.search(text)
    if m:
        opts = _split_options(m.group(1))
        if 2 <= len(opts) <= 12 and all(len(o) <= 40 for o in opts):
            out["choices"] = opts
            out["answer_type"] = "choice"
            return out

    norm = normalize_text(text)
    if re.search(r"فقط (?:یک )?حرف|only the letter|single letter|one letter", norm):
        out["answer_type"] = "letter"
    elif re.search(r"فقط (?:یک )?عدد صحیح|only the integer|integer only|"
                   r"return (?:only )?(?:the |an |one )?integer|whole number", norm):
        out["answer_type"] = "integer"
    elif re.search(r"(?:فقط|only|return)[^.\n]{0,25}(?:کد|رمز|\bcode\b|\bpin\b|password)", norm):
        out["answer_type"] = "code"
        m = (re.search(r"(\d+|[؀-ۿ]+)\s*رقمی", norm) or
             re.search(r"\b(\d+|one|two|three|four|five|six|seven|eight|nine|ten)[- ]digit", norm))
        if m:
            word = m.group(1)
            n = int(word) if word.isdigit() else _NUM_WORDS.get(word, 0)
            if n:
                out["code_length"] = n
    elif re.search(r"عدد صحیح|\binteger\b", norm):
        out["answer_type"] = "integer"
    return out


# --------------------------------------------------------------- spec build
def default_spec(problem):
    return Spec("text", [], "", "Only the final answer, exactly as the problem requests.", problem)


def _grounded_choices(raw, prob_norm):
    ch = []
    if isinstance(raw, list):
        for c in raw:
            c = squash(str(c))
            if c and key_of(c) in prob_norm and c not in ch:
                ch.append(c)
    return ch if len(ch) >= 2 else []


def _grounded_template(raw, prob_norm):
    tmpl = sanitize_template(str(raw or ""))
    if tmpl and _PLACEHOLDER.search(tmpl):
        lits = [v.strip() for k, v in _split_template(tmpl) if k == "lit" and v.strip()]
        if all(key_of(v) in prob_norm for v in lits):
            return tmpl
    return ""


def build_spec(data, problem, rules=None):
    """Create a Spec from the analyst's JSON plus the rule-based detection.

    Rules win for templates and option lists (they are exact copies of the problem text);
    anything the analyst claims is discarded unless it is grounded in the problem.
    """
    spec = default_spec(problem)
    rules = rules if isinstance(rules, dict) else {}
    data = data if isinstance(data, dict) else {}
    prob_norm = key_of(problem)

    atype = str(data.get("answer_type", "text")).strip().lower()
    spec.answer_type = atype if atype in TYPES else "text"
    instr = squash(str(data.get("format_instruction") or ""))
    if instr:
        spec.instruction = instr[:300]

    spec.choices = _grounded_choices(rules.get("choices"), prob_norm) or \
        _grounded_choices(data.get("choices"), prob_norm)
    spec.template = _grounded_template(rules.get("template"), prob_norm) or \
        _grounded_template(data.get("template"), prob_norm)

    rule_type = rules.get("answer_type")
    if spec.template:
        spec.answer_type = "template"
    elif spec.choices:
        spec.answer_type = "choice"
    elif rule_type in TYPES and rule_type not in ("choice", "template"):
        spec.answer_type = rule_type
    elif spec.answer_type in ("choice", "template"):
        spec.answer_type = "text"  # claimed but not grounded in the problem
    if spec.answer_type == "code":
        try:
            spec.code_length = max(0, int(rules.get("code_length") or data.get("code_length") or 0))
        except (TypeError, ValueError):
            spec.code_length = 0
    return spec


def describe_spec(spec):
    out = []
    if spec.instruction:
        out.append(f"Required answer format: {spec.instruction}")
    if spec.choices:
        out.append("The final answer must be EXACTLY one of these strings "
                   "(copy it character for character): " + " | ".join(spec.choices))
    if spec.template:
        out.append("The final answer must follow this exact pattern, replacing each "
                   f"<...> placeholder with a value (keep every other character, including "
                   f"any outer brackets): {spec.template}")
    if spec.answer_type == "integer":
        out.append("The final answer is one integer written with digits only (no units, no words).")
    elif spec.answer_type == "letter":
        out.append("The final answer is a single letter.")
    elif spec.answer_type == "code":
        n = f"{spec.code_length} " if spec.code_length else ""
        out.append(f"The final answer is the {n}character code written as one token without "
                   "spaces, commas or brackets (for example 123).")
    if not out:
        out.append("Print only the final answer, exactly as the problem requests.")
    return "\n".join(out)


# ---------------------------------------------------------------- cleaning
_MARKER = re.compile(r"(?:final\s+answer|answer|chosen|verified|پاسخ\s*نهایی|پاسخ)\s*[:：]\s*(.+)", re.I)


def _unwrap_literal(s):
    """'["C"]' -> 'C', '(2,)' -> '2'; longer lists are returned unchanged."""
    if not s or s[0] not in "[(" or s[-1] not in "])":
        return s
    try:
        val = ast.literal_eval(s)
    except Exception:
        return s
    if isinstance(val, (list, tuple)) and len(val) == 1 and not isinstance(val[0], (list, tuple, dict)):
        return str(val[0]).strip()
    return s


def loose_clean(raw):
    """Strip markers/markdown/quotes without validating against a format."""
    s = ascii_digits(raw or "").strip()
    if not s:
        return ""
    found = _MARKER.findall(s)
    if found:
        s = found[-1]
    else:
        lines = [ln for ln in s.splitlines() if ln.strip()]
        s = lines[-1] if lines else ""
    s = re.sub(r"\\boxed\{(.*)\}", r"\1", s)
    s = s.replace("**", "").replace("__", "").replace("`", "").replace("$", "")
    s = s.strip().strip("\"'“”«»").strip()
    s = re.sub(r"\.\s*$", "", s)
    s = _unwrap_literal(squash(s))
    return squash(s.strip("\"'“”«»"))


def _fmt_number(s, int_only):
    s = s.lstrip("+")
    if "." in s:
        a, b = s.split(".", 1)
        b = b.rstrip("0")
        if b:
            if int_only:
                val = float(s)
                if abs(val - round(val)) < 1e-6:  # floating artefact such as 599.9999999
                    return str(int(round(val)))
                return None
            return f"{a}.{b}"
        s = a
    if s in ("-0", ""):
        s = "0"
    return s


def _clean_number(s, int_only):
    s = s.strip()
    if re.fullmatch(r"[-+]?\d{1,3}(,\d{3})+(\.\d+)?", s):
        s = s.replace(",", "")
    if not int_only and re.fullmatch(r"[-+]?\d+/\d+", s):
        return s.lstrip("+")
    if not re.fullmatch(r"[-+]?\d+(\.\d+)?", s):
        nums = re.findall(r"[-+]?\d+(?:\.\d+)?", s)
        if len(nums) != 1:
            return None
        s = nums[0]
    return _fmt_number(s, int_only)


def _clean_code(s, length):
    """'4,1,3' / '[4, 1, 3]' / '(4 1 3)' / 'code 413' -> '413' (length checked when known)."""
    toks = re.findall(r"[A-Za-z0-9]+", s)
    if not toks:
        return None
    if len(toks) == 1:
        code = toks[0]
    elif all(len(t) == 1 for t in toks):
        code = "".join(toks)
    else:
        sized = [t for t in toks if length and len(t) == length]
        if len(sized) != 1:
            return None
        code = sized[0]
    if length and len(code) != length:
        return None
    return code.upper() if code.isalpha() else code


def clean_answer(raw, spec):
    """Return the normalised answer if it satisfies the spec, else None."""
    s = loose_clean(raw)
    if not s:
        return None
    t = spec.answer_type
    if t == "choice" and spec.choices:
        for probe in (s, days_to_english(s)):  # 'سه‌شنبه' answers an English day list
            ns = key_of(probe)
            for c in spec.choices:
                if key_of(c) == ns:
                    return c
            hits = [c for c in spec.choices if key_of(c) in ns]
            if hits:
                longest = max(hits, key=lambda c: len(key_of(c)))
                if all(key_of(h) in key_of(longest) for h in hits):
                    return longest
        return None
    if t == "template" and spec.template:
        return _match_template(s, spec.template)
    if t == "integer":
        return _clean_number(s, True)
    if t == "number":
        return _clean_number(s, False)
    if t == "code":
        return _clean_code(s, spec.code_length)
    if t == "letter":
        if len(s) == 1 and s.isalpha():
            letter = s
        else:
            toks = set(re.findall(r"(?<![A-Za-z])[A-Za-z](?![A-Za-z])", s))
            if len(toks) != 1:
                return None
            letter = toks.pop()
        up = letter.upper()
        if re.search(rf"(?<![A-Za-z]){up}(?![A-Za-z])", spec.problem) and \
                not re.search(rf"(?<![A-Za-z]){letter.lower()}(?![A-Za-z])", spec.problem):
            return up
        return letter
    return s
