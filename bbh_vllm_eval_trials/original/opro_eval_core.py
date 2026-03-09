import re


# ============================================================================
# Vendored OPRO-style evaluation core (metrics + prompting-again helpers)
# ----------------------------------------------------------------------------
# This module is intentionally isolated from local vLLM / BBH I/O.
# The adapter layer should only pass raw predictions/targets in and out.
# ============================================================================

FINAL_ANSWER_BEHIND_PATTERNS_PRIMARY = ["answer is ", "answer: ", "answer is: "]
FINAL_ANSWER_BEHIND_PATTERNS_SECONDARY = [" is: ", " are: "]
FINAL_ANSWER_AHEAD_PATTERNS = [
    " is the correct answer",
    " is the right answer",
    " is the final answer",
    " is the answer",
]
NEXT_QUESTION_DELIMITERS = ["\nq:", "\nquestion:", "\n\nq:", "\n\nquestion:"]
PROMPTING_AGAIN_SUFFIX = "So the final answer is"


def _strip_short_answer_wrappers(value):
    prev = None
    s = str(value)
    while s != prev:
        prev = s
        s = s.strip()
        s = s.strip("`*")
        s = s.strip()
        if len(s) >= 2 and ((s[0], s[-1]) in {("'", "'"), ('"', '"'), ("(", ")"), ("[", "]")}):
            s = s[1:-1].strip()
    return s


def _strip_trailing_short_answer_punct(value):
    return str(value).rstrip(" \t\r\n.,;:!?)]}\"'")


def _extract_bracketed_choice_from_string(value):
    lower = str(value).lower()
    matches = re.findall(r"\([a-z]\)", lower)
    unique = sorted(set(matches))
    if len(unique) == 1:
        return unique[0]
    return lower


def _strip_final_answer_indicators(value):
    s = str(value).lower().strip()
    for delim in NEXT_QUESTION_DELIMITERS:
        s = s.split(delim)[0]

    primary_found = any(pat in s for pat in FINAL_ANSWER_BEHIND_PATTERNS_PRIMARY)
    behind_patterns = (
        FINAL_ANSWER_BEHIND_PATTERNS_PRIMARY
        if primary_found
        else FINAL_ANSWER_BEHIND_PATTERNS_SECONDARY
    )

    answer_indicated = False
    for pat in behind_patterns:
        if pat in s:
            s = s.split(pat)[-1]
            answer_indicated = True

    for pat in FINAL_ANSWER_AHEAD_PATTERNS:
        if pat in s:
            s = s.split(pat)[0]
            answer_indicated = True

    s = s.strip()
    # Matches OPRO-style cleanup used around evaluation strings.
    if "this is the solution:" in s:
        s = s.split("this is the solution:")[0].strip()
    while s.endswith("."):
        s = s[:-1]
    while s.endswith("'"):
        s = s[:-1]
    return s.strip(), answer_indicated


def _normalize_opro_style_base(value):
    s = _strip_short_answer_wrappers(value)
    s, answer_indicated = _strip_final_answer_indicators(s)
    s = _strip_short_answer_wrappers(s)
    return s.strip(), answer_indicated


def extract_second_round_answer_for_scoring(text):
    return str(text).strip(":").strip().split("\n")[0].split("Q:")[0]


def _normalize_multiple_choice_text(raw_value, valid_letters):
    s, _ = _normalize_opro_style_base(raw_value)
    if not s:
        return ""

    choice_text = _extract_bracketed_choice_from_string(s)
    m = re.fullmatch(r"\(([a-z])\)", choice_text)
    if m:
        candidate = m.group(1).upper()
        return candidate if candidate in valid_letters else ""

    choice_text = _strip_trailing_short_answer_punct(choice_text)
    if len(choice_text) == 1 and choice_text.upper() in valid_letters:
        return choice_text.upper()
    return ""


def _normalize_numeric_text(raw_value):
    s, answer_indicated = _normalize_opro_style_base(raw_value)
    if not s:
        return ""
    raw = s
    for c in [",", "$", "%", "€", "£"]:
        raw = raw.replace(c, "")
    numeric_tokens = re.findall(r"-?\d+(?:\.\d+)?", raw)
    if numeric_tokens:
        chosen = numeric_tokens[0] if answer_indicated else numeric_tokens[-1]
        if chosen.endswith(".0"):
            chosen = chosen[:-2]
        if chosen.startswith("+"):
            chosen = chosen[1:]
        return chosen
    return _strip_trailing_short_answer_punct(raw.strip())


def _normalize_bool_text(raw_value):
    s, _ = _normalize_opro_style_base(raw_value)
    low = _strip_trailing_short_answer_punct(s).lower().strip()
    if not low:
        return ""
    # Conservative boolean normalization to stay close to OPRO-style metrics usage.
    if low in {"yes", "true", "1", "1.0"}:
        return "Yes"
    if low in {"no", "false", "0", "0.0"}:
        return "No"
    return ""


def normalize_prediction(
    prediction,
    *,
    is_multiple_choice=False,
    valid_letters=None,
    treat_as_number=False,
    treat_as_bool=False,
):
    if is_multiple_choice:
        if valid_letters is None:
            valid_letters = set("ABCDE")
        return _normalize_multiple_choice_text(prediction, valid_letters)
    if treat_as_number:
        return _normalize_numeric_text(prediction)
    if treat_as_bool:
        return _normalize_bool_text(prediction)

    s, _ = _normalize_opro_style_base(prediction)
    return _strip_trailing_short_answer_punct(s)


def normalize_target(
    target,
    *,
    is_multiple_choice=False,
    valid_letters=None,
    treat_as_number=False,
    treat_as_bool=False,
):
    s = str(target).strip()
    if len(s) == 3 and s.startswith("(") and s.endswith(")"):
        s = s[1:-1]

    if is_multiple_choice:
        if valid_letters is None:
            valid_letters = set("ABCDE")
        s = _strip_short_answer_wrappers(s).strip().upper()
        return s if s in valid_letters else s
    if treat_as_number:
        return _normalize_numeric_text(s)
    if treat_as_bool:
        return _normalize_bool_text(s)
    return _strip_trailing_short_answer_punct(s)


def get_accuracy_of_list(
    predictions,
    targets,
    *,
    is_multiple_choice=False,
    valid_letters=None,
    prediction_treat_as_number=False,
    prediction_treat_as_bool=False,
    target_treat_as_number=False,
    target_treat_as_bool=False,
):
    if len(predictions) != len(targets):
        raise ValueError(f"Length mismatch: {len(predictions)} != {len(targets)}")

    normalized_predictions = []
    normalized_targets = []
    correct = 0
    parse_failures = 0

    for pred, tgt in zip(predictions, targets):
        n_pred = normalize_prediction(
            pred,
            is_multiple_choice=is_multiple_choice,
            valid_letters=valid_letters,
            treat_as_number=prediction_treat_as_number,
            treat_as_bool=prediction_treat_as_bool,
        )
        n_tgt = normalize_target(
            tgt,
            is_multiple_choice=is_multiple_choice,
            valid_letters=valid_letters,
            treat_as_number=target_treat_as_number,
            treat_as_bool=target_treat_as_bool,
        )

        normalized_predictions.append(n_pred)
        normalized_targets.append(n_tgt)

        if not n_pred or not n_tgt:
            parse_failures += 1
            continue
        if n_pred == n_tgt:
            correct += 1

    total = len(targets)
    accuracy = float(correct) / float(total) if total > 0 else 0.0
    parsed_count = total - parse_failures
    parsed_rate = float(parsed_count) / float(total) if total > 0 else 0.0
    return {
        "accuracy": accuracy,
        "correct": correct,
        "total": total,
        "parse_failures": parse_failures,
        "parsed_rate": parsed_rate,
        "normalized_predictions": normalized_predictions,
        "normalized_targets": normalized_targets,
    }


def append_prompting_again_suffix(stage1_prompt, stage1_answer):
    stage1_answer = str(stage1_answer or "")
    sep = "" if (not stage1_answer or stage1_answer.endswith((" ", "\n"))) else " "
    return f"{stage1_prompt}{stage1_answer}{sep}{PROMPTING_AGAIN_SUFFIX}"
