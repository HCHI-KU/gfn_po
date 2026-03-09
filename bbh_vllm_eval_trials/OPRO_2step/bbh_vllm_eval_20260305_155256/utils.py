import json
import re
import time
from pathlib import Path

import pandas as pd
from fastchat.model import get_conversation_template
from vllm import SamplingParams


TASK_EXTRACTOR_TEXT = {
    "tracking_shuffled_objects_five_objects": (
        "Therefore, the final answer is (output exactly one letter: A, B, C, D, or E; no explanation) "
    ),
    "object_counting": (
        "Therefore, the final answer is (output exactly one positive integer using digits only; no explanation) "
    ),
    "causal_judgement": (
        "Therefore, the final answer is (output exactly Yes or No; no explanation) "
    ),
    "movie_recommendation": (
        "Therefore, the final answer is (output exactly one letter: A, B, C, D, or E; no explanation) "
    ),
    "hyperbaton": "Therefore, the final answer is (output exactly one letter: A or B; no explanation) ",
}


TASK_STRICT_PATTERNS = {
    "tracking_shuffled_objects_five_objects": re.compile(r"^[A-E]$"),
    "object_counting": re.compile(r"^[0-9]+$"),
    "causal_judgement": re.compile(r"^(Yes|No)$"),
    "movie_recommendation": re.compile(r"^[A-E]$"),
    "hyperbaton": re.compile(r"^[A-B]$"),
}

# Paper-main decoding stabilization (strict parsing still used).
TASK_ANSWER_MAX_TOKENS = {
    "tracking_shuffled_objects_five_objects": 4,
    "object_counting": 1,
    "causal_judgement": 4,
    "movie_recommendation": 4,
    "hyperbaton": 4,
}

# OPRO-style second-round extraction is usually a short answer only.
# Keep this tighter than the first extraction to reduce verbose spillover.
TASK_OPRO_SECOND_ROUND_MAX_TOKENS = {
    "tracking_shuffled_objects_five_objects": 4,
    "object_counting": 8,
    "causal_judgement": 4,
    "movie_recommendation": 4,
    "hyperbaton": 4,
}

# Stop early on line breaks / turn markers during answer extraction.
ANSWER_EXTRACTION_STOPS = ["\n", "<|eot_id|>", "<end_of_turn>", "</s>"]

# OPRO-style answer indication patterns (parser-side normalization).
OPRO_FINAL_ANSWER_BEHIND_PATTERNS_PRIMARY = ["answer is ", "answer: ", "answer is: "]
OPRO_FINAL_ANSWER_BEHIND_PATTERNS_SECONDARY = [" is: ", " are: "]
OPRO_FINAL_ANSWER_AHEAD_PATTERNS = [
    " is the correct answer",
    " is the right answer",
    " is the final answer",
    " is the answer",
]
OPRO_NEXT_QUESTION_DELIMITERS = ["\nq:", "\nquestion:", "\n\nq:", "\n\nquestion:"]
OPRO_SECOND_ROUND_EXTRACTION_SUFFIX = "So the final answer is"
INVALID_PARSED_ANSWER = "__INVALID_PARSED_ANSWER__"


PAPER_ACC = {
    "movie_recommendation": 0.57,
    "object_counting": 0.90,
    "tracking_shuffled_objects_five_objects": 0.70,
    "hyperbaton": 0.84,
    "causal_judgement": 0.57,
}


PAPER_OPT_PROMPTS = {
    "movie_recommendation": "Use movie ratings data available here above movies for reference. ThisHOFF has an interesting analysis based solely options to options based movies ratings expect from the other movies you are asked ones mentioned here you.",
    "object_counting": "Use only addition. Add think step by step. Finally give the actual correct answer.",
    "tracking_shuffled_objects_five_objects": "Use this process as an explanation stepwise for each step until you get to as given above Alice has got originaly the following as follows.",
    "hyperbaton": "Use the reasoning and examples you would step. Finally give the actual correct answer.",
    "causal_judgement": "Use causal diagram. The correct option ask about whether there the variable C of about whether a specific cause is sufficient. The answer a causal relationship between C to D if the probability P that C occurs given E changes.",
}


def ensure_dir(path):
    Path(path).mkdir(parents=True, exist_ok=True)


def save_json(path, payload):
    path = Path(path)
    ensure_dir(path.parent)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def append_jsonl(path, row):
    path = Path(path)
    ensure_dir(path.parent)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")


def load_json(path):
    with Path(path).open("r", encoding="utf-8") as f:
        return json.load(f)


def remove_parentheses_if_single_char(text):
    text = str(text).strip()
    if text.startswith("(") and text.endswith(")") and len(text) == 3:
        return text[1:-1]
    return text


def normalize_answer(text):
    return remove_parentheses_if_single_char(str(text).strip())


def _strip_short_answer_wrappers(value):
    prev = None
    while value != prev:
        prev = value
        value = value.strip()
        value = value.strip("`*")
        value = value.strip()
        if len(value) >= 2 and ((value[0], value[-1]) in {("'", "'"), ('"', '"'), ("(", ")"), ("[", "]")}):
            value = value[1:-1].strip()
    return value


def _strip_trailing_short_answer_punct(value):
    return value.rstrip(" \t\r\n.,;:!?)]}\"'")


def _extract_bracketed_choice_from_string(value):
    """OPRO-style bracketed choice extraction, e.g., '(A) apple' -> '(a)'."""
    lower = value.lower()
    matches = re.findall(r"\([a-z]\)", lower)
    unique = sorted(set(matches))
    if len(unique) == 1:
        return unique[0]
    return lower


def _normalize_text_for_option_match(text):
    s = str(text).lower()
    s = s.replace("&", " and ")
    s = re.sub(r"[^a-z0-9\s]", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def _extract_option_map_from_goal(goal_text):
    """Extract {(A): text, ...} from a BBH goal string containing 'Options:'."""
    goal = str(goal_text)
    if "Options:" not in goal:
        return {}
    options_part = goal.split("Options:", 1)[1]
    matches = list(re.finditer(r"\(([A-E])\)\s*", options_part))
    if not matches:
        return {}
    option_map = {}
    for idx, m in enumerate(matches):
        letter = m.group(1)
        start = m.end()
        end = matches[idx + 1].start() if idx + 1 < len(matches) else len(options_part)
        text = options_part[start:end].strip()
        text = text.rstrip(" .")
        option_map[letter] = text
    return option_map


def _strip_opro_answer_indicators(value):
    """Strip answer-indicating prefixes/suffixes similar to OPRO metrics."""
    s = value.lower().strip()

    for delim in OPRO_NEXT_QUESTION_DELIMITERS:
        s = s.split(delim)[0]

    primary_found = any(pat in s for pat in OPRO_FINAL_ANSWER_BEHIND_PATTERNS_PRIMARY)
    behind_patterns = (
        OPRO_FINAL_ANSWER_BEHIND_PATTERNS_PRIMARY
        if primary_found
        else OPRO_FINAL_ANSWER_BEHIND_PATTERNS_SECONDARY
    )

    answer_indicated = False
    for pat in behind_patterns:
        if pat in s:
            s = s.split(pat)[-1]
            answer_indicated = True

    for pat in OPRO_FINAL_ANSWER_AHEAD_PATTERNS:
        if pat in s:
            s = s.split(pat)[0]
            answer_indicated = True

    s = s.strip()
    s = s.split("this is the solution:")[0].strip()
    while s.endswith("."):
        s = s[:-1]
    while s.endswith("'"):
        s = s[:-1]

    return s.strip(), answer_indicated


def _normalize_opro_style_base(value):
    """Apply OPRO-style parser normalization before task-specific strict parsing."""
    s = _strip_short_answer_wrappers(value)
    s, answer_indicated = _strip_opro_answer_indicators(s)
    s = _strip_short_answer_wrappers(s)
    s = s.strip()
    return s, answer_indicated


def _extract_second_round_answer_for_parsing(ans):
    """Match OPRO second-round extraction cleanup before parsing."""
    return str(ans).strip(":").strip().split("\n")[0].split("Q:")[0]


def _parse_multiple_choice_with_goal_fallback(task_name, raw_text, goal_text):
    """Map option descriptions or ordinal mentions back to choice letters."""
    text = str(raw_text).strip()
    if not text:
        return None

    # Handle common explicit mentions not covered by the strict parser.
    low = text.lower()
    for pattern in [
        r"\b(?:option|choice|letter)\s*[:\-]?\s*([a-e])\b",
        r"^\s*([a-e])\s*[\)\].,:;\-]",
    ]:
        m = re.search(pattern, low)
        if m:
            letter = m.group(1).upper()
            if task_name == "hyperbaton" and letter not in {"A", "B"}:
                return None
            return letter

    # Hyperbaton outputs often mention "the first/second sentence".
    if task_name == "hyperbaton":
        if re.search(r"\b(first|1st)\b", low):
            return "A"
        if re.search(r"\b(second|2nd)\b", low):
            return "B"
        # Treat explicit non-choice responses as parsed-but-invalid (count as wrong,
        # not parser failure) so parse-fail reflects extraction failures only.
        if re.search(r"\b(neither|both|none|not among|not given|cannot determine|can't determine)\b", low):
            return INVALID_PARSED_ANSWER

    option_map = _extract_option_map_from_goal(goal_text)
    if not option_map:
        return None

    parsed_text, _ = _normalize_opro_style_base(text)
    pred_norm = _normalize_text_for_option_match(parsed_text)
    if not pred_norm:
        return None

    candidates = []
    for letter, option_text in option_map.items():
        option_norm = _normalize_text_for_option_match(option_text)
        if not option_norm:
            continue
        if pred_norm == option_norm:
            candidates.append((letter, 4, len(option_norm)))
            continue
        if pred_norm.startswith(option_norm):
            candidates.append((letter, 3, len(option_norm)))
            continue
        if option_norm in pred_norm:
            candidates.append((letter, 2, len(option_norm)))
            continue
        # Handle truncated second-round outputs like "None" for
        # an option text "None of the above".
        if len(pred_norm) >= 3 and option_norm.startswith(pred_norm):
            candidates.append((letter, 1, len(option_norm)))

    if not candidates:
        # Final fallback: choose the option with the strongest token overlap.
        pred_tokens = set(pred_norm.split())
        if pred_tokens:
            scored = []
            for letter, option_text in option_map.items():
                option_tokens = set(_normalize_text_for_option_match(option_text).split())
                if not option_tokens:
                    continue
                overlap = len(pred_tokens & option_tokens)
                if overlap <= 0:
                    continue
                score = (overlap / max(len(option_tokens), 1), overlap, len(option_tokens))
                scored.append((letter, score))
            if scored:
                scored.sort(key=lambda x: x[1], reverse=True)
                top_letter, top_score = scored[0]
                tied = [s for s in scored if s[1] == top_score]
                if len(tied) == 1:
                    return top_letter
        return None

    # Prefer exact/prefix, then longer option-text matches to reduce ambiguity.
    candidates.sort(key=lambda x: (x[1], x[2]), reverse=True)
    top = candidates[0]
    tied = [c for c in candidates if c[1] == top[1] and c[2] == top[2]]
    if len(tied) != 1:
        return None
    return top[0]


def _parse_explicit_invalid_mc_response(task_name, raw_text):
    """Treat explicit non-choice responses as parsed-but-invalid (wrong, not parse-fail)."""
    if task_name not in {"tracking_shuffled_objects_five_objects", "movie_recommendation", "hyperbaton"}:
        return None
    text = str(raw_text).strip()
    if not text:
        return None
    parsed_text, _ = _normalize_opro_style_base(text)
    low = parsed_text.lower().strip()
    if not low:
        return None

    # Explicit invalid choice statements that are still semantically parseable.
    if re.search(r"\b(neither|both)\b", low):
        return INVALID_PARSED_ANSWER
    if re.search(r"\b(not among|not one of|not given|none of the options|none of these)\b", low):
        return INVALID_PARSED_ANSWER
    if re.search(r"\b(cannot|can't)\s+determine\b", low):
        return INVALID_PARSED_ANSWER
    return None


def _parse_causal_judgement_fallback(raw_text):
    text = str(raw_text).strip()
    if not text:
        return None
    parsed_text, _ = _normalize_opro_style_base(text)
    low = parsed_text.lower().strip()
    token_map = {
        "yes": "Yes",
        "no": "No",
        "y": "Yes",
        "n": "No",
        "yeah": "Yes",
        "yep": "Yes",
        "nope": "No",
        "true": "Yes",
        "false": "No",
        "valid": "Yes",
        "invalid": "No",
        "1": "Yes",
        "0": "No",
        "1.0": "Yes",
        "0.0": "No",
    }
    for token, mapped in token_map.items():
        if re.match(rf"^{re.escape(token)}(\b|[^a-z0-9])", low):
            return mapped

    # OPRO-style fallback doesn't force a choice, but for binary tasks we can
    # still recover many cases by reading the first unambiguous boolean token.
    hits = []
    for token, mapped in token_map.items():
        for m in re.finditer(rf"\b{re.escape(token)}\b", low):
            hits.append((m.start(), token, mapped))
    if hits:
        hits.sort(key=lambda x: x[0])
        first_pos = hits[0][0]
        first_mapped = hits[0][2]
        # If multiple different labels appear later, trust the earliest mention
        # when it is near the start (common output: "No, because ...").
        if first_pos <= 24:
            return first_mapped

    # Last-resort binary coercion to reduce parse-fail: use negation cues.
    if re.search(r"\b(no|not|never|cannot|can't|doesn't|isn't|insufficient|unlikely)\b", low):
        return "No"
    if low:
        return "Yes"
    return None


def normalize_for_strict_match(task_name, text):
    value = normalize_answer(text)
    if not value:
        return value

    # OPRO-style parser normalization (no second-round prompting).
    value, answer_indicated = _normalize_opro_style_base(value)
    if not value:
        return value

    if task_name == "object_counting":
        raw = value
        for c in ["$", ",", "%", "€", "£"]:
            raw = raw.replace(c, "")
        raw = raw.strip()
        # OPRO-like behavior: if an answer indicator was found, prioritize earlier
        # numeric tokens; otherwise look from the end.
        numeric_tokens = re.findall(r"-?\d+(?:\.\d+)?", raw)
        if numeric_tokens:
            chosen = numeric_tokens[0] if answer_indicated else numeric_tokens[-1]
            # object_counting expects positive integer; keep strict regex downstream.
            if chosen.startswith("+"):
                chosen = chosen[1:]
            return chosen
        return _strip_trailing_short_answer_punct(raw)

    if task_name in {"tracking_shuffled_objects_five_objects", "movie_recommendation", "hyperbaton"}:
        choice_text = _extract_bracketed_choice_from_string(value)
        m = re.fullmatch(r"\(([a-z])\)", choice_text)
        if m:
            return m.group(1).upper()
        choice_text = _strip_trailing_short_answer_punct(choice_text)
        return choice_text.upper() if len(choice_text) == 1 else choice_text

    if task_name == "causal_judgement":
        value = _strip_trailing_short_answer_punct(value)
        if value in {"0", "1", "0.0", "1.0"}:
            return "No" if value in {"0", "0.0"} else "Yes"
        if value.lower() in {"false", "true"}:
            return "No" if value.lower() == "false" else "Yes"
        if value.lower() in {"yes", "no"}:
            return value.capitalize()
        return value

    return value


def parse_strict_answer(task_name, text):
    value = normalize_for_strict_match(task_name, text)
    pattern = TASK_STRICT_PATTERNS.get(task_name)
    if pattern is None:
        return value
    if pattern.fullmatch(value):
        return value
    return None


def parse_prediction_answer(task_name, raw_text, goal_text=None):
    """Paper-main parser: OPRO-style normalization + task regex/boolean parsing."""
    parsed = parse_strict_answer(task_name, raw_text)
    if parsed is not None:
        return parsed

    if task_name == "causal_judgement":
        return _parse_causal_judgement_fallback(raw_text)

    invalid_mc = _parse_explicit_invalid_mc_response(task_name, raw_text)
    if invalid_mc is not None:
        return invalid_mc

    return None


def get_goals_and_targets(data_path, extractor_text, conversation_template_name, n_train_data, n_test_data):
    addition_goal = "."
    addition_target = extractor_text
    offset = 0

    train_goals, train_targets = [], []
    test_goals, test_targets = [], []
    train_final_targets, test_final_targets = [], []

    data_path = str(data_path)
    if data_path.endswith(".tsv"):
        train_data = pd.read_csv(data_path, sep="\t", dtype=str)
    else:
        train_data = pd.read_csv(data_path, dtype=str)

    train_targets = train_data["final_target"].astype(str).tolist()[offset:offset + n_train_data]
    if addition_target:
        train_targets = [addition_target + remove_parentheses_if_single_char(x) for x in train_targets]

    if "goal" in train_data.columns:
        train_goals = train_data["goal"].astype(str).tolist()[offset:offset + n_train_data]
        if addition_goal:
            train_goals = [goal + addition_goal for goal in train_goals]
    else:
        train_goals = [""] * len(train_targets)

    if "final_target" in train_data.columns:
        train_final_targets = train_data["final_target"].astype(str).tolist()[offset:offset + n_train_data]
        if "llama-3" in conversation_template_name:
            train_final_targets = [remove_parentheses_if_single_char(x) for x in train_final_targets]
        elif "llama-2" in conversation_template_name or "gemma-2" in conversation_template_name:
            train_final_targets = [remove_parentheses_if_single_char(x) for x in train_final_targets]
    else:
        train_final_targets = [""] * len(train_targets)

    if n_test_data > 0:
        if data_path.endswith(".tsv"):
            test_data = pd.read_csv(data_path, sep="\t", dtype=str)
        else:
            test_data = pd.read_csv(data_path, dtype=str)

        test_targets = test_data["final_target"].astype(str).tolist()[
            offset + n_train_data:offset + n_train_data + n_test_data
        ]
        if addition_target:
            test_targets = [addition_target + remove_parentheses_if_single_char(x) for x in test_targets]

        if "goal" in test_data.columns:
            test_goals = test_data["goal"].astype(str).tolist()[
                offset + n_train_data:offset + n_train_data + n_test_data
            ]
            if addition_goal:
                test_goals = [goal + addition_goal for goal in test_goals]
        else:
            test_goals = [""] * len(test_targets)

        if "final_target" in test_data.columns:
            test_final_targets = test_data["final_target"].astype(str).tolist()[
                offset + n_train_data:offset + n_train_data + n_test_data
            ]
            if "llama-3" in conversation_template_name:
                test_final_targets = [remove_parentheses_if_single_char(x) for x in test_final_targets]
            elif "llama-2" in conversation_template_name or "gemma-2" in conversation_template_name:
                test_final_targets = [remove_parentheses_if_single_char(x) for x in test_final_targets]
        else:
            test_final_targets = [""] * len(test_targets)

    assert len(train_goals) == len(train_targets)
    assert len(test_goals) == len(test_targets)
    return train_goals, train_targets, test_goals, test_targets, train_final_targets, test_final_targets


def get_conv_template(conversation_template_name):
    conv = get_conversation_template(conversation_template_name)

    if "gemma-2" in conversation_template_name:
        conv.name = "gemma-2"
    elif "llama-3" in conversation_template_name:
        conv.name = "llama-3"
    elif "gpt2" in conversation_template_name:
        conv.name = "gpt2"
    else:
        raise NotImplementedError(f"Conversation template {conversation_template_name} not implemented")

    if conv.name == "llama-3":
        conv.system = " "
    elif conv.name == "gpt2":
        conv.system = " "
    return conv


def _compose_user_content(goal, control, control_pos="post"):
    goal = str(goal)
    control = str(control)
    if control_pos == "post":
        content = goal
        control = control.lstrip()
        if control:
            content += " " + control
        return content
    if control_pos == "pre":
        control = control.lstrip()
        if not control:
            return goal
        return control + " " + goal
    raise ValueError(f"Unsupported control_pos: {control_pos}")


def _apply_hf_chat_template_prompt(hf_tokenizer, user_content, assistant_prefix_role="assistant"):
    if hf_tokenizer is None or not hasattr(hf_tokenizer, "apply_chat_template"):
        return None

    # Try common role naming first; some templates (e.g., Gemma) may prefer "model".
    role_candidates = ["user"]
    for user_role in role_candidates:
        messages = [{"role": user_role, "content": user_content}]
        try:
            return hf_tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
        except Exception:
            continue

    # Fallback for tokenizer-specific role naming edge cases (Gemma chat templates sometimes use `model`)
    # We only need a user message + generation prompt, but some templates validate role names tightly.
    role_pairs = [
        ("user", "assistant"),
        ("user", "model"),
        ("human", "assistant"),
    ]
    for user_role, _ in role_pairs:
        messages = [{"role": user_role, "content": user_content}]
        try:
            return hf_tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
        except Exception:
            continue
    return None


def supports_vllm_chat(llm):
    return callable(getattr(llm, "chat", None))


def should_use_vllm_chat(chat_backend, llm, conv_template_name):
    if conv_template_name not in {"llama-3", "gemma-2"}:
        return False
    if chat_backend == "manual":
        return False
    if chat_backend == "hf_chat_template":
        return False
    if chat_backend == "vllm_chat":
        return supports_vllm_chat(llm)
    # auto
    return supports_vllm_chat(llm)


def _chat_user_batch(user_contents):
    return [[{"role": "user", "content": content}] for content in user_contents]


def _chat_continue_assistant_batch(user_contents, assistant_prefixes):
    return [
        [
            {"role": "user", "content": user_content},
            {"role": "assistant", "content": assistant_prefix},
        ]
        for user_content, assistant_prefix in zip(user_contents, assistant_prefixes)
    ]


def _append_answer_with_suffix(prefix, answer, suffix):
    answer = str(answer or "")
    if answer and not answer.endswith((" ", "\n")):
        answer = answer + " "
    return f"{prefix}{answer}{suffix}"


def apply_conv_template(conv_template, goal, control, control_pos="post", hf_tokenizer=None):
    user_content = _compose_user_content(goal, control, control_pos=control_pos)

    # Prefer the model's official HF chat template when available, but keep manual formatting
    # as a compatibility fallback (e.g., older tokenizer templates or non-chat models).
    if conv_template.name in {"llama-3", "gemma-2"}:
        rendered = _apply_hf_chat_template_prompt(hf_tokenizer, user_content)
        if rendered is not None:
            return rendered

    if conv_template.name == "llama-3":
        full_input = "<|start_header_id|>user<|end_header_id|>\n\n"
        full_input += user_content
        full_input += "<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
        return full_input

    if conv_template.name == "gemma-2":
        full_input = "<bos><start_of_turn>user\n"
        if control_pos != "post":
            raise NotImplementedError("Only control_pos='post' is supported for gemma-2.")
        full_input += user_content
        full_input += "<end_of_turn>\n<start_of_turn>model\n"
        return full_input

    if conv_template.name == "gpt2":
        instruction = user_content.strip()
        return (
            "Below is an instruction that describes a task. "
            "Write a response that appropriately completes the request.\n\n"
            f"### Instruction:\n{instruction}\n\n### Response:\n"
        )

    raise NotImplementedError(f"Conversation template {conv_template.name} not implemented")


def apply_template(conv_template, goal_control, reasoning, target):
    if conv_template.name == "llama-3":
        return goal_control + reasoning + " " + target
    if conv_template.name == "gemma-2":
        separator = "" if (not reasoning or str(reasoning).endswith((" ", "\n"))) else " "
        return goal_control + reasoning + separator + target
    if conv_template.name == "gpt2":
        return f"{goal_control.rstrip()}{reasoning.rstrip()} {target.rstrip()}"
    raise NotImplementedError(f"Conversation template {conv_template.name} not implemented")


def clean_generated_prompt_text(text):
    cleaned = str(text).strip()
    cleaned = cleaned.split("\n")[0].strip()
    cleaned = re.sub(r"^(Instruction|Prompt)\s*[:\-]\s*", "", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"^\s*[\-\*\d\.\)\(]+\s*", "", cleaned)
    cleaned = cleaned.strip("`\"' ")
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    return cleaned


def build_meta_prompt_generation_request(task_name, train_goals, train_final_targets, num_examples):
    n_examples = min(max(1, num_examples), len(train_goals))
    lines = [
        f"You are creating a single instruction prompt for a reasoning task named '{task_name}'.",
        "Infer the task behavior from the input/output examples below.",
        "Write exactly one concise instruction that helps solve this task.",
        "Output only the instruction text. Do not output numbering, quotes, or explanations.",
        "",
        "Examples:",
    ]
    for i in range(n_examples):
        lines.append(f"Input: {train_goals[i]}")
        lines.append(f"Output: {train_final_targets[i]}")
        lines.append("")
    lines.append("Instruction:")
    return "\n".join(lines)


def generate_meta_prompts(
    llm,
    conversation_template_name,
    task_name,
    train_goals,
    train_final_targets,
    target_count,
    temperature,
    top_p,
    max_tokens,
    num_examples,
    max_batch_size=128,
    max_rounds=6,
    hf_tokenizer=None,
    chat_backend="auto",
):
    if target_count <= 0:
        return {}, ""

    conv_template = get_conv_template(conversation_template_name)
    meta_request = build_meta_prompt_generation_request(
        task_name=task_name,
        train_goals=train_goals,
        train_final_targets=train_final_targets,
        num_examples=num_examples,
    )
    gen_params = SamplingParams(
        temperature=temperature,
        top_p=top_p,
        max_tokens=max_tokens,
    )

    unique_prompts = []
    seen = set()
    max_batch_size = max(1, int(max_batch_size))
    max_rounds = max(1, int(max_rounds))
    prompt_input = apply_conv_template(conv_template, meta_request, "", hf_tokenizer=hf_tokenizer)
    use_vllm_chat = should_use_vllm_chat(chat_backend, llm, conv_template.name)
    meta_chat_messages = [[{"role": "user", "content": meta_request}]]

    for _ in range(max_rounds):
        if len(unique_prompts) >= target_count:
            break
        remaining = max(target_count - len(unique_prompts), 1)
        batch_size = max(1, min(max_batch_size, max(remaining * 2, 8)))
        if use_vllm_chat:
            outputs = llm.chat(meta_chat_messages * batch_size, gen_params, use_tqdm=False)
        else:
            outputs = llm.generate([prompt_input] * batch_size, gen_params, use_tqdm=False)
        for output in outputs:
            candidate = clean_generated_prompt_text(output.outputs[0].text)
            if len(candidate) < 8:
                continue
            key = candidate.lower()
            if key in seen:
                continue
            seen.add(key)
            unique_prompts.append(candidate)
            if len(unique_prompts) >= target_count:
                break

    prompt_dict = {
        f"Meta-{idx:02d}": prompt
        for idx, prompt in enumerate(unique_prompts[:target_count], start=1)
    }
    return prompt_dict, meta_request


def normalize_prompt_entries(entries, prefix="Meta"):
    prompts = {}
    if isinstance(entries, dict):
        for key, value in entries.items():
            prompts[str(key)] = str(value)
        return prompts

    if isinstance(entries, list):
        for idx, item in enumerate(entries, start=1):
            if isinstance(item, str):
                prompts[f"{prefix}-{idx:02d}"] = item
            elif isinstance(item, dict):
                if "prompt" in item:
                    name = str(item.get("name", f"{prefix}-{idx:02d}"))
                    prompts[name] = str(item["prompt"])
                elif len(item) == 1:
                    name, prompt_text = next(iter(item.items()))
                    prompts[str(name)] = str(prompt_text)
                else:
                    raise ValueError(
                        f"Invalid prompt entry at index {idx}. "
                        "Use {'name': ..., 'prompt': ...} or single-key dict."
                    )
            else:
                raise ValueError(f"Unsupported prompt type at index {idx}: {type(item)}")
        return prompts

    raise ValueError(f"Unsupported prompt container type: {type(entries)}")


def resolve_prompts_from_payload(payload, task_name, known_task_names):
    if isinstance(payload, dict):
        if "*" in payload or any(k in known_task_names for k in payload.keys()):
            task_entries = payload.get(task_name, payload.get("*", []))
        else:
            task_entries = payload
    else:
        task_entries = payload
    return normalize_prompt_entries(task_entries)


def evaluate_prompt(
    llm,
    conversation_template_name,
    task_name,
    goals,
    final_targets,
    control_prompt,
    reasoning_params,
    answer_params,
    opro_second_round_extraction=False,
    opro_second_round_answer_params=None,
    collect_debug_rows=False,
    debug_sample_limit=5,
    hf_tokenizer=None,
    chat_backend="auto",
):
    conv_template = get_conv_template(conversation_template_name)
    user_contents = [_compose_user_content(goal, control_prompt, control_pos="post") for goal in goals]
    use_vllm_chat = should_use_vllm_chat(chat_backend, llm, conv_template.name)

    if use_vllm_chat:
        chat_stage1_messages = _chat_user_batch(user_contents)
        outputs = llm.chat(chat_stage1_messages, reasoning_params, use_tqdm=False)
        goals_controls = chat_stage1_messages
    else:
        goals_controls = [
            apply_conv_template(conv_template, goal, control_prompt, hf_tokenizer=hf_tokenizer)
            for goal in goals
        ]
        outputs = llm.generate(goals_controls, reasoning_params, use_tqdm=False)
    reasonings = [output.outputs[0].text for output in outputs]

    extractor_text = TASK_EXTRACTOR_TEXT[task_name]
    if use_vllm_chat:
        extraction_assistant_prefixes = []
        for reasoning in reasonings:
            separator = "" if (not reasoning or str(reasoning).endswith((" ", "\n"))) else " "
            extraction_assistant_prefixes.append(f"{reasoning}{separator}{extractor_text}")
        extraction_prompts = _chat_continue_assistant_batch(user_contents, extraction_assistant_prefixes)
        outputs = llm.chat(
            extraction_prompts,
            answer_params,
            use_tqdm=False,
            add_generation_prompt=False,
            continue_final_message=True,
        )
    else:
        extraction_prompts = [
            apply_template(conv_template, goal_control, reasoning, extractor_text)
            for goal_control, reasoning in zip(goals_controls, reasonings)
        ]
        outputs = llm.generate(extraction_prompts, answer_params, use_tqdm=False)
    raw_answers = [output.outputs[0].text.strip() for output in outputs]

    raw_answers_to_parse = raw_answers
    second_round_prompts = None
    raw_answers_second_round = None
    if opro_second_round_extraction:
        second_round_suffix = OPRO_SECOND_ROUND_EXTRACTION_SUFFIX
        second_round_params = opro_second_round_answer_params or answer_params
        if use_vllm_chat:
            second_round_assistant_prefixes = [
                _append_answer_with_suffix(prefix, raw_answer, second_round_suffix)
                for prefix, raw_answer in zip(extraction_assistant_prefixes, raw_answers)
            ]
            second_round_prompts = _chat_continue_assistant_batch(user_contents, second_round_assistant_prefixes)
            outputs_second_round = llm.chat(
                second_round_prompts,
                second_round_params,
                use_tqdm=False,
                add_generation_prompt=False,
                continue_final_message=True,
            )
        else:
            second_round_prompts = [
                f"{prompt}{raw_answer} {second_round_suffix}"
                for prompt, raw_answer in zip(extraction_prompts, raw_answers)
            ]
            outputs_second_round = llm.generate(second_round_prompts, second_round_params, use_tqdm=False)
        raw_answers_second_round = [output.outputs[0].text for output in outputs_second_round]
        raw_answers_to_parse = [_extract_second_round_answer_for_parsing(x) for x in raw_answers_second_round]

    parsed_answers = [
        parse_prediction_answer(task_name, raw_answer, goal_text=goal)
        for raw_answer, goal in zip(raw_answers_to_parse, goals)
    ]
    parsed_targets = [parse_strict_answer(task_name, x) for x in final_targets]

    correct = 0
    parse_failures = 0
    invalid_predictions = 0
    debug_rows = []
    sample_limit = None if int(debug_sample_limit) < 0 else int(debug_sample_limit)
    for pred, target in zip(parsed_answers, parsed_targets):
        if pred is None or target is None:
            parse_failures += 1
            continue
        if pred == INVALID_PARSED_ANSWER:
            invalid_predictions += 1
            continue
        if pred == target:
            correct += 1

    if collect_debug_rows:
        for idx, (goal, final_target, parsed_target, goal_control, extraction_prompt, reasoning, raw_stage1, raw_for_parse, pred) in enumerate(
            zip(
                goals,
                final_targets,
                parsed_targets,
                goals_controls,
                extraction_prompts,
                reasonings,
                raw_answers,
                raw_answers_to_parse,
                parsed_answers,
            ),
            start=1,
        ):
            if sample_limit is not None and len(debug_rows) >= sample_limit:
                break
            is_parse_failure = (pred is None) or (parsed_target is None)
            is_invalid = pred == INVALID_PARSED_ANSWER
            is_correct = (not is_parse_failure) and (not is_invalid) and (pred == parsed_target)
            row = {
                "idx": idx,
                "goal": goal,
                "target_raw": final_target,
                "target_parsed": parsed_target,
                "stage1_prompt": goal_control,
                "stage2_extractor_prompt": extraction_prompt,
                "reasoning": reasoning,
                "raw_answer_stage1": raw_stage1,
                "raw_answer_for_parse": raw_for_parse,
                "parsed_answer": pred,
                "is_parse_failure": bool(is_parse_failure),
                "is_invalid": bool(is_invalid),
                "is_correct": bool(is_correct),
            }
            if use_vllm_chat:
                row["prompt_backend"] = "vllm_chat"
            if raw_answers_second_round is not None:
                row["raw_answer_stage2"] = raw_answers_second_round[idx - 1]
            if second_round_prompts is not None:
                row["stage3_second_round_prompt"] = second_round_prompts[idx - 1]
            debug_rows.append(row)

    total = len(parsed_targets)
    acc = float(correct) / float(total) if total > 0 else 0.0
    result = {
        "accuracy": acc,
        "correct": correct,
        "total": total,
        "parse_failures": parse_failures,
        "invalid_predictions": invalid_predictions,
        "opro_second_round_extraction": bool(opro_second_round_extraction),
        "prompt_backend": "vllm_chat" if use_vllm_chat else ("hf_chat_template" if hf_tokenizer is not None else "manual"),
    }
    if collect_debug_rows:
        result["debug_rows"] = debug_rows
    return result


def print_summary_table(all_results):
    if not all_results:
        print("[BBH Eval] No results.")
        return

    method_names = sorted(
        {k for task_result in all_results.values() for k in task_result.keys() if k != "paper_acc"}
    )
    task_col_width = 40
    score_col_width = 10

    header = f"{'Task':<{task_col_width}} {'Paper':>8}"
    for method_name in method_names:
        name = method_name if len(method_name) <= score_col_width else method_name[:score_col_width - 1] + "~"
        header += f" {name:>{score_col_width}}"
    line = "-" * len(header)

    print("\n" + "=" * len(header))
    print("Logit Argmax Evaluation (vLLM)")
    print("=" * len(header))
    print(header)
    print(line)

    avg_paper = []
    avg_method = {m: [] for m in method_names}
    for task_name, task_result in all_results.items():
        paper = task_result.get("paper_acc", 0.0)
        avg_paper.append(paper)
        short_name = task_name.replace("tracking_shuffled_objects_five_objects", "tracking_shuffled_objects")
        row = f"{short_name:<{task_col_width}} {paper:>8.2f}"
        for method_name in method_names:
            acc = task_result.get(method_name, {}).get("accuracy", 0.0)
            avg_method[method_name].append(acc)
            row += f" {acc:>{score_col_width}.2f}"
        print(row)

    avg_row = f"{'AVERAGE':<{task_col_width}} {sum(avg_paper) / len(avg_paper):>8.3f}"
    for method_name in method_names:
        values = avg_method[method_name]
        avg = sum(values) / len(values) if values else 0.0
        avg_row += f" {avg:>{score_col_width}.3f}"
    print(line)
    print(avg_row)
    print("=" * len(header))


def now():
    return time.strftime("%Y-%m-%d %H:%M:%S")
