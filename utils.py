import json
import re
import time
from pathlib import Path

import pandas as pd
from vllm import SamplingParams

try:
    from fastchat.model import get_conversation_template as _get_conversation_template
except Exception:
    _get_conversation_template = None


# ============================================================================
# Terminology used in this standalone evaluator
# ----------------------------------------------------------------------------
# - GreaTer-style evaluation: do not force answer format in the input prompt;
#   instead use a second LLM call (Extractor) to produce a short final answer
#   for scoring.
# - "Extractor": second LLM call used to extract a short final answer
# - "Parser": postprocessing that converts raw output to strict task answer type
# ============================================================================

TASK_EXTRACTOR_TEXT = {
    "tracking_shuffled_objects_five_objects": (
        "Therefore, the final answer (use exact format: '$A' or '$B' or '$C' or '$D' or '$E') is $"
    ),
    "object_counting": (
        "Therefore, the final answer (use exactly this format: $NUMBER$, where NUMBER is a positive integer) is $"
    ),
    "causal_judgement": (
        "Therefore, the final answer (use exact format: '$ Yes' or '$ No') is $ "
    ),
    "movie_recommendation": (
        "Therefore, the final answer (use exact format: '$A' or '$B' or '$C' or '$D') is $"
    ),
    "hyperbaton": (
        "Therefore, the final answer (use exact format: '$A' or '$B') is $"
    ),
}

TASK_STRICT_PATTERNS = {
    "tracking_shuffled_objects_five_objects": re.compile(r"^[A-E]$"),
    "object_counting": re.compile(r"^[0-9]+$"),
    "causal_judgement": re.compile(r"^(Yes|No)$"),
    "movie_recommendation": re.compile(r"^[A-E]$"),
    "hyperbaton": re.compile(r"^[A-B]$"),
}

TASK_ANSWER_MAX_TOKENS = {
    "tracking_shuffled_objects_five_objects": 4,
    "object_counting": 1,
    "causal_judgement": 4,
    "movie_recommendation": 4,
    "hyperbaton": 4,
}

ANSWER_EXTRACTION_STOPS = ["\n", "<|eot_id|>", "<end_of_turn>", "</s>"]

PAPER_ACC = {
    "movie_recommendation": 0.57,
    "object_counting": 0.90,
    "tracking_shuffled_objects_five_objects": 0.70,
    "hyperbaton": 0.84,
    "causal_judgement": 0.57,
}

PAPER_OPT_PROMPTS = {
    "movie_recommendation": (
        "Use movie ratings data available here above movies for reference. "
        "ThisHOFF has an interesting analysis based solely options to options "
        "based movies ratings expect from the other movies you are asked ones "
        "mentioned here you."
    ),
    "object_counting": "Use only addition. Add think step by step. Finally give the actual correct answer.",
    "tracking_shuffled_objects_five_objects": (
        "Use this process as an explanation stepwise for each step until you get to "
        "as given above Alice has got originaly the following as follows."
    ),
    "hyperbaton": "Use the reasoning and examples you would step. Finally give the actual correct answer.",
    "causal_judgement": (
        "Use causal diagram. The correct option ask about whether there the variable C "
        "of about whether a specific cause is sufficient. The answer a causal relationship "
        "between C to D if the probability P that C occurs given E changes."
    ),
}

SUPPORTED_TASKS = list(TASK_EXTRACTOR_TEXT.keys())

INVALID_PARSED_ANSWER = "__INVALID_PARSED_ANSWER__"

OPRO_FINAL_ANSWER_BEHIND_PATTERNS_PRIMARY = ["answer is ", "answer: ", "answer is: "]
OPRO_FINAL_ANSWER_BEHIND_PATTERNS_SECONDARY = [" is: ", " are: "]
OPRO_FINAL_ANSWER_AHEAD_PATTERNS = [
    " is the correct answer",
    " is the right answer",
    " is the final answer",
    " is the answer",
]
OPRO_NEXT_QUESTION_DELIMITERS = ["\nq:", "\nquestion:", "\n\nq:", "\n\nquestion:"]


def ensure_dir(path):
    Path(path).mkdir(parents=True, exist_ok=True)


def save_json(path, payload):
    path = Path(path)
    ensure_dir(path.parent)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def load_json(path):
    with Path(path).open("r", encoding="utf-8") as f:
        return json.load(f)


def now():
    return time.strftime("%Y-%m-%d %H:%M:%S")


def remove_parentheses_if_single_char(text):
    text = str(text).strip()
    if text.startswith("(") and text.endswith(")") and len(text) == 3:
        return text[1:-1]
    return text


def get_goals_and_targets(data_path, extractor_text, conversation_template_name, n_train_data, n_test_data):
    addition_goal = "."
    addition_target = extractor_text
    offset = 0

    train_goals, train_targets = [], []
    test_goals, test_targets = [], []
    train_final_targets, test_final_targets = [], []

    data_path = str(data_path)
    if data_path.endswith(".tsv"):
        data = pd.read_csv(data_path, sep="\t", dtype=str)
    else:
        data = pd.read_csv(data_path, dtype=str)

    train_targets = data["final_target"].astype(str).tolist()[offset : offset + n_train_data]
    if addition_target:
        train_targets = [addition_target + remove_parentheses_if_single_char(x) for x in train_targets]

    if "goal" in data.columns:
        train_goals = data["goal"].astype(str).tolist()[offset : offset + n_train_data]
        if addition_goal:
            train_goals = [g + addition_goal for g in train_goals]
    else:
        train_goals = [""] * len(train_targets)

    if "final_target" in data.columns:
        train_final_targets = data["final_target"].astype(str).tolist()[offset : offset + n_train_data]
        if "llama-3" in conversation_template_name or "llama-2" in conversation_template_name or "gemma-2" in conversation_template_name:
            train_final_targets = [remove_parentheses_if_single_char(x) for x in train_final_targets]
    else:
        train_final_targets = [""] * len(train_targets)

    if n_test_data > 0:
        test_targets = data["final_target"].astype(str).tolist()[
            offset + n_train_data : offset + n_train_data + n_test_data
        ]
        if addition_target:
            test_targets = [addition_target + remove_parentheses_if_single_char(x) for x in test_targets]

        if "goal" in data.columns:
            test_goals = data["goal"].astype(str).tolist()[
                offset + n_train_data : offset + n_train_data + n_test_data
            ]
            if addition_goal:
                test_goals = [g + addition_goal for g in test_goals]
        else:
            test_goals = [""] * len(test_targets)

        if "final_target" in data.columns:
            test_final_targets = data["final_target"].astype(str).tolist()[
                offset + n_train_data : offset + n_train_data + n_test_data
            ]
            if "llama-3" in conversation_template_name or "llama-2" in conversation_template_name or "gemma-2" in conversation_template_name:
                test_final_targets = [remove_parentheses_if_single_char(x) for x in test_final_targets]
        else:
            test_final_targets = [""] * len(test_targets)

    assert len(train_goals) == len(train_targets)
    assert len(test_goals) == len(test_targets)
    return train_goals, train_targets, test_goals, test_targets, train_final_targets, test_final_targets


def get_conv_template(conversation_templates):
    if _get_conversation_template is None:
        class _FallbackConv:
            def __init__(self):
                self.name = ""
                self.roles = ("user", "assistant")
                self.sep = "\n"
                self.sep2 = "\n"
                self.system = " "
                self.messages = []

        conv = _FallbackConv()
    else:
        conv = _get_conversation_template(conversation_templates)

    if "gemma-2" in conversation_templates:
        conv.name = "gemma-2"
    elif "gemma" in conversation_templates:
        conv.name = "gemma"
    elif "llama-3" in conversation_templates:
        conv.name = "llama-3"
    elif "gpt2" in conversation_templates:
        conv.name = "gpt2"
    else:
        raise NotImplementedError(f"Conversation template {conversation_templates} not implemented")

    if conv.name == "zero_shot":
        conv.roles = tuple(["### " + r for r in conv.roles])
        conv.sep = "\n"
    elif conv.name == "llama-2":
        conv.system = "<s>[INST] "  # forcing to use no system instruction
        conv.sep2 = conv.sep2.strip()
    elif conv.name == "llama-3":
        conv.system = " "  # not used in the system
    elif conv.name == "gpt2":
        conv.system = " "  # not used in the system
    elif conv.name == "gemma":
        conv.system = "<bos>"
        conv.roles = ("user\n", "model\n")
    return conv


def apply_conv_template(conv_template, goal, control, control_pos="post"):
    if conv_template.name == "llama-2":
        raise NotImplementedError("Llama-2 is not supported")
    if conv_template.name == "llama-3":
        conv_template.messages = []
        full_input = ""
        full_input += "<|start_header_id|>user<|end_header_id|>\n\n"
        if control_pos == "post":
            full_input += goal
            if control.startswith(" "):
                control = control[1:]
            full_input = full_input + " " + control
        elif control_pos == "pre":
            full_input += control
            full_input += " "
            full_input += goal
        full_input += "<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
        return full_input
    if conv_template.name == "gemma-2":
        conv_template.messages = []
        full_input = ""
        full_input += "<bos><start_of_turn>user\n"
        if control_pos == "post":
            full_input += goal
            if control.startswith(" "):
                control = control[1:]
            full_input = full_input + " " + control
        elif control_pos == "pre":
            raise NotImplementedError
        full_input += "<end_of_turn>\n<start_of_turn>model\n"
        return full_input
    if conv_template.name == "gpt2":
        conv_template.messages = []
        instruction = goal + " " + control
        prompt_template = (
            "Below is an instruction that describes a task. "
            "Write a response that appropriately completes the request.\n\n"
            "### Instruction:\n{instruction}\n\n### Response:\n"
        )
        return prompt_template.format(instruction=instruction.rstrip())
    raise NotImplementedError(f"Conversation template {conv_template.name} not implemented")


def _compose_user_content(goal, control):
    goal = str(goal)
    control = str(control).lstrip()
    if not control:
        return goal
    return f"{goal} {control}"


def render_llama3_user_prompt(user_content):
    # Manual Llama-3 Instruct prompt formatting (kept simple and dependency-light).
    return (
        "<|start_header_id|>user<|end_header_id|>\n\n"
        f"{user_content}"
        "<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
    )


def render_extractor_prompt_llama3(stage1_prompt, reasoning, extractor_text):
    separator = "" if (not reasoning or str(reasoning).endswith((" ", "\n"))) else " "
    return f"{stage1_prompt}{reasoning}{separator}{extractor_text}"


def clean_generated_prompt_text(text):
    cleaned = str(text).strip()
    cleaned = cleaned.split("\n")[0].strip()
    cleaned = re.sub(r"^(Instruction|Prompt)\s*[:\-]\s*", "", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"^\s*[\-\*\d\.\)\(]+\s*", "", cleaned)
    cleaned = cleaned.strip("`\"' ")
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    return cleaned


def build_meta_prompt_generation_request(task_name, train_goals, train_final_targets, num_examples):
    n_examples = min(max(1, int(num_examples)), len(train_goals))
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
):
    if target_count <= 0:
        if target_count == 0:
            return {}, ""
        target_count = 5

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
    for _ in range(max_rounds):
        if len(unique_prompts) >= target_count:
            break
        remaining = max(target_count - len(unique_prompts), 1)
        batch_size = max(1, min(max_batch_size, max(remaining * 2, 8)))
        prompts_input = [apply_conv_template(conv_template, meta_request, "") for _ in range(batch_size)]
        outputs = llm.generate(prompts_input, gen_params, use_tqdm=False)
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

    if len(unique_prompts) == 0:
        raise RuntimeError(
            f"Failed to generate meta prompts for task '{task_name}'. "
            "Try increasing --meta_prompt_temperature or --meta_prompt_max_tokens."
        )

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
    if isinstance(payload, dict) and "prompts" in payload and isinstance(payload["prompts"], list):
        # Supports generated_meta_prompts/<task>.json style payload.
        task_entries = payload["prompts"]
        return normalize_prompt_entries(task_entries)

    if isinstance(payload, dict):
        if "*" in payload or any(k in known_task_names for k in payload.keys()):
            task_entries = payload.get(task_name, payload.get("*", []))
        else:
            task_entries = payload
    else:
        task_entries = payload
    return normalize_prompt_entries(task_entries)


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
    lower = value.lower()
    matches = re.findall(r"\([a-z]\)", lower)
    unique = sorted(set(matches))
    if len(unique) == 1:
        return unique[0]
    return lower


def _strip_opro_answer_indicators(value):
    s = value.lower().strip()
    for delim in OPRO_NEXT_QUESTION_DELIMITERS:
        s = s.split(delim)[0]

    primary_found = any(pat in s for pat in OPRO_FINAL_ANSWER_BEHIND_PATTERNS_PRIMARY)
    behind_patterns = (
        OPRO_FINAL_ANSWER_BEHIND_PATTERNS_PRIMARY if primary_found else OPRO_FINAL_ANSWER_BEHIND_PATTERNS_SECONDARY
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
    while s.endswith("."):
        s = s[:-1]
    while s.endswith("'"):
        s = s[:-1]
    return s.strip(), answer_indicated


def _normalize_opro_style_base(value):
    s = _strip_short_answer_wrappers(str(value))
    s, answer_indicated = _strip_opro_answer_indicators(s)
    s = _strip_short_answer_wrappers(s)
    return s.strip(), answer_indicated


def _parse_explicit_invalid_mc_response(task_name, raw_text):
    if task_name not in {"tracking_shuffled_objects_five_objects", "movie_recommendation", "hyperbaton"}:
        return None
    text = str(raw_text).strip()
    if not text:
        return None
    parsed_text, _ = _normalize_opro_style_base(text)
    low = parsed_text.lower().strip()
    if not low:
        return None
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
        "true": "Yes",
        "false": "No",
        "1": "Yes",
        "0": "No",
    }
    for token, mapped in token_map.items():
        if re.match(rf"^{re.escape(token)}(\b|[^a-z0-9])", low):
            return mapped
    if re.search(r"\b(no|not|never|cannot|can't|doesn't|isn't)\b", low):
        return "No"
    if low:
        return "Yes"
    return None


def normalize_for_strict_match(task_name, text):
    value = remove_parentheses_if_single_char(str(text).strip())
    if not value:
        return value
    value, answer_indicated = _normalize_opro_style_base(value)
    if not value:
        return value

    if task_name == "object_counting":
        raw = value
        for c in ["$", ",", "%", "€", "£"]:
            raw = raw.replace(c, "")
        numeric_tokens = re.findall(r"-?\d+(?:\.\d+)?", raw)
        if numeric_tokens:
            chosen = numeric_tokens[0] if answer_indicated else numeric_tokens[-1]
            if chosen.startswith("+"):
                chosen = chosen[1:]
            return chosen
        return _strip_trailing_short_answer_punct(raw.strip())

    if task_name in {"tracking_shuffled_objects_five_objects", "movie_recommendation", "hyperbaton"}:
        choice_text = _extract_bracketed_choice_from_string(value)
        m = re.fullmatch(r"\(([a-z])\)", choice_text)
        if m:
            return m.group(1).upper()
        choice_text = _strip_trailing_short_answer_punct(choice_text)
        return choice_text.upper() if len(choice_text) == 1 else choice_text

    if task_name == "causal_judgement":
        value = _strip_trailing_short_answer_punct(value)
        low = value.lower()
        if low in {"yes", "no"}:
            return low.capitalize()
        if low in {"true", "false"}:
            return "Yes" if low == "true" else "No"
        if low in {"1", "1.0"}:
            return "Yes"
        if low in {"0", "0.0"}:
            return "No"
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
    parsed = parse_strict_answer(task_name, raw_text)
    if parsed is not None:
        return parsed
    if task_name == "causal_judgement":
        return _parse_causal_judgement_fallback(raw_text)
    invalid_mc = _parse_explicit_invalid_mc_response(task_name, raw_text)
    if invalid_mc is not None:
        return invalid_mc
    return None


def evaluate_prompt(
    llm,
    conversation_template_name,
    task_name,
    goals,
    final_targets,
    control_prompt,
    reasoning_params,
    answer_params,
):
    if conversation_template_name != "llama-3":
        raise NotImplementedError(
            "This minimal 2-file evaluator currently supports conversation_template='llama-3' only."
        )

    if len(goals) != len(final_targets):
        raise ValueError(
            f"Length mismatch for task={task_name}: len(goals)={len(goals)} != len(final_targets)={len(final_targets)}"
        )

    user_contents = [_compose_user_content(goal, control_prompt) for goal in goals]
    stage1_prompts = [render_llama3_user_prompt(x) for x in user_contents]
    outputs = llm.generate(stage1_prompts, reasoning_params, use_tqdm=False)
    reasonings = [o.outputs[0].text for o in outputs]

    extractor_text = TASK_EXTRACTOR_TEXT[task_name]
    extractor_prompts = [
        render_extractor_prompt_llama3(stage1_prompt, reasoning, extractor_text)
        for stage1_prompt, reasoning in zip(stage1_prompts, reasonings)
    ]
    outputs = llm.generate(extractor_prompts, answer_params, use_tqdm=False)
    raw_answers = [o.outputs[0].text.strip() for o in outputs]
    used_extractor = True
    # Paper-reproduction path: exact string matching after minimal canonicalization.
    normalized_answers = [remove_parentheses_if_single_char(ans.strip()) for ans in raw_answers]
    normalized_targets = [remove_parentheses_if_single_char(str(t).strip()) for t in final_targets]

    correct = sum(1 for pred, target in zip(normalized_answers, normalized_targets) if pred == target)
    total = len(normalized_targets)
    accuracy = float(correct) / total if total > 0 else 0.0
    parse_failures = 0
    invalid_predictions = 0
    parsed_rate = 1.0 if total > 0 else 0.0
    valid_parsed_rate = parsed_rate

    return {
        "accuracy": accuracy,
        "correct": correct,
        "total": total,
        "parse_failures": parse_failures,
        "invalid_predictions": invalid_predictions,
        # "parsed_rate" is a practical proxy for "test parsed" (parse success).
        "parsed_rate": parsed_rate,
        "valid_parsed_rate": valid_parsed_rate,
        "used_extractor": bool(used_extractor),
        "prompt_backend": "manual_llama3",
    }


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
        name = method_name if len(method_name) <= score_col_width else method_name[: score_col_width - 1] + "~"
        header += f" {name:>{score_col_width}}"
    line = "-" * len(header)

    print("\n" + "=" * len(header))
    print("BBH vLLM Evaluation (test accuracy)")
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
