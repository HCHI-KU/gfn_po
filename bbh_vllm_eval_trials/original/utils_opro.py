from opro_eval_core import (
    PROMPTING_AGAIN_SUFFIX as OPRO_PROMPTING_AGAIN_SUFFIX,
    append_prompting_again_suffix,
    extract_second_round_answer_for_scoring,
    get_accuracy_of_list as opro_get_accuracy_of_list,
)
from utils_final import (
    ANSWER_EXTRACTION_STOPS,
    PAPER_ACC,
    PAPER_OPT_PROMPTS,
    SUPPORTED_TASKS,
    TASK_ANSWER_MAX_TOKENS,
    TASK_EXTRACTOR_TEXT,  # reused for task list / parity only
    ensure_dir,
    get_goals_and_targets,
    load_json,
    now,
    print_summary_table,
    resolve_prompts_from_payload,
    save_json,
    _compose_user_content,
    render_llama3_user_prompt,
)


# Adapter-only task typing (used to feed OPRO core flags).
TASK_IS_MULTIPLE_CHOICE = {
    "tracking_shuffled_objects_five_objects": True,
    "object_counting": False,
    "causal_judgement": False,
    "movie_recommendation": True,
    "hyperbaton": True,
}

TASK_MC_VALID_LETTERS = {
    "tracking_shuffled_objects_five_objects": set("ABCDE"),
    "movie_recommendation": set("ABCDE"),
    "hyperbaton": set("AB"),
}

TASK_TREAT_AS_NUMBER = {
    "object_counting": True,
    "tracking_shuffled_objects_five_objects": False,
    "causal_judgement": False,
    "movie_recommendation": False,
    "hyperbaton": False,
}

TASK_TREAT_AS_BOOL = {
    "causal_judgement": True,
    "tracking_shuffled_objects_five_objects": False,
    "object_counting": False,
    "movie_recommendation": False,
    "hyperbaton": False,
}


def _task_metric_flags(task_name):
    return {
        "is_multiple_choice": bool(TASK_IS_MULTIPLE_CHOICE.get(task_name, False)),
        "valid_letters": TASK_MC_VALID_LETTERS.get(task_name, set("ABCDE")),
        "prediction_treat_as_number": bool(TASK_TREAT_AS_NUMBER.get(task_name, False)),
        "prediction_treat_as_bool": bool(TASK_TREAT_AS_BOOL.get(task_name, False)),
        "target_treat_as_number": bool(TASK_TREAT_AS_NUMBER.get(task_name, False)),
        "target_treat_as_bool": bool(TASK_TREAT_AS_BOOL.get(task_name, False)),
    }


def evaluate_prompt(
    llm,
    conversation_template_name,
    task_name,
    goals,
    final_targets,
    control_prompt,
    reasoning_params,
    answer_params,
    extract_final_answer_by_prompting_again=False,
):
    """
    OPRO-eval adapter:
    - Evaluation logic (normalization / scoring / prompting-again suffix handling)
      is delegated to `opro_eval_core` and should remain unchanged.
    - This function only adapts local vLLM + BBH input/output plumbing.
    """
    if conversation_template_name != "llama-3":
        raise NotImplementedError(
            "This OPRO adapter currently supports conversation_template='llama-3' only."
        )

    if len(goals) != len(final_targets):
        raise ValueError(
            f"Length mismatch for task={task_name}: len(goals)={len(goals)} != len(final_targets)={len(final_targets)}"
        )

    user_contents = [_compose_user_content(goal, control_prompt) for goal in goals]
    stage1_prompts = [render_llama3_user_prompt(x) for x in user_contents]
    outputs = llm.generate(stage1_prompts, reasoning_params, use_tqdm=False)
    stage1_outputs = [o.outputs[0].text.strip() for o in outputs]

    scored_outputs = stage1_outputs
    prompting_again_used = False

    if extract_final_answer_by_prompting_again:
        stage2_prompts = [
            append_prompting_again_suffix(stage1_prompt, stage1_output)
            for stage1_prompt, stage1_output in zip(stage1_prompts, stage1_outputs)
        ]
        outputs2 = llm.generate(stage2_prompts, answer_params, use_tqdm=False)
        stage2_outputs = [o.outputs[0].text for o in outputs2]
        scored_outputs = [extract_second_round_answer_for_scoring(x) for x in stage2_outputs]
        prompting_again_used = True

    metrics = opro_get_accuracy_of_list(
        scored_outputs,
        final_targets,
        **_task_metric_flags(task_name),
    )

    parsed_rate = metrics.get("parsed_rate", 0.0)
    parse_failures = metrics.get("parse_failures", 0)
    total = metrics.get("total", len(final_targets))

    metrics.update(
        {
            "invalid_predictions": 0,  # OPRO metrics-style path does not track this separately.
            "valid_parsed_rate": parsed_rate,
            "used_extractor": bool(prompting_again_used),
            "extract_final_answer_by_prompting_again": bool(prompting_again_used),
            "prompt_backend": "manual_llama3",
            "evaluation_protocol": "opro_eval_adapter_strict_core",
            "oprotask_total": total,  # diagnostic alias, non-scoring
        }
    )
    return metrics
