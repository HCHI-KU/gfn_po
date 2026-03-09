import argparse
import time
from pathlib import Path

from transformers import GenerationConfig
from vllm import LLM, SamplingParams

from utils_final import (
    ANSWER_EXTRACTION_STOPS,
    PAPER_ACC,
    PAPER_OPT_PROMPTS,
    SUPPORTED_TASKS,
    TASK_ANSWER_MAX_TOKENS,
    TASK_EXTRACTOR_TEXT,
    ensure_dir,
    evaluate_prompt,
    get_goals_and_targets,
    load_json,
    now,
    print_summary_table,
    resolve_prompts_from_payload,
    save_json,
)


def str2bool(value):
    if isinstance(value, bool):
        return value
    value = str(value).strip().lower()
    if value in {"1", "true", "t", "yes", "y"}:
        return True
    if value in {"0", "false", "f", "no", "n"}:
        return False
    raise argparse.ArgumentTypeError(f"Invalid bool value: {value}")


def parse_args():
    p = argparse.ArgumentParser(
        description=(
            "Standalone 2-file BBH vLLM evaluator (GreaTer-style extractor evaluation). "
            "This is a re-implementation for internal fair comparison, not an exact "
            "reproduction of GreaTer's original eval code."
        )
    )

    # Model / backend
    p.add_argument("--model_path", type=str, default="meta-llama/Meta-Llama-3-8B-Instruct")
    p.add_argument("--conversation_template", type=str, default="llama-3")
    p.add_argument("--dtype", type=str, default="bfloat16")
    p.add_argument("--tensor_parallel_size", type=int, default=1)
    p.add_argument("--gpu_memory_utilization", type=float, default=0.90)

    # Data / tasks
    p.add_argument("--task_name", type=str, default="")
    p.add_argument("--data_root", type=str, default="../bbh_vllm_eval/data/GreaTer_data/BBH")
    p.add_argument("--n_train_data", type=int, default=50)
    p.add_argument("--n_test_data", type=int, default=100)

    # Experiment output
    p.add_argument("--save_dir", type=str, default="save")
    p.add_argument("--exp_name", type=str, default="bbh_eval_vllm_twofile")

    # Prompt sources (evaluation only; no prompt generation in this script)
    p.add_argument("--include_paper_opt_prompt", type=str2bool, default=True)
    p.add_argument(
        "--meta_prompt_file",
        type=str,
        default="",
        help="Optional JSON prompt payload (task-keyed dict/list or generated_meta_prompts/<task>.json format).",
    )
    p.add_argument(
        "--num_meta_prompts",
        type=int,
        default=0,
        help="Limit loaded meta prompts. 0 means do not load any meta prompts.",
    )

    p.add_argument(
        "--evaluate_train_split",
        type=str2bool,
        default=False,
        help="Also evaluate train split per prompt (useful for later train-test correlation analysis).",
    )

    # Decoding (argmax-style by default)
    p.add_argument("--temperature", type=float, default=0.0)
    p.add_argument("--max_reasoning_tokens", type=int, default=256)
    p.add_argument("--answer_max_tokens", type=int, default=1)
    return p.parse_args()


def script_root():
    return Path(__file__).resolve().parent


def resolve_path(path_like, base_dir):
    path = Path(path_like)
    if path.is_absolute():
        return path
    cwd_candidate = Path.cwd() / path
    if cwd_candidate.exists():
        return cwd_candidate
    return base_dir / path


def insert_prompts_unique(dst, src):
    for name, prompt in src.items():
        name = str(name)
        prompt = str(prompt).strip()
        if not prompt:
            continue
        unique_name = name
        suffix = 2
        while unique_name in dst:
            unique_name = f"{name}_{suffix}"
            suffix += 1
        dst[unique_name] = prompt


def main():
    args = parse_args()
    base_dir = script_root()

    if args.conversation_template != "llama-3":
        raise ValueError("This minimal 2-file version currently supports --conversation_template llama-3 only.")

    if args.task_name and args.task_name not in TASK_EXTRACTOR_TEXT:
        raise ValueError(f"Unknown task_name: {args.task_name}. Choices: {SUPPORTED_TASKS}")

    data_root = resolve_path(args.data_root, base_dir)
    save_dir = resolve_path(args.save_dir, base_dir)
    output_dir = save_dir / args.exp_name
    ensure_dir(output_dir)

    print(f"[BBH Eval] output_dir: {output_dir}")
    print(f"[BBH Eval] model_path: {args.model_path}")
    print(f"[BBH Eval] conversation_template: {args.conversation_template}")
    print(f"[BBH Eval] decoding: temperature={args.temperature} (argmax-style when 0.0)")
    print(
        "[BBH Eval] note: standalone extractor-based evaluation for internal fair comparison "
        "(not exact reproduction of original GreaTer eval code)"
    )

    try:
        gen_config = GenerationConfig.from_pretrained(args.model_path)
        eval_top_p = gen_config.top_p if gen_config.top_p is not None else 1.0
    except Exception:
        eval_top_p = 1.0

    llm = LLM(
        model=args.model_path,
        tokenizer=args.model_path,
        dtype=args.dtype,
        trust_remote_code=True,
        tensor_parallel_size=args.tensor_parallel_size,
        gpu_memory_utilization=args.gpu_memory_utilization,
    )

    reasoning_params = SamplingParams(
        temperature=args.temperature,
        top_p=eval_top_p,
        max_tokens=args.max_reasoning_tokens,
    )

    tasks_to_eval = (
        {args.task_name: TASK_EXTRACTOR_TEXT[args.task_name]}
        if args.task_name
        else TASK_EXTRACTOR_TEXT.copy()
    )
    print(f"[BBH Eval] tasks: {list(tasks_to_eval.keys())}")

    prompt_payload = None
    if args.meta_prompt_file and args.num_meta_prompts != 0:
        prompt_payload = load_json(resolve_path(args.meta_prompt_file, base_dir))

    task_data = {}
    eval_prompts = {}

    for task_name, extractor_text in tasks_to_eval.items():
        data_file = data_root / f"{task_name}.json"
        if not data_file.exists():
            raise FileNotFoundError(f"Missing data file: {data_file}")

        train_goals, _, test_goals, _, train_final_targets, test_final_targets = get_goals_and_targets(
            data_path=str(data_file),
            extractor_text=extractor_text,
            conversation_template_name=args.conversation_template,
            n_train_data=args.n_train_data,
            n_test_data=args.n_test_data,
        )
        task_data[task_name] = {
            "train_goals": train_goals,
            "train_final_targets": train_final_targets,
            "test_goals": test_goals,
            "test_final_targets": test_final_targets,
        }

        prompt_dict = {}
        if args.include_paper_opt_prompt and task_name in PAPER_OPT_PROMPTS:
            prompt_dict["Opt"] = PAPER_OPT_PROMPTS[task_name]

        if prompt_payload is not None and args.num_meta_prompts != 0:
            loaded = resolve_prompts_from_payload(
                payload=prompt_payload,
                task_name=task_name,
                known_task_names=set(tasks_to_eval.keys()),
            )
            if args.num_meta_prompts > 0:
                loaded = dict(list(loaded.items())[: args.num_meta_prompts])
            insert_prompts_unique(prompt_dict, loaded)

        eval_prompts[task_name] = prompt_dict
        print(f"[BBH Eval] {task_name}: {len(prompt_dict)} prompts ready")

    save_json(
        output_dir / "eval_prompts_snapshot.json",
        {
            "generated_at": now(),
            "args": vars(args),
            "terminology": {
                "extractor": "Second LLM call that extracts a short final answer.",
                "parser": "Postprocessing logic that converts raw output to strict answer type.",
                "test_acc": "Fraction of correct answers on test split.",
                "test_parsed": "Proxy parse-success rate on test split (1 - parse_failures / total).",
            },
            "evaluation_note": (
                "Standalone vLLM evaluator for internal fair comparison. "
                "Describe as GreaTer-compatible baseline re-evaluation, not exact reproduction."
            ),
            "tasks": eval_prompts,
        },
    )
    print("[BBH Eval] saved eval prompt snapshot")

    all_results = {}
    for task_name, prompts in eval_prompts.items():
        if not prompts:
            print(f"[SKIP] {task_name}: no prompts")
            continue

        task_results = {"paper_acc": PAPER_ACC.get(task_name, None)}

        train_goals = task_data[task_name]["train_goals"]
        train_final_targets = task_data[task_name]["train_final_targets"]
        test_goals = task_data[task_name]["test_goals"]
        test_final_targets = task_data[task_name]["test_final_targets"]

        answer_max_tokens = TASK_ANSWER_MAX_TOKENS.get(task_name, args.answer_max_tokens)
        answer_params = SamplingParams(
            temperature=args.temperature,
            top_p=eval_top_p,
            max_tokens=answer_max_tokens,
            min_tokens=1,
            stop=ANSWER_EXTRACTION_STOPS,
        )
        print(f"[BBH Eval] {task_name}: answer_max_tokens={answer_max_tokens}")
        if args.evaluate_train_split:
            print(f"[BBH Eval] {task_name}: train split evaluation enabled")

        for method_name, prompt in prompts.items():
            print(f"\n{'=' * 60}")
            print(f"Task: {task_name} | Method: {method_name}")
            print(f"Prompt: {prompt[:120]}{'...' if len(prompt) > 120 else ''}")
            print(f"{'=' * 60}")

            train_result = None
            train_elapsed = None
            if args.evaluate_train_split:
                t_train = time.time()
                train_result = evaluate_prompt(
                    llm=llm,
                    conversation_template_name=args.conversation_template,
                    task_name=task_name,
                    goals=train_goals,
                    final_targets=train_final_targets,
                    control_prompt=prompt,
                    reasoning_params=reasoning_params,
                    answer_params=answer_params,
                )
                train_elapsed = time.time() - t_train
                print(
                    f"  [train] acc={train_result['accuracy']:.4f} "
                    f"({train_result['correct']}/{train_result['total']}) "
                    f"| parsed={train_result['parsed_rate']:.4f} "
                    f"| parse_failures={train_result['parse_failures']} "
                    f"| invalid={train_result['invalid_predictions']} "
                    f"| time={train_elapsed:.1f}s"
                )

            t0 = time.time()
            test_result = evaluate_prompt(
                llm=llm,
                conversation_template_name=args.conversation_template,
                task_name=task_name,
                goals=test_goals,
                final_targets=test_final_targets,
                control_prompt=prompt,
                reasoning_params=reasoning_params,
                answer_params=answer_params,
            )
            test_elapsed = time.time() - t0
            print(
                f"  [test] acc={test_result['accuracy']:.4f} "
                f"({test_result['correct']}/{test_result['total']}) "
                f"| parsed={test_result['parsed_rate']:.4f} "
                f"| parse_failures={test_result['parse_failures']} "
                f"| invalid={test_result['invalid_predictions']} "
                f"| extractor={test_result['used_extractor']} "
                f"| time={test_elapsed:.1f}s"
            )

            entry = {
                "prompt": prompt,
                # keep compatibility with existing summary-table code (`accuracy` means test acc)
                "accuracy": test_result["accuracy"],
                "inference_time": test_elapsed,
                "correct": test_result["correct"],
                "total": test_result["total"],
                "parse_failures": test_result["parse_failures"],
                "invalid_predictions": test_result["invalid_predictions"],
                "parsed_rate": test_result["parsed_rate"],
                "valid_parsed_rate": test_result["valid_parsed_rate"],
                "used_extractor": test_result["used_extractor"],
                "prompt_backend": test_result["prompt_backend"],
                # aliases for clarity with your terminology
                "test_acc": test_result["accuracy"],
                "test_parsed": test_result["parsed_rate"],
                "answer_max_tokens": answer_max_tokens,
            }
            if train_result is not None:
                entry.update(
                    {
                        "train_accuracy": train_result["accuracy"],
                        "train_inference_time": train_elapsed,
                        "train_correct": train_result["correct"],
                        "train_total": train_result["total"],
                        "train_parse_failures": train_result["parse_failures"],
                        "train_invalid_predictions": train_result["invalid_predictions"],
                        "train_parsed_rate": train_result["parsed_rate"],
                        "train_valid_parsed_rate": train_result["valid_parsed_rate"],
                        "train_acc": train_result["accuracy"],
                        "train_parsed": train_result["parsed_rate"],
                    }
                )

            task_results[method_name] = entry

        all_results[task_name] = task_results
        save_json(output_dir / f"paper_prompts_{task_name}.json", task_results)

    print_summary_table(all_results)
    save_json(output_dir / "paper_prompts_all_results.json", all_results)
    print(f"[BBH Eval] done. results saved to: {output_dir}")


if __name__ == "__main__":
    main()
