import argparse
import time
from pathlib import Path

from transformers import AutoTokenizer, GenerationConfig
from vllm import LLM, SamplingParams

from utils import (
    ANSWER_EXTRACTION_STOPS,
    PAPER_ACC,
    PAPER_OPT_PROMPTS,
    TASK_ANSWER_MAX_TOKENS,
    TASK_OPRO_SECOND_ROUND_MAX_TOKENS,
    TASK_EXTRACTOR_TEXT,
    ensure_dir,
    evaluate_prompt,
    generate_meta_prompts,
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
    parser = argparse.ArgumentParser(description="Standalone BBH vLLM evaluator")
    parser.add_argument("--model_path", type=str, default="meta-llama/Meta-Llama-3-8B-Instruct")
    parser.add_argument("--conversation_template", type=str, default="llama-3")
    parser.add_argument(
        "--chat_backend",
        type=str,
        choices=["auto", "vllm_chat", "hf_chat_template", "manual"],
        default="hf_chat_template",
    )
    parser.add_argument("--task_name", type=str, default="")
    parser.add_argument("--data_root", type=str, default="data/GreaTer_data/BBH")

    parser.add_argument("--save_dir", type=str, default="save")
    parser.add_argument("--exp_name", type=str, default="bbh_eval_vllm_standalone")

    parser.add_argument("--n_train_data", type=int, default=50)
    parser.add_argument("--n_test_data", type=int, default=100)

    parser.add_argument("--generate_meta_prompts", type=str2bool, default=True)
    parser.add_argument("--meta_prompt_file", type=str, default="")
    parser.add_argument("--num_meta_prompts", type=int, default=5)
    parser.add_argument("--include_paper_opt_prompt", type=str2bool, default=False)

    parser.add_argument("--meta_prompt_temperature", type=float, default=0.8)
    parser.add_argument("--meta_prompt_top_p", type=float, default=0.95)
    parser.add_argument("--meta_prompt_max_tokens", type=int, default=96)
    parser.add_argument("--meta_prompt_num_examples", type=int, default=3)
    parser.add_argument("--meta_prompt_generation_batch_size", type=int, default=128)
    parser.add_argument("--meta_prompt_generation_max_rounds", type=int, default=6)

    parser.add_argument("--max_reasoning_tokens", type=int, default=1024)
    parser.add_argument("--answer_max_tokens", type=int, default=1)
    parser.add_argument("--opro_second_round_extraction", type=str2bool, default=False)
    parser.add_argument("--opro_second_round_max_tokens", type=int, default=50)
    parser.add_argument(
        "--evaluate_train_split",
        type=str2bool,
        default=False,
        help="Also evaluate and save train-split metrics per prompt (doubles eval cost).",
    )

    parser.add_argument("--dtype", type=str, default="bfloat16")
    parser.add_argument("--tensor_parallel_size", type=int, default=1)
    parser.add_argument("--gpu_memory_utilization", type=float, default=0.90)
    return parser.parse_args()


def script_root():
    return Path(__file__).resolve().parent


def resolve_path(path_like, base_dir):
    path = Path(path_like)
    if path.is_absolute():
        return path

    # Prefer caller cwd for relative paths when it already exists.
    cwd_candidate = Path.cwd() / path
    if cwd_candidate.exists():
        return cwd_candidate

    # Fallback to script directory.
    return base_dir / path


def insert_prompts_unique(dst, src):
    for name, prompt in src.items():
        key = str(name)
        text = str(prompt).strip()
        if not text:
            continue
        unique_key = key
        suffix = 2
        while unique_key in dst:
            unique_key = f"{key}_{suffix}"
            suffix += 1
        dst[unique_key] = text


def main():
    args = parse_args()
    base_dir = script_root()

    if args.task_name and args.task_name not in TASK_EXTRACTOR_TEXT:
        raise ValueError(
            f"Unknown task_name: {args.task_name}. "
            f"Choices: {sorted(TASK_EXTRACTOR_TEXT.keys())}"
        )

    save_base = resolve_path(args.save_dir, base_dir)
    output_dir = save_base / args.exp_name
    generated_prompts_dir = output_dir / "generated_meta_prompts"
    ensure_dir(output_dir)
    ensure_dir(generated_prompts_dir)

    print(f"[BBH Eval] output_dir: {output_dir}")
    print(f"[BBH Eval] model_path: {args.model_path}")
    print(f"[BBH Eval] dtype: {args.dtype} (default is bfloat16)")
    print(f"[BBH Eval] chat_backend: {args.chat_backend}")

    try:
        gen_config = GenerationConfig.from_pretrained(args.model_path)
        eval_top_p = gen_config.top_p if gen_config.top_p is not None else 1.0
    except Exception:
        eval_top_p = 1.0

    hf_tokenizer = None
    if args.chat_backend in {"auto", "hf_chat_template"}:
        try:
            hf_tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
            print(
                "[BBH Eval] HF tokenizer loaded for chat-template rendering "
                f"(chat_template={'yes' if bool(getattr(hf_tokenizer, 'chat_template', None)) else 'no'})"
            )
        except Exception as e:
            if args.chat_backend == "hf_chat_template":
                raise RuntimeError(
                    "chat_backend=hf_chat_template was requested, but tokenizer loading failed."
                ) from e
            print(f"[BBH Eval] HF tokenizer load failed, fallback to manual formatting may be used: {e}")

    llm = LLM(
        model=args.model_path,
        tokenizer=args.model_path,
        dtype=args.dtype,
        trust_remote_code=True,
        tensor_parallel_size=args.tensor_parallel_size,
        gpu_memory_utilization=args.gpu_memory_utilization,
    )
    if args.chat_backend == "vllm_chat" and not callable(getattr(llm, "chat", None)):
        print("[BBH Eval] WARNING: this vLLM version does not support LLM.chat(); fallback path will be used.")
    reasoning_params = SamplingParams(
        temperature=0.0,
        top_p=eval_top_p,
        max_tokens=args.max_reasoning_tokens,
    )

    tasks_to_eval = (
        {args.task_name: TASK_EXTRACTOR_TEXT[args.task_name]}
        if args.task_name
        else TASK_EXTRACTOR_TEXT.copy()
    )
    print(f"[BBH Eval] tasks: {list(tasks_to_eval.keys())}")

    payload = None
    if args.generate_meta_prompts:
        if args.meta_prompt_file:
            print("[BBH Eval] generate_meta_prompts=True, so meta_prompt_file is ignored.")
        if args.num_meta_prompts <= 0:
            raise ValueError("--num_meta_prompts must be > 0 when --generate_meta_prompts=True.")
    else:
        if not args.meta_prompt_file:
            raise ValueError("Set --meta_prompt_file when --generate_meta_prompts=False.")
        payload = load_json(resolve_path(args.meta_prompt_file, base_dir))

    task_data = {}
    eval_prompts = {}

    for task_name, extractor_text in tasks_to_eval.items():
        data_root = resolve_path(args.data_root, base_dir)
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

        if args.generate_meta_prompts:
            generated, meta_request = generate_meta_prompts(
                llm=llm,
                conversation_template_name=args.conversation_template,
                task_name=task_name,
                train_goals=train_goals,
                train_final_targets=train_final_targets,
                target_count=args.num_meta_prompts,
                temperature=args.meta_prompt_temperature,
                top_p=args.meta_prompt_top_p,
                max_tokens=args.meta_prompt_max_tokens,
                num_examples=args.meta_prompt_num_examples,
                max_batch_size=args.meta_prompt_generation_batch_size,
                max_rounds=args.meta_prompt_generation_max_rounds,
                hf_tokenizer=hf_tokenizer,
                chat_backend=args.chat_backend,
            )
            insert_prompts_unique(prompt_dict, generated)
            if len(generated) < args.num_meta_prompts:
                print(
                    f"[WARN] {task_name}: generated {len(generated)} prompts "
                    f"(requested {args.num_meta_prompts})."
                )
            save_json(
                generated_prompts_dir / f"{task_name}.json",
                {
                    "task_name": task_name,
                    "generated_at": now(),
                    "generation_config": {
                        "model_path": args.model_path,
                        "conversation_template": args.conversation_template,
                        "temperature": args.meta_prompt_temperature,
                        "top_p": args.meta_prompt_top_p,
                        "max_tokens": args.meta_prompt_max_tokens,
                        "num_examples": args.meta_prompt_num_examples,
                        "requested_num_prompts": args.num_meta_prompts,
                        "actual_num_prompts": len(generated),
                        "generation_batch_size_cap": args.meta_prompt_generation_batch_size,
                        "generation_max_rounds": args.meta_prompt_generation_max_rounds,
                    },
                    "meta_prompt_request": meta_request,
                    "prompts": [
                        {"name": name, "prompt": prompt}
                        for name, prompt in generated.items()
                    ],
                },
            )
            print(f"[BBH Eval] {task_name}: saved generated prompts")
        else:
            loaded = resolve_prompts_from_payload(
                payload=payload,
                task_name=task_name,
                known_task_names=set(tasks_to_eval.keys()),
            )
            if args.num_meta_prompts > 0:
                loaded = dict(list(loaded.items())[:args.num_meta_prompts])
            elif args.num_meta_prompts == 0:
                loaded = {}
            insert_prompts_unique(prompt_dict, loaded)

        eval_prompts[task_name] = prompt_dict
        print(f"[BBH Eval] {task_name}: {len(prompt_dict)} prompts ready")

    save_json(
        output_dir / "eval_prompts_snapshot.json",
        {
            "generated_at": now(),
            "args": vars(args),
            "tasks": eval_prompts,
        },
    )
    print("[BBH Eval] saved eval prompt snapshot")

    all_results = {}
    for task_name, prompts in eval_prompts.items():
        if not prompts:
            print(f"[SKIP] {task_name}: no prompts")
            continue

        test_goals = task_data[task_name]["test_goals"]
        test_final_targets = task_data[task_name]["test_final_targets"]
        train_goals = task_data[task_name]["train_goals"]
        train_final_targets = task_data[task_name]["train_final_targets"]
        answer_max_tokens = TASK_ANSWER_MAX_TOKENS.get(task_name, args.answer_max_tokens)
        second_round_max_tokens = TASK_OPRO_SECOND_ROUND_MAX_TOKENS.get(
            task_name, args.opro_second_round_max_tokens
        )
        answer_params = SamplingParams(
            temperature=0.0,
            top_p=eval_top_p,
            max_tokens=answer_max_tokens,
            min_tokens=1,
            stop=ANSWER_EXTRACTION_STOPS,
        )
        print(f"[BBH Eval] {task_name}: answer_max_tokens={answer_max_tokens}")
        if args.opro_second_round_extraction:
            print(f"[BBH Eval] {task_name}: opro_second_round_max_tokens={second_round_max_tokens}")
        if args.evaluate_train_split:
            print(f"[BBH Eval] {task_name}: train split evaluation enabled")

        task_results = {"paper_acc": PAPER_ACC.get(task_name, None)}
        for method_name, prompt in prompts.items():
            print(f"\n{'=' * 60}")
            print(f"Task: {task_name} | Method: {method_name}")
            print(f"Prompt: {prompt[:120]}{'...' if len(prompt) > 120 else ''}")
            print(f"{'=' * 60}")

            second_round_params = SamplingParams(
                temperature=0.0,
                top_p=eval_top_p,
                max_tokens=second_round_max_tokens,
                min_tokens=1,
                stop=ANSWER_EXTRACTION_STOPS,
            ) if args.opro_second_round_extraction else None

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
                    opro_second_round_extraction=args.opro_second_round_extraction,
                    opro_second_round_answer_params=second_round_params,
                    hf_tokenizer=hf_tokenizer,
                    chat_backend=args.chat_backend,
                )
                train_elapsed = time.time() - t_train
                print(
                    f"  [train] Accuracy: {train_result['accuracy']:.4f} "
                    f"({train_result['correct']}/{train_result['total']}) "
                    f"| Parse failures: {train_result.get('parse_failures', 0)} "
                    f"| Invalid: {train_result.get('invalid_predictions', 0)} "
                    f"| Time: {train_elapsed:.1f}s"
                )

            t0 = time.time()
            result = evaluate_prompt(
                llm=llm,
                conversation_template_name=args.conversation_template,
                task_name=task_name,
                goals=test_goals,
                final_targets=test_final_targets,
                control_prompt=prompt,
                reasoning_params=reasoning_params,
                answer_params=answer_params,
                opro_second_round_extraction=args.opro_second_round_extraction,
                opro_second_round_answer_params=second_round_params,
                hf_tokenizer=hf_tokenizer,
                chat_backend=args.chat_backend,
            )
            elapsed = time.time() - t0
            entry = {
                "prompt": prompt,
                "accuracy": result["accuracy"],
                "inference_time": elapsed,
                "correct": result["correct"],
                "total": result["total"],
                "parse_failures": result.get("parse_failures", 0),
                "invalid_predictions": result.get("invalid_predictions", 0),
                "prompt_backend": result.get("prompt_backend"),
                "answer_max_tokens": answer_max_tokens,
                "opro_second_round_extraction": bool(args.opro_second_round_extraction),
                "opro_second_round_max_tokens": second_round_max_tokens if args.opro_second_round_extraction else None,
            }
            if train_result is not None:
                entry.update(
                    {
                        "train_accuracy": train_result["accuracy"],
                        "train_inference_time": train_elapsed,
                        "train_correct": train_result["correct"],
                        "train_total": train_result["total"],
                        "train_parse_failures": train_result.get("parse_failures", 0),
                        "train_invalid_predictions": train_result.get("invalid_predictions", 0),
                    }
                )
            task_results[method_name] = entry
            print(
                f"  [test] Accuracy: {result['accuracy']:.4f} "
                f"({result['correct']}/{result['total']}) "
                f"| Backend: {result.get('prompt_backend')} "
                f"| Parse failures: {result.get('parse_failures', 0)} "
                f"| Invalid: {result.get('invalid_predictions', 0)} "
                f"| Time: {elapsed:.1f}s"
            )

        all_results[task_name] = task_results
        save_json(output_dir / f"paper_prompts_{task_name}.json", task_results)

    print_summary_table(all_results)
    save_json(output_dir / "paper_prompts_all_results.json", all_results)
    print(f"[BBH Eval] done. results saved to: {output_dir}")


if __name__ == "__main__":
    main()
