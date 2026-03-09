import argparse
import json
from pathlib import Path

from transformers import AutoTokenizer
from vllm import LLM

from utils import TASK_EXTRACTOR_TEXT, ensure_dir, generate_meta_prompts, get_goals_and_targets


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
    p = argparse.ArgumentParser(description="Generate BBH meta prompts and split into shard JSON files.")
    p.add_argument("--task_name", required=True, choices=sorted(TASK_EXTRACTOR_TEXT.keys()))
    p.add_argument("--num_meta_prompts", type=int, default=1000)
    p.add_argument("--num_shards", type=int, default=4)
    p.add_argument("--output_root", type=str, default="tmp")
    p.add_argument("--data_root", type=str, default="data/GreaTer_data/BBH")

    p.add_argument("--model_path", type=str, default="google/gemma-2-9b-it")
    p.add_argument("--conversation_template", type=str, default="gemma-2")
    p.add_argument("--dtype", type=str, default="bfloat16")
    p.add_argument("--tensor_parallel_size", type=int, default=1)
    p.add_argument("--gpu_memory_utilization", type=float, default=0.90)
    p.add_argument("--chat_backend", type=str, default="hf_chat_template")

    p.add_argument("--n_train_data", type=int, default=50)
    p.add_argument("--n_test_data", type=int, default=100)
    p.add_argument("--meta_prompt_temperature", type=float, default=0.8)
    p.add_argument("--meta_prompt_top_p", type=float, default=0.95)
    p.add_argument("--meta_prompt_max_tokens", type=int, default=96)
    p.add_argument("--meta_prompt_num_examples", type=int, default=3)
    p.add_argument("--meta_prompt_generation_batch_size", type=int, default=128)
    p.add_argument("--meta_prompt_generation_max_rounds", type=int, default=6)
    p.add_argument("--backfill_duplicates", type=str2bool, default=True)
    return p.parse_args()


def _resolve(path_like, base_dir):
    path = Path(path_like)
    if path.is_absolute():
        return path
    cwd_candidate = Path.cwd() / path
    if cwd_candidate.exists():
        return cwd_candidate
    return base_dir / path


def _chunk_sizes(total, n_shards):
    base = total // n_shards
    rem = total % n_shards
    return [base + (1 if i < rem else 0) for i in range(n_shards)]


def main():
    args = parse_args()
    base_dir = Path(__file__).resolve().parent
    output_root = _resolve(args.output_root, base_dir)
    data_root = _resolve(args.data_root, base_dir)

    if args.num_meta_prompts <= 0:
        raise ValueError("--num_meta_prompts must be > 0")
    if args.num_shards <= 0:
        raise ValueError("--num_shards must be > 0")

    task = args.task_name
    data_file = data_root / f"{task}.json"
    if not data_file.exists():
        raise FileNotFoundError(f"Missing data file: {data_file}")

    print(f"[prepare] task={task}")
    print(f"[prepare] model={args.model_path}")
    print(f"[prepare] output_root={output_root}")

    hf_tokenizer = None
    if args.chat_backend in {"auto", "hf_chat_template"}:
        hf_tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
        print(
            "[prepare] hf tokenizer loaded "
            f"(chat_template={'yes' if bool(getattr(hf_tokenizer, 'chat_template', None)) else 'no'})"
        )

    llm = LLM(
        model=args.model_path,
        tokenizer=args.model_path,
        dtype=args.dtype,
        trust_remote_code=True,
        tensor_parallel_size=args.tensor_parallel_size,
        gpu_memory_utilization=args.gpu_memory_utilization,
    )

    extractor_text = TASK_EXTRACTOR_TEXT[task]
    train_goals, _, _, _, train_targets, _ = get_goals_and_targets(
        data_path=str(data_file),
        extractor_text=extractor_text,
        conversation_template_name=args.conversation_template,
        n_train_data=args.n_train_data,
        n_test_data=args.n_test_data,
    )

    generated_dict, meta_request = generate_meta_prompts(
        llm=llm,
        conversation_template_name=args.conversation_template,
        task_name=task,
        train_goals=train_goals,
        train_final_targets=train_targets,
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
    prompts = [str(p).strip() for p in generated_dict.values() if str(p).strip()]
    unique_count = len(prompts)

    if not prompts:
        raise RuntimeError("No meta prompts were generated.")

    if len(prompts) < args.num_meta_prompts:
        if not args.backfill_duplicates:
            raise RuntimeError(
                f"Generated only {len(prompts)} prompts (requested {args.num_meta_prompts}) and backfill is disabled."
            )
        print(
            f"[prepare] generated {len(prompts)} unique prompts; backfilling duplicates to reach {args.num_meta_prompts}"
        )
        idx = 0
        while len(prompts) < args.num_meta_prompts:
            prompts.append(prompts[idx % unique_count])
            idx += 1

    prompts = prompts[: args.num_meta_prompts]
    shard_dir = output_root / f"{task}_corr{args.num_meta_prompts}_bbh_eval_shards"
    ensure_dir(shard_dir)

    prompts_list_path = output_root / f"{task}_corr{args.num_meta_prompts}_prompts_list.json"
    prompts_meta_path = output_root / f"{task}_corr{args.num_meta_prompts}_prompts_meta.json"

    with prompts_list_path.open("w", encoding="utf-8") as f:
        json.dump(prompts, f, ensure_ascii=False, indent=2)

    with prompts_meta_path.open("w", encoding="utf-8") as f:
        json.dump(
            {
                "task_name": task,
                "num_requested": args.num_meta_prompts,
                "num_generated_unique": unique_count,
                "num_final": len(prompts),
                "num_shards": args.num_shards,
                "generator": {
                    "model_path": args.model_path,
                    "conversation_template": args.conversation_template,
                    "temperature": args.meta_prompt_temperature,
                    "top_p": args.meta_prompt_top_p,
                    "max_tokens": args.meta_prompt_max_tokens,
                    "num_examples": args.meta_prompt_num_examples,
                    "generation_batch_size_cap": args.meta_prompt_generation_batch_size,
                    "generation_max_rounds": args.meta_prompt_generation_max_rounds,
                    "chat_backend": args.chat_backend,
                },
                "meta_prompt_request": meta_request,
            },
            f,
            ensure_ascii=False,
            indent=2,
        )

    sizes = _chunk_sizes(len(prompts), args.num_shards)
    start = 0
    for shard_idx, size in enumerate(sizes):
        chunk = prompts[start : start + size]
        payload = {
            f"Meta-{start + i + 1:04d}": prompt
            for i, prompt in enumerate(chunk)
        }
        shard_path = shard_dir / f"prompts_shard_{shard_idx}.json"
        with shard_path.open("w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)
        print(f"[prepare] shard{shard_idx}: {len(chunk)} prompts -> {shard_path}")
        start += size

    print(f"[prepare] done. unique={unique_count}, final={len(prompts)}")
    print(f"[prepare] prompts_list={prompts_list_path}")
    print(f"[prepare] prompt_shards_dir={shard_dir}")


if __name__ == "__main__":
    main()
