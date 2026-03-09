import argparse
import csv
import json
import math
from pathlib import Path

import matplotlib.pyplot as plt


def parse_args():
    p = argparse.ArgumentParser(description="Merge sharded bbh_vllm_eval results and plot summary.")
    p.add_argument("--task_name", required=True)
    p.add_argument("--output_dir", required=True)
    p.add_argument("--shard_dirs", nargs="+", required=True)
    p.add_argument("--expected_rows", type=int, default=0)
    return p.parse_args()


def _safe_float(x, default=0.0):
    try:
        return float(x)
    except Exception:
        return default


def _safe_int(x, default=0):
    try:
        return int(x)
    except Exception:
        return default


def load_rows(task_name, shard_dirs):
    rows = []
    paper_acc = None
    for shard_dir in shard_dirs:
        shard_path = Path(shard_dir)
        json_path = shard_path / f"paper_prompts_{task_name}.json"
        if not json_path.exists():
            raise FileNotFoundError(f"Missing result file: {json_path}")
        data = json.loads(json_path.read_text())
        if paper_acc is None and "paper_acc" in data:
            paper_acc = data["paper_acc"]
        for method_name, item in data.items():
            if method_name == "paper_acc":
                continue
            rows.append(
                {
                    "method_name": method_name,
                    "prompt": str(item.get("prompt", "")),
                    "accuracy": _safe_float(item.get("accuracy")),
                    "correct": _safe_int(item.get("correct")),
                    "total": _safe_int(item.get("total")),
                    "parse_failures": _safe_int(item.get("parse_failures")),
                    "invalid_predictions": _safe_int(item.get("invalid_predictions")),
                    "inference_time": _safe_float(item.get("inference_time")),
                    "prompt_backend": item.get("prompt_backend"),
                    "answer_max_tokens": item.get("answer_max_tokens"),
                    "opro_second_round_extraction": bool(item.get("opro_second_round_extraction", False)),
                    "opro_second_round_max_tokens": item.get("opro_second_round_max_tokens"),
                    "train_accuracy": _safe_float(item.get("train_accuracy")) if "train_accuracy" in item else None,
                    "train_correct": _safe_int(item.get("train_correct")) if "train_correct" in item else None,
                    "train_total": _safe_int(item.get("train_total")) if "train_total" in item else None,
                    "train_parse_failures": _safe_int(item.get("train_parse_failures"))
                    if "train_parse_failures" in item else None,
                    "train_invalid_predictions": _safe_int(item.get("train_invalid_predictions"))
                    if "train_invalid_predictions" in item else None,
                    "train_inference_time": _safe_float(item.get("train_inference_time"))
                    if "train_inference_time" in item else None,
                    "source_dir": shard_path.name,
                }
            )
    return rows, paper_acc


def write_csv(rows, path):
    fieldnames = [
        "method_name",
        "prompt",
        "accuracy",
        "correct",
        "total",
        "parse_failures",
        "invalid_predictions",
        "inference_time",
        "prompt_backend",
        "answer_max_tokens",
        "opro_second_round_extraction",
        "opro_second_round_max_tokens",
        "train_accuracy",
        "train_correct",
        "train_total",
        "train_parse_failures",
        "train_invalid_predictions",
        "train_inference_time",
        "source_dir",
    ]
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)


def summarize(rows, paper_acc):
    accs = [r["accuracy"] for r in rows]
    pfs = [r["parse_failures"] for r in rows]
    invs = [r["invalid_predictions"] for r in rows]
    times = [r["inference_time"] for r in rows]
    train_accs = [r["train_accuracy"] for r in rows if r.get("train_accuracy") is not None]
    n = len(rows)

    def mean(xs):
        return sum(xs) / len(xs) if xs else 0.0

    def std(xs):
        if len(xs) < 2:
            return 0.0
        m = mean(xs)
        return math.sqrt(sum((x - m) ** 2 for x in xs) / (len(xs) - 1))

    by_acc_desc = sorted(rows, key=lambda r: (r["accuracy"], -r["parse_failures"]), reverse=True)
    by_acc_asc = sorted(rows, key=lambda r: (r["accuracy"], r["parse_failures"]))
    summary = {
        "num_rows": n,
        "num_unique_method_names": len({r["method_name"] for r in rows}),
        "num_unique_prompts": len({r["prompt"] for r in rows}),
        "paper_acc": paper_acc,
        "accuracy_mean": mean(accs),
        "accuracy_std": std(accs),
        "accuracy_min": min(accs) if accs else None,
        "accuracy_max": max(accs) if accs else None,
        "parse_failures_mean": mean(pfs),
        "parse_failures_total": sum(pfs),
        "invalid_predictions_total": sum(invs),
        "inference_time_mean_sec": mean(times),
        "top5": [
            {
                "method_name": r["method_name"],
                "accuracy": r["accuracy"],
                "parse_failures": r["parse_failures"],
                "prompt": r["prompt"],
            }
            for r in by_acc_desc[:5]
        ],
        "bottom5": [
            {
                "method_name": r["method_name"],
                "accuracy": r["accuracy"],
                "parse_failures": r["parse_failures"],
                "prompt": r["prompt"],
            }
            for r in by_acc_asc[:5]
        ],
    }
    if train_accs:
        summary["train_accuracy_mean"] = mean(train_accs)
        summary["train_accuracy_std"] = std(train_accs)
        summary["train_accuracy_min"] = min(train_accs)
        summary["train_accuracy_max"] = max(train_accs)
    return summary


def write_summary_txt(summary, path):
    lines = [
        "BBH vLLM Eval Shard Merge Summary",
        f"num_rows: {summary['num_rows']}",
        f"num_unique_method_names: {summary['num_unique_method_names']}",
        f"num_unique_prompts: {summary['num_unique_prompts']}",
        f"paper_acc: {summary['paper_acc']}",
        f"accuracy_mean: {summary['accuracy_mean']:.6f}",
        f"accuracy_std: {summary['accuracy_std']:.6f}",
        f"accuracy_min: {summary['accuracy_min']}",
        f"accuracy_max: {summary['accuracy_max']}",
        f"parse_failures_mean: {summary['parse_failures_mean']:.6f}",
        f"parse_failures_total: {summary['parse_failures_total']}",
        f"invalid_predictions_total: {summary['invalid_predictions_total']}",
        f"inference_time_mean_sec: {summary['inference_time_mean_sec']:.4f}",
    ]
    if "train_accuracy_mean" in summary:
        lines.extend(
            [
                f"train_accuracy_mean: {summary['train_accuracy_mean']:.6f}",
                f"train_accuracy_std: {summary['train_accuracy_std']:.6f}",
                f"train_accuracy_min: {summary['train_accuracy_min']}",
                f"train_accuracy_max: {summary['train_accuracy_max']}",
            ]
        )
    lines.extend(["", "Top 5 by accuracy:"])
    for row in summary["top5"]:
        lines.append(
            f"- {row['method_name']}: acc={row['accuracy']:.3f}, parse_failures={row['parse_failures']} | {row['prompt']}"
        )
    lines.append("")
    lines.append("Bottom 5 by accuracy:")
    for row in summary["bottom5"]:
        lines.append(
            f"- {row['method_name']}: acc={row['accuracy']:.3f}, parse_failures={row['parse_failures']} | {row['prompt']}"
        )
    path.write_text("\n".join(lines) + "\n")


def plot_summary(rows, summary, out_path):
    accs = [r["accuracy"] for r in rows]
    pfs = [r["parse_failures"] for r in rows]
    times = [r["inference_time"] for r in rows]
    sorted_accs = sorted(accs, reverse=True)

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))

    ax = axes[0, 0]
    ax.hist(accs, bins=20, color="#1f77b4", alpha=0.85, edgecolor="white")
    ax.axvline(summary["accuracy_mean"], color="#d62728", linestyle="--", label=f"mean={summary['accuracy_mean']:.3f}")
    if summary.get("paper_acc") is not None:
        ax.axvline(summary["paper_acc"], color="#2ca02c", linestyle=":", label=f"paper={summary['paper_acc']:.2f}")
    ax.set_title("Accuracy Distribution")
    ax.set_xlabel("Accuracy")
    ax.set_ylabel("Count")
    ax.legend(fontsize=8)

    ax = axes[0, 1]
    ax.plot(range(1, len(sorted_accs) + 1), sorted_accs, color="#ff7f0e", linewidth=1.5)
    ax.set_title("Sorted Accuracy Curve")
    ax.set_xlabel("Rank")
    ax.set_ylabel("Accuracy")
    ax.grid(alpha=0.2)

    ax = axes[1, 0]
    ax.scatter(pfs, accs, s=10, alpha=0.5, color="#9467bd")
    ax.set_title("Accuracy vs Parse Failures")
    ax.set_xlabel("Parse Failures (out of 100)")
    ax.set_ylabel("Accuracy")
    ax.grid(alpha=0.2)

    ax = axes[1, 1]
    ax.scatter(times, accs, s=10, alpha=0.5, color="#2ca02c")
    ax.set_title("Accuracy vs Inference Time")
    ax.set_xlabel("Inference Time (sec / prompt)")
    ax.set_ylabel("Accuracy")
    ax.grid(alpha=0.2)

    fig.suptitle(
        f"bbh_vllm_eval merged results ({len(rows)} prompts)\nmean={summary['accuracy_mean']:.3f}, "
        f"max={summary['accuracy_max']:.3f}, unique_prompts={summary['num_unique_prompts']}",
        fontsize=11,
    )
    fig.tight_layout()
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def _pearson(xs, ys):
    n = len(xs)
    if n < 2:
        return float("nan")
    mx = sum(xs) / n
    my = sum(ys) / n
    vx = sum((x - mx) ** 2 for x in xs)
    vy = sum((y - my) ** 2 for y in ys)
    if vx <= 0 or vy <= 0:
        return float("nan")
    cov = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    return cov / math.sqrt(vx * vy)


def plot_train_vs_test_if_available(rows, out_path):
    xs = []
    ys = []
    for r in rows:
        train_acc = r.get("train_accuracy")
        test_acc = r.get("accuracy")
        if train_acc is None or test_acc is None:
            continue
        xs.append(train_acc)
        ys.append(test_acc)
    if not xs:
        return None
    p = _pearson(xs, ys)
    plt.figure(figsize=(6.5, 5.5))
    plt.scatter(xs, ys, s=12, alpha=0.5)
    plt.xlabel("train_accuracy")
    plt.ylabel("test_accuracy")
    plt.title("Train vs Test Accuracy (bbh_vllm_eval)")
    plt.grid(alpha=0.2)
    plt.text(0.02, 0.98, f"n={len(xs)}\npearson={p:.4f}", transform=plt.gca().transAxes, va="top")
    plt.tight_layout()
    plt.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close()
    return {"n_pairs": len(xs), "pearson_train_vs_test": p}


def main():
    args = parse_args()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    rows, paper_acc = load_rows(args.task_name, args.shard_dirs)
    if args.expected_rows and len(rows) != args.expected_rows:
        print(f"[WARN] expected_rows={args.expected_rows}, got {len(rows)}")

    rows_sorted = sorted(rows, key=lambda r: r["method_name"])
    summary = summarize(rows_sorted, paper_acc)

    csv_path = out_dir / f"merged_{args.task_name}_results.csv"
    json_path = out_dir / f"merged_{args.task_name}_summary.json"
    txt_path = out_dir / f"merged_{args.task_name}_summary.txt"
    png_path = out_dir / f"merged_{args.task_name}_summary.png"
    train_test_png_path = out_dir / f"merged_{args.task_name}_train_vs_test.png"

    write_csv(rows_sorted, csv_path)
    json_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2))
    write_summary_txt(summary, txt_path)
    plot_summary(rows_sorted, summary, png_path)
    train_test_info = plot_train_vs_test_if_available(rows_sorted, train_test_png_path)
    if train_test_info is not None:
        summary["train_vs_test_plot"] = train_test_info
        json_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2))

    print(f"Merged rows: {len(rows_sorted)}")
    print(f"Unique prompts: {summary['num_unique_prompts']}")
    print(f"Accuracy mean/std: {summary['accuracy_mean']:.4f}/{summary['accuracy_std']:.4f}")
    print(f"Accuracy min/max: {summary['accuracy_min']:.4f}/{summary['accuracy_max']:.4f}")
    print(f"Parse failures total: {summary['parse_failures_total']}")
    print(f"Saved CSV: {csv_path}")
    print(f"Saved summary: {txt_path}")
    print(f"Saved plot: {png_path}")
    if train_test_info is not None:
        print(f"Saved train/test plot: {train_test_png_path}")


if __name__ == "__main__":
    main()
