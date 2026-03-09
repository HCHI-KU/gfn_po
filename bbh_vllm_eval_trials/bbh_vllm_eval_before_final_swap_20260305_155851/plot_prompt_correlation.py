import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd


def safe_corrcoef(xs, ys):
    if len(xs) < 2:
        return None
    x = pd.Series(xs, dtype="float64")
    y = pd.Series(ys, dtype="float64")
    if x.nunique() <= 1 or y.nunique() <= 1:
        return None
    return float(x.corr(y))


def compute_stats(df):
    if df.empty:
        return {"num_points": 0, "pearson": None, "spearman": None}

    train_vals = df["train_accuracy"].astype(float)
    test_vals = df["test_accuracy"].astype(float)
    pearson = safe_corrcoef(train_vals, test_vals)
    spearman = safe_corrcoef(train_vals.rank(method="average"), test_vals.rank(method="average"))
    return {
        "num_points": int(len(df)),
        "pearson": pearson,
        "spearman": spearman,
        "train_mean": float(train_vals.mean()),
        "test_mean": float(test_vals.mean()),
        "train_std": float(train_vals.std(ddof=0)),
        "test_std": float(test_vals.std(ddof=0)),
    }


def load_prompt_rows(task_json_path):
    payload = json.loads(task_json_path.read_text(encoding="utf-8"))
    rows = []
    for method_name, entry in payload.items():
        if method_name == "paper_acc" or not isinstance(entry, dict):
            continue
        train_acc = entry.get("train_accuracy", entry.get("train_acc"))
        test_acc = entry.get("test_acc", entry.get("accuracy"))
        if train_acc is None or test_acc is None:
            continue
        rows.append(
            {
                "method_name": method_name,
                "train_accuracy": float(train_acc),
                "test_accuracy": float(test_acc),
                "train_correct": entry.get("train_correct"),
                "train_total": entry.get("train_total"),
                "test_correct": entry.get("correct"),
                "test_total": entry.get("total"),
                "prompt": entry.get("prompt", ""),
            }
        )
    return rows


def save_task_artifacts(task_name, task_dir, rows):
    df = pd.DataFrame(rows)
    stats = compute_stats(df)

    csv_path = task_dir / f"train_test_correlation_{task_name}.csv"
    json_path = task_dir / f"train_test_correlation_{task_name}.json"
    png_path = task_dir / f"train_test_correlation_{task_name}.png"

    df.to_csv(csv_path, index=False)
    json_path.write_text(
        json.dumps(
            {
                "task_name": task_name,
                "stats": stats,
                "records": rows,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    fig, ax = plt.subplots(figsize=(6.5, 6.0))
    ax.scatter(df["train_accuracy"], df["test_accuracy"], s=28, alpha=0.75)
    ax.plot([0, 1], [0, 1], linestyle="--", linewidth=1.0, alpha=0.6)
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.02, 1.02)
    ax.set_xlabel("Train Accuracy")
    ax.set_ylabel("Test Accuracy")
    ax.grid(True, alpha=0.25)
    pearson_txt = "NA" if stats["pearson"] is None else f"{stats['pearson']:.3f}"
    spearman_txt = "NA" if stats["spearman"] is None else f"{stats['spearman']:.3f}"
    ax.set_title(
        f"{task_name}\nTrain-Test Prompt Correlation "
        f"(n={stats['num_points']}, r={pearson_txt}, rho={spearman_txt})"
    )
    fig.tight_layout()
    fig.savefig(png_path, dpi=180)
    plt.close(fig)

    return {
        "task_name": task_name,
        "stats": stats,
        "csv_path": str(csv_path),
        "json_path": str(json_path),
        "plot_path": str(png_path),
    }


def main():
    parser = argparse.ArgumentParser(description="Create train/test correlation artifacts from final eval outputs.")
    parser.add_argument("--run_root", type=str, required=True, help="Parent save dir that contains per-task output dirs.")
    parser.add_argument(
        "--run_prefix",
        type=str,
        required=True,
        help="Shared prefix of per-task output dirs, e.g. meta10_train_test_newgen_all5_20260227_162439",
    )
    args = parser.parse_args()

    run_root = Path(args.run_root)
    task_dirs = sorted(
        p for p in run_root.glob(f"{args.run_prefix}_*")
        if p.is_dir() and p.name != args.run_prefix
    )
    if not task_dirs:
        raise FileNotFoundError(f"No task dirs found for prefix: {args.run_prefix}")

    summary = {"run_prefix": args.run_prefix, "tasks": []}
    for task_dir in task_dirs:
        task_name = task_dir.name.replace(f"{args.run_prefix}_", "", 1)
        task_json_path = task_dir / f"paper_prompts_{task_name}.json"
        if not task_json_path.exists():
            continue
        rows = load_prompt_rows(task_json_path)
        if not rows:
            continue
        summary["tasks"].append(save_task_artifacts(task_name, task_dir, rows))

    summary_path = run_root / args.run_prefix / "correlation_summary_all5.json"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(summary_path)


if __name__ == "__main__":
    main()
