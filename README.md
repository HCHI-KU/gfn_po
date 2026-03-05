# BBH vLLM Eval (2-file standalone)

Standalone BBH evaluator for vLLM using two files only:
- `main.py`: CLI / runner / result saving
- `utils.py`: data loading, prompt templating, parsing, evaluation helpers

This is intended for reproducible baseline comparisons (e.g., GreaTer baseline vs. your method under the same evaluation code).

## What This Code Is For

- Evaluate BBH tasks with vLLM (`Meta-Llama-3-8B-Instruct`, etc.)
- Run:
  - `GreaTer-compatible anchored 1-round` main evaluation (recommended baseline protocol)
  - `OPRO-style` auxiliary evaluation with optional second-round extraction
- Report:
  - `accuracy`
  - `parse_failures`
  - `invalid_predictions`

## Evaluation Protocols

### 1) Main baseline protocol (recommended)

This is the default when `--opro_second_round_extraction false`.

- 1-round anchored extraction (extract final answer once)
- strict task-specific parsing (regex)
- minimal normalization
- decoding stabilization (`min_tokens=1`, fixed extractor format instruction)

Use this for:
- baseline vs method comparison in your paper (under your fixed protocol)

### 2) OPRO-style auxiliary protocol

Enable with `--opro_second_round_extraction true`.

- After the first extraction, asks again with:
  - `So the final answer is`
- Helps reduce parse failures
- Useful as auxiliary/ablation analysis (not recommended as main GreaTer baseline protocol)

## Metrics

- `accuracy`: exact match after parsing
- `parse_failures`: parser could not parse the model output into a valid answer format
- `invalid_predictions`: parseable but invalid-for-task outputs (e.g., `Neither A nor B` for an `A/B` task)

Both `parse_failures` and `invalid_predictions` count as wrong predictions in accuracy.

## Requirements

Use `requirements.txt` in this folder.

Tested environment (this workspace):
- Python `3.12`
- `vllm 0.11.1`
- `transformers 4.57.1`
- `fschat 0.2.36`

## Setup

```bash
cd bbh_vllm_eval
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

If using gated Llama models:

```bash
huggingface-cli login
```

## Data Layout

`--data_root` must contain BBH task files like:

```text
<data_root>/tracking_shuffled_objects_five_objects.json
<data_root>/object_counting.json
<data_root>/causal_judgement.json
<data_root>/movie_recommendation.json
<data_root>/hyperbaton.json
```

Bundled in this repo:
- `./data/GreaTer_data/BBH`

External example (previous workspace path):
- `/home/work/jihun/gfn_po/data/GreaTer_data/BBH`

## Run Examples

### A. Main baseline (GreaTer-compatible, 1-round anchored extraction)

All 5 tasks, `Opt` prompt only, `test=100`, 4 GPUs:

```bash
cd bbh_vllm_eval
mkdir -p tmp && printf '{}\n' > tmp/empty_prompts.json

NCCL_SHM_DISABLE=1 NCCL_P2P_DISABLE=1 NCCL_IB_DISABLE=1 \
CUDA_VISIBLE_DEVICES=0,1,2,3 \
python3 -u main.py \
  --data_root ./data/GreaTer_data/BBH \
  --save_dir save \
  --exp_name run_opt_all5_train50_test100_tp4_greater_main \
  --model_path meta-llama/Meta-Llama-3-8B-Instruct \
  --conversation_template llama-3 \
  --n_train_data 50 \
  --n_test_data 100 \
  --generate_meta_prompts false \
  --meta_prompt_file tmp/empty_prompts.json \
  --num_meta_prompts 0 \
  --include_paper_opt_prompt true \
  --tensor_parallel_size 4 \
  --dtype bfloat16 \
  --opro_second_round_extraction false
```

### B. OPRO-style auxiliary evaluation (2nd-round extraction ON)

Same setup as above, but enable second-round extraction:

```bash
NCCL_SHM_DISABLE=1 NCCL_P2P_DISABLE=1 NCCL_IB_DISABLE=1 \
CUDA_VISIBLE_DEVICES=0,1,2,3 \
python3 -u main.py \
  --data_root ./data/GreaTer_data/BBH \
  --save_dir save \
  --exp_name run_opt_all5_train50_test100_tp4_opro_aux \
  --model_path meta-llama/Meta-Llama-3-8B-Instruct \
  --conversation_template llama-3 \
  --n_train_data 50 \
  --n_test_data 100 \
  --generate_meta_prompts false \
  --meta_prompt_file tmp/empty_prompts.json \
  --num_meta_prompts 0 \
  --include_paper_opt_prompt true \
  --tensor_parallel_size 4 \
  --dtype bfloat16 \
  --opro_second_round_extraction true \
  --opro_second_round_max_tokens 50
```

Note:
- The code now uses task-specific second-round max tokens internally.
- `--opro_second_round_max_tokens` acts as a fallback default.

### C. Single task run (main protocol)

```bash
CUDA_VISIBLE_DEVICES=0 python3 -u main.py \
  --task_name object_counting \
  --data_root ./data/GreaTer_data/BBH \
  --generate_meta_prompts false \
  --meta_prompt_file tmp/empty_prompts.json \
  --num_meta_prompts 0 \
  --include_paper_opt_prompt true \
  --exp_name run_object_counting_opt_main \
  --opro_second_round_extraction false
```

### D. Meta prompt generation + evaluation

```bash
CUDA_VISIBLE_DEVICES=0 python3 -u main.py \
  --task_name object_counting \
  --data_root ./data/GreaTer_data/BBH \
  --n_train_data 50 \
  --n_test_data 100 \
  --generate_meta_prompts true \
  --num_meta_prompts 100 \
  --include_paper_opt_prompt true \
  --exp_name run_meta100_object_counting
```

## Output Files

Saved under `save/<exp_name>/`:

- `eval_prompts_snapshot.json`
- `paper_prompts_<task>.json`
- `paper_prompts_all_results.json`
- `generated_meta_prompts/<task>.json` (when prompt generation is enabled)

## Important Notes (for paper writing)

- This is suitable for **internal fair comparison** if you evaluate all methods with the same fixed protocol.
- Do **not** describe this as an exact reproduction of GreaTer's original evaluation code.
- Recommended wording:
  - `GreaTer-compatible baseline (our re-evaluation under a fixed anchored one-round strict protocol)`
- `main.py` currently evaluates the **test split only** (it still loads `n_train_data` for prompt generation / split offset).
