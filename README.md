# gl_cde

Inference engine and example scripts for `guidelabs/steerling-8b-instruct`.

**Maintenance disclaimer: this repo is not maintained.**

## Requirements
- [uv](https://docs.astral.sh/uv/)
- Python 3.13–3.14
- A NVIDIA GPU with CUDA 13 (CPU is not supported)

Model weights and the tokenizer are downloaded from Hugging Face on first run.

## Setup
```bash
uv sync --extra cuda
```

## Scripts
Run all scripts from the repo root.

### Example generation
Generates a single chat response.
```bash
uv run --extra cuda python example.py
```

### Throughput
Measures generation throughput and latency on random prompts.
```bash
uv run --extra cuda python throughput.py
```
Flags (defaults in parentheses): `--num-requests` (100), `--num-warmup` (10), `--min-len` / `--max-len`
prompt length in tokens (128 / 1024), `--max-tokens` (256), `--concurrency` (4), `--seed` (0).

### GSM8K benchmark
Runs `inference_lm_eval.py` through lm-eval (GSM8K CoT, 8-shot, 4 concurrent requests, seed 1234).
Run it on a GPU node.
```bash
bash run_inference_benchmark.sh                  # full benchmark (1319 questions)
bash run_inference_benchmark.sh --limit 20       # extra arguments are passed to lm-eval
PROGRESS=false bash run_inference_benchmark.sh   # hide the progress bar
```
Prompts come from `gsm8k_processed.jsonl`, which only supports the `gsm8k_cot_llama` task.

## Metrics
Both `throughput.py` and the benchmark print the following metrics, averaged over requests:

| Metric | Meaning |
|---|---|
| ttfb | Time to first block: from submitting a request to its first block of tokens |
| ibl | Inter-block latency: time between consecutive blocks |
| ttlb | Time to last block: from submitting a request to its final block |
| throughput | Total generated tokens / wall-clock time of the timed run |
