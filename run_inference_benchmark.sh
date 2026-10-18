#!/bin/bash
# PROGRESS=false hides the progress bar
PROGRESS=${PROGRESS:-true}

# extra arguments are passed through to lm-eval, e.g. --limit 20
uv run --group benchmark --extra cuda \
    python inference_lm_eval.py \
    --model modal \
    --model_args "progress=$PROGRESS" \
    --batch_size 4 \
    --tasks gsm8k_cot_llama --apply_chat_template --fewshot_as_multiturn \
    --num_fewshot 8 \
    --gen_kwargs "seed=1234" \
    "$@"
