#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

: "${OPENAI_API_KEY:?OPENAI_API_KEY is required for the GPT-3.5 helper}"
: "${GEMINI_API_KEY:?GEMINI_API_KEY is required for Gemini target calls}"

python train_policy_seq.py \
  --env-name gemini31_flash_lite_ahrl \
  --target_model gemini-3.1-flash-lite \
  --model_path gpt-3.5-turbo \
  --index "${INDEX:-0}" \
  --datasets advbench \
  --history_size 4 \
  --seqattn simple \
  --attention_weight_mode learned \
  --max_query 10000 \
  --num_processes "${NUM_PROCESSES:-4}" \
  --cuda_id "${CUDA_ID:-0}"
