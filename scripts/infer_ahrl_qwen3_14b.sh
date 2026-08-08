#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

: "${OPENAI_API_KEY:?OPENAI_API_KEY is required for GPT-3.5 helper and GPT-4o judge}"
: "${DEEPINFRA_API_KEY:?DEEPINFRA_API_KEY is required for DeepInfra target calls}"

INDEX="${INDEX:-0}"
CKPT_PATH="${CKPT_PATH:-trained_models/ppo_seq/qwen3_14b_ahrl_index${INDEX}_best.pt}"

python test_policy_seq.py \
  --target_model Qwen/Qwen3-14B \
  --model_path gpt-3.5-turbo \
  --ckpt_path "${CKPT_PATH}" \
  --index "${INDEX}" \
  --datasets advbench \
  --history_size 4 \
  --seqattn simple \
  --max_query 10000 \
  --max_attempts_per_question 50 \
  --K 1000 \
  --num_processes "${NUM_PROCESSES:-4}" \
  --cuda_id "${CUDA_ID:-0}"
