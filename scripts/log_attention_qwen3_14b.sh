#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

: "${OPENAI_API_KEY:?OPENAI_API_KEY is required for GPT-3.5 helper and GPT-4o judge}"
: "${DEEPINFRA_API_KEY:?DEEPINFRA_API_KEY is required for DeepInfra target calls}"

INDEX="${INDEX:-0}"
CKPT_PATH="${CKPT_PATH:-trained_models/ppo_seq/qwen3_14b_ahrl_index${INDEX}_best.pt}"
ATTN_LOG="${ATTN_LOG:-reports/qwen3_attention.jsonl}"

python test_policy_seq_attention_log.py \
  --target_model Qwen/Qwen3-14B \
  --model_path gpt-3.5-turbo \
  --ckpt_path "${CKPT_PATH}" \
  --index "${INDEX}" \
  --datasets advbench \
  --history_size 4 \
  --seqattn simple \
  --combined_judge \
  --max_query 10000 \
  --max_attempts_per_question 50 \
  --K 1000 \
  --attention_log_path "${ATTN_LOG}" \
  --num_processes "${NUM_PROCESSES:-4}" \
  --cuda_id "${CUDA_ID:-0}"
