#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

: "${OPENAI_API_KEY:?OPENAI_API_KEY is required for GPT-4o vulnerability scoring}"

ATTN_LOG="${ATTN_LOG:-reports/qwen3_attention.jsonl}"
GRADED_CACHE="${GRADED_CACHE:-reports/qwen3_graded_cache.jsonl}"
BGE_CACHE="${BGE_CACHE:-reports/qwen3_bge_cache.jsonl}"

python analyze_attention_per_question.py \
  --attention_log_path "${ATTN_LOG}" \
  --graded_cache_path "${GRADED_CACHE}" \
  --bge_cache_path "${BGE_CACHE}" \
  --reference_path datasets/processed_unalign.csv \
  --grade_missing \
  --compute_bge \
  --bge_device "${BGE_DEVICE:-cpu}"
