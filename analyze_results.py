import argparse
import json
import os

os.environ["TOKENIZERS_PARALLELISM"] = "true"

import openai
import pandas as pd
import torch
from sentence_transformers import SentenceTransformer

from utils import check_keywords, check_with_unaligned, judge_prompt, openai_request, test_prefixes


def parse_binary_judge(text):
    text = str(text).strip()
    return 1 if ("1" in text and "0" not in text) else 0


def main():
    parser = argparse.ArgumentParser(description="Evaluate saved target-model responses with GPT-4o judge")
    parser.add_argument("--file_path", type=str, required=True, help="CSV file produced by test_policy_seq.py")
    parser.add_argument("--target_model", type=str, required=True, help="Target model name used for output naming")
    parser.add_argument("--openai_key", type=str, default=None, help="OpenAI key; defaults to OPENAI_API_KEY")
    parser.add_argument("--reference_path", type=str, default="datasets/processed_unalign.csv",
                        help="CSV with question,response columns for BGE similarity reporting")
    parser.add_argument("--judge_model", type=str, default="gpt-4o")
    parser.add_argument("--output_path", type=str, default=None)
    parser.add_argument("--cuda_id", type=int, default=0)
    args = parser.parse_args()

    openai.api_key = args.openai_key or os.environ.get("OPENAI_API_KEY")

    responses_df = pd.read_csv(args.file_path)
    responses = responses_df["response"].astype(str).tolist()
    questions = responses_df["question"].astype(str).tolist()

    reference_df = pd.read_csv(args.reference_path)
    reference_map = dict(
        zip(reference_df["question"].astype(str), reference_df["response"].astype(str))
    )

    device = torch.device(f"cuda:{args.cuda_id}" if torch.cuda.is_available() else "cpu")
    embedder = SentenceTransformer("BAAI/bge-large-en-v1.5", device=device)

    keyword_asr = []
    similarities = []
    gpt_judge_success = 0
    log = []

    for i, (question, response) in enumerate(zip(questions, responses), start=1):
        print(f"[{i}/{len(questions)}] judging response_chars={len(response)}", flush=True)
        keyword_success = check_keywords(response, test_prefixes)
        keyword_asr.append(keyword_success)

        if question in reference_map:
            similarities.append(check_with_unaligned(embedder, response, reference_map[question]))

        judge_result_text = openai_request(
            judge_prompt.format(question=question, response=response),
            model=args.judge_model,
        ).choices[0].message.content
        judge_success = parse_binary_judge(judge_result_text)
        gpt_judge_success += judge_success

        log.append(
            {
                "question": question,
                "response": response,
                "GPT-Judge Result": judge_result_text,
                "GPT-Judge Success": judge_success,
            }
        )

    total = len(questions)
    results = {
        "test_data_size": total,
        "GPT-Judge-Success": gpt_judge_success,
        "GPT-Judge-ASR": gpt_judge_success / total if total else 0.0,
        "Keyword-ASR": float(sum(keyword_asr) / len(keyword_asr)) if keyword_asr else 0.0,
        "Mean-BGE-Similarity": float(sum(similarities) / len(similarities)) if similarities else None,
        "log": log,
    }

    print(f"{args.target_model} Keyword ASR: {results['Keyword-ASR']}")
    print(f"{args.target_model} Mean BGE Similarity: {results['Mean-BGE-Similarity']}")
    print(f"{args.target_model} GPT-Judge ASR: {results['GPT-Judge-ASR']}")

    output_path = args.output_path or f"{args.target_model.split('/')[-1]}.json"
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"Saved results to {output_path}")


if __name__ == "__main__":
    main()
