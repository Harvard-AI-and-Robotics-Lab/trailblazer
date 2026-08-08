import argparse
import hashlib
import json
import os
import random
import time

os.environ["TOKENIZERS_PARALLELISM"] = "true"

import numpy as np
import openai
import pandas as pd
import torch
from fastchat.model import add_model_args

from a2c_ppo_acktr.envs_seq import make_vec_envs_seq
from test_policy_seq import _redact_keys, preprocess_templates


def get_base_env(vec_env):
    """Return the underlying MutatorSelectSeq env from VecNormalize/DummyVecEnv wrappers."""
    if hasattr(vec_env, "venv") and hasattr(vec_env.venv, "envs"):
        return vec_env.venv.envs[0]
    if hasattr(vec_env, "envs"):
        return vec_env.envs[0]
    raise RuntimeError("Could not locate base environment for attention diagnostics")


def observation_size(history_size):
    base_embedding_size = 1024
    current_embedding_size = base_embedding_size
    history_embedding_size = history_size * base_embedding_size
    history_response_size = history_size * 4
    history_reward_size = history_size
    history_mutator_size = history_size
    metadata_size = 3
    return (
        current_embedding_size
        + history_embedding_size
        + history_response_size
        + history_reward_size
        + history_mutator_size
        + metadata_size
    )


def prepare_template_file(args):
    source_model = args.source_model or args.target_model
    template_index = args.template_index if args.template_index is not None else args.index
    training_templates_path = (
        f"datasets/prompts_generated/RL_{source_model.split('/')[-1]}_{template_index}.csv"
    )
    default_processed_path = (
        f"datasets/prompts_generated/RL_{source_model.split('/')[-1]}_{template_index}_processed.csv"
    )
    args.template_path = args.template_output_path or default_processed_path
    if args.preserve_processed_order:
        args.template_path = default_processed_path
        if not os.path.exists(args.template_path):
            raise FileNotFoundError(
                f"--preserve_processed_order requires an existing processed template file: {args.template_path}"
            )
        processed_df = pd.read_csv(args.template_path)
        if "processed_templates" not in processed_df.columns:
            raise RuntimeError(f"{args.template_path} must contain a processed_templates column")
        print(
            f"Preserving existing processed template order from {args.template_path}",
            flush=True,
        )
        return source_model, args.template_path

    df = pd.read_csv(training_templates_path)
    templates = df["template"].tolist()

    if "processed_templates" not in df.columns:
        df["processed_templates"] = preprocess_templates(templates, source_model)
        df.to_csv(training_templates_path, index=False)

    try:
        sorted_df = df.sort_values(by="success_q", ascending=False)
    except Exception:
        sorted_df = df

    sorted_df["processed_templates"].to_csv(args.template_path, index=False)
    if args.template_output_path:
        print(
            f"Wrote legacy success_q-sorted processed templates to {args.template_path}",
            flush=True,
        )
    return source_model, args.template_path


def attention_to_numpy(actor_critic):
    attn_module = getattr(actor_critic, "sequence_attention", None)
    if attn_module is None:
        return None
    attn = getattr(attn_module, "last_attention_weights_raw", None)
    if attn is None:
        return None
    attn = attn.detach().cpu()
    if attn.dim() == 4:
        # [B, heads, 1, K] -> [B, K]
        attn = attn.mean(dim=1).squeeze(1)
    elif attn.dim() == 3:
        # [B, 1, K] -> [B, K]
        attn = attn.squeeze(1)
    else:
        raise RuntimeError(f"Unexpected attention weight shape: {tuple(attn.shape)}")
    return attn.numpy()


def tensorize_obs(obs, device):
    if torch.is_tensor(obs):
        return obs.float().to(device)
    return torch.FloatTensor(obs).to(device)


def run_attention_unit_smoke(args):
    device = torch.device(f"cuda:{args.cuda_id}" if torch.cuda.is_available() else "cpu")
    actor_critic = torch.load(args.ckpt_path, map_location=device, weights_only=False)[0]
    actor_critic.to(device)

    obs_size = observation_size(args.history_size)
    dummy_obs = torch.zeros(args.num_processes, obs_size, device=device)
    # Make each history slot distinct so softmax is not only testing all-zero padding.
    for slot in range(args.history_size):
        start = 1024 + slot * 1024
        dummy_obs[:, start : start + 1024] = float(slot + 1) / args.history_size
    dummy_obs[:, :1024] = 0.5

    recurrent_hidden_states = torch.zeros(
        args.num_processes, actor_critic.recurrent_hidden_state_size, device=device
    )
    masks = torch.zeros(args.num_processes, 1, device=device)
    with torch.no_grad():
        actor_critic.act(dummy_obs, recurrent_hidden_states, masks)

    attn = attention_to_numpy(actor_critic)
    if attn is None:
        raise RuntimeError("No attention weights were captured")
    row_sums = attn.sum(axis=1)
    print(f"[unit-smoke] attention shape: {attn.shape}")
    print(f"[unit-smoke] row-sum min/max: {row_sums.min():.6f}/{row_sums.max():.6f}")
    if attn.shape != (args.num_processes, args.history_size):
        raise RuntimeError(
            f"Expected attention shape {(args.num_processes, args.history_size)}, got {attn.shape}"
        )
    if not np.allclose(row_sums, 1.0, atol=1e-5):
        raise RuntimeError("Attention weights do not sum to 1")


def write_records(path, records):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


def summarize_current_step(diag):
    if not diag:
        return {
            "current_step_success_max": None,
            "current_step_success_mean": None,
            "current_step_vulnerability_max": None,
            "current_step_vulnerability_mean": None,
            "current_step_judge_parse_success_count": 0,
            "current_step_response_count": 0,
            "current_step_question_count": 0,
            "current_step_action": None,
            "current_step_mutation": None,
            "current_step_question_hashes": [],
        }
    attack_results = diag.get("attack_results") or []
    questions = diag.get("questions") or []
    numeric_results = []
    for value in attack_results:
        try:
            numeric_results.append(float(value))
        except Exception:
            pass
    judge_results = diag.get("judge_results") or []
    vulnerability_scores = []
    judge_parse_success_count = 0
    for item in judge_results:
        if not isinstance(item, dict):
            continue
        if item.get("parse_success"):
            judge_parse_success_count += 1
        try:
            score = item.get("vulnerability_score")
            if score is not None:
                vulnerability_scores.append(float(score))
        except Exception:
            pass
    return {
        "current_step_success_max": max(numeric_results) if numeric_results else None,
        "current_step_success_mean": float(np.mean(numeric_results)) if numeric_results else None,
        "current_step_vulnerability_max": max(vulnerability_scores) if vulnerability_scores else None,
        "current_step_vulnerability_mean": float(np.mean(vulnerability_scores)) if vulnerability_scores else None,
        "current_step_judge_parse_success_count": judge_parse_success_count,
        "current_step_response_count": len(diag.get("responses") or []),
        "current_step_question_count": len(diag.get("questions") or []),
        "current_step_action": diag.get("action"),
        "current_step_mutation": diag.get("mutation"),
        "current_step_question_hashes": [
            hashlib.sha256(str(question).encode("utf-8")).hexdigest()[:16]
            for question in questions
        ],
    }


def main():
    parser = argparse.ArgumentParser(description="Test TrailBlazer AHRL policy while logging attention")
    parser.add_argument("--openai_key", type=str, default=None)
    parser.add_argument("--deepinfra_key", type=str, default=None)
    parser.add_argument("--anthropic_key", type=str, default=None)
    parser.add_argument("--gemini_key", type=str, default=None)
    parser.add_argument("--model_path", type=str, default="gpt-3.5-turbo")
    parser.add_argument("--target_model", type=str, required=True)
    parser.add_argument("--max_query", type=int, default=10000)
    parser.add_argument("--num_processes", type=int, default=4)
    parser.add_argument("--datasets", default="advbench")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--cuda_id", type=int, default=0)
    parser.add_argument("--index", type=int, required=True)
    parser.add_argument(
        "--template_index",
        type=int,
        default=None,
        help="Template-pool index to load. Defaults to --index; output CSVs still use --index.",
    )
    parser.add_argument("--defense", type=str, default="none")
    parser.add_argument("--history_size", type=int, required=True)
    parser.add_argument("--ckpt_path", type=str, required=True)
    parser.add_argument("--K", type=int, default=1000)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--max_attempts_per_question", type=int, default=50)
    parser.add_argument("--source_model", type=str, default=None)
    parser.add_argument("--seqattn", choices=["none", "simple", "multi_head"], default="simple")
    parser.add_argument("--seqattn_heads", type=int, default=4)
    parser.add_argument("--seqattn_dropout", type=float, default=0.1)
    parser.add_argument("--use_step_improvement", action="store_true", default=False)
    parser.add_argument("--lambda_coef", type=float, default=0.1)
    parser.add_argument(
        "--combined_judge",
        action="store_true",
        default=False,
        help="Use one GPT-4o judge call that returns both binary success and 0-10 vulnerability score.",
    )
    parser.add_argument("--attention_log_path", type=str, required=True)
    parser.add_argument(
        "--preserve_processed_order",
        action="store_true",
        default=False,
        help="Use the existing *_processed.csv order directly instead of sorting raw templates by success_q.",
    )
    parser.add_argument(
        "--template_output_path",
        type=str,
        default=None,
        help="Optional processed-template output path when using legacy success_q sorting. This avoids overwriting the canonical *_processed.csv.",
    )
    parser.add_argument(
        "--legacy_unseeded_random",
        action="store_true",
        default=False,
        help="Match legacy test_policy_seq.py RNG behavior: do not seed NumPy/Python random in this script.",
    )
    parser.add_argument(
        "--max_policy_decisions",
        type=int,
        default=0,
        help="Maximum policy decisions to log; <=0 runs until the environment stop condition.",
    )
    parser.add_argument("--unit_smoke_only", action="store_true")
    add_model_args(parser)
    args = parser.parse_args()

    openai.api_key = args.openai_key or os.environ.get("OPENAI_API_KEY")
    args.num_gpus = 1

    device = torch.device(f"cuda:{args.cuda_id}" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    if args.legacy_unseeded_random:
        print("Legacy RNG mode: not seeding NumPy/Python random", flush=True)
    else:
        np.random.seed(args.seed)
        random.seed(args.seed)

    print("Experiment arguments ", _redact_keys(args), flush=True)
    print(f"Sequence attention mechanism: {args.seqattn}", flush=True)
    print(f"Step-wise Improvement Reward: {'Enabled' if args.use_step_improvement else 'Disabled'}", flush=True)

    if args.unit_smoke_only:
        run_attention_unit_smoke(args)
        return

    source_model, template_path = prepare_template_file(args)
    print(f"Using source model: {source_model} (for template path)", flush=True)
    print("================================================", flush=True)
    print(f"Loading training templates from {template_path}", flush=True)
    print("================================================", flush=True)

    total_obs_size = observation_size(args.history_size)
    print(f"Total observation size (raw): {total_obs_size}", flush=True)
    envs = make_vec_envs_seq(
        args,
        args.num_processes,
        total_obs_size,
        args.cuda_id,
        eval=True,
        history_size=args.history_size,
    )
    base_env = get_base_env(envs)

    actor_critic = torch.load(args.ckpt_path, map_location=device, weights_only=False)[0]
    actor_critic.to(device)
    if not (hasattr(actor_critic, "use_sequence_attention") and actor_critic.use_sequence_attention):
        raise RuntimeError("Loaded policy does not have sequence attention enabled")

    if os.path.exists(args.attention_log_path):
        os.remove(args.attention_log_path)

    obs = envs.reset()
    recurrent_hidden_states = torch.zeros(
        args.num_processes, actor_critic.recurrent_hidden_state_size, device=device
    )
    masks = torch.zeros(args.num_processes, 1, device=device)

    start = time.time()
    decisions = 0
    episodes = 0

    while args.max_policy_decisions <= 0 or decisions < args.max_policy_decisions:
        diagnostics_before = base_env.get_attention_diagnostics()
        with torch.no_grad():
            _, action, _, recurrent_hidden_states = actor_critic.act(
                tensorize_obs(obs, device), recurrent_hidden_states, masks
            )

        attn = attention_to_numpy(actor_critic)
        if attn is None:
            raise RuntimeError("No attention weights were captured after actor_critic.act")

        records = []
        action_list = action.detach().cpu().view(-1).tolist()
        for proc_idx in range(min(args.num_processes, attn.shape[0])):
            proc_diags = diagnostics_before[proc_idx] if proc_idx < len(diagnostics_before) else []
            for slot_idx in range(args.history_size):
                diag = proc_diags[slot_idx] if slot_idx < len(proc_diags) else {}
                records.append(
                    {
                        "decision_id": decisions,
                        "process_index": proc_idx,
                        "slot_index": slot_idx,
                        "history_size": args.history_size,
                        "alpha": float(attn[proc_idx, slot_idx]),
                        "selected_action": int(action_list[proc_idx]) if proc_idx < len(action_list) else None,
                        "is_padding": not bool(diag),
                        "history_diagnostic": diag,
                    }
                )

        obs, _, done, infos = envs.step(action)
        current_diags = getattr(base_env, "last_step_diagnostics", [{} for _ in range(args.num_processes)])
        for record in records:
            proc_idx = record["process_index"]
            current_diag = current_diags[proc_idx] if proc_idx < len(current_diags) else {}
            record.update(summarize_current_step(current_diag))
        write_records(args.attention_log_path, records)
        decisions += 1

        masks = torch.tensor(
            [[0.0] if done_ else [1.0] for done_ in done],
            dtype=torch.float32,
            device=device,
        )

        if done[0]:
            episodes += 1
            obs = envs.reset()
            recurrent_hidden_states = torch.zeros(
                args.num_processes, actor_critic.recurrent_hidden_state_size, device=device
            )
            masks = torch.zeros(args.num_processes, 1, device=device)

        if len(infos) > 0 and isinstance(infos[0], dict) and "stop" in infos[0]:
            print("Evaluation stop condition reached.", flush=True)
            break

    elapsed = time.time() - start
    print(f"Attention logging completed in {elapsed:.2f}s", flush=True)
    print(f"Policy decisions logged: {decisions}", flush=True)
    print(f"Episodes completed: {episodes}", flush=True)
    print(f"Attention log: {args.attention_log_path}", flush=True)


if __name__ == "__main__":
    torch.multiprocessing.set_start_method("spawn", force=True)
    main()
