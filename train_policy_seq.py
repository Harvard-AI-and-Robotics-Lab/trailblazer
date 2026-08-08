import os
os.environ["TOKENIZERS_PARALLELISM"] = "true"
import openai
import time
import argparse
import json
import numpy as np
import random
from collections import deque

import torch
from torch.utils.tensorboard import SummaryWriter
from a2c_ppo_acktr import algo, utils
from a2c_ppo_acktr.envs import make_vec_envs
from a2c_ppo_acktr.model import Policy
from a2c_ppo_acktr.storage import RolloutStorage
from fastchat.model import add_model_args

# TrailBlazer sequence-aware PPO training entry point.
# See README.md and scripts/ for camera-ready reproduction commands.


def _redact_keys(args):
    display_args = argparse.Namespace(**vars(args))
    for key_name in ("openai_key", "deepinfra_key", "anthropic_key", "gemini_key"):
        if getattr(display_args, key_name, None):
            setattr(display_args, key_name, "***")
    return display_args


def get_base_env(vec_env):
    if hasattr(vec_env, "venv") and hasattr(vec_env.venv, "envs"):
        return vec_env.venv.envs[0]
    if hasattr(vec_env, "envs"):
        return vec_env.envs[0]
    raise RuntimeError("Could not locate base environment for attention diagnostics")


def attention_to_numpy(actor_critic):
    attn_module = getattr(actor_critic, "sequence_attention", None)
    if attn_module is None:
        return None
    attn = getattr(attn_module, "last_attention_weights_raw", None)
    if attn is None:
        return None
    attn = attn.detach().cpu()
    if attn.dim() == 4:
        attn = attn.mean(dim=1).squeeze(1)
    elif attn.dim() == 3:
        attn = attn.squeeze(1)
    else:
        raise RuntimeError(f"Unexpected attention weight shape: {tuple(attn.shape)}")
    return attn.numpy()


def summarize_history_diagnostic(diag):
    if not diag:
        return {
            "is_padding": True,
            "history_step": None,
            "history_process_index": None,
            "history_action": None,
            "history_mutation": None,
            "binary_max": None,
            "binary_mean": None,
            "response_count": 0,
            "question_count": 0,
        }
    attack_results = diag.get("attack_results") or []
    numeric_results = []
    for value in attack_results:
        try:
            numeric_results.append(float(value))
        except Exception:
            pass
    return {
        "is_padding": False,
        "history_step": diag.get("step"),
        "history_process_index": diag.get("process_index"),
        "history_action": diag.get("action"),
        "history_mutation": diag.get("mutation"),
        "binary_max": max(numeric_results) if numeric_results else None,
        "binary_mean": float(np.mean(numeric_results)) if numeric_results else None,
        "response_count": len(diag.get("responses") or []),
        "question_count": len(diag.get("questions") or []),
    }


def write_training_attention_records(path, records):
    if not path:
        return
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


def main():
    parser = argparse.ArgumentParser(description='RL parameters for sequence-based training')
    parser.add_argument('--openai_key', type=str, default=None, help='OpenAI key')
    parser.add_argument('--deepinfra_key', type=str, default=None, help='Deepinfra key')
    parser.add_argument('--anthropic_key', type=str, default=None, help='Anthropic key')
    parser.add_argument('--gemini_key', type=str, default=None, help='Gemini API key')
    parser.add_argument('--model_path', type=str, default='meta-llama/Llama-2-7b-chat-hf', help='openai model or open-sourced LLMs')
    parser.add_argument('--target_model', type=str, default='meta-llama/Llama-2-7b-chat-hf', help='The target model, openai model or open-sourced LLMs')
    parser.add_argument('--max_query', type=int, default=10000, help='The maximum number of queries')
    parser.add_argument('--num_processes',type=int,default=4,help='how many training CPU processes to use (default: 4)')
    parser.add_argument('--num_mini_batch',type=int,default=4,help='number of mini batches for PPO (default: 4)')
    parser.add_argument('--datasets', dest='datasets', action='store', default='advbench', help='name of dataset(s), e.g., agnews')
    parser.add_argument('--seed', type=int, default=1, help='random seed (default: 1)')
    parser.add_argument('--cuda_id',type=int, default=0)
    parser.add_argument('--index', type=int, default=10, help='task id')
    parser.add_argument('--defense', type=str, default="none", help='defense method')
    parser.add_argument('--history_size', type=int, default=4, help='history buffer size for sequence observation')
    parser.add_argument('--use_vec_normalize', action='store_true', default=True, help='use vectorized environment normalization')
    parser.add_argument('--reference_path', type=str, default='./datasets/processed_unalign.csv',
                       help='CSV with question,response columns for BGE reward references')
    parser.add_argument('--reward_embedder_device', type=str, default=None,
                       help='Device for the BGE reward embedder, e.g. cuda:0 or cpu')
    
    # Sequence attention mechanism options (환경 레벨 히스토리 압축용)
    parser.add_argument('--seqattn', type=str, default='none', 
                       choices=['none', 'simple', 'multi_head'], 
                       help='sequence attention mechanism for history compression: none (flatten concat), simple (single-head attention), multi_head (multi-head attention)')
    parser.add_argument('--seqattn_heads', type=int, default=4, help='number of attention heads for sequence attention (only for multi_head)')
    parser.add_argument('--seqattn_dropout', type=float, default=0.1, help='dropout rate for sequence attention mechanism')
    parser.add_argument('--attention_weight_mode', type=str, default='learned',
                       choices=['learned', 'uniform', 'random'],
                       help='attention weighting for ablation: learned (full AHRL), uniform, or random')
    
    # Step-wise Improvement Reward options
    parser.add_argument('--use_step_improvement', action='store_true', default=False,
                       help='Enable step-wise improvement reward (Δ Reward)')
    parser.add_argument('--lambda_coef', type=float, default=0.1, 
                       help='λ coefficient for step-wise improvement reward (default: 0.1)')
    parser.add_argument('--attention_log_path', type=str, default=None,
                       help='Optional sanitized JSONL path for training-time attention diagnostics')
    parser.add_argument('--attention_log_every', type=int, default=1,
                       help='Log one training attention snapshot every N policy decisions')
    
    add_model_args(parser)
    utils.add_train_args(parser)
    args = parser.parse_args()
    
    openai.api_key = args.openai_key or os.environ.get("OPENAI_API_KEY")
    
    args.num_gpus = 1
    device = torch.device("cuda:{}".format(args.cuda_id))

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    print("Experiment arguments ", _redact_keys(args), flush=True)
    print(f"Using sequence-based observation with history size: {args.history_size}", flush=True)
    print(f"Sequence attention mechanism: {args.seqattn}", flush=True)
    print(f"Attention weight mode: {args.attention_weight_mode}", flush=True)
    print(f"Step-wise Improvement Reward: {'Enabled' if args.use_step_improvement else 'Disabled'}", flush=True)
    if args.use_step_improvement:
        print(f"Lambda coefficient (λ): {args.lambda_coef}", flush=True)

    if args.use_value:
        print("use value network in PPO.")
    else:
        print("Not use value network in PPO.")

    # 시퀀스 기반 관찰값 크기 계산 (환경에서는 항상 raw 히스토리 전달)
    base_embedding_size = 1024  # BAAI/bge-large-en-v1.5 임베딩 크기
    
    # 환경에서 전달하는 관찰값 구성:
    # - 현재 템플릿 임베딩: 1024
    # - 히스토리 템플릿 임베딩 시퀀스: history_size * 1024
    # - 히스토리 응답 피처 시퀀스: history_size * 4
    # - 히스토리 보상 시퀀스: history_size
    # - 히스토리 mutator ID 시퀀스: history_size
    # - 스텝 수: 1
    # - 종료 플래그: 1
    # - 이전 액션: 1
    
    history_embedding_size = args.history_size * base_embedding_size
    history_response_size = args.history_size * 4
    history_reward_size = args.history_size
    history_mutator_size = args.history_size
    current_embedding_size = base_embedding_size
    metadata_size = 3  # steps + terminate + prev_action
    
    total_obs_size = (history_embedding_size + history_response_size + 
                     history_reward_size + history_mutator_size + 
                     current_embedding_size + metadata_size)
    
    print(f"Total observation size (raw): {total_obs_size}", flush=True)
    print(f"Sequence attention will be applied in policy network", flush=True)
    
    # 벡터화된 환경 생성 (시퀀스 버전 사용)
    from a2c_ppo_acktr.envs_seq import make_vec_envs_seq
    envs = make_vec_envs_seq(args, args.num_processes, total_obs_size, args.cuda_id, eval=False, history_size=args.history_size)
    base_env = get_base_env(envs)

    # 관찰 공간 크기 계산
    obs_shape = (total_obs_size,)
    num_blocks = int(obs_shape[0] / base_embedding_size)  # 블록 수는 기본 임베딩 크기 기준
    
    print(f"Observation shape: {obs_shape}", flush=True)
    print(f"Number of blocks: {num_blocks}", flush=True)
    
    # 정책 네트워크 생성
    actor_critic = Policy(
        obs_shape,
        envs.action_space,
        args.use_attention,
        device,
        num_blocks,
        base_kwargs={"recurrent": False, "hidden_size": 1024},  # 시퀀스 정보를 위해 recurrent=True
        use_sequence_attention=(args.seqattn != "none"),
        seqattn_type=args.seqattn,
        history_size=args.history_size,
        seqattn_weight_mode=args.attention_weight_mode,
        seqattn_heads=args.seqattn_heads,
        seqattn_dropout=args.seqattn_dropout,
    )
    actor_critic.to(device)

    # PPO 에이전트 생성
    agent = algo.PPO(
        actor_critic,
        args.clip_param,
        args.ppo_epoch,
        args.num_mini_batch,
        args.value_loss_coef,
        args.entropy_coef,
        lr=args.lr,
        eps=args.eps,
        max_grad_norm=args.max_grad_norm,
    )

    # 롤아웃 스토리지 생성
    rollouts = RolloutStorage(
        args.num_steps,
        args.num_processes,
        obs_shape,
        envs.action_space,
        actor_critic.recurrent_hidden_state_size,
    )

    obs = envs.reset()
    rollouts.obs[0].copy_(obs)
    rollouts.to(device)

    episode_rewards = deque(maxlen=10)

    # TensorBoard writer 설정
    log_dir = f"runs/{args.env_name}_index{args.index}"
    writer = SummaryWriter(log_dir)
    print(f"TensorBoard logging to: {log_dir}", flush=True)

    start = time.time()
    num_updates = int(args.num_env_steps) // args.num_steps // args.num_processes
    # num_updates = 20

    # import pdb; pdb.set_trace()
    cur_best_ep_reward = 0.0
    final_checkpoint_saved = False

    print(f"Starting training with {num_updates} updates...", flush=True)
    if args.attention_log_path:
        if os.path.exists(args.attention_log_path):
            os.remove(args.attention_log_path)
        print(f"Training attention diagnostics: {args.attention_log_path}", flush=True)

    for j in range(num_updates): #6000//32//4 = 46
        utils.update_linear_schedule(
            agent.optimizer,
            j,
            num_updates,
            agent.optimizer.lr if args.algo == "acktr" else args.lr,
        )

        for step in range(args.num_steps): 
            # 액션 샘플링
            diagnostics_before = base_env.get_attention_diagnostics() if args.attention_log_path else None
            with torch.no_grad():
                value, action, action_log_prob, recurrent_hidden_states = actor_critic.act(
                    rollouts.obs[step], rollouts.recurrent_hidden_states[step],
                    rollouts.masks[step])

            if args.attention_log_path:
                decision_id = j * args.num_steps + step
                if args.attention_log_every <= 1 or decision_id % args.attention_log_every == 0:
                    attn = attention_to_numpy(actor_critic)
                    if attn is not None:
                        action_list = action.detach().cpu().view(-1).tolist()
                        records = []
                        for proc_idx in range(min(args.num_processes, attn.shape[0])):
                            proc_diags = diagnostics_before[proc_idx] if proc_idx < len(diagnostics_before) else []
                            for slot_idx in range(args.history_size):
                                diag = proc_diags[slot_idx] if slot_idx < len(proc_diags) else {}
                                records.append(
                                    {
                                        "phase": "train",
                                        "update": int(j),
                                        "step": int(step),
                                        "decision_id": int(decision_id),
                                        "process_index": int(proc_idx),
                                        "slot_index": int(slot_idx),
                                        "history_size": int(args.history_size),
                                        "alpha": float(attn[proc_idx, slot_idx]),
                                        "selected_action": int(action_list[proc_idx]) if proc_idx < len(action_list) else None,
                                        **summarize_history_diagnostic(diag),
                                    }
                                )
                        write_training_attention_records(args.attention_log_path, records)

            ## 보상 계산
            obs, reward, done, infos = envs.step(action)

            # 정규화 업데이트 (시퀀스 어텐션 적용 후 데이터로)
            if actor_critic.use_sequence_attention:
                # 시퀀스 어텐션 적용 후 정규화기 통계 업데이트
                with torch.no_grad():
                    obs_tensor = torch.FloatTensor(obs).to(device)
                    processed_obs = actor_critic._apply_sequence_attention(obs_tensor)
                    actor_critic.base.normalizer.update_stats(processed_obs.cpu().numpy())
            else:
                actor_critic.base.normalizer.update_stats(obs)

            # 에피소드 보상 수집
            for info in infos:
                if isinstance(info, dict) and "episode" in info.keys():
                    episode_rewards.append(info["episode"]["r"])
            if done[0] and len(infos) > 0:
                if isinstance(infos[0], dict) and "episode_r" in infos[0]:
                    episode_rewards.append(float(infos[0]["episode_r"].max()))

            # 조기 종료 조건
            if len(infos) > 0 and isinstance(infos[0], dict) and "stop" in infos[0].keys():
                break

            # 마스크 생성
            masks = torch.FloatTensor([[0.0] if done_ else [1.0] for done_ in done])
            bad_masks = torch.FloatTensor(
                [[0.0] if isinstance(info, dict) and "bad_transition" in info.keys() else [1.0] for info in infos]
            )
            
            # 롤아웃에 데이터 삽입
            # reward를 텐서로 변환
            reward_tensor = torch.FloatTensor(reward).unsqueeze(1) if not isinstance(reward, torch.Tensor) else reward
            
            rollouts.insert(
                obs,
                recurrent_hidden_states,
                action,
                action_log_prob,
                value,
                reward_tensor,
                masks,
                bad_masks,
            )

        # 조기 종료 처리
        if len(infos) > 0 and isinstance(infos[0], dict) and "stop" in infos[0].keys():
            print("Training finished!")
            end = time.time()
            print(f"total running time: {end-start}\n")
            save_path = os.path.join(args.save_dir, args.algo + "_seq")
            try:
                os.makedirs(save_path)
            except OSError:
                pass
            torch.save(
                [actor_critic, getattr(utils.get_vec_normalize(envs), "obs_rms", None)],
                os.path.join(save_path, args.env_name + f"_index{args.index}_final.pt"),
            )
            final_checkpoint_saved = True
            writer.close()
            break

        # 다음 가치 계산
        with torch.no_grad():
            next_value = actor_critic.get_value(
                rollouts.obs[-1],
                rollouts.recurrent_hidden_states[-1],
                rollouts.masks[-1],
            ).detach()

        # 리턴 계산
        rollouts.compute_returns(
            next_value,
            args.use_gae,
            args.gamma,
            args.gae_lambda,
            None,
            args.use_proper_time_limits,
        )

        # 정책 업데이트
        value_loss, action_loss, dist_entropy = agent.update(rollouts, args.use_value)

        rollouts.after_update()

        # 주기적 저장
        if (
            (j % args.save_interval == 0 or j == num_updates - 1)
            and args.save_dir != ""
            and j > 0
        ):
            save_path = os.path.join(args.save_dir, args.algo + "_seq")
            try:
                os.makedirs(save_path)
            except OSError:
                pass

            torch.save(
                [actor_critic, getattr(utils.get_vec_normalize(envs), "obs_rms", None)],
                os.path.join(save_path, args.env_name + f"_index{args.index}_iter{j}.pt"),
            )

        # 로깅
        if j % args.log_interval == 0 and len(episode_rewards) > 1:
            total_num_steps = (j + 1) * args.num_processes * args.num_steps
            end = time.time()
            print(f"time elapsed now: {end-start}\n")
            print(
                "Updates {}, num timesteps {}, FPS {} \n Last {} training episodes: mean/median reward {:.3f}/{:.3f}, min/max reward {:.3f}/{:.3f}\n Entropy: {:.3f}, Value Loss: {:.3f}, Action Loss: {:.3f}".format(
                    j,
                    total_num_steps,
                    int(total_num_steps / (end - start)),
                    len(episode_rewards),
                    np.mean(episode_rewards),
                    np.median(episode_rewards),
                    np.min(episode_rewards),
                    np.max(episode_rewards),
                    dist_entropy,
                    value_loss,
                    action_loss,
                ),
                flush=True,
            )
            
            # TensorBoard 로깅
            writer.add_scalar('Reward/mean', np.mean(episode_rewards), j)
            writer.add_scalar('Reward/median', np.median(episode_rewards), j)
            writer.add_scalar('Reward/min', np.min(episode_rewards), j)
            writer.add_scalar('Reward/max', np.max(episode_rewards), j)
            writer.add_scalar('Loss/value', value_loss, j)
            writer.add_scalar('Loss/action', action_loss, j)
            writer.add_scalar('Entropy', dist_entropy, j)
            writer.add_scalar('LearningRate', agent.optimizer.param_groups[0]['lr'], j)
            writer.add_scalar('FPS', int(total_num_steps / (end - start)), j)
            
            # 최고 성능 모델 저장
            ep_reward = np.mean(episode_rewards)
            if ep_reward > cur_best_ep_reward:
                print("Updates {}, new max mean reward {}".format(j, ep_reward))
                save_path = os.path.join(args.save_dir, args.algo + "_seq")
                try:
                    os.makedirs(save_path)
                except OSError:
                    pass

                torch.save(
                    [
                        actor_critic,
                        getattr(utils.get_vec_normalize(envs), "obs_rms", None),
                    ],
                    os.path.join(save_path, args.env_name + f"_index{args.index}_best.pt"),
                )

                cur_best_ep_reward = ep_reward

    if not final_checkpoint_saved and args.save_dir != "":
        save_path = os.path.join(args.save_dir, args.algo + "_seq")
        try:
            os.makedirs(save_path)
        except OSError:
            pass
        torch.save(
            [actor_critic, getattr(utils.get_vec_normalize(envs), "obs_rms", None)],
            os.path.join(save_path, args.env_name + f"_index{args.index}_final.pt"),
        )
        print(
            f"Saved final checkpoint to {os.path.join(save_path, args.env_name + f'_index{args.index}_final.pt')}",
            flush=True,
        )

    print("Training completed!", flush=True)
    writer.close()


if __name__ == "__main__":
    torch.multiprocessing.set_start_method("spawn", force=True)
    main()
