import os
os.environ["TOKENIZERS_PARALLELISM"] = "true"
import openai
import time
import argparse
import numpy as np
import pandas as pd
import torch
import random

from a2c_ppo_acktr.envs_seq import make_vec_envs_seq
from fastchat.model import add_model_args
from utils import *


# TrailBlazer sequence-aware PPO inference entry point.
# See README.md and scripts/ for camera-ready reproduction commands.


def remove_prompt(text):
    prompt = "[INSERT PROMPT HERE]"
    stripped_text = text.strip()
    if stripped_text.startswith(prompt):
        # Remove the prompt and any leading whitespace after the prompt
        return text[len(prompt):].lstrip()
    return text


def _redact_keys(args):
    display_args = argparse.Namespace(**vars(args))
    for key_name in ("openai_key", "deepinfra_key", "anthropic_key", "gemini_key"):
        if getattr(display_args, key_name, None):
            setattr(display_args, key_name, "***")
    return display_args


def get_template_limit_by_model(model_name):
    """모델별 템플릿 토큰 제한 반환"""
    
    # OpenAI 모델
    if 'gpt-3.5' in model_name:
        return 2500  # 4096 * 0.8 - 여유분
    elif 'gpt-4' in model_name:
        return 6000  # 8192 * 0.8 - 여유분
    
    # 로컬 모델
    elif 'llama' in model_name.lower():
        return 6000  # 8192 * 0.8 - 여유분
    elif 'vicuna' in model_name.lower():
        return 1500  # 2048 * 0.8 - 여유분
    
    # 기본값
    else:
        # return 2000
        return 6000

def estimate_tokens(text):
    """텍스트의 대략적인 토큰 수 추정"""
    return int(len(text.split()) * 1.3)

def preprocess_templates(templates, model_name=None):
    processed_t = []
    begin_marker = "====Template begins===="
    end_marker = "====Template ends===="

    # 모델별 토큰 제한 가져오기
    max_tokens = get_template_limit_by_model(model_name) if model_name else 2000

    for text in templates:
        if begin_marker in text:
            text = text.replace(begin_marker, "").replace(end_marker, "")
        if text.startswith("Sure") or text.startswith(" Of course") \
            or text.startswith(" Sure") or text.startswith(" Of course!"):
            colon_index = text.find(":")
            if colon_index != -1:
                new_text = text[colon_index + 1:].strip()  # strip to remove any leading whitespace
            else:
                # not find
                new_text = text
        elif text.startswith(" I apologize"):
            newline_index = text.find("\n")
            if newline_index != -1:
                new_text = text[newline_index + 1:].strip()
            else:
                new_text = text
            
        else:
            new_text = text
        assert type(new_text) is str
        new_text = remove_prompt(new_text)
        
        # 토큰 수 제한 적용
        if estimate_tokens(new_text) > max_tokens:
            words = new_text.split()
            target_words = int(max_tokens / 1.3)
            if target_words < len(words):
                new_text = ' '.join(words[:target_words]) + "..."
                print(f"Template truncated to {target_words} words (estimated {max_tokens} tokens)")
        
        processed_t.append(new_text)
    assert len(processed_t) == len(templates)
    return processed_t


def main():
    parser = argparse.ArgumentParser(description='Test sequence-based RL policy')
    parser.add_argument('--openai_key', type=str, default=None, help='OpenAI key')
    parser.add_argument('--deepinfra_key', type=str, default=None, help='Deepinfra key')
    parser.add_argument('--anthropic_key', type=str, default=None, help='Anthropic key')
    parser.add_argument('--gemini_key', type=str, default=None, help='Gemini API key')
    parser.add_argument('--model_path', type=str, default='meta-llama/Llama-2-7b-chat-hf', help='openai model or open-sourced LLMs')
    parser.add_argument('--target_model', type=str, default='meta-llama/Llama-2-7b-chat-hf', help='The target model, openai model or open-sourced LLMs')
    parser.add_argument('--max_query', type=int, default=10000, help='The maximum number of queries')
    parser.add_argument('--num_processes', type=int, default=4, help='how many training CPU processes to use')
    parser.add_argument('--datasets', dest='datasets', action='store', default='advbench', help='name of dataset(s),support advbench, harmbench')
    parser.add_argument('--seed', type=int, default=1, help='random seed')
    parser.add_argument('--cuda_id', type=int, default=0)
    parser.add_argument('--index', type=int, default=10, help='task id')
    parser.add_argument('--defense', type=str, default="none", help='defense method')
    parser.add_argument('--history_size', type=int, default=4, help='history buffer size for sequence observation')
    parser.add_argument('--ckpt_path', type=str, required=True, help='path to checkpoint file')
    parser.add_argument('--K', type=int, default=1000, help='number of top generated templates to use at inference')
    parser.add_argument('--resume', action='store_true', help='resume evaluation, only run unanswered questions')
    parser.add_argument('--max_attempts_per_question', type=int, default=50, help='maximum attempts per question during evaluation')
    parser.add_argument('--source_model', type=str, default=None, help='Source model used for training (for template path). If not specified, uses target_model')
    parser.add_argument('--combined_judge', action='store_true', default=False,
                       help='Use one GPT-4o judge call that returns binary success and a 0-10 vulnerability score')
    
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
    
    add_model_args(parser)
    args = parser.parse_args()
    
    openai.api_key = args.openai_key or os.environ.get("OPENAI_API_KEY")
    
    args.num_gpus = 1
    device = torch.device("cuda:{}".format(args.cuda_id))

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    
    print('Experiment arguments ', _redact_keys(args), flush=True)
    print(f"Sequence attention mechanism: {args.seqattn}", flush=True)
    print(f"Attention weight mode: {args.attention_weight_mode}", flush=True)
    print(f"Step-wise Improvement Reward: {'Enabled' if args.use_step_improvement else 'Disabled'}", flush=True)
    if args.use_step_improvement:
        print(f"Lambda coefficient (λ): {args.lambda_coef}", flush=True)

    # source_model이 지정되지 않으면 target_model 사용
    source_model = args.source_model or args.target_model
    print(f"Using source model: {source_model} (for template path)", flush=True)

    # 훈련된 템플릿 로딩
    training_templates_path = f"datasets/prompts_generated/RL_{source_model.split('/')[-1]}_{args.index}.csv"


    df = pd.read_csv(training_templates_path)
    templates = df['template'].tolist()
    
    if 'processed_templates' not in df.columns:
        processed_t = preprocess_templates(templates, source_model)
        df['processed_templates'] = processed_t
        df.to_csv(training_templates_path, index=False)
    
    try:
        sorted_df = df.sort_values(by='success_q', ascending=False)
    except:
        sorted_df = df
        
    sorted_templates = sorted_df['processed_templates']
    args.template_path = f"datasets/prompts_generated/RL_{source_model.split('/')[-1]}_{args.index}_processed.csv"
    sorted_templates.to_csv(args.template_path, index=False)
    
    print("================================================")
    print(f"Loading training templates from {args.template_path}", flush=True)
    print("================================================")
    
    # 시퀀스 기반 관찰값 크기 계산 (환경에서는 항상 raw 히스토리 전달)
    base_embedding_size = 1024
    
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
    
    # 시퀀스 기반 환경 생성
    envs = make_vec_envs_seq(args, args.num_processes, total_obs_size, args.cuda_id, eval=True, history_size=args.history_size)
    
    # 훈련된 모델 로딩 (PyTorch 2.6+ 호환)
    actor_critic = torch.load(args.ckpt_path, map_location=device, weights_only=False)[0]
    if hasattr(actor_critic, 'sequence_attention') and actor_critic.sequence_attention is not None:
        actor_critic.sequence_attention.weight_mode = args.attention_weight_mode
        print(f"Applied attention weight mode: {args.attention_weight_mode}", flush=True)
    
    # 시퀀스 어텐션 상태 확인
    if hasattr(actor_critic, 'use_sequence_attention') and actor_critic.use_sequence_attention:
        print(f"로딩된 모델에 시퀀스 어텐션이 활성화되어 있습니다: {type(actor_critic.sequence_attention).__name__}")
    else:
        print("로딩된 모델에는 시퀀스 어텐션이 없습니다.")
    
    # 시퀀스 어텐션 설정 확인
    if args.seqattn != 'none':
        if not (hasattr(actor_critic, 'use_sequence_attention') and actor_critic.use_sequence_attention):
            print(f"경고: --seqattn={args.seqattn}으로 설정했지만 로딩된 모델에는 시퀀스 어텐션이 없습니다.")
        else:
            print(f"시퀀스 어텐션 설정이 일치합니다: {args.seqattn}")

    obs = envs.reset()

    recurrent_hidden_states = torch.zeros(
        args.num_processes, actor_critic.recurrent_hidden_state_size, device=device)
    masks = torch.zeros(args.num_processes, 1, device=device)
    
    start = time.time()
    
    eval_episode_rewards = []
    while len(eval_episode_rewards) < 600:
        
        # 액션 샘플링
        with torch.no_grad():
            # obs를 올바른 디바이스로 이동
            obs_tensor = torch.FloatTensor(obs).to(device)
            _, action, _, recurrent_hidden_states = actor_critic.act(
                obs_tensor, recurrent_hidden_states, masks)
            
        obs, _, done, infos = envs.step(action)
        
        masks = torch.tensor(
                [[0.0] if done_ else [1.0] for done_ in done],
                dtype=torch.float32,
                device=device)
        
        if done[0]:
            if len(infos) > 0 and isinstance(infos[0], dict) and "episode_r" in infos[0]:
                eval_episode_rewards.append(float(infos[0]["episode_r"].max()))
                print(f"Episode {len(eval_episode_rewards)}: Reward = {eval_episode_rewards[-1]:.3f}")
            else:
                # 기본값으로 1.0 추가 (성공한 공격이므로)
                eval_episode_rewards.append(1.0)
                print(f"Episode {len(eval_episode_rewards)}: Reward = 1.0")
            
            # 환경 리셋
            obs = envs.reset()
            recurrent_hidden_states = torch.zeros(
                args.num_processes, actor_critic.recurrent_hidden_state_size, device=device)
            masks = torch.zeros(args.num_processes, 1, device=device)
        
        # 조기 종료 조건
        if len(infos) > 0 and isinstance(infos[0], dict) and "stop" in infos[0].keys():
            print("Evaluation finished!")
            break
    
    end = time.time()
    print(f"Evaluation completed in {end-start:.2f} seconds")
    print(f"Total episodes: {len(eval_episode_rewards)}")
    if len(eval_episode_rewards) > 0:
        print(f"Mean reward: {np.mean(eval_episode_rewards):.3f}")
        print(f"Std reward: {np.std(eval_episode_rewards):.3f}")
        print(f"Min reward: {np.min(eval_episode_rewards):.3f}")
        print(f"Max reward: {np.max(eval_episode_rewards):.3f}")
    else:
        print("No completed episodes before stop condition.")


if __name__ == "__main__":
    torch.multiprocessing.set_start_method("spawn", force=True)
    main()
