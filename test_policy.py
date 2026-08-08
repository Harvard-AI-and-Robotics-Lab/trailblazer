import os
import time
import openai
import argparse
import torch
import pandas as pd

from a2c_ppo_acktr import utils
from a2c_ppo_acktr.envs import make_vec_envs
from torch import multiprocessing as mp
from fastchat.model import add_model_args

from llm_utils.model_inference import LLM_response
from utils import *

# python test_policy.py \
# --index=0 \
# --target_model=meta-llama/Meta-Llama-3-8B-Instruct \
# --model_path=gpt-3.5-turbo \
# --ckpt_path=trained_models/ppo/llama3_8bbest.pt \
# --max_query=50
# --K=1000


def remove_prompt(text):
    prompt = "[INSERT PROMPT HERE]"
    stripped_text = text.strip()
    if stripped_text.startswith(prompt):
        # Remove the prompt and any leading whitespace after the prompt
        return text[len(prompt):].lstrip()
    return text   
   
def preprocess_templates(templates):
    processed_t = []
    begin_marker = "====Template begins===="
    end_marker = "====Template ends===="

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
        processed_t.append(new_text)
    assert len(processed_t) == len(templates)
    return processed_t

def replace_template(test_question, prompt):
    if '[INSERT PROMPT HERE]' in prompt:
        jailbreak_input = prompt.replace('[INSERT PROMPT HERE]', test_question)
        return jailbreak_input
    else:
        return False
    
def execute_query(args, mutant, MODEL, TOK):
    if 'gpt' in args.model_path and 'gpt-oss' not in args.model_path:
        for _ in range(10):
            try:
                response = openai.ChatCompletion.create(
                    model=MODEL,
                    messages=[
                        {"role": "system", "content": "You are a helpful assistant."},
                        {"role": "user", "content": mutant},
                    ],
                    temperature=0,
                )
                return response['choices'][0]['message']['content']
            except Exception as e:
                print("Error is: ", e)
                # Rate limit 에러인 경우 더 짧은 대기시간 사용
                if "rate_limit_exceeded" in str(e) or "429" in str(e):
                    # 에러 메시지에서 대기시간 추출 시도
                    import re
                    wait_time = re.search(r'try again in ([\d.]+)s', str(e))
                    if wait_time:
                        sleep_time = float(wait_time.group(1)) + 2  # 2초 여유분 추가
                        print(f"Rate limit hit, waiting {sleep_time} seconds...")
                        time.sleep(sleep_time)
                    else:
                        # TPM 제한의 경우 보통 1분 정도 기다려야 함
                        print("TPM rate limit hit, waiting 65 seconds...")
                        time.sleep(65)  # 1분 + 5초 여유분
                else:
                    time.sleep(30)  # 다른 에러는 30초 대기
                continue
        print('system busy...')
        return "Sorry, I cannot help with this request. The system is busy now."
    
    else:
        return LLM_response(args, MODEL, TOK, args.model_path, mutant, target=True)
    
def main():
    ctx = mp.get_context('spawn')
    parser = argparse.ArgumentParser(description='RL parameters')
    parser.add_argument('--openai_key', type=str, default=None, help='OpenAI key')
    parser.add_argument('--deepinfra_key', type=str, default=None, help='Deepinfra key')
    parser.add_argument('--anthropic_key', type=str, default=None, help='Anthropic key')
    parser.add_argument('--model_path', type=str, default='meta-llama/Llama-2-7b-chat-hf', help='openai model or open-sourced LLMs')
    parser.add_argument('--target_model', type=str, default='meta-llama/Llama-2-7b-chat-hf', help='The target model, openai model or open-sourced LLMs')
    parser.add_argument('--max_query', type=int, default=10000, help='The maximum number of queries')
    parser.add_argument('--num_processes',type=int,default=4,help='how many training CPU processes to use (default: 16)')
    parser.add_argument('--datasets', dest='datasets', action='store', default='advbench', help='name of dataset(s), e.g., agnews')
    parser.add_argument('--seed', type=int, default=1, help='random seed (default: 1)')
    parser.add_argument('--cuda_id',type=int, default=0)
    parser.add_argument('--index', type=int, default=10, help='task id')
    parser.add_argument('--ckpt_path', type=str, default=None, help='path to trained policy')
    parser.add_argument('--defense', type=str, default="none", help='defense method')
    parser.add_argument('--K', type=int, default=200, help='top k templates from training')
    parser.add_argument('--source_model', type=str, default=None)
    parser.add_argument('--max_attempts_per_question', type=int, default=50, 
                       help='Maximum attempts per question during testing (default: 50)')
    add_model_args(parser)
    args = parser.parse_args()
    openai.api_key = args.openai_key or os.environ.get("OPENAI_API_KEY")
 
    args.num_gpus = 1
    device = torch.device("cuda:{}".format(args.cuda_id))

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    
    print('Experiment arguments ', args, flush=True)

    training_templates_path = f"datasets/prompts_generated/RL_{args.target_model.split('/')[-1]}_{args.index}.csv"
    df = pd.read_csv(training_templates_path)
    templates = df['template'].tolist()
    if 'processed_templates' not in df.columns:
        processed_t = preprocess_templates(templates)
        df['processed_templates'] = processed_t
        df.to_csv(training_templates_path, index=False)
    try:
        sorted_df = df.sort_values(by='success_q', ascending=False)
    except:
        sorted_df = df
    sorted_templates = sorted_df['processed_templates']
    args.template_path = f"datasets/prompts_generated/RL_{args.target_model.split('/')[-1]}_{args.index}_processed.csv"
    sorted_templates.to_csv(args.template_path, index=False)
        
    
    obs_size = 1024
    envs = make_vec_envs(args, args.num_processes, obs_size, args.cuda_id, eval=True)         
    actor_critic = torch.load(args.ckpt_path, map_location=device)[0]
 
    obs = envs.reset()

    recurrent_hidden_states = torch.zeros(
        args.num_processes, actor_critic.recurrent_hidden_state_size, device=device)
    masks = torch.zeros(args.num_processes, 1, device=device)
    
    start = time.time()
    
    eval_episode_rewards = []
    while len(eval_episode_rewards) < 600:
        
        # Sample actions
        with torch.no_grad():
            _, action, _, recurrent_hidden_states = actor_critic.act(
                obs, recurrent_hidden_states,
                masks)
            
        obs, _, done, infos = envs.step(action)
        
        masks = torch.tensor(
                [[0.0] if done_ else [1.0] for done_ in done],
                dtype=torch.float32,
                device=device)
        
        if done[0]:
            eval_episode_rewards.append(infos[0]['episode_r'])
            
        if 'stop' in infos[0].keys():
            print("finished!")
            break
    
    

if __name__ == "__main__":
    torch.multiprocessing.set_start_method('spawn', force=True)# good solution !!!!
    main()
