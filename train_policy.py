import os
os.environ["TOKENIZERS_PARALLELISM"] = "true"
import openai
import time
import argparse
import numpy as np
from collections import deque

import torch
from a2c_ppo_acktr import algo, utils
from a2c_ppo_acktr.envs import make_vec_envs
from a2c_ppo_acktr.model import Policy
from a2c_ppo_acktr.storage import RolloutStorage
from fastchat.model import add_model_args


# python train_policy.py \
# --index=0 \
# --env-name=llama3_8b \
# --target_model=meta-llama/Meta-Llama-3-8B-Instruct \
# --model_path=gpt-3.5-turbo \


def main():
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
    parser.add_argument('--defense', type=str, default="none", help='defense method')
    add_model_args(parser)
    utils.add_train_args(parser)
    args = parser.parse_args()
    
    openai.api_key = args.openai_key or os.environ.get("OPENAI_API_KEY")
    
    args.num_gpus = 1
    device = torch.device("cuda:{}".format(args.cuda_id))

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    print("Experiment arguments ", args, flush=True)

    if args.use_value:
        print("use value network in PPO.")
    else:
        print("Not use value network in PPO.")

    obs_size = 1024
    #this will create a vectorized environment : RL_env.py will be called for each process
    envs = make_vec_envs(args, args.num_processes, obs_size, args.cuda_id)

    num_blocks = int(envs.observation_space.shape[0] / obs_size)
    actor_critic = Policy(
        envs.observation_space.shape,
        envs.action_space,
        args.use_attention,
        device,
        num_blocks,
        base_kwargs={"recurrent": False, "hidden_size": 1024},
    )
    actor_critic.to(device)

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

    rollouts = RolloutStorage(
        args.num_steps,
        args.num_processes,
        envs.observation_space.shape,
        envs.action_space,
        actor_critic.recurrent_hidden_state_size,
    )

    obs = envs.reset()
    rollouts.obs[0].copy_(obs)
    rollouts.to(device)

    episode_rewards = deque(maxlen=10)

    start = time.time()
    num_updates = int(args.num_env_steps) // args.num_steps // args.num_processes

    cur_best_ep_reward = 0.0

    for j in range(num_updates): #6000//32//4 = 46
        utils.update_linear_schedule(
            agent.optimizer,
            j,
            num_updates,
            agent.optimizer.lr if args.algo == "acktr" else args.lr,
        )

        for step in range(args.num_steps): 
            # Sample actions
            with torch.no_grad():
                value, action, action_log_prob, recurrent_hidden_states = actor_critic.act(
                    rollouts.obs[step], rollouts.recurrent_hidden_states[step],
                    rollouts.masks[step])

            ##Reward computation
            obs, reward, done, infos = envs.step(action)

            actor_critic.base.normalizer.update_stats(obs)

            for info in infos:
                if "episode" in info.keys():
                    episode_rewards.append(info["episode"]["r"])
            if done[0]:
                episode_rewards.append(float(infos[0]["episode_r"].max()))

            if "stop" in infos[0].keys():
                break

            masks = torch.FloatTensor([[0.0] if done_ else [1.0] for done_ in done])
            bad_masks = torch.FloatTensor(
                [[0.0] if "bad_transition" in info.keys() else [1.0] for info in infos]
            )
            rollouts.insert(
                obs,
                recurrent_hidden_states,
                action,
                action_log_prob,
                value,
                reward,
                masks,
                bad_masks,
            )

        if "stop" in infos[0].keys():
            print("finished!")
            end = time.time()
            print(f"total running time: {end-start}\n")
            save_path = os.path.join(args.save_dir, args.algo)
            try:
                os.makedirs(save_path)
            except OSError:
                pass
            torch.save(
                [actor_critic, getattr(utils.get_vec_normalize(envs), "obs_rms", None)],
                os.path.join(save_path, args.env_name + "_final.pt"),
            )
            break

        with torch.no_grad():
            next_value = actor_critic.get_value(
                rollouts.obs[-1],
                rollouts.recurrent_hidden_states[-1],
                rollouts.masks[-1],
            ).detach()

        rollouts.compute_returns(
            next_value,
            args.use_gae,
            args.gamma,
            args.gae_lambda,
            None,
            args.use_proper_time_limits,
        )

        value_loss, action_loss, dist_entropy = agent.update(rollouts, args.use_value)

        rollouts.after_update()

        # save for every interval-th episode or for the last epoch
        if (
            (j % args.save_interval == 0 or j == num_updates - 1)
            and args.save_dir != ""
            and j > 0
        ):
            save_path = os.path.join(args.save_dir, args.algo)
            try:
                os.makedirs(save_path)
            except OSError:
                pass

            torch.save(
                [actor_critic, getattr(utils.get_vec_normalize(envs), "obs_rms", None)],
                os.path.join(save_path, args.env_name + "_iter%d.pt" % j),
            )

        if j % args.log_interval == 0 and len(episode_rewards) > 1:
            total_num_steps = (j + 1) * args.num_processes * args.num_steps
            end = time.time()
            print(f"time elapsed now: {end-start}\n")
            print(
                "Updates {}, num timesteps {}, FPS {} \n Last {} training episodes: mean/median reward {:.3f}/{:.3f}, min/max reward {:.3f}/{:.3f}".format(
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
            ep_reward = np.mean(episode_rewards)
            if ep_reward > cur_best_ep_reward:
                print("Updates {}, new max mean reward {}".format(j, ep_reward))
                save_path = os.path.join(args.save_dir, args.algo)
                try:
                    os.makedirs(save_path)
                except OSError:
                    pass

                torch.save(
                    [
                        actor_critic,
                        getattr(utils.get_vec_normalize(envs), "obs_rms", None),
                    ],
                    os.path.join(save_path, args.env_name + "best.pt"),
                )

                cur_best_ep_reward = ep_reward


if __name__ == "__main__":
    torch.multiprocessing.set_start_method("spawn", force=True)
    main()
