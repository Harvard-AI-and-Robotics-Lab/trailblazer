import os
import gymnasium as gym
import torch
import copy
import numpy as np
import pandas as pd
from gymnasium import spaces
from sentence_transformers import SentenceTransformer

from utils import *
from llm_utils.creat_model import prepare_model_and_tok


class MutatorSelect(gym.Env):
    def __init__(self, args, obs_size, gpu_id, eval=False) -> None:
        super(MutatorSelect, self).__init__()
        self.args = args
        self.num_processes = args.num_processes
        self.device = torch.device(
            "cuda:{}".format(gpu_id) if torch.cuda.is_available() else "cpu"
        )
        if args.datasets == "advbench":
            print(f"using advbench")
            if eval:
                question_path = "./datasets/questions/processed_unalign_test_questions.csv"
                self.questions_pool = pd.read_csv(question_path)["text"].tolist()
            else:
                question_path = "./datasets/questions/processed_unalign_train_questions.csv"
                self.questions_pool = pd.read_csv(question_path)["text"].tolist()
        else:
            print(f"using val")
            question_path = "./datasets/questions/processed_unalign_val_questions.csv"
            self.questions_pool = pd.read_csv(question_path)["text"].tolist()
        if eval:
            print(f"using top {args.K} training generated templates!")
            self.initial_seed = pd.read_csv(args.template_path)[
                "processed_templates"
            ].tolist()[: args.K]
        else:
            self.initial_seed = pd.read_excel(
                "./datasets/prompts/jailbreak-prompt.xlsx"
            )["text"].tolist()
        self.status = fuzzing_status(
            self.questions_pool[:4],
            initial_seed=self.initial_seed,
            max_query=args.max_query,
        )

        self.args_target = copy.deepcopy(args)
        self.args_target.model_path = args.target_model
        self.args_target.temperature = (
            0.01  
        )
        self.MODEL, self.TOK = prepare_model_and_tok(args)
        self.MODEL_TARGET, self.TOK_TARGET = prepare_model_and_tok(
            self.args_target, target=True
        )

        self.embedder = SentenceTransformer(
            "BAAI/bge-large-en-v1.5", device="cuda:{}".format(gpu_id)
        )

        self.refusal_signal = test_prefixes
        self.uppercased_refusal_signal = [word.upper() for word in self.refusal_signal]

        self.observation_space = spaces.Box(-np.inf, np.inf, (obs_size + 3,))
        self.action_space = spaces.Discrete(len(list(mutator)))

        self.steps = 0
        self.max_step = 5
        self.terminate = []
        self.save_len = len(self.status.seed_queue)
        self.eval = eval

        if eval:
            directory_path = "datasets/eval"
            if not os.path.exists(directory_path):
                os.makedirs(directory_path)
            # source와 target 모델 정보 모두 포함
            target_model_name = self.args_target.model_path.split('/')[-1]
            source_model = getattr(self.args, 'source_model', self.args_target.model_path)
            source_model_name = source_model.split('/')[-1] if source_model is not None else target_model_name
            if source_model_name != target_model_name:
                # Cross-model evaluation
                self.result_csv_path = f"{directory_path}/RL_{source_model_name}_to_{target_model_name}_{self.args.index}_eval_{self.args.defense}.csv"
                self.responses_csv_path = f"{directory_path}/RL_{source_model_name}_to_{target_model_name}_{self.args.index}_responses_{self.args.defense}.csv"
            else:
                # Same model evaluation
                self.result_csv_path = f"{directory_path}/RL_{target_model_name}_{self.args.index}_eval_{self.args.defense}.csv"
                self.responses_csv_path = f"{directory_path}/RL_{target_model_name}_{self.args.index}_responses_{self.args.defense}.csv"
            self.target_responses = []
            print(f"using defense: {self.args.defense}")
            
            # max attempts per question
            self.max_attempts_per_question = getattr(args, 'max_attempts_per_question', 50)
            # maintain the number of attempts for each question
            self.question_attempts = {}
            # record the last output of the question that reached the maximum number of attempts
            self.max_attempt_responses = {}
            print(f"Setting max attempts per question: {self.max_attempts_per_question}")
            
            # resume 옵션: 기존 응답 CSV가 있고 resume이면, 응답된 질문은 풀에서 제거
            self.answered_questions = set()
            if getattr(self.args, 'resume', False) and os.path.exists(self.responses_csv_path):
                try:
                    prev_df = pd.read_csv(self.responses_csv_path)
                    if {'question','response','prompt'}.issubset(set(prev_df.columns)):
                        self.answered_questions = set(prev_df['question'].astype(str).tolist())
                        # 질문 풀에서 제거
                        before = len(self.questions_pool)
                        self.questions_pool = [q for q in self.questions_pool if str(q) not in self.answered_questions]
                        after = len(self.questions_pool)
                        print(f"[resume] Skipped {before - after} already-answered questions. Remaining: {after}")
                except Exception as e:
                    print(f"[resume] Failed to load previous responses: {e}")

            # 응답 CSV 헤더 준비 (파일 없으면 생성)
            if not os.path.exists(self.responses_csv_path):
                with open(self.responses_csv_path, "w", newline="") as outfile:
                    writer = csv.writer(outfile)
                    writer.writerow(["question", "response", "prompt"])
        else:
            directory_path = "datasets/prompts_generated"
            if not os.path.exists(directory_path):
                os.makedirs(directory_path)
            # Training 시에는 target 모델만 사용
            target_model_name = self.args_target.model_path.split('/')[-1]
            self.result_csv_path = f"{directory_path}/RL_{target_model_name}_{self.args.index}.csv"
        
        # result csv 헤더: resume이고 파일이 이미 존재하면 헤더를 다시 쓰지 않음
        need_write_header = True
        if getattr(self.args, 'resume', False) and os.path.exists(self.result_csv_path):
            need_write_header = False
        if need_write_header:
            with open(self.result_csv_path, "w", newline="") as outfile:
                writer = csv.writer(outfile)
                writer.writerow(
                    ["template", "mutation", "generation", "success_q", "total_num_query"]
                )
        self.start_time = time.time()

    def reset(self):
        self.steps = 0
        self.terminate = [False for _ in range(self.num_processes)]
        self.prev_actions = np.zeros((self.num_processes))
        try:
            random_idx = np.random.choice(
                range(len(self.questions_pool)), self.num_processes, replace=False
            )
        except:
            # number of questions avaliable < num_processes
            random_idx = np.random.choice(
                range(len(self.questions_pool)), self.num_processes, replace=True
            )
        setattr(
            self.status, "questions", [self.questions_pool[idx] for idx in random_idx]
        )
        self.current_questions = [self.questions_pool[idx] for idx in random_idx]
        # select templates
        self.selected_seed = [
            self.status.seed_selection_strategy() for _ in range(self.num_processes)
        ]
        self.current_embeddings = []
        for seed in self.selected_seed:
            self.current_embeddings.append(self.embedder.encode(seed))
        new_obs = self.get_obs(np.array(self.current_embeddings), self.prev_actions)
        self.reward = np.zeros((self.num_processes))

        return new_obs

    def step(self, actions):
        reward = np.zeros((self.num_processes))
        current_templates = []

        for i in range(self.num_processes):
            if not self.terminate[i] and len(self.status.questions) > 0:
                mutate = list(mutator)[actions[i][0]]
                mutate_results, mutation = mutate_single(
                    self.selected_seed[i],
                    self.status,
                    mutate,
                    self.MODEL,
                    self.TOK,
                    self.args,
                )
                attack_results, valid_input_index, data, complete_prompts = execute(
                    self.status,
                    mutate_results,
                    self.args_target,
                    self.MODEL_TARGET,
                    self.TOK_TARGET,
                    eval=self.eval,
                    args=self.args,
                )
                self.status.update(
                    attack_results, mutate_results, mutation, valid_input_index, data
                )
                if type(mutate_results) == list:
                    mutate_results = mutate_results[0]
                else:
                    mutate_results = mutate_results.choices[0].message.content
                accepted = check_keywords(mutate_results, test_prefixes_for_templates)
                # if there are newly generated templates, change it to new, otherwise still use last step
                if accepted:
                    self.selected_seed[i] = mutate_results
                    current_templates.append(mutate_results)
                else:
                    current_templates.append(self.selected_seed[i])

                successful_num = sum(attack_results)
                reward[i] = successful_num / len(self.status.questions)
                # if in eval mode, remove success questions from question pool
                if self.eval:
                    remove_idx = []
                    for idx, result in enumerate(attack_results):
                        current_question = self.status.questions[idx]
                        
                        # initialize the counter (if not already)
                        if current_question not in self.question_attempts:
                            self.question_attempts[current_question] = 0
                        
                        # add the number of attempts
                        self.question_attempts[current_question] += 1
                        
                        if result == 1:  # successful jailbreak
                            print(len(attack_results), len(self.status.questions))
                            print(f"removing question {current_question} (successful after {self.question_attempts[current_question]} attempts)")
                            remove_idx.append(idx)
                            self.terminate[i] = True
                            try:
                                self.questions_pool.remove(current_question)
                                self.target_responses.append(data[idx])
                                append_to_csv(
                                    [
                                        current_question,
                                        data[idx],
                                        complete_prompts[idx],
                                    ],
                                    self.responses_csv_path,
                                )
                                # clean the counter
                                if current_question in self.question_attempts:
                                    del self.question_attempts[current_question]
                            except:
                                pass
                        elif self.question_attempts[current_question] >= self.max_attempts_per_question:
                            # reached the maximum number of attempts, record the last output
                            print(f"removing question {current_question} (max attempts {self.max_attempts_per_question} reached)")
                            remove_idx.append(idx)
                            self.terminate[i] = True
                            try:
                                self.questions_pool.remove(current_question)
                                # record the last output of the question that reached the maximum number of attempts
                                self.target_responses.append(data[idx])
                                self.max_attempt_responses[current_question] = data[idx]
                                append_to_csv(
                                    [
                                        current_question,
                                        data[idx],
                                        complete_prompts[idx],
                                    ],
                                    self.responses_csv_path,
                                )
                                # clean the counter
                                if current_question in self.question_attempts:
                                    del self.question_attempts[current_question]
                            except:
                                pass
                    for idx in remove_idx:
                        try:
                            self.status.questions.remove(self.current_questions[idx])
                        except:
                            pass
                elif reward[i] > 0:
                    # there is at least one question succeeds, we will terminate the current trajectory
                    self.terminate[i] = True
            else:
                current_templates.append(self.selected_seed[i])

        if len(self.status.seed_queue) > self.save_len:
            for i in range(self.save_len, len(self.status.seed_queue)):
                if self.status.seed_queue[i].parent != "root":
                    append_to_csv(
                        [
                            self.status.seed_queue[i].text,
                            self.status.seed_queue[i].mutation,
                            self.status.seed_queue[i].generation,
                            self.status.seed_queue[i].response,
                            self.status.query,
                        ],
                        self.result_csv_path,
                    )
            self.save_len = len(self.status.seed_queue)

        self.steps += 1

        if self.steps >= self.max_step:
            done = np.ones(self.num_processes)
            info = {"episode_r": reward, "step_r": reward}
        else:
            done = np.zeros(self.num_processes)
            info = {"episode_r": reward, "step_r": reward}

        if self.status.stop_condition() or len(self.questions_pool) == 0:
            info["stop"] = 1
            if self.eval:
                print(f"left testing questions: {len(self.questions_pool)}")
                info["left_q"] = self.questions_pool
                info["tar_responses"] = self.target_responses
                info["response_csv"] = self.responses_csv_path

        current_templates_embed = self.embedder.encode(current_templates)
        return_obs = self.get_obs(current_templates_embed, self.prev_actions)
        self.prev_actions = copy.deepcopy(np.array(actions))

        return return_obs, reward, done, info

    def get_obs(self, obs, actions):
        all_obs = obs if isinstance(obs, np.ndarray) else obs.detach().cpu().numpy()
        all_obs = np.concatenate(
            [
                all_obs,
                np.expand_dims(
                    np.array(self.terminate).astype(float) * 0 + self.steps, -1
                ),
            ],
            axis=-1,
        )
        all_obs = np.concatenate(
            [all_obs, np.expand_dims(np.array(self.terminate).astype(float), -1)],
            axis=-1,
        )
        all_obs = np.concatenate(
            [all_obs, np.array(actions).reshape(all_obs.shape[0], -1)], axis=-1
        )

        return all_obs
