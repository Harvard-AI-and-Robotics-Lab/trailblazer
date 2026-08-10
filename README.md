<p align="center">
  <h1 align="center">TrailBlazer:<br>History-Guided Reinforcement Learning for<br>Black-Box LLM Jailbreaking</h1>
  <p align="center">
    <strong>Sung-Hoon Yoon</strong><sup>*</sup>
    ·
    <strong>Ruizhi Qian</strong><sup>*</sup>
    ·
    <strong>Minda Zhao</strong><sup>*</sup>
    ·
    <strong>Weiyue Li</strong><sup>*</sup>
    ·
    <strong>Mengyu Wang</strong><sup>†</sup>
  </p>
  <p align="center">
    <sup>*</sup>Equal contribution. <sup>†</sup>Corresponding author.
  </p>
  <p align="center">
    <strong><i>COLM 2026</i></strong>
  </p>
  <p align="center">
    <i>Official repository for the COLM 2026 paper.</i>
  </p>
</p>

<h3 align="center">
  <a href="https://arxiv.org/abs/2602.06440"><strong>Paper</strong></a>
  |
  <a href="https://arxiv.org/pdf/2602.06440"><strong>PDF</strong></a>
  |
  <a href="https://github.com/Harvard-AI-and-Robotics-Lab/trailblazer"><strong>Code</strong></a>
</h3>

## 📚 Citation

If you find this repository useful, please cite:

```bibtex
@article{yoon2026trailblazer,
  title={TrailBlazer: History-Guided Reinforcement Learning for Black-Box LLM Jailbreaking},
  author={Yoon, Sung-Hoon and Qian, Ruizhi and Zhao, Minda and Li, Weiyue and Wang, Mengyu},
  journal={arXiv preprint arXiv:2602.06440},
  year={2026}
}
```

## 📌 Overview

TrailBlazer extends the RLbreaker search framework with sequence-aware state representations:

- **HRL**: the policy observes the current embedding plus a fixed window of previous template embeddings, response features, rewards, and mutator IDs.
- **AHRL / TrailBlazer**: a learned step-level attention module reweights the recent history before the PPO policy selects the next mutator.

The code is intended for controlled red-team evaluation and robustness research. Do not use it to generate or deploy harmful content.

## 📁 Repository Layout

```text
trailblazer/
├── train_policy_seq.py              # HRL/AHRL PPO training
├── test_policy_seq.py               # HRL/AHRL inference
├── test_policy_seq_attention_log.py # inference with attention logging
├── RL_env_seq.py                    # history-aware RL environment
├── sequence_attention.py            # simple and multi-head step-level attention
├── utils.py                         # mutators, API calls, rewards, judges
├── a2c_ppo_acktr/                   # PPO policy implementation
├── llm_utils/                       # model/API wrappers
├── scripts/                         # reproducible command templates
├── configs/                         # camera-ready hyperparameter record
└── datasets/                        # user-supplied data; generated files are ignored
```

## 💻 Environment Setup

```bash
conda create -n trailblazer python=3.9
conda activate trailblazer
pip install -r requirements.txt
```

Set API keys through environment variables. Do not hard-code keys in scripts or commits.

```bash
export OPENAI_API_KEY="<YOUR_OPENAI_API_KEY>"
export DEEPINFRA_API_KEY="<YOUR_DEEPINFRA_API_KEY>"
export GEMINI_API_KEY="<YOUR_GEMINI_API_KEY>"          # optional
export ANTHROPIC_API_KEY="<YOUR_ANTHROPIC_API_KEY>"    # optional
```

## 📦 Data

Place the data files in the following structure:

```text
datasets/
├── processed_unalign.csv
├── questions/
│   ├── processed_unalign_train_questions.csv
│   ├── processed_unalign_test_questions.csv
│   └── harmbench_questions_test.csv
└── prompts/
    └── jailbreak-prompt.xlsx
```

Expected schemas:

- `processed_unalign.csv`: columns `question,response`. The `response` column is the unaligned reference answer used by the BGE reward.
- `datasets/questions/*_questions.csv`: column `text`.
- `datasets/prompts/jailbreak-prompt.xlsx`: column `text` containing initial prompt templates.

Generated templates, evaluation responses, logs, TensorBoard runs, and checkpoints are intentionally ignored by git.

If `datasets/processed_unalign.csv` is already available, the AdvBench question split files can be generated with:

```bash
python prepare_datasets.py \
  --input datasets/processed_unalign.csv \
  --train_size 364 \
  --seed 1
```

## ⚙️ Camera-Ready Settings

The main paper configuration is:

| Component | Setting |
|---|---|
| Helper LLM / mutator executor | `gpt-3.5-turbo` |
| Mutators | `generate_similar`, `crossover`, `expand`, `shorten`, `rephrase` |
| Training reward | BGE-large cosine similarity to unaligned reference response |
| Training success threshold | `0.7` cosine similarity |
| PPO max episode length | `T=5` environment steps |
| AHRL attention | `--seqattn simple --attention_weight_mode learned` |
| History size | `4` by default; `5` for LLaMA-3.2-11B in the paper |
| Step-improvement reward | off by default; do not pass `--use_step_improvement` for the main setting |
| Inference query budget | `--max_query 10000` and `--max_attempts_per_question 50` |
| Inference template pool | `--K 1000` top generated training templates |
| Inference judge | GPT-4o binary compliance judge |

Important: `--history_size` is the HRL/AHRL history window. `--K` is the number of generated training templates loaded at inference time. They are different parameters.

## 🚀 Training

AHRL / TrailBlazer on Qwen3-14B:

```bash
python train_policy_seq.py \
  --env-name qwen3_14b_ahrl \
  --target_model Qwen/Qwen3-14B \
  --model_path gpt-3.5-turbo \
  --index 0 \
  --datasets advbench \
  --history_size 4 \
  --seqattn simple \
  --attention_weight_mode learned \
  --max_query 10000 \
  --num_processes 4
```

The best and final checkpoints are saved to:

```text
trained_models/ppo_seq/{env_name}_index{index}_best.pt
trained_models/ppo_seq/{env_name}_index{index}_final.pt
```

Use `best.pt` for the paper-style inference unless you intentionally want the final policy snapshot.

## 🔎 Inference

```bash
python test_policy_seq.py \
  --target_model Qwen/Qwen3-14B \
  --model_path gpt-3.5-turbo \
  --ckpt_path trained_models/ppo_seq/qwen3_14b_ahrl_index0_best.pt \
  --index 0 \
  --datasets advbench \
  --history_size 4 \
  --seqattn simple \
  --max_query 10000 \
  --max_attempts_per_question 50 \
  --K 1000
```

Outputs are written to:

```text
datasets/prompts_generated/RL_{target_model_name}_{index}.csv
datasets/prompts_generated/RL_{target_model_name}_{index}_processed.csv
datasets/eval/RL_{target_model_name}_{index}_responses_none.csv
datasets/eval/RL_{target_model_name}_{index}_eval_none.csv
```

## 📊 Evaluation

Run GPT-4o judging on saved responses:

```bash
python analyze_results.py \
  --file_path datasets/eval/RL_Qwen3-14B_0_responses_none.csv \
  --target_model Qwen/Qwen3-14B
```

This writes `{target_model_name}.json` with `GPT-Judge-Success` and `GPT-Judge-ASR`.
