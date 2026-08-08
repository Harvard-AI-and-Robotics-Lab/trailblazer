# Dataset Placement

This repository does not commit benchmark prompts, generated attack templates, model responses, or checkpoints.

Prepare the following files before training or inference:

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

Schemas:

- `processed_unalign.csv`: `question,response`
- `datasets/questions/*_questions.csv`: `text`
- `datasets/prompts/jailbreak-prompt.xlsx`: `text`

Runtime outputs are written under `datasets/prompts_generated/` and `datasets/eval/`.
