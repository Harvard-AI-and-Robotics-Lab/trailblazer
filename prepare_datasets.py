import argparse
import os

import pandas as pd


def write_question_file(df, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    out = pd.DataFrame({"index": range(1, len(df) + 1), "text": df["question"].astype(str)})
    out.to_csv(path, index=False)
    print(f"wrote {len(out)} questions to {path}")


def split_processed_unalign(input_path, output_dir, train_size, seed):
    df = pd.read_csv(input_path)
    if not {"question", "response"}.issubset(df.columns):
        raise ValueError(f"{input_path} must contain question,response columns")

    df = df.sample(frac=1.0, random_state=seed).reset_index(drop=True)
    train_df = df.iloc[:train_size].reset_index(drop=True)
    test_df = df.iloc[train_size:].reset_index(drop=True)

    os.makedirs(output_dir, exist_ok=True)
    train_path = os.path.join(output_dir, "processed_unalign_train.csv")
    test_path = os.path.join(output_dir, "processed_unalign_test.csv")
    train_df.to_csv(train_path, index=False)
    test_df.to_csv(test_path, index=False)

    write_question_file(
        train_df,
        os.path.join(output_dir, "questions", "processed_unalign_train_questions.csv"),
    )
    write_question_file(
        test_df,
        os.path.join(output_dir, "questions", "processed_unalign_test_questions.csv"),
    )

    print(f"train rows: {len(train_df)} -> {train_path}")
    print(f"test rows: {len(test_df)} -> {test_path}")


def extract_questions(input_path, output_path):
    df = pd.read_csv(input_path)
    if "question" not in df.columns:
        raise ValueError(f"{input_path} must contain a question column")
    write_question_file(df, output_path)


def main():
    parser = argparse.ArgumentParser(description="Prepare TrailBlazer dataset files")
    parser.add_argument("--input", default="datasets/processed_unalign.csv",
                        help="CSV with question,response columns")
    parser.add_argument("--output_dir", default="datasets")
    parser.add_argument("--train_size", type=int, default=364,
                        help="Default AdvBench train size used in the paper")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--extract_only", action="store_true",
                        help="Only extract question CSV from --input")
    parser.add_argument("--question_output", default=None,
                        help="Required with --extract_only unless using the default test path")
    args = parser.parse_args()

    if args.extract_only:
        output_path = args.question_output or os.path.join(
            args.output_dir, "questions", "processed_unalign_test_questions.csv"
        )
        extract_questions(args.input, output_path)
    else:
        split_processed_unalign(args.input, args.output_dir, args.train_size, args.seed)


if __name__ == "__main__":
    main()
