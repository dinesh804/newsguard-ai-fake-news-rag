"""CPU-friendly RoBERTa fine-tuning for NewsGuard AI.

Expected CSV columns: text,label
Labels accepted: REAL/FAKE, TRUE/FALSE, genuine/fabricated, or 0/1.
IMPORTANT: Confirm that numeric 0 means REAL and 1 means FAKE in your dataset.

Quick CPU test:
python train_roberta.py --csv data/news.csv --max-samples 500 --epochs 1 --max-length 96 --quick

Full dataset training:
python train_roberta.py --csv data/news.csv --max-samples 0 --epochs 2
"""

import argparse
import json
import os
import random
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    cohen_kappa_score,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split


def normalize_label(value):
    """Map supported labels to REAL=0 and FAKE=1."""
    label = str(value).strip().lower()
    if label in {"fake", "false", "fabricated", "1", "fake news"}:
        return 1
    if label in {"real", "true", "genuine", "0", "real news"}:
        return 0
    return None


def parse_args():
    parser = argparse.ArgumentParser(
        description="Fine-tune RoBERTa on a labelled REAL/FAKE news CSV."
    )
    parser.add_argument("--csv", required=True, help="Path to labelled CSV")
    parser.add_argument("--text-col", default="text")
    parser.add_argument("--label-col", default="label")
    parser.add_argument("--model-name", default="FacebookAI/roberta-base")
    parser.add_argument("--epochs", type=float, default=1.0)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--max-length", type=int, default=128)
    parser.add_argument(
        "--max-samples",
        type=int,
        default=1000,
        help="Stratified development sample size; use 0 for the full dataset",
    )
    parser.add_argument("--cpu-threads", type=int, default=4)
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Skip validation during training and intermediate checkpoints",
    )
    parser.add_argument("--output-dir", default="models/roberta-fake-news")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main():
    args = parse_args()
    if args.epochs <= 0 or args.batch_size < 1 or args.max_length < 8:
        raise SystemExit("epochs, batch-size, and max-length must be positive.")
    if args.max_samples < 0 or args.cpu_threads < 1:
        raise SystemExit("max-samples must be >= 0 and cpu-threads >= 1.")

    # Set thread limits before importing/loading PyTorch model operations.
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    os.environ.setdefault("OMP_NUM_THREADS", str(args.cpu_threads))
    os.environ.setdefault("MKL_NUM_THREADS", str(args.cpu_threads))

    try:
        import torch
        from datasets import Dataset
        from transformers import (
            AutoModelForSequenceClassification,
            AutoTokenizer,
            DataCollatorWithPadding,
            Trainer,
            TrainingArguments,
        )
    except ImportError as exc:
        raise SystemExit(
            f"Missing dependency: {exc}\n"
            "Activate your virtual environment and run: pip install -r requirements.txt"
        ) from exc

    torch.set_num_threads(args.cpu_threads)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    start = time.time()
    csv_path = Path(args.csv)
    if not csv_path.is_file():
        raise SystemExit(f"Dataset not found: {csv_path.resolve()}")

    print(f"[1/6] Reading dataset: {csv_path}")
    # Read only requested columns to reduce memory use.
    header = pd.read_csv(csv_path, nrows=0).columns.tolist()
    if args.text_col not in header or args.label_col not in header:
        raise SystemExit(
            f"Columns not found. Available columns: {header}\n"
            f"Requested text={args.text_col!r}, label={args.label_col!r}"
        )

    df = pd.read_csv(csv_path, usecols=[args.text_col, args.label_col])
    df = df.dropna(subset=[args.text_col, args.label_col]).copy()
    df["text"] = df[args.text_col].astype(str).str.strip()
    df["label"] = df[args.label_col].map(normalize_label)
    df = df.dropna(subset=["label"])
    df = df[df["text"].str.len() >= 20]
    df = df.drop_duplicates(subset=["text"])
    df = df[["text", "label"]].copy()
    df["label"] = df["label"].astype("int64")
    df = df.reset_index(drop=True)

    if df["label"].nunique() != 2 or df["label"].value_counts().min() < 4:
        raise SystemExit(
            "Need at least 4 examples in each class after cleaning. "
            "Check the CSV and numeric label mapping (0=REAL, 1=FAKE)."
        )

    if args.max_samples > 0 and len(df) > args.max_samples:
        # Stratify before splitting so both classes remain represented.
        df, _ = train_test_split(
            df,
            train_size=args.max_samples,
            random_state=args.seed,
            stratify=df["label"],
        )
        df = df.reset_index(drop=True)

    print(f"Clean rows selected: {len(df):,}")
    print(f"Class counts:\n{df['label'].map({0: 'REAL', 1: 'FAKE'}).value_counts()}")

    train_df, temp_df = train_test_split(
        df,
        test_size=0.20,
        random_state=args.seed,
        stratify=df["label"],
    )
    val_df, test_df = train_test_split(
        temp_df,
        test_size=0.50,
        random_state=args.seed,
        stratify=temp_df["label"],
    )

    print("[2/6] Loading tokenizer (first run downloads model files if needed)...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)

    def tokenize_batch(batch):
        # Dynamic padding happens later in DataCollatorWithPadding.
        return tokenizer(
            batch["text"],
            truncation=True,
            max_length=args.max_length,
            padding=False,
        )

    def to_dataset(frame, name):
        dataset = Dataset.from_pandas(
            frame[["text", "label"]].reset_index(drop=True),
            preserve_index=False,
        )
        dataset = dataset.map(
            tokenize_batch,
            batched=True,
            batch_size=128,
            remove_columns=["text"],
            desc=f"Tokenizing {name}",
        )
        return dataset.rename_column("label", "labels")

    print("[3/6] Tokenizing splits...")
    train_ds = to_dataset(train_df, "train")
    val_ds = to_dataset(val_df, "validation")
    test_ds = to_dataset(test_df, "test")

    print("[4/6] Loading pretrained model...")
    model = AutoModelForSequenceClassification.from_pretrained(
        args.model_name,
        num_labels=2,
        id2label={0: "REAL", 1: "FAKE"},
        label2id={"REAL": 0, "FAKE": 1},
    )

    def compute_metrics(eval_prediction):
        logits = eval_prediction.predictions
        if isinstance(logits, tuple):
            logits = logits[0]
        labels = eval_prediction.label_ids
        predictions = np.argmax(logits, axis=1)
        return {
            "accuracy": accuracy_score(labels, predictions),
            "precision_fake": precision_score(labels, predictions, zero_division=0),
            "recall_fake": recall_score(labels, predictions, zero_division=0),
            "f1_fake": f1_score(labels, predictions, zero_division=0),
        }

    os.makedirs("reports", exist_ok=True)
    os.makedirs(args.output_dir, exist_ok=True)

    training_kwargs = dict(
        output_dir="reports/roberta-training",
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        learning_rate=2e-5,
        weight_decay=0.01,
        logging_steps=10,
        report_to="none",
        fp16=False,
        dataloader_num_workers=0,  # safest default on Windows
        seed=args.seed,
        save_safetensors=True,
    )

    if args.quick:
        training_kwargs.update(
            eval_strategy="no",
            save_strategy="no",
            load_best_model_at_end=False,
        )
    else:
        training_kwargs.update(
            eval_strategy="epoch",
            save_strategy="epoch",
            load_best_model_at_end=True,
            metric_for_best_model="f1_fake",
            greater_is_better=True,
            save_total_limit=1,
        )

    training_args = TrainingArguments(**training_kwargs)
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        processing_class=tokenizer,
        data_collator=DataCollatorWithPadding(tokenizer=tokenizer),
        compute_metrics=compute_metrics,
    )

    print("[5/6] Training started. CPU training can still take time.")
    trainer.train()

    if not args.quick:
        print("Validation metrics:")
        print(json.dumps(trainer.evaluate(), indent=2))

    print("[6/6] Evaluating once on held-out test set...")
    output = trainer.predict(test_ds)
    logits = output.predictions
    if isinstance(logits, tuple):
        logits = logits[0]
    y_true = output.label_ids
    y_pred = np.argmax(logits, axis=1)

    # Stable softmax.
    shifted_logits = logits - np.max(logits, axis=1, keepdims=True)
    exp_logits = np.exp(shifted_logits)
    probabilities = exp_logits / np.sum(exp_logits, axis=1, keepdims=True)

    matrix = confusion_matrix(y_true, y_pred, labels=[0, 1])
    report = {
        "model_name": args.model_name,
        "rows_total": int(len(df)),
        "train_rows": int(len(train_df)),
        "validation_rows": int(len(val_df)),
        "test_rows": int(len(test_df)),
        "epochs": args.epochs,
        "max_length": args.max_length,
        "max_samples": args.max_samples,
        "quick_mode": args.quick,
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision_fake": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall_fake": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1_fake": float(f1_score(y_true, y_pred, zero_division=0)),
        "specificity": float(recall_score(y_true, y_pred, pos_label=0, zero_division=0)),
        "false_positive_rate": float(matrix[0, 1] / max(1, matrix[0].sum())),
        "false_negative_rate": float(matrix[1, 0] / max(1, matrix[1].sum())),
        "mcc": float(matthews_corrcoef(y_true, y_pred)),
        "cohen_kappa": float(cohen_kappa_score(y_true, y_pred)),
        "roc_auc": (
            float(roc_auc_score(y_true, probabilities[:, 1]))
            if len(np.unique(y_true)) == 2
            else None
        ),
        "confusion_matrix_labels_real_fake": matrix.tolist(),
        "classification_report": classification_report(
            y_true,
            y_pred,
            labels=[0, 1],
            target_names=["REAL", "FAKE"],
            output_dict=True,
            zero_division=0,
        ),
        "elapsed_seconds": round(time.time() - start, 2),
    }

    print("Saving model and tokenizer...")
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)

    report_path = Path("reports/evaluation.json")
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print("\nFinal test metrics:")
    print(json.dumps(
        {key: value for key, value in report.items()
         if key not in ("classification_report", "confusion_matrix_labels_real_fake")},
        indent=2,
    ))
    print(f"\nConfusion matrix [REAL, FAKE]: {matrix.tolist()}")
    print(f"Model saved to: {Path(args.output_dir).resolve()}")
    print(f"Evaluation saved to: {report_path.resolve()}")


if __name__ == "__main__":
    main()
