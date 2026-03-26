"""
RoBERTa fine-tuning script for prompt injection detection.

Supports roberta-base and roberta-large (and any HuggingFace seq-classification
compatible model).  Designed to run on:
  - Local GPU  (RTX 4090, 16 GB VRAM) — with fp16 enabled by default
  - Sol HPC    — pass --model-path to a local directory instead of a HF hub ID

Usage:
  # Train roberta-base (default, runs on your 4090)
  python src/train_roberta.py

  # Train roberta-large
  python src/train_roberta.py --model roberta-large

  # Use a locally downloaded model (for Sol — no internet)
  python src/train_roberta.py --model /path/to/roberta-base

  # Quick smoke test (1 epoch, small batch)
  python src/train_roberta.py --epochs 1 --batch-size 8 --smoke-test

Outputs saved to results/roberta/<model-tag>/:
  metrics.json                - full metrics for every eval split
  confusion_<split>.png       - confusion matrix per split
  roc.png                     - ROC curves across all splits
  pr.png                      - Precision-Recall curves

Model checkpoint saved to models/roberta/<model-tag>/
"""

import argparse
import json
import logging
import os
from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
    precision_recall_curve,
)
from sklearn.utils.class_weight import compute_class_weight
from torch.utils.data import DataLoader, Dataset
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    EarlyStoppingCallback,
    Trainer,
    TrainingArguments,
    set_seed,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

PROJECT_ROOT  = Path(__file__).resolve().parent.parent
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
MODELS_DIR    = PROJECT_ROOT / "models" / "roberta"
RESULTS_DIR   = PROJECT_ROOT / "results" / "roberta"

RANDOM_SEED = 42


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class PromptDataset(Dataset):
    def __init__(self, encodings: dict, labels: list[int]):
        self.encodings = encodings
        self.labels    = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        item = {k: torch.tensor(v[idx]) for k, v in self.encodings.items()}
        item["labels"] = torch.tensor(self.labels[idx], dtype=torch.long)
        return item


def tokenize(
    tokenizer,
    texts: pd.Series,
    max_length: int,
) -> dict:
    return tokenizer(
        texts.tolist(),
        truncation=True,
        padding="max_length",
        max_length=max_length,
        return_tensors=None,   # return plain lists — Trainer handles tensor conversion
    )


# ---------------------------------------------------------------------------
# Class-weighted Trainer
# Applies higher loss weight to the minority class (malicious prompts)
# so the model doesn't trade recall for precision.
# ---------------------------------------------------------------------------

class WeightedTrainer(Trainer):
    def __init__(self, class_weights: torch.Tensor, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.class_weights = class_weights

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        logits  = outputs.logits
        weights = self.class_weights.to(logits.device)
        loss_fn = torch.nn.CrossEntropyLoss(weight=weights)
        loss    = loss_fn(logits, labels)
        return (loss, outputs) if return_outputs else loss


# ---------------------------------------------------------------------------
# Metrics callback used by Trainer during validation
# ---------------------------------------------------------------------------

def make_compute_metrics(threshold: float = 0.5):
    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        probs = torch.softmax(torch.tensor(logits), dim=-1).numpy()[:, 1]
        preds = (probs >= threshold).astype(int)
        return {
            "accuracy":  accuracy_score(labels, preds),
            "f1":        f1_score(labels, preds, zero_division=0),
            "precision": precision_score(labels, preds, zero_division=0),
            "recall":    recall_score(labels, preds, zero_division=0),
            "roc_auc":   roc_auc_score(labels, probs),
        }
    return compute_metrics


# ---------------------------------------------------------------------------
# Full evaluation on a split (post-training)
# ---------------------------------------------------------------------------

@torch.no_grad()
def evaluate_split(
    model,
    tokenizer,
    df: pd.DataFrame,
    split_name: str,
    max_length: int,
    batch_size: int,
    device: torch.device,
) -> dict:
    if df.empty:
        log.warning(f"  {split_name} is empty — skipping")
        return {}

    model.eval()
    encodings = tokenize(tokenizer, df["text"], max_length)
    dataset   = PromptDataset(encodings, df["label"].tolist())
    loader    = DataLoader(dataset, batch_size=batch_size * 2, shuffle=False)

    all_probs  = []
    all_labels = []

    for batch in loader:
        input_ids      = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        token_type_ids = batch.get("token_type_ids")
        labels         = batch["labels"].numpy()

        kwargs = {"input_ids": input_ids, "attention_mask": attention_mask}
        if token_type_ids is not None:
            kwargs["token_type_ids"] = token_type_ids.to(device)

        outputs = model(**kwargs)
        probs   = torch.softmax(outputs.logits, dim=-1)[:, 1].cpu().numpy()

        all_probs.extend(probs.tolist())
        all_labels.extend(labels.tolist())

    all_probs  = np.array(all_probs)
    all_labels = np.array(all_labels)
    all_preds  = (all_probs >= 0.5).astype(int)

    metrics = {
        "split":         split_name,
        "n_samples":     len(all_labels),
        "accuracy":      round(float(accuracy_score(all_labels, all_preds)), 4),
        "precision":     round(float(precision_score(all_labels, all_preds, zero_division=0)), 4),
        "recall":        round(float(recall_score(all_labels, all_preds, zero_division=0)), 4),
        "f1":            round(float(f1_score(all_labels, all_preds, zero_division=0)), 4),
        "roc_auc":       round(float(roc_auc_score(all_labels, all_probs)), 4),
        "avg_precision": round(float(average_precision_score(all_labels, all_probs)), 4),
    }
    metrics["report"] = classification_report(
        all_labels, all_preds, target_names=["benign", "malicious"]
    )

    log.info(
        f"  {split_name:20s}  acc={metrics['accuracy']:.4f}  "
        f"f1={metrics['f1']:.4f}  auc={metrics['roc_auc']:.4f}"
    )
    print(metrics["report"])

    return {**metrics, "_probs": all_probs, "_labels": all_labels, "_preds": all_preds}


# ---------------------------------------------------------------------------
# Plotting (same style as baseline.py for easy comparison)
# ---------------------------------------------------------------------------

def plot_confusion(labels, preds, split_name: str, out_dir: Path) -> None:
    cm = confusion_matrix(labels, preds)
    fig, ax = plt.subplots(figsize=(5, 4))
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues", ax=ax,
        xticklabels=["benign", "malicious"],
        yticklabels=["benign", "malicious"],
    )
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title(f"Confusion Matrix — RoBERTa / {split_name}")
    fig.tight_layout()
    fig.savefig(out_dir / f"confusion_{split_name}.png", dpi=150)
    plt.close(fig)


def plot_roc(results: list[dict], out_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(7, 5))
    for r in results:
        if not r:
            continue
        fpr, tpr, _ = roc_curve(r["_labels"], r["_probs"])
        ax.plot(fpr, tpr, label=f"{r['split']}  (AUC={r['roc_auc']:.3f})")
    ax.plot([0, 1], [0, 1], "k--", linewidth=0.8)
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curves — RoBERTa")
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(out_dir / "roc.png", dpi=150)
    plt.close(fig)


def plot_pr(results: list[dict], out_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(7, 5))
    for r in results:
        if not r:
            continue
        prec, rec, _ = precision_recall_curve(r["_labels"], r["_probs"])
        ax.plot(rec, prec, label=f"{r['split']}  (AP={r['avg_precision']:.3f})")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("Precision-Recall Curves — RoBERTa")
    ax.legend(loc="upper right")
    fig.tight_layout()
    fig.savefig(out_dir / "pr.png", dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Fine-tune RoBERTa for prompt injection detection")
    parser.add_argument("--model",        default="roberta-base",
                        help="HuggingFace model ID or local path (default: roberta-base)")
    parser.add_argument("--epochs",       type=int,   default=5)
    parser.add_argument("--batch-size",   type=int,   default=16,
                        help="Per-device batch size (default: 16)")
    parser.add_argument("--grad-accum",   type=int,   default=2,
                        help="Gradient accumulation steps (effective batch = batch_size * grad_accum)")
    parser.add_argument("--max-length",   type=int,   default=256,
                        help="Max token length (default: 256; RoBERTa max is 512)")
    parser.add_argument("--lr",           type=float, default=2e-5,
                        help="Peak learning rate (default: 2e-5)")
    parser.add_argument("--warmup-ratio", type=float, default=0.1)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--no-fp16",      action="store_true",
                        help="Disable mixed-precision training")
    parser.add_argument("--patience",     type=int,   default=3,
                        help="Early stopping patience in eval steps (default: 3)")
    parser.add_argument("--smoke-test",   action="store_true",
                        help="Use 1000 training samples for a quick sanity check")
    args = parser.parse_args()

    set_seed(RANDOM_SEED)

    # ---- Device -----------------------------------------------------------
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        log.info(f"GPU: {torch.cuda.get_device_name(0)}  "
                 f"({torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB VRAM)")
    else:
        log.warning("No GPU detected — training on CPU (will be slow)")

    use_fp16 = (device.type == "cuda") and (not args.no_fp16)

    # ---- Paths ------------------------------------------------------------
    model_tag  = Path(args.model).name          # e.g. "roberta-base"
    result_dir = RESULTS_DIR / model_tag
    model_dir  = MODELS_DIR  / model_tag
    ckpt_dir   = model_dir / "checkpoints"
    result_dir.mkdir(parents=True, exist_ok=True)
    model_dir.mkdir(parents=True, exist_ok=True)

    # ---- Load data --------------------------------------------------------
    log.info("Loading processed splits...")
    split_files = {
        "train":         PROCESSED_DIR / "train.parquet",
        "val":           PROCESSED_DIR / "val.parquet",
        "test":          PROCESSED_DIR / "test.parquet",
        "test_deepset":  PROCESSED_DIR / "test_deepset.parquet",
        "test_wildcard": PROCESSED_DIR / "test_wildcard.parquet",
    }
    splits = {}
    for name, path in split_files.items():
        if path.exists():
            splits[name] = pd.read_parquet(path)
            log.info(f"  {name:20s}  {len(splits[name]):>8,} rows")
        else:
            log.warning(f"  {name} not found — skipping")
            splits[name] = pd.DataFrame(columns=["text", "label"])

    train_df = splits["train"]
    val_df   = splits["val"]

    if train_df.empty:
        log.error("Training data missing. Run prepare_data.py first.")
        raise SystemExit(1)

    if args.smoke_test:
        log.info("Smoke-test mode: using 1,000 training samples")
        train_df = train_df.sample(n=min(1000, len(train_df)), random_state=RANDOM_SEED)

    # ---- Tokenizer --------------------------------------------------------
    log.info(f"Loading tokenizer: {args.model}")
    tokenizer = AutoTokenizer.from_pretrained(args.model)

    log.info("Tokenizing train split...")
    train_enc = tokenize(tokenizer, train_df["text"], args.max_length)
    train_ds  = PromptDataset(train_enc, train_df["label"].tolist())

    log.info("Tokenizing val split...")
    val_enc = tokenize(tokenizer, val_df["text"], args.max_length)
    val_ds  = PromptDataset(val_enc, val_df["label"].tolist())

    # ---- Class weights (handles any residual imbalance) -------------------
    y_train = train_df["label"].values
    cw = compute_class_weight("balanced", classes=np.array([0, 1]), y=y_train)
    class_weights = torch.tensor(cw, dtype=torch.float32)
    log.info(f"Class weights  benign={cw[0]:.3f}  malicious={cw[1]:.3f}")

    # ---- Model ------------------------------------------------------------
    log.info(f"Loading model: {args.model}")
    model = AutoModelForSequenceClassification.from_pretrained(
        args.model,
        num_labels=2,
        ignore_mismatched_sizes=True,
    )

    # ---- Training arguments -----------------------------------------------
    training_args = TrainingArguments(
        output_dir=str(ckpt_dir),
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size * 2,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        warmup_ratio=args.warmup_ratio,
        weight_decay=args.weight_decay,
        fp16=use_fp16,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        greater_is_better=True,
        logging_dir=str(result_dir / "logs"),
        logging_steps=50,
        report_to="none",            # set to "wandb" if you want experiment tracking
        seed=RANDOM_SEED,
        dataloader_num_workers=4 if device.type == "cuda" else 0,
        dataloader_pin_memory=(device.type == "cuda"),
    )

    log.info(
        f"Training config: epochs={args.epochs}  "
        f"batch={args.batch_size}  accum={args.grad_accum}  "
        f"effective_batch={args.batch_size * args.grad_accum}  "
        f"lr={args.lr}  fp16={use_fp16}  max_len={args.max_length}"
    )

    # ---- Trainer ----------------------------------------------------------
    trainer = WeightedTrainer(
        class_weights=class_weights,
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        compute_metrics=make_compute_metrics(),
        callbacks=[EarlyStoppingCallback(early_stopping_patience=args.patience)],
    )

    # ---- Train ------------------------------------------------------------
    log.info("=" * 60)
    log.info("Starting training...")
    log.info("=" * 60)
    trainer.train()

    # ---- Save best model --------------------------------------------------
    trainer.save_model(str(model_dir))
    tokenizer.save_pretrained(str(model_dir))
    log.info(f"Best model saved to {model_dir}")

    # ---- Evaluate all splits with the best model --------------------------
    log.info("=" * 60)
    log.info("Evaluating all splits with best checkpoint...")
    log.info("=" * 60)

    best_model = trainer.model.to(device)

    eval_splits = {k: v for k, v in splits.items() if k != "train" and not v.empty}
    all_results = []

    for split_name, df in eval_splits.items():
        result = evaluate_split(
            best_model, tokenizer, df, split_name,
            args.max_length, args.batch_size, device,
        )
        if result:
            all_results.append(result)
            plot_confusion(result["_labels"], result["_preds"], split_name, result_dir)

    # ---- Plots ------------------------------------------------------------
    plot_roc(all_results, result_dir)
    plot_pr(all_results, result_dir)

    # ---- Save metrics -----------------------------------------------------
    clean_results = [
        {k: v for k, v in r.items() if not k.startswith("_")}
        for r in all_results
    ]

    metrics_path = result_dir / "metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(clean_results, f, indent=2)
    log.info(f"Metrics saved to {metrics_path}")

    # ---- Save curve data for interactive charts ---------------------------
    curves = {"roc": [], "pr": [], "confusions": []}
    for r in all_results:
        fpr, tpr, _ = roc_curve(r["_labels"], r["_probs"])
        curves["roc"].append({"split": r["split"], "fpr": fpr.tolist(), "tpr": tpr.tolist(), "auc": r["roc_auc"]})
        prec, rec, _ = precision_recall_curve(r["_labels"], r["_probs"])
        curves["pr"].append({"split": r["split"], "precision": prec.tolist(), "recall": rec.tolist(), "ap": r["avg_precision"]})
        cm = confusion_matrix(r["_labels"], r["_preds"])
        curves["confusions"].append({"split": r["split"], "matrix": cm.tolist(), "labels": ["Benign", "Malicious"]})
    with open(result_dir / "curves.json", "w") as f:
        json.dump(curves, f)
    log.info(f"Curve data saved to {result_dir / 'curves.json'}")

    summary = pd.DataFrame(clean_results).drop(columns=["report"], errors="ignore")
    print("\nSummary:")
    print(summary.to_string(index=False))

    # ---- Save training args for reproducibility ---------------------------
    config = vars(args)
    config["fp16_used"] = use_fp16
    config["device"]    = str(device)
    if device.type == "cuda":
        config["gpu_name"] = torch.cuda.get_device_name(0)
    with open(result_dir / "train_config.json", "w") as f:
        json.dump(config, f, indent=2)

    log.info("RoBERTa fine-tuning complete.")


if __name__ == "__main__":
    main()
