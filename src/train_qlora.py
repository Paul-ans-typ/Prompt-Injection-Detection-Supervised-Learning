"""
QLoRA fine-tuning script for large LLMs (Qwen2.5, DeepSeek, etc.)
for prompt injection detection.

Uses 4-bit NF4 quantization (bitsandbytes) + LoRA adapters (PEFT) so that
a 7B model trains in ~6 GB VRAM and a 32B model fits on a single A100 80 GB.

Designed to run on:
  - RTX 4090 16 GB  →  7B / 14B models
  - Sol HPC A100    →  32B model (use SLURM script: scripts/slurm_qlora.sh)

Usage:
  # 7B on your 4090 (default)
  python src/train_qlora.py

  # 14B — still fits on 4090 with batch 2 + grad_accum 16
  python src/train_qlora.py --model Qwen/Qwen2.5-14B-Instruct --batch-size 2 --grad-accum 16

  # 32B on Sol (after downloading weights locally + scp to Sol)
  python src/train_qlora.py --model /path/to/Qwen2.5-32B-Instruct --batch-size 4 --grad-accum 8

  # Smoke test
  python src/train_qlora.py --smoke-test --epochs 1

  # Merge LoRA adapter into base model and save (for inference)
  python src/train_qlora.py --merge-and-save

Outputs saved to results/qlora/<model-tag>/:
  metrics.json            - full metrics for every eval split
  confusion_<split>.png   - confusion matrices
  roc.png / pr.png        - curves across all splits
  train_config.json       - every hyperparameter used

LoRA adapter saved to  models/qlora/<model-tag>/adapter/
Merged model saved to  models/qlora/<model-tag>/merged/   (if --merge-and-save)
"""

import argparse
import json
import logging
import os
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
from peft import (
    LoraConfig,
    TaskType,
    get_peft_model,
    prepare_model_for_kbit_training,
)
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.utils.class_weight import compute_class_weight
from torch.utils.data import DataLoader, Dataset
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    BitsAndBytesConfig,
    EarlyStoppingCallback,
    Trainer,
    TrainingArguments,
    set_seed,
)

warnings.filterwarnings("ignore")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

PROJECT_ROOT  = Path(__file__).resolve().parent.parent
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
MODELS_DIR    = PROJECT_ROOT / "models" / "qlora"
RESULTS_DIR   = PROJECT_ROOT / "results" / "qlora"

RANDOM_SEED = 42

# ---------------------------------------------------------------------------
# LoRA target modules per model family
# Extend this dict if you add more model families later.
# ---------------------------------------------------------------------------
LORA_TARGET_MAP = {
    "qwen":     ["q_proj", "k_proj", "v_proj", "o_proj"],
    "deepseek": ["q_proj", "k_proj", "v_proj", "o_proj"],
    "llama":    ["q_proj", "k_proj", "v_proj", "o_proj"],
    "mistral":  ["q_proj", "k_proj", "v_proj", "o_proj"],
    "default":  ["q_proj", "v_proj"],
}


def get_lora_targets(model_name: str) -> list[str]:
    name = model_name.lower()
    for family, targets in LORA_TARGET_MAP.items():
        if family in name:
            return targets
    return LORA_TARGET_MAP["default"]


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


def tokenize(tokenizer, texts: pd.Series, max_length: int) -> dict:
    return tokenizer(
        texts.tolist(),
        truncation=True,
        padding="max_length",
        max_length=max_length,
        return_tensors=None,
    )


# ---------------------------------------------------------------------------
# Class-weighted Trainer  (same pattern as train_roberta.py)
# ---------------------------------------------------------------------------

class WeightedTrainer(Trainer):
    def __init__(self, class_weights: torch.Tensor, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.class_weights = class_weights

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels  = inputs.pop("labels")
        outputs = model(**inputs)
        logits  = outputs.logits
        weights = self.class_weights.to(logits.device)
        loss    = torch.nn.CrossEntropyLoss(weight=weights)(logits, labels)
        return (loss, outputs) if return_outputs else loss


# ---------------------------------------------------------------------------
# Compute metrics callback (used during training evals)
# ---------------------------------------------------------------------------

def make_compute_metrics():
    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        probs = torch.softmax(torch.tensor(logits, dtype=torch.float32), dim=-1).numpy()[:, 1]
        preds = (probs >= 0.5).astype(int)
        return {
            "accuracy":  float(accuracy_score(labels, preds)),
            "f1":        float(f1_score(labels, preds, zero_division=0)),
            "precision": float(precision_score(labels, preds, zero_division=0)),
            "recall":    float(recall_score(labels, preds, zero_division=0)),
            "roc_auc":   float(roc_auc_score(labels, probs)),
        }
    return compute_metrics


# ---------------------------------------------------------------------------
# Full post-training evaluation
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

    all_probs, all_labels = [], []

    for batch in loader:
        input_ids      = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels         = batch["labels"].numpy()

        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        probs   = torch.softmax(outputs.logits.float(), dim=-1)[:, 1].cpu().numpy()

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
        "report":        classification_report(
                             all_labels, all_preds,
                             target_names=["benign", "malicious"]
                         ),
    }

    log.info(
        f"  {split_name:20s}  acc={metrics['accuracy']:.4f}  "
        f"f1={metrics['f1']:.4f}  auc={metrics['roc_auc']:.4f}"
    )
    print(metrics["report"])

    return {**metrics, "_probs": all_probs, "_labels": all_labels, "_preds": all_preds}


# ---------------------------------------------------------------------------
# Plots (consistent style across all model scripts)
# ---------------------------------------------------------------------------

def plot_confusion(labels, preds, split_name: str, model_tag: str, out_dir: Path) -> None:
    cm = confusion_matrix(labels, preds)
    fig, ax = plt.subplots(figsize=(5, 4))
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues", ax=ax,
        xticklabels=["benign", "malicious"],
        yticklabels=["benign", "malicious"],
    )
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title(f"Confusion Matrix — {model_tag} / {split_name}")
    fig.tight_layout()
    fig.savefig(out_dir / f"confusion_{split_name}.png", dpi=150)
    plt.close(fig)


def plot_roc(results: list[dict], model_tag: str, out_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(7, 5))
    for r in results:
        fpr, tpr, _ = roc_curve(r["_labels"], r["_probs"])
        ax.plot(fpr, tpr, label=f"{r['split']}  (AUC={r['roc_auc']:.3f})")
    ax.plot([0, 1], [0, 1], "k--", linewidth=0.8)
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title(f"ROC Curves — {model_tag}")
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(out_dir / "roc.png", dpi=150)
    plt.close(fig)


def plot_pr(results: list[dict], model_tag: str, out_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(7, 5))
    for r in results:
        prec, rec, _ = precision_recall_curve(r["_labels"], r["_probs"])
        ax.plot(rec, prec, label=f"{r['split']}  (AP={r['avg_precision']:.3f})")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title(f"Precision-Recall Curves — {model_tag}")
    ax.legend(loc="upper right")
    fig.tight_layout()
    fig.savefig(out_dir / "pr.png", dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="QLoRA fine-tuning for prompt injection detection"
    )
    parser.add_argument(
        "--model", default="Qwen/Qwen2.5-7B-Instruct",
        help="HuggingFace model ID or local path (default: Qwen/Qwen2.5-7B-Instruct)",
    )
    parser.add_argument("--epochs",       type=int,   default=3)
    parser.add_argument("--batch-size",   type=int,   default=4,
                        help="Per-device batch size (default: 4)")
    parser.add_argument("--grad-accum",   type=int,   default=8,
                        help="Gradient accumulation steps (effective batch = batch_size * grad_accum)")
    parser.add_argument("--max-length",   type=int,   default=256)
    parser.add_argument("--lr",           type=float, default=2e-4,
                        help="LoRA learning rate (higher than full fine-tune, default: 2e-4)")
    parser.add_argument("--warmup-ratio", type=float, default=0.05)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--lora-r",       type=int,   default=16,
                        help="LoRA rank (default: 16; increase for harder tasks)")
    parser.add_argument("--lora-alpha",   type=int,   default=32,
                        help="LoRA alpha — scaling = alpha/r (default: 32)")
    parser.add_argument("--lora-dropout", type=float, default=0.05)
    parser.add_argument("--lora-targets", type=str,   default=None,
                        help="Comma-separated LoRA target modules. "
                             "Auto-detected from model name if not set.")
    parser.add_argument("--no-4bit",      action="store_true",
                        help="Disable 4-bit quantization (use full bf16 — needs more VRAM)")
    parser.add_argument("--patience",     type=int,   default=2,
                        help="Early stopping patience in eval epochs (default: 2)")
    parser.add_argument("--merge-and-save", action="store_true",
                        help="After training, merge LoRA weights into base and save full model")
    parser.add_argument("--smoke-test",   action="store_true",
                        help="Use 500 training samples for a quick sanity check")
    args = parser.parse_args()

    set_seed(RANDOM_SEED)

    # ---- Device -----------------------------------------------------------
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        log.warning("No GPU detected. QLoRA training on CPU is not practical.")

    if device.type == "cuda":
        gpu_name = torch.cuda.get_device_name(0)
        vram_gb  = torch.cuda.get_device_properties(0).total_memory / 1e9
        log.info(f"GPU: {gpu_name}  ({vram_gb:.1f} GB VRAM)")
        # bf16 is preferred on Ampere/Ada (RTX 30xx/40xx) and A100
        use_bf16 = torch.cuda.is_bf16_supported()
        compute_dtype = torch.bfloat16 if use_bf16 else torch.float16
        log.info(f"Compute dtype: {'bfloat16' if use_bf16 else 'float16'}")
    else:
        compute_dtype = torch.float32
        use_bf16      = False

    # ---- Paths ------------------------------------------------------------
    model_tag  = Path(args.model).name
    result_dir = RESULTS_DIR / model_tag
    adapter_dir = MODELS_DIR / model_tag / "adapter"
    merged_dir  = MODELS_DIR / model_tag / "merged"
    ckpt_dir    = MODELS_DIR / model_tag / "checkpoints"

    result_dir.mkdir(parents=True, exist_ok=True)
    adapter_dir.mkdir(parents=True, exist_ok=True)

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
        log.info("Smoke-test mode: using 500 training samples")
        train_df = train_df.sample(n=min(500, len(train_df)), random_state=RANDOM_SEED)
        val_df   = val_df.sample(n=min(200, len(val_df)),   random_state=RANDOM_SEED)

    # ---- BitsAndBytes config (4-bit NF4 QLoRA) ----------------------------
    bnb_config = None
    if not args.no_4bit:
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",           # NF4 is best for normal-distributed weights
            bnb_4bit_compute_dtype=compute_dtype, # actual matmul dtype
            bnb_4bit_use_double_quant=True,       # double quantization saves ~0.4 bits/param extra
        )
        log.info("4-bit NF4 QLoRA enabled (double quantization ON)")
    else:
        log.info("4-bit quantization disabled — loading in bf16/fp16")

    # ---- Tokenizer --------------------------------------------------------
    log.info(f"Loading tokenizer: {args.model}")
    tokenizer = AutoTokenizer.from_pretrained(
        args.model,
        trust_remote_code=True,
        padding_side="right",   # right-pad for classification
    )

    # Many decoder-only models lack a dedicated pad token
    if tokenizer.pad_token is None:
        tokenizer.pad_token    = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
        log.info("Set pad_token = eos_token")

    # ---- Model ------------------------------------------------------------
    log.info(f"Loading model: {args.model}")
    model = AutoModelForSequenceClassification.from_pretrained(
        args.model,
        num_labels=2,
        quantization_config=bnb_config,
        device_map="auto",              # spreads across all available GPUs automatically
        trust_remote_code=True,
        torch_dtype=compute_dtype if args.no_4bit else None,
        ignore_mismatched_sizes=True,
    )

    # Sync model config pad token with tokenizer (required for batched inference)
    model.config.pad_token_id = tokenizer.pad_token_id

    # Required before applying LoRA to a quantized model
    model = prepare_model_for_kbit_training(
        model,
        use_gradient_checkpointing=True,
    )

    # Gradient checkpointing trades compute for memory — essential for 7B+
    model.config.use_cache = False   # incompatible with gradient checkpointing

    # Ensure the new classification head is in full precision
    if hasattr(model, "score"):
        model.score.weight.data = model.score.weight.data.to(torch.float32)

    # ---- LoRA -------------------------------------------------------------
    lora_targets = (
        args.lora_targets.split(",") if args.lora_targets
        else get_lora_targets(args.model)
    )
    log.info(f"LoRA targets: {lora_targets}")

    lora_config = LoraConfig(
        task_type=TaskType.SEQ_CLS,
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        target_modules=lora_targets,
        bias="none",
        inference_mode=False,
    )

    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    # ---- Tokenise ---------------------------------------------------------
    log.info(f"Tokenizing train ({len(train_df):,} samples, max_length={args.max_length})...")
    train_enc = tokenize(tokenizer, train_df["text"], args.max_length)
    train_ds  = PromptDataset(train_enc, train_df["label"].tolist())

    log.info(f"Tokenizing val ({len(val_df):,} samples)...")
    val_enc = tokenize(tokenizer, val_df["text"], args.max_length)
    val_ds  = PromptDataset(val_enc, val_df["label"].tolist())

    # ---- Class weights ----------------------------------------------------
    y_train = train_df["label"].values
    cw = compute_class_weight("balanced", classes=np.array([0, 1]), y=y_train)
    class_weights = torch.tensor(cw, dtype=torch.float32)
    log.info(f"Class weights  benign={cw[0]:.3f}  malicious={cw[1]:.3f}")

    # ---- Training arguments -----------------------------------------------
    effective_batch = args.batch_size * args.grad_accum
    log.info(
        f"Training config:  epochs={args.epochs}  batch={args.batch_size}  "
        f"accum={args.grad_accum}  effective_batch={effective_batch}  "
        f"lr={args.lr}  lora_r={args.lora_r}  lora_alpha={args.lora_alpha}"
    )

    training_args = TrainingArguments(
        output_dir=str(ckpt_dir),
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size * 2,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        warmup_ratio=args.warmup_ratio,
        weight_decay=args.weight_decay,
        bf16=use_bf16,
        fp16=(not use_bf16 and device.type == "cuda"),
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        greater_is_better=True,
        logging_dir=str(result_dir / "logs"),
        logging_steps=25,
        report_to="none",
        seed=RANDOM_SEED,
        gradient_checkpointing=True,
        optim="paged_adamw_8bit",   # 8-bit paged AdamW saves optimizer state memory
        lr_scheduler_type="cosine",
        dataloader_num_workers=4 if device.type == "cuda" else 0,
        dataloader_pin_memory=(device.type == "cuda"),
        remove_unused_columns=False,
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
    log.info("Starting QLoRA training...")
    log.info("=" * 60)

    # Resume from checkpoint if SLURM job was restarted after a timeout
    resume_ckpt = os.environ.get("RESUME_FROM_CHECKPOINT", None)
    if resume_ckpt:
        log.info(f"Resuming from checkpoint: {resume_ckpt}")
    trainer.train(resume_from_checkpoint=resume_ckpt or None)

    # ---- Save LoRA adapter ------------------------------------------------
    trainer.model.save_pretrained(str(adapter_dir))
    tokenizer.save_pretrained(str(adapter_dir))
    log.info(f"LoRA adapter saved to {adapter_dir}")

    # ---- Optionally merge adapter into base and save full model -----------
    if args.merge_and_save:
        log.info("Merging LoRA weights into base model...")
        from peft import PeftModel
        merged = trainer.model.merge_and_unload()
        merged_dir.mkdir(parents=True, exist_ok=True)
        merged.save_pretrained(str(merged_dir), safe_serialization=True)
        tokenizer.save_pretrained(str(merged_dir))
        log.info(f"Merged model saved to {merged_dir}")

    # ---- Evaluate all splits ----------------------------------------------
    log.info("=" * 60)
    log.info("Evaluating all splits with best checkpoint...")
    log.info("=" * 60)

    best_model = trainer.model
    eval_splits = {k: v for k, v in splits.items() if k != "train" and not v.empty}
    all_results = []

    for split_name, df in eval_splits.items():
        result = evaluate_split(
            best_model, tokenizer, df, split_name,
            args.max_length, args.batch_size, device,
        )
        if result:
            all_results.append(result)
            plot_confusion(result["_labels"], result["_preds"], split_name, model_tag, result_dir)

    if all_results:
        plot_roc(all_results, model_tag, result_dir)
        plot_pr(all_results, model_tag, result_dir)

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

    # ---- Save config for reproducibility ----------------------------------
    config = vars(args)
    config.update({
        "lora_targets_used": lora_targets,
        "compute_dtype":     str(compute_dtype),
        "device":            str(device),
        "effective_batch":   effective_batch,
    })
    if device.type == "cuda":
        config["gpu_name"] = torch.cuda.get_device_name(0)
    with open(result_dir / "train_config.json", "w") as f:
        json.dump(config, f, indent=2)

    log.info("QLoRA fine-tuning complete.")


if __name__ == "__main__":
    main()
