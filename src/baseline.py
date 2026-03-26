"""
Baseline model: TF-IDF + Logistic Regression for prompt injection detection.

Two feature modes are compared side-by-side:
  simple   — word unigrams/bigrams only (standard TF-IDF)
  enhanced — word n-grams + character n-grams + handcrafted security features

A grid search over the LR regularisation strength (C) is run on the training
set to find the best hyperparameter, then the model is evaluated on four sets:
  val            - validation split of the training pool
  test           - held-out split of the training pool
  test_deepset   - deepset benchmark (generalization)
  test_wildcard  - real-world scraped prompts (hardest generalization)

Usage:
  python src/baseline.py                        # run both modes (default)
  python src/baseline.py --mode simple          # only simple TF-IDF
  python src/baseline.py --mode enhanced        # only enhanced features
  python src/baseline.py --no-search            # skip grid search, use C=1.0

Outputs saved to results/baseline/:
  metrics_<mode>.json          - full metrics for every eval set
  confusion_<mode>_<split>.png - confusion matrix heatmaps
  roc_<mode>.png               - ROC curves across all splits
  pr_<mode>.png                - Precision-Recall curves
  model_<mode>.joblib          - saved pipeline (vectorizer + classifier)
"""

import argparse
import json
import logging
import warnings
from pathlib import Path
from typing import Optional

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.sparse import hstack, csr_matrix
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
    precision_recall_curve,
    average_precision_score,
)
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MaxAbsScaler
from tqdm import tqdm

warnings.filterwarnings("ignore")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
RESULTS_DIR   = PROJECT_ROOT / "results" / "baseline"
MODELS_DIR    = PROJECT_ROOT / "models"

RANDOM_SEED = 42

# ---------------------------------------------------------------------------
# Security-aware handcrafted features
# ---------------------------------------------------------------------------

# Keywords that appear disproportionately in injection/jailbreak prompts
INJECTION_KEYWORDS = [
    "ignore", "disregard", "forget", "pretend", "act as", "you are now",
    "dan", "jailbreak", "bypass", "override", "system prompt", "new instructions",
    "do anything now", "no restrictions", "without limitations", "roleplay",
    "hypothetically", "simulate", "imagine you", "from now on",
    "developer mode", "unrestricted", "ignore previous", "ignore all",
    "you must", "you will", "you should ignore", "print your instructions",
    "repeat after me", "say the word", "tell me your", "reveal your",
]


def _keyword_hit_rate(text: str) -> float:
    t = text.lower()
    return sum(1 for kw in INJECTION_KEYWORDS if kw in t) / len(INJECTION_KEYWORDS)


def extract_handcrafted(texts: pd.Series) -> np.ndarray:
    """
    Returns an (N, 7) float array of interpretable security features.

    Feature index  Description
    0              Prompt length in characters (log-scaled)
    1              Token count (whitespace split, log-scaled)
    2              Special character ratio  (non-alphanumeric / total chars)
    3              Uppercase ratio
    4              Injection keyword hit rate  (fraction of INJECTION_KEYWORDS present)
    5              Contains role-override pattern  (binary)
    6              Sentence count (rough — split on ". ")
    """
    feats = np.zeros((len(texts), 7), dtype=np.float32)

    for i, text in enumerate(tqdm(texts, desc="  handcrafted features", leave=False)):
        text = str(text)
        n_chars = max(len(text), 1)
        tokens  = text.split()
        n_tok   = max(len(tokens), 1)

        special_chars = sum(1 for c in text if not c.isalnum() and not c.isspace())
        upper_chars   = sum(1 for c in text if c.isupper())
        sentences     = len(text.split(". "))

        role_override = int(
            any(p in text.lower() for p in (
                "you are now", "act as", "pretend to be",
                "from now on you", "ignore your", "disregard your",
            ))
        )

        feats[i] = [
            np.log1p(n_chars),
            np.log1p(n_tok),
            special_chars / n_chars,
            upper_chars   / n_chars,
            _keyword_hit_rate(text),
            role_override,
            np.log1p(sentences),
        ]

    return feats


# ---------------------------------------------------------------------------
# Feature builders
# ---------------------------------------------------------------------------

def build_simple_vectorizer() -> TfidfVectorizer:
    """Word unigrams + bigrams — standard TF-IDF baseline."""
    return TfidfVectorizer(
        analyzer="word",
        ngram_range=(1, 2),
        max_features=50_000,
        sublinear_tf=True,
        min_df=2,
        strip_accents="unicode",
    )


def build_char_vectorizer() -> TfidfVectorizer:
    """Character n-grams — captures obfuscated/misspelled keywords."""
    return TfidfVectorizer(
        analyzer="char_wb",
        ngram_range=(3, 5),
        max_features=40_000,
        sublinear_tf=True,
        min_df=3,
        strip_accents="unicode",
    )


def fit_transform_enhanced(
    train_texts: pd.Series,
    val_texts:   Optional[pd.Series] = None,
) -> tuple:
    """
    Fits word + char vectorizers on train_texts and returns:
      (word_vec, char_vec, X_train)   if val_texts is None
      (word_vec, char_vec, X_train, X_val)   if val_texts supplied
    """
    log.info("  Fitting word TF-IDF...")
    word_vec = build_simple_vectorizer()
    X_word_train = word_vec.fit_transform(train_texts)

    log.info("  Fitting char TF-IDF...")
    char_vec = build_char_vectorizer()
    X_char_train = char_vec.fit_transform(train_texts)

    log.info("  Extracting handcrafted features (train)...")
    X_hand_train = csr_matrix(extract_handcrafted(train_texts))

    X_train = hstack([X_word_train, X_char_train, X_hand_train])

    if val_texts is not None:
        X_word_val = word_vec.transform(val_texts)
        X_char_val = char_vec.transform(val_texts)
        log.info("  Extracting handcrafted features (eval)...")
        X_hand_val = csr_matrix(extract_handcrafted(val_texts))
        X_val = hstack([X_word_val, X_char_val, X_hand_val])
        return word_vec, char_vec, X_train, X_val

    return word_vec, char_vec, X_train


def transform_enhanced(word_vec, char_vec, texts: pd.Series):
    X_word = word_vec.transform(texts)
    X_char = char_vec.transform(texts)
    X_hand = csr_matrix(extract_handcrafted(texts))
    return hstack([X_word, X_char, X_hand])


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def compute_metrics(y_true, y_pred, y_prob, split_name: str) -> dict:
    return {
        "split":     split_name,
        "accuracy":  round(accuracy_score(y_true, y_pred), 4),
        "precision": round(precision_score(y_true, y_pred, zero_division=0), 4),
        "recall":    round(recall_score(y_true, y_pred, zero_division=0), 4),
        "f1":        round(f1_score(y_true, y_pred, zero_division=0), 4),
        "roc_auc":   round(roc_auc_score(y_true, y_prob), 4),
        "avg_precision": round(average_precision_score(y_true, y_prob), 4),
        "report":    classification_report(y_true, y_pred, target_names=["benign", "malicious"]),
    }


# ---------------------------------------------------------------------------
# Plotting helpers
# ---------------------------------------------------------------------------

def plot_confusion(y_true, y_pred, split_name: str, mode: str, out_dir: Path) -> None:
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(5, 4))
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues", ax=ax,
        xticklabels=["benign", "malicious"],
        yticklabels=["benign", "malicious"],
    )
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title(f"Confusion Matrix — {mode} / {split_name}")
    fig.tight_layout()
    fig.savefig(out_dir / f"confusion_{mode}_{split_name}.png", dpi=150)
    plt.close(fig)


def plot_roc(roc_data: list[dict], mode: str, out_dir: Path) -> None:
    """roc_data: list of {split, fpr, tpr, auc}"""
    fig, ax = plt.subplots(figsize=(7, 5))
    for d in roc_data:
        ax.plot(d["fpr"], d["tpr"], label=f"{d['split']}  (AUC={d['auc']:.3f})")
    ax.plot([0, 1], [0, 1], "k--", linewidth=0.8)
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title(f"ROC Curves — {mode}")
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(out_dir / f"roc_{mode}.png", dpi=150)
    plt.close(fig)


def plot_pr(pr_data: list[dict], mode: str, out_dir: Path) -> None:
    """pr_data: list of {split, precision, recall, ap}"""
    fig, ax = plt.subplots(figsize=(7, 5))
    for d in pr_data:
        ax.plot(d["recall"], d["precision"], label=f"{d['split']}  (AP={d['ap']:.3f})")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title(f"Precision-Recall Curves — {mode}")
    ax.legend(loc="upper right")
    fig.tight_layout()
    fig.savefig(out_dir / f"pr_{mode}.png", dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Grid search
# ---------------------------------------------------------------------------

def grid_search_C(X_train, y_train) -> float:
    log.info("  Running 5-fold grid search over C...")
    param_grid = {"C": [0.01, 0.1, 1.0, 5.0, 10.0]}
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_SEED)
    gs = GridSearchCV(
        LogisticRegression(
            solver="lbfgs",
            max_iter=500,
            class_weight="balanced",
            random_state=RANDOM_SEED,
            n_jobs=1,
        ),
        param_grid,
        cv=cv,
        scoring="f1",
        n_jobs=1,
        verbose=1,
    )
    gs.fit(X_train, y_train)
    best_C = gs.best_params_["C"]
    log.info(f"  Best C = {best_C}  (cv F1 = {gs.best_score_:.4f})")
    return best_C


# ---------------------------------------------------------------------------
# Main training + evaluation routine
# ---------------------------------------------------------------------------

def run_mode(mode: str, splits: dict, do_search: bool) -> None:
    log.info("=" * 60)
    log.info(f"MODE: {mode.upper()}")
    log.info("=" * 60)

    train_df = splits["train"]
    y_train  = train_df["label"].values

    # ---- Featurise -------------------------------------------------------
    if mode == "simple":
        log.info("Fitting simple word TF-IDF vectorizer...")
        vec = build_simple_vectorizer()
        X_train = vec.fit_transform(train_df["text"])

        def transform(df):
            return vec.transform(df["text"])

        artifact = {"vec": vec}

    else:  # enhanced
        log.info("Fitting enhanced (word + char + handcrafted) features...")
        word_vec, char_vec, X_train = fit_transform_enhanced(train_df["text"])

        def transform(df):
            return transform_enhanced(word_vec, char_vec, df["text"])

        artifact = {"word_vec": word_vec, "char_vec": char_vec}

    # ---- Scale (helps saga converge faster) ------------------------------
    scaler = MaxAbsScaler()
    X_train_s = scaler.fit_transform(X_train)

    # ---- Hyperparameter search -------------------------------------------
    best_C = grid_search_C(X_train_s, y_train) if do_search else 1.0

    # ---- Train final model -----------------------------------------------
    log.info(f"Training final LogisticRegression (C={best_C})...")
    clf = LogisticRegression(
        C=best_C,
        solver="lbfgs",
        max_iter=1000,
        class_weight="balanced",
        random_state=RANDOM_SEED,
        n_jobs=1,
    )
    clf.fit(X_train_s, y_train)

    # ---- Evaluate on all available splits --------------------------------
    all_metrics = []
    roc_data    = []
    pr_data     = []
    confusions  = []

    eval_splits = {k: v for k, v in splits.items() if k != "train" and not v.empty}

    for split_name, df in eval_splits.items():
        log.info(f"Evaluating on {split_name}...")
        X = scaler.transform(transform(df))
        y_true = df["label"].values
        y_pred = clf.predict(X)
        y_prob = clf.predict_proba(X)[:, 1]

        metrics = compute_metrics(y_true, y_pred, y_prob, split_name)
        all_metrics.append({k: v for k, v in metrics.items() if k != "report"})
        log.info(
            f"  {split_name:20s}  acc={metrics['accuracy']:.4f}  "
            f"f1={metrics['f1']:.4f}  auc={metrics['roc_auc']:.4f}"
        )
        print(metrics["report"])

        # Plots
        plot_confusion(y_true, y_pred, split_name, mode, RESULTS_DIR)

        fpr, tpr, _ = roc_curve(y_true, y_prob)
        roc_data.append({"split": split_name, "fpr": fpr, "tpr": tpr, "auc": metrics["roc_auc"]})

        precision, recall, _ = precision_recall_curve(y_true, y_prob)
        pr_data.append({"split": split_name, "precision": precision, "recall": recall, "ap": metrics["avg_precision"]})

        cm = confusion_matrix(y_true, y_pred)
        confusions.append({"split": split_name, "matrix": cm.tolist(), "labels": ["Benign", "Malicious"]})

    plot_roc(roc_data, mode, RESULTS_DIR)
    plot_pr(pr_data, mode, RESULTS_DIR)

    # ---- Save curve data for interactive charts --------------------------
    curves = {
        "roc": [
            {"split": d["split"], "fpr": d["fpr"].tolist(), "tpr": d["tpr"].tolist(), "auc": d["auc"]}
            for d in roc_data
        ],
        "pr": [
            {"split": d["split"], "precision": d["precision"].tolist(), "recall": d["recall"].tolist(), "ap": d["ap"]}
            for d in pr_data
        ],
        "confusions": confusions,
    }
    with open(RESULTS_DIR / f"curves_{mode}.json", "w") as f:
        json.dump(curves, f)
    log.info(f"Curve data saved to {RESULTS_DIR / f'curves_{mode}.json'}")

    # ---- Save metrics ----------------------------------------------------
    metrics_path = RESULTS_DIR / f"metrics_{mode}.json"
    with open(metrics_path, "w") as f:
        json.dump(all_metrics, f, indent=2)
    log.info(f"Metrics saved to {metrics_path}")

    # Summary table
    summary = pd.DataFrame(all_metrics)
    print("\nSummary:")
    print(summary.to_string(index=False))

    # ---- Save model artifacts --------------------------------------------
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    model_path = MODELS_DIR / f"model_{mode}.joblib"
    joblib.dump({"artifact": artifact, "scaler": scaler, "clf": clf, "best_C": best_C}, model_path)
    log.info(f"Model saved to {model_path}")


# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Baseline TF-IDF + LR model")
    parser.add_argument(
        "--mode",
        choices=["simple", "enhanced", "both"],
        default="both",
        help="Feature mode to run (default: both)",
    )
    parser.add_argument(
        "--no-search",
        action="store_true",
        help="Skip grid search and use C=1.0",
    )
    args = parser.parse_args()

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    # ---- Load processed splits -------------------------------------------
    log.info("Loading processed data splits...")
    split_files = {
        "train":          PROCESSED_DIR / "train.parquet",
        "val":            PROCESSED_DIR / "val.parquet",
        "test":           PROCESSED_DIR / "test.parquet",
        "test_deepset":   PROCESSED_DIR / "test_deepset.parquet",
        "test_wildcard":  PROCESSED_DIR / "test_wildcard.parquet",
    }

    splits = {}
    for name, path in split_files.items():
        if path.exists():
            splits[name] = pd.read_parquet(path)
            log.info(f"  Loaded {name:20s}  ({len(splits[name]):>8,} rows)")
        else:
            log.warning(f"  {name} not found at {path} — skipping")
            splits[name] = pd.DataFrame(columns=["text", "label", "source"])

    if splits.get("train", pd.DataFrame()).empty:
        log.error("Training data not found. Run prepare_data.py first.")
        raise SystemExit(1)

    # ---- Run selected modes ----------------------------------------------
    modes = ["simple", "enhanced"] if args.mode == "both" else [args.mode]
    for mode in modes:
        run_mode(mode, splits, do_search=not args.no_search)

    log.info("Baseline training complete.")


if __name__ == "__main__":
    main()
