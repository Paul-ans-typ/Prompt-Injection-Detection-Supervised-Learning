"""
Data pipeline for Prompt Injection Detection project.

Downloads, standardizes, merges, deduplicates, and splits five datasets
into train/val/test and held-out generalization test sets.

Usage:
  python src/prepare_data.py              # download + process (requires internet)
  python src/prepare_data.py --offline    # process from local data/raw/ only (for Sol HPC)

Output (saved to data/processed/):
  train.parquet        - 80% of merged pool (geekyrakshit + neuralchemy + SPML)
  val.parquet          - 10% of merged pool
  test.parquet         - 10% of merged pool
  test_deepset.parquet - Held-out generalization test (deepset benchmark)
  test_wildcard.parquet- Held-out generalization test (real-world verazuo prompts)
  label_stats.csv      - Class distribution summary across all splits
"""

import argparse
import hashlib
import logging
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import requests
from datasets import load_dataset
from sklearn.model_selection import train_test_split
from tqdm import tqdm

# ---------------------------------------------------------------------------
# Setup
# ---------------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = PROJECT_ROOT / "data" / "raw"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"

RANDOM_SEED = 42


# ---------------------------------------------------------------------------
# Individual dataset loaders
# Each returns a DataFrame with exactly three columns:
#   text   (str)  - the prompt text
#   label  (int)  - 0 = benign, 1 = malicious/injection
#   source (str)  - dataset identifier for provenance tracking
# ---------------------------------------------------------------------------

def load_geekyrakshit(offline: bool) -> pd.DataFrame:
    """
    geekyrakshit/prompt-injection-dataset  (~534k samples)
    Aggregated from deepset, xTRam1, jayavibhav datasets.
    Label: 0 = legitimate, 1 = injection
    """
    name = "geekyrakshit"
    cache_path = RAW_DIR / f"{name}.parquet"

    if cache_path.exists():
        log.info(f"[{name}] Loading from cache: {cache_path}")
        return pd.read_parquet(cache_path)

    if offline:
        log.warning(f"[{name}] Offline mode — cache not found, skipping.")
        return pd.DataFrame(columns=["text", "label", "source"])

    log.info(f"[{name}] Downloading from Hugging Face...")
    ds = load_dataset("geekyrakshit/prompt-injection-dataset", split="train")
    df = ds.to_pandas()

    # Normalise column names (HF dataset uses 'prompt' and 'label')
    text_col = next(c for c in df.columns if c.lower() in ("prompt", "text"))
    label_col = next(c for c in df.columns if c.lower() == "label")

    out = pd.DataFrame({
        "text":   df[text_col].astype(str),
        "label":  df[label_col].astype(int),
        "source": name,
    })

    out.to_parquet(cache_path, index=False)
    log.info(f"[{name}] {len(out):,} rows  |  label dist: {out['label'].value_counts().to_dict()}")
    return out


def load_neuralchemy(offline: bool) -> pd.DataFrame:
    """
    neuralchemy/Prompt-injection-dataset  (~22k samples)
    Zero-leakage verified; includes attack categories and severity levels.
    Label: 0 = benign, 1 = malicious
    Config 'full' used for maximum coverage.
    """
    name = "neuralchemy"
    cache_path = RAW_DIR / f"{name}.parquet"

    if cache_path.exists():
        log.info(f"[{name}] Loading from cache: {cache_path}")
        return pd.read_parquet(cache_path)

    if offline:
        log.warning(f"[{name}] Offline mode — cache not found, skipping.")
        return pd.DataFrame(columns=["text", "label", "source"])

    log.info(f"[{name}] Downloading from Hugging Face...")
    # Try 'full' config first, fall back to default
    try:
        ds = load_dataset("neuralchemy/Prompt-injection-dataset", "full", split="train")
    except Exception:
        ds = load_dataset("neuralchemy/Prompt-injection-dataset", split="train")

    df = ds.to_pandas()
    log.info(f"[{name}] Columns available: {list(df.columns)}")

    text_col = next(
        (c for c in df.columns if c.lower() in ("prompt", "text", "input")),
        df.columns[0],
    )
    label_col = next(
        (c for c in df.columns if c.lower() in ("label", "is_injection", "malicious")),
        None,
    )

    if label_col is None:
        raise ValueError(f"[{name}] Cannot find label column. Columns: {list(df.columns)}")

    out = pd.DataFrame({
        "text":   df[text_col].astype(str),
        "label":  df[label_col].astype(int),
        "source": name,
    })

    out.to_parquet(cache_path, index=False)
    log.info(f"[{name}] {len(out):,} rows  |  label dist: {out['label'].value_counts().to_dict()}")
    return out


def load_spml_chatbot(offline: bool) -> pd.DataFrame:
    """
    reshabhs/SPML_Chatbot_Prompt_Injection  (~16k samples)
    Realistic chatbot scenarios with 840 unique system prompts.
    Combines system prompt + user prompt as the full input text.
    Label: 0 = no injection, 1 = injection
    """
    name = "spml_chatbot"
    cache_path = RAW_DIR / f"{name}.parquet"

    if cache_path.exists():
        log.info(f"[{name}] Loading from cache: {cache_path}")
        return pd.read_parquet(cache_path)

    if offline:
        log.warning(f"[{name}] Offline mode — cache not found, skipping.")
        return pd.DataFrame(columns=["text", "label", "source"])

    log.info(f"[{name}] Downloading from Hugging Face...")
    ds = load_dataset("reshabhs/SPML_Chatbot_Prompt_Injection", split="train")
    df = ds.to_pandas()
    log.info(f"[{name}] Columns available: {list(df.columns)}")

    # Concatenate system prompt + user prompt — both matter for injection detection
    sys_col  = next(c for c in df.columns if "system" in c.lower())
    user_col = next(c for c in df.columns if "user" in c.lower())
    label_col = next(c for c in df.columns if "inject" in c.lower() or c.lower() == "label")

    combined_text = (
        "[SYSTEM]: " + df[sys_col].astype(str) +
        " [USER]: "  + df[user_col].astype(str)
    )

    out = pd.DataFrame({
        "text":   combined_text,
        "label":  df[label_col].astype(int),
        "source": name,
    })

    out.to_parquet(cache_path, index=False)
    log.info(f"[{name}] {len(out):,} rows  |  label dist: {out['label'].value_counts().to_dict()}")
    return out


def load_deepset(offline: bool) -> pd.DataFrame:
    """
    deepset/prompt-injections  (~662 samples)
    Widely used benchmark — kept as a held-out generalization test set only.
    Label: 0 = benign, 1 = injection
    """
    name = "deepset"
    cache_path = RAW_DIR / f"{name}.parquet"

    if cache_path.exists():
        log.info(f"[{name}] Loading from cache: {cache_path}")
        return pd.read_parquet(cache_path)

    if offline:
        log.warning(f"[{name}] Offline mode — cache not found, skipping.")
        return pd.DataFrame(columns=["text", "label", "source"])

    log.info(f"[{name}] Downloading from Hugging Face...")
    # Load both splits and combine — we use the full dataset as a test set
    splits = []
    for split in ("train", "test"):
        try:
            ds = load_dataset("deepset/prompt-injections", split=split)
            splits.append(ds.to_pandas())
        except Exception:
            pass

    df = pd.concat(splits, ignore_index=True)
    log.info(f"[{name}] Columns available: {list(df.columns)}")

    text_col  = next(c for c in df.columns if c.lower() in ("text", "prompt", "input"))
    label_col = next(c for c in df.columns if c.lower() == "label")

    out = pd.DataFrame({
        "text":   df[text_col].astype(str),
        "label":  df[label_col].astype(int),
        "source": name,
    })

    out.to_parquet(cache_path, index=False)
    log.info(f"[{name}] {len(out):,} rows  |  label dist: {out['label'].value_counts().to_dict()}")
    return out


def load_verazuo(offline: bool) -> pd.DataFrame:
    """
    verazuo/jailbreak_llms  (~15k samples, ACM CCS'24)
    Real-world prompts scraped from Reddit, Discord, and jailbreak sites.
    Kept as a held-out wildcard generalization test set only.
    Label: 0 = regular prompt, 1 = jailbreak/malicious
    """
    name = "verazuo"
    cache_path = RAW_DIR / f"{name}.parquet"

    if cache_path.exists():
        log.info(f"[{name}] Loading from cache: {cache_path}")
        return pd.read_parquet(cache_path)

    if offline:
        log.warning(f"[{name}] Offline mode — cache not found, skipping.")
        return pd.DataFrame(columns=["text", "label", "source"])

    log.info(f"[{name}] Attempting Hugging Face load...")
    df = None

    # Try HuggingFace first
    try:
        ds = load_dataset("verazuo/jailbreak_llms", split="train")
        df = ds.to_pandas()
    except Exception as e:
        log.warning(f"[{name}] HF load failed ({e}), trying GitHub CSV...")

    # Fall back to raw GitHub CSV
    if df is None:
        csv_url = (
            "https://raw.githubusercontent.com/verazuo/jailbreak_llms"
            "/main/data/jailbreak_prompts.csv"
        )
        try:
            resp = requests.get(csv_url, timeout=60)
            resp.raise_for_status()
            from io import StringIO
            df = pd.read_csv(StringIO(resp.text))
        except Exception as e:
            log.warning(f"[{name}] GitHub CSV download failed ({e}), skipping.")
            return pd.DataFrame(columns=["text", "label", "source"])

    log.info(f"[{name}] Columns available: {list(df.columns)}")

    # Map columns — verazuo CSV has a 'jailbreak' boolean/int column
    text_col = next(
        (c for c in df.columns if c.lower() in ("prompt", "text", "content")),
        df.columns[0],
    )
    label_col = next(
        (c for c in df.columns if c.lower() in ("jailbreak", "label", "is_jailbreak")),
        None,
    )

    if label_col is None:
        log.warning(f"[{name}] Cannot find label column, skipping.")
        return pd.DataFrame(columns=["text", "label", "source"])

    out = pd.DataFrame({
        "text":   df[text_col].astype(str),
        "label":  pd.to_numeric(df[label_col], errors="coerce").fillna(0).astype(int),
        "source": name,
    })

    out.to_parquet(cache_path, index=False)
    log.info(f"[{name}] {len(out):,} rows  |  label dist: {out['label'].value_counts().to_dict()}")
    return out


# ---------------------------------------------------------------------------
# Deduplication
# ---------------------------------------------------------------------------

def deduplicate(df: pd.DataFrame) -> pd.DataFrame:
    """
    Remove duplicate prompts using exact match after normalization.
    Normalization: lowercase + strip whitespace + collapse internal spaces.
    Keeps the first occurrence (preserves source provenance).
    """
    original_len = len(df)
    df = df.dropna(subset=["text"]).reset_index(drop=True)
    normalized = df["text"].astype(str).str.lower().str.strip().str.replace(r"\s+", " ", regex=True)
    hashes = normalized.apply(lambda t: hashlib.md5(t.encode()).hexdigest())
    df = df[~hashes.duplicated(keep="first")].reset_index(drop=True)
    removed = original_len - len(df)
    log.info(f"Deduplication: removed {removed:,} duplicates → {len(df):,} unique rows")
    return df


# ---------------------------------------------------------------------------
# Splitting
# ---------------------------------------------------------------------------

def split_main_pool(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Stratified 80 / 10 / 10 train / val / test split on the main training pool.
    Stratification ensures label balance is preserved in each split.
    """
    train, temp = train_test_split(
        df, test_size=0.20, random_state=RANDOM_SEED, stratify=df["label"]
    )
    val, test = train_test_split(
        temp, test_size=0.50, random_state=RANDOM_SEED, stratify=temp["label"]
    )
    log.info(
        f"Split sizes  —  train: {len(train):,}  val: {len(val):,}  test: {len(test):,}"
    )
    return (
        train.reset_index(drop=True),
        val.reset_index(drop=True),
        test.reset_index(drop=True),
    )


# ---------------------------------------------------------------------------
# Statistics helper
# ---------------------------------------------------------------------------

def label_stats(splits: dict[str, pd.DataFrame]) -> pd.DataFrame:
    rows = []
    for split_name, df in splits.items():
        if df.empty:
            continue
        counts = df["label"].value_counts().sort_index()
        total = len(df)
        rows.append({
            "split":          split_name,
            "total":          total,
            "benign_n":       counts.get(0, 0),
            "malicious_n":    counts.get(1, 0),
            "benign_pct":     round(counts.get(0, 0) / total * 100, 1),
            "malicious_pct":  round(counts.get(1, 0) / total * 100, 1),
        })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(offline: bool) -> None:
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # 1. Load all datasets
    # ------------------------------------------------------------------
    log.info("=" * 60)
    log.info("STEP 1 — Loading datasets")
    log.info("=" * 60)

    # Training pool
    geeky     = load_geekyrakshit(offline)
    neura     = load_neuralchemy(offline)
    spml      = load_spml_chatbot(offline)

    # Held-out generalization test sets (never used in training)
    deepset   = load_deepset(offline)
    verazuo   = load_verazuo(offline)

    # ------------------------------------------------------------------
    # 2. Merge training pool + deduplicate
    # ------------------------------------------------------------------
    log.info("=" * 60)
    log.info("STEP 2 — Merging and deduplicating training pool")
    log.info("=" * 60)

    pool = pd.concat([geeky, neura, spml], ignore_index=True)
    log.info(f"Pool before dedup: {len(pool):,} rows")
    pool = deduplicate(pool)

    # Also remove any text that appears in the held-out test sets
    # to prevent data leakage from held-out sets into training
    if not deepset.empty or not verazuo.empty:
        held_out_texts = set(
            pd.concat([deepset, verazuo], ignore_index=True)["text"]
            .str.lower().str.strip()
            .str.replace(r"\s+", " ", regex=True)
        )
        before = len(pool)
        normalized_pool = pool["text"].str.lower().str.strip().str.replace(r"\s+", " ", regex=True)
        pool = pool[~normalized_pool.isin(held_out_texts)].reset_index(drop=True)
        log.info(f"Removed {before - len(pool):,} rows that appeared in held-out sets")

    # ------------------------------------------------------------------
    # 3. Split training pool
    # ------------------------------------------------------------------
    log.info("=" * 60)
    log.info("STEP 3 — Splitting training pool (80 / 10 / 10)")
    log.info("=" * 60)

    train, val, test = split_main_pool(pool)

    # ------------------------------------------------------------------
    # 4. Save all splits
    # ------------------------------------------------------------------
    log.info("=" * 60)
    log.info("STEP 4 — Saving to data/processed/")
    log.info("=" * 60)

    saves = {
        "train":          train,
        "val":            val,
        "test":           test,
        "test_deepset":   deepset,
        "test_wildcard":  verazuo,
    }

    for fname, df in saves.items():
        if df.empty:
            log.warning(f"  Skipping {fname} — empty DataFrame")
            continue
        out_path = PROCESSED_DIR / f"{fname}.parquet"
        df.to_parquet(out_path, index=False)
        log.info(f"  Saved {fname}.parquet  ({len(df):,} rows)")

    # ------------------------------------------------------------------
    # 5. Print + save label distribution summary
    # ------------------------------------------------------------------
    log.info("=" * 60)
    log.info("STEP 5 — Label distribution summary")
    log.info("=" * 60)

    stats = label_stats(saves)
    print("\n" + stats.to_string(index=False))

    stats_path = PROCESSED_DIR / "label_stats.csv"
    stats.to_csv(stats_path, index=False)
    log.info(f"\nStats saved to {stats_path}")
    log.info("Pipeline complete.")


# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Prompt injection data pipeline")
    parser.add_argument(
        "--offline",
        action="store_true",
        help="Skip downloads; load only from existing data/raw/ cache (use on Sol HPC)",
    )
    args = parser.parse_args()
    main(offline=args.offline)
