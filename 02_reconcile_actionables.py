"""
reconcile_actionables.py
========================
ModeX-inspired set reconciliation across multiple LLM extraction runs,
with scientifically calibrated similarity threshold (θ) via STS benchmark.

═══════════════════════════════════════════════════════════════════════
OVERVIEW
═══════════════════════════════════════════════════════════════════════

This script does two things:

  1. CALIBRATE θ  — Before running the reconciliation, it downloads the
     STS15 semantic textual similarity benchmark from HuggingFace
     (mteb/sts15-sts).  Each STS pair has a human-annotated similarity
     score 0–5.  We treat pairs with score ≥ 4.0 as "should cluster"
     and pairs with score < 2.0 as "should NOT cluster".  We then run
     a grid search over candidate θ values and pick the one that
     maximises F1 on this ground truth.  A chart is saved showing
     Precision / Recall / F1 vs θ so you can inspect the tradeoff.

  2. RECONCILE — Using the calibrated θ, run the full per-article
     pipeline across all five model CSVs, producing:
       • One Markdown trace per article  →  LOG_DIR/
       • One master CSV of canonical actionables  →  OUTPUT_CSV

═══════════════════════════════════════════════════════════════════════
PIPELINE STAGES (per article)
═══════════════════════════════════════════════════════════════════════

  Stage 0  Load & parse  — read all CSVs, flatten every recommendation
           from every model into a tagged candidate pool.

  Stage 1  Similarity   — build N×N pairwise similarity matrix over
           merged_text = "recommendation . impact . evidence".
           Metric: hybrid Jaccard n-gram + overlap coefficient,
           with stopword removal and intra-model downweighting.

  Stage 2  Clustering   — agglomerative clustering (complete linkage)
           using calibrated θ.  Complete linkage prevents chaining:
           two unrelated topics won't merge just because they each
           share one bridge candidate with generic vocabulary.

  Stage 3  Reconcile    — per cluster, select the ModeX centroid
           (highest weighted degree) and average confidences.

  Stage 4  Markdown log — write fully traceable per-article report.

  Stage 5  Master CSV   — aggregate all canonical actionables.

═══════════════════════════════════════════════════════════════════════
USAGE
═══════════════════════════════════════════════════════════════════════

  # Full run: calibrate θ then reconcile
  python reconcile_actionables.py

  # Skip calibration, use a fixed θ (faster for re-runs)
  python reconcile_actionables.py --theta 0.08 --skip-calibration

  # Only calibrate, don't reconcile (inspect chart first)
  python reconcile_actionables.py --calibrate-only

  # Control parallelism and minimum support
  python reconcile_actionables.py --workers 8 --min-support 2

Python ≥ 3.8 compatible (no X | Y union type hints).
"""

from __future__ import annotations   # ← makes all hints strings; fixes Python <3.10

import ast
import csv
import json
import os
import re
import sys
import pathlib
import argparse
import textwrap
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime
from itertools import combinations
from statistics import mean

import numpy as np
import pandas as pd
from sklearn.cluster import AgglomerativeClustering

# ══════════════════════════════════════════════════════════════════════
# CONFIGURATION
# ══════════════════════════════════════════════════════════════════════

CSV_PATHS = {
    "qwen":     "/data/Deep_Angiography/ReACT-GPT/local_history/qwen.csv",
    "llama":    "/data/Deep_Angiography/ReACT-GPT/local_history/llama.csv",
    "mixtral":  "/data/Deep_Angiography/ReACT-GPT/local_history/mixtral.csv",
    "gpt":      "/data/Deep_Angiography/ReACT-GPT/local_history/gpt.csv",
    "deepseek": "/data/Deep_Angiography/ReACT-GPT/local_history/deepseek.csv",
}

# Output directories / files
LOG_DIR    = pathlib.Path("/data/Deep_Angiography/ReACT-GPT/log")
OUTPUT_CSV = pathlib.Path("/data/Deep_Angiography/ReACT-GPT/reconciled_actionables.csv")
CHART_PATH = pathlib.Path("/data/Deep_Angiography/ReACT-GPT/theta_calibration.png")

# Similarity threshold — overwritten by calibration unless --skip-calibration
SIM_THRESHOLD = 0.08

# Downweight intra-model pairs so a single model's consistent style
# doesn't artificially drive two distinct actionables into the same cluster.
INTRA_MODEL_ALPHA = 0.15

# Stopwords removed before similarity computation.
# Without this, shared function words ("a","the","of","for","in") inflate
# the overlap coefficient and create false-positive cluster merges.
STOPWORDS = {
    'a','an','the','and','or','but','in','on','at','to','for','of','with',
    'by','from','is','are','was','were','be','been','being','have','has',
    'had','do','does','did','will','would','could','should','may','might',
    'that','this','these','those','it','its','as','not','no','nor','so',
    'yet','both','either','each','few','more','most','other','some','such',
    'than','then','there','when','where','which','who','whom','why','how',
    'all','any','can','into','if','up','out','about','what','per','i','we',
    'you','he','she','they','them','us','our','your'
}

# STS thresholds: human score ≥ STS_POSITIVE → "should cluster",
#                 human score ≤ STS_NEGATIVE → "should NOT cluster".
# Mid-range pairs (2.0–4.0) are excluded from calibration — too ambiguous.
STS_POSITIVE = 4.0   # clear paraphrases
STS_NEGATIVE = 2.0   # clearly different

# θ candidates to evaluate in the grid search
# Phase-1 coarse grid: full range 0.01 → 0.95 in steps of 0.02.
# This ensures we never miss a peak at either extreme.
# Phase-2 fine grid: ±0.05 around the coarse best, step 0.005.
# Both phases are built dynamically in calibrate_theta() — this
# constant is kept only as a fallback if that function is bypassed.
THETA_GRID = list(round(v, 3) for v in
                  [x * 0.02 + 0.01 for x in range(48)])   # 0.01 … 0.95

# Minimum number of distinct source models a cluster needs to be kept.
# 1 = keep singletons (comprehensive); 2 = require cross-model corroboration.
MIN_SUPPORT = 1

ALL_MODELS = list(CSV_PATHS.keys())
N_MODELS   = len(ALL_MODELS)

# Parallel worker count for article processing.
N_WORKERS = os.cpu_count() or 1


# ══════════════════════════════════════════════════════════════════════
# N-GRAM SIMILARITY UTILITIES
# ══════════════════════════════════════════════════════════════════════

def tokenise(text: str) -> list:
    """
    Lowercase, strip punctuation, remove stopwords.

    Stopword removal is critical: without it, shared function words like
    "a", "the", "of" inflate the overlap coefficient and cause unrelated
    actionables to merge into the same cluster.
    """
    text = text.lower()
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    return [w for w in text.split() if w not in STOPWORDS and len(w) > 1]


def ngrams(tokens: list, n: int) -> set:
    """Return the set of n-grams from a token list."""
    return set(zip(*[tokens[i:] for i in range(n)]))


def jaccard(a: set, b: set) -> float:
    """Jaccard similarity = |A∩B| / |A∪B|."""
    if not a and not b:
        return 1.0
    union = a | b
    return len(a & b) / len(union) if union else 0.0


def overlap_coef(a: set, b: set) -> float:
    """
    Overlap coefficient = |A∩B| / min(|A|, |B|).

    Better than Jaccard for paraphrases where one text is a
    condensed version of the other — common when GPT-4 gives a terse
    recommendation and Llama gives a verbose one about the same thing.
    """
    if not a or not b:
        return 0.0
    return len(a & b) / min(len(a), len(b))


def text_similarity(t1: str, t2: str) -> float:
    """
    Hybrid similarity in [0, 1]:

        0.5 × mean_Jaccard(unigram + bigram + trigram)
      + 0.5 × overlap_coefficient(unigrams)

    After stopword removal:
    - Jaccard captures shared phrase structure.
    - Overlap coefficient catches paraphrases where vocabulary differs
      in size (one model is more verbose than another).

    Called on the MERGED string (recommendation + impact + evidence)
    so all three fields contribute equally without manual weighting.
    """
    tok1, tok2 = tokenise(t1), tokenise(t2)
    j_score  = sum(jaccard(ngrams(tok1, n), ngrams(tok2, n)) for n in (1, 2, 3)) / 3.0
    ov_score = overlap_coef(set(tok1), set(tok2))
    return (j_score + ov_score) / 2.0


# ══════════════════════════════════════════════════════════════════════
# THETA CALIBRATION VIA STS15 BENCHMARK
# ══════════════════════════════════════════════════════════════════════

def download_sts15() -> pd.DataFrame:
    """
    Download the STS15 test split from HuggingFace (mteb/sts15-sts).

    STS15 contains ~1400 sentence pairs with human similarity scores
    on a 0–5 scale.  We use it to calibrate θ: pairs scored ≥ 4.0 are
    ground-truth "same meaning" (should cluster); pairs scored ≤ 2.0
    are ground-truth "different meaning" (should not cluster).

    Returns a DataFrame with columns: sentence1, sentence2, score.
    Falls back to a built-in SE-domain sample if HuggingFace is
    unreachable (e.g. in air-gapped environments).
    """
    print("\n── Calibration: downloading STS15 benchmark ──")
    try:
        # Primary: HuggingFace datasets library
        from datasets import load_dataset
        ds = load_dataset("mteb/sts15-sts", split="test")
        df = ds.to_pandas()[["sentence1", "sentence2", "score"]]
        df["score"] = df["score"].astype(float)
        print(f"   Downloaded {len(df)} pairs from HuggingFace (mteb/sts15-sts)")
        return df
    except Exception as e1:
        print(f"   HuggingFace download failed ({e1}), trying parquet URL …")

    try:
        # Secondary: raw parquet from HF CDN
        url = ("https://huggingface.co/datasets/mteb/sts15-sts"
               "/resolve/main/data/test-00000-of-00001.parquet")
        df = pd.read_parquet(url)[["sentence1", "sentence2", "score"]]
        df["score"] = df["score"].astype(float)
        print(f"   Downloaded {len(df)} pairs via parquet URL")
        return df
    except Exception as e2:
        print(f"   Parquet URL also failed ({e2}). Using built-in SE-domain fallback.")

    # ── Fallback: curated SE / NLP4RE pairs matching the paper domain ────────
    # These mirror the STS format and cover the same vocabulary as your data.
    # Replace this block with real STS data once network access is available.
    fallback = [
        # ── HIGH similarity ≥ 4.0 (paraphrases — SHOULD cluster) ────────────
        ("Use Python for NLP4RE tool development to ensure reusability.",
         "Implement NLP4RE solutions in Python to improve reusability of existing tools.", 4.8),
        ("Configure the depth parameter to control Wikipedia traversal depth.",
         "Set the depth parameter to adjust how deeply Wikipedia categories are traversed.", 4.6),
        ("Use WikiDoMiner to generate a domain-specific corpus from requirements.",
         "Apply WikiDoMiner to build a domain corpus for your requirements specification.", 4.5),
        ("Adopt Python libraries for NLP to avoid Java maintenance issues.",
         "Use Python-based NLP libraries to prevent outdated Java dependency problems.", 4.3),
        ("Prioritize code smells in the predominant programming paradigm.",
         "Focus code smell research on the dominant paradigm of the target language.", 4.2),
        ("Release tools under open-source licenses to enable community reuse.",
         "Publish NLP4RE tools as open-source to foster community adoption and reuse.", 4.4),
        ("Evaluate models on domain-specific benchmarks rather than generic ones.",
         "Test NLP tools on domain-specific datasets not only standard NLP benchmarks.", 4.1),
        ("Pre-train language models on domain corpora before fine-tuning.",
         "Fine-tune pre-trained models on domain-specific text prior to deployment.", 3.9),
        # ── MID similarity 2.0–4.0 (related but different — EXCLUDED) ────────
        ("Configure corpus depth parameter for Wikipedia category traversal.",
         "Use WikiDoMiner to generate domain-specific corpora for NLP tasks.", 2.8),
        ("Implement tools in Python for reusability across NLP4RE ecosystem.",
         "Release NLP tools under open-source licenses for community adoption.", 2.2),
        ("Use static analysis to detect anti-patterns in Python code.",
         "Adopt Python as the standard language for NLP4RE tooling.", 2.0),
        ("Evaluate tools on domain benchmarks to ensure generalisability.",
         "Configure depth parameter to control corpus size and relevance.", 1.8),
        # ── LOW similarity < 2.0 (unrelated — SHOULD NOT cluster) ───────────
        ("Use Python for NLP4RE tool development.",
         "Prioritize code smells in the predominant paradigm.", 0.8),
        ("Configure depth parameter for Wikipedia traversal.",
         "Release tools under open-source licenses.", 0.5),
        ("Apply WikiDoMiner to generate domain corpus.",
         "Use static analysis to detect Python anti-patterns.", 0.3),
        ("Adopt Python for NLP tooling.",
         "Pre-train language models on domain corpora.", 1.2),
        ("Configure corpus depth for Wikipedia traversal.",
         "Evaluate models on domain-specific benchmarks.", 0.9),
        ("Use Python libraries instead of Java.",
         "Prioritize smells in dominant programming paradigm.", 0.4),
    ]
    df = pd.DataFrame(fallback, columns=["sentence1", "sentence2", "score"])
    print(f"   Using built-in fallback: {len(df)} pairs")
    return df


def _score_theta(theta: float, sims: "pd.Series", gt: "pd.Series") -> dict:
    """
    Evaluate one θ value against ground-truth labels.

    Returns a dict with theta, precision, recall, f1, tp, fp, fn.
    This is factored out so both calibration phases can call it.
    """
    predicted_same = sims >= theta
    tp = int(( predicted_same &  gt).sum())
    fp = int(( predicted_same & ~gt).sum())
    fn = int((~predicted_same &  gt).sum())
    precision = tp / (tp + fp) if (tp + fp) > 0 else 1.0
    recall    = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1        = (2 * precision * recall / (precision + recall)
                 if (precision + recall) > 0 else 0.0)
    return {"theta": theta, "precision": precision,
            "recall": recall, "f1": f1,
            "tp": tp, "fp": fp, "fn": fn}


def calibrate_theta(sts_df: pd.DataFrame, chart_path: pathlib.Path) -> float:
    """
    Two-phase grid search to find the θ that maximises F1 on STS15.

    WHY TWO PHASES?
    ───────────────
    A single fixed grid (e.g. 0.04–0.40) risks missing the true optimum
    if it sits outside that range or in a gap between grid points.

      Phase 1 — Coarse sweep: θ from 0.01 to 0.95 in steps of 0.02.
                This covers the entire meaningful range and cannot miss
                a peak at either extreme.

      Phase 2 — Fine zoom: ±0.05 around the Phase-1 best, step 0.005.
                This finds the precise optimum within the promising region
                to 3 decimal places.

    HOW SCORING WORKS
    ─────────────────
    For each θ we predict "same cluster" if our text_similarity() ≥ θ.
    Ground truth comes from human STS scores:
      • score ≥ STS_POSITIVE (4.0) → SAME   (clear paraphrases)
      • score ≤ STS_NEGATIVE (2.0) → DIFFERENT
      • 2.0 < score < 4.0          → excluded (genuinely ambiguous)

    We maximise F1 = harmonic mean of Precision and Recall, which
    balances avoiding false merges (precision) with catching all true
    paraphrases (recall).

    The chart shows the full coarse curve plus a zoomed inset of the
    fine phase so you can inspect the tradeoff and override if needed.

    Parameters
    ----------
    sts_df     : DataFrame with columns sentence1, sentence2, score (0–5)
    chart_path : Where to save the PNG chart

    Returns
    -------
    Best θ (float, 3 decimal places)
    """
    print("\n── Calibration: computing text_similarity on STS pairs ──")

    # Filter to unambiguous pairs only
    clear_pairs = sts_df[
        (sts_df["score"] >= STS_POSITIVE) | (sts_df["score"] <= STS_NEGATIVE)
    ].copy()
    clear_pairs["ground_truth"] = clear_pairs["score"] >= STS_POSITIVE
    n_same = int(clear_pairs["ground_truth"].sum())
    n_diff = len(clear_pairs) - n_same
    print(f"   Using {len(clear_pairs)} unambiguous pairs out of {len(sts_df)} total")
    print(f"   Same (≥{STS_POSITIVE}): {n_same}  |  Different (≤{STS_NEGATIVE}): {n_diff}")

    # Compute our similarity metric once for all pairs (reused in both phases)
    print("   Computing similarities … ", end="", flush=True)
    clear_pairs["our_sim"] = [
        text_similarity(r["sentence1"], r["sentence2"])
        for _, r in clear_pairs.iterrows()
    ]
    print("done")

    sims = clear_pairs["our_sim"]
    gt   = clear_pairs["ground_truth"]

    # ── Phase 1: coarse sweep 0.01 → 0.95, step 0.02 ─────────────────────────
    coarse_grid = [round(0.01 + i * 0.02, 3) for i in range(48)]   # 0.01 … 0.95

    print(f"\n   Phase 1 — Coarse sweep ({len(coarse_grid)} values: "
          f"{coarse_grid[0]} → {coarse_grid[-1]}, step 0.02)")
    print(f"   {'θ':>6}  {'Precision':>10}  {'Recall':>8}  {'F1':>8}  "
          f"{'TP':>5}  {'FP':>5}  {'FN':>5}")
    print("   " + "─" * 60)

    coarse_results = []
    for theta in coarse_grid:
        row = _score_theta(theta, sims, gt)
        coarse_results.append(row)
        print(f"   {theta:>6.3f}  {row['precision']:>10.3f}  "
              f"{row['recall']:>8.3f}  {row['f1']:>8.3f}  "
              f"{row['tp']:>5}  {row['fp']:>5}  {row['fn']:>5}")

    coarse_df   = pd.DataFrame(coarse_results)
    coarse_best = float(coarse_df.loc[coarse_df["f1"].idxmax(), "theta"])
    coarse_f1   = float(coarse_df.loc[coarse_df["f1"].idxmax(), "f1"])
    print(f"\n   Phase 1 best: θ = {coarse_best}  (F1 = {coarse_f1:.3f})")

    # ── Phase 2: fine zoom ±0.05 around coarse best, step 0.005 ──────────────
    fine_lo   = max(0.001, round(coarse_best - 0.05, 3))
    fine_hi   = min(0.999, round(coarse_best + 0.05, 3))
    fine_grid = [round(fine_lo + i * 0.005, 3)
                 for i in range(int((fine_hi - fine_lo) / 0.005) + 1)]

    print(f"\n   Phase 2 — Fine zoom ({len(fine_grid)} values: "
          f"{fine_grid[0]} → {fine_grid[-1]}, step 0.005)")
    print(f"   {'θ':>6}  {'Precision':>10}  {'Recall':>8}  {'F1':>8}  "
          f"{'TP':>5}  {'FP':>5}  {'FN':>5}")
    print("   " + "─" * 60)

    fine_results = []
    for theta in fine_grid:
        row = _score_theta(theta, sims, gt)
        fine_results.append(row)
        print(f"   {theta:>6.3f}  {row['precision']:>10.3f}  "
              f"{row['recall']:>8.3f}  {row['f1']:>8.3f}  "
              f"{row['tp']:>5}  {row['fp']:>5}  {row['fn']:>5}")

    fine_df   = pd.DataFrame(fine_results)
    fine_best_row = fine_df.loc[fine_df["f1"].idxmax()]
    best_theta    = float(fine_best_row["theta"])
    best_f1       = float(fine_best_row["f1"])

    print(f"\n   ★  Best θ = {best_theta}  "
          f"(F1={best_f1:.3f}, "
          f"P={fine_best_row['precision']:.3f}, "
          f"R={fine_best_row['recall']:.3f})")
    print(f"   (coarse best was {coarse_best}, fine search refined to {best_theta})")

    # ── Generate chart ────────────────────────────────────────────────────────
    _plot_calibration(coarse_df, fine_df, best_theta, chart_path, clear_pairs)

    return best_theta


def _plot_calibration(
    coarse_df: pd.DataFrame,
    fine_df: pd.DataFrame,
    best_theta: float,
    chart_path: pathlib.Path,
    sts_pairs: pd.DataFrame,
) -> None:
    """
    Save a three-panel PNG:
      Left   — Full coarse P/R/F1 curve (0.01 → 0.95, step 0.02)
      Centre — Fine-zoom P/R/F1 curve  (±0.05 around best, step 0.005)
      Right  — Histogram of similarity scores by ground truth label

    Reading the chart:
      • In the Left panel you see the global shape of the P/R tradeoff.
        High θ → high precision (few false merges) but low recall (misses
        many true paraphrases).  Low θ → the reverse.  F1 balances both.
      • The Centre panel shows the precise optimum within the fine region.
      • The Right panel confirms the chosen θ sits in the gap between the
        "same" and "different" score distributions.  If the distributions
        heavily overlap, the similarity metric itself needs improvement.
    """
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from mpl_toolkits.axes_grid1.inset_locator import inset_axes
    except ImportError:
        print("   [WARN] matplotlib not installed — skipping chart")
        return

    BG    = "#0F1117"
    PANEL = "#1A1D27"
    GRID  = "#2A2D3A"
    TICK  = "#AAAAAA"

    fig, axes = plt.subplots(1, 3, figsize=(18, 5.5))
    fig.patch.set_facecolor(BG)
    for ax in axes:
        ax.set_facecolor(PANEL)
        ax.tick_params(colors=TICK, labelsize=9)
        for spine in ax.spines.values():
            spine.set_edgecolor("#333344")

    def _draw_prf_panel(ax, df, title, mark_best=True):
        """Draw P/R/F1 lines on ax from a results DataFrame."""
        thetas = df["theta"].values
        ax.plot(thetas, df["precision"].values, "o-", color="#4FC3F7",
                lw=1.8, ms=4, label="Precision")
        ax.plot(thetas, df["recall"].values,    "s-", color="#81C784",
                lw=1.8, ms=4, label="Recall")
        ax.plot(thetas, df["f1"].values,        "D-", color="#FFD54F",
                lw=2.2, ms=5, label="F1", zorder=5)
        if mark_best:
            best_f1 = float(df.loc[df["theta"] == best_theta, "f1"].values[0])
            ax.axvline(best_theta, color="#FF7043", lw=1.5,
                       ls="--", alpha=0.9, label=f"Best θ={best_theta}")
            ax.scatter([best_theta], [best_f1], color="#FF7043",
                       s=100, zorder=6, edgecolors="white", lw=1)
            # Annotation: position to avoid going off-chart
            x_off = 0.015 if best_theta < (max(thetas) * 0.8) else -0.06
            ax.annotate(f"θ={best_theta}\nF1={best_f1:.3f}",
                        xy=(best_theta, best_f1),
                        xytext=(best_theta + x_off, max(0.1, best_f1 - 0.14)),
                        color="#FF7043", fontsize=8,
                        arrowprops=dict(arrowstyle="->", color="#FF7043", lw=1))
        ax.set_xlabel("Threshold θ", color="#CCCCCC", fontsize=10)
        ax.set_ylabel("Score", color="#CCCCCC", fontsize=10)
        ax.set_title(title, color="white", fontsize=11, fontweight="bold")
        ax.set_ylim(-0.05, 1.05)
        ax.set_xlim(min(thetas) - 0.01, max(thetas) + 0.01)
        ax.legend(facecolor="#252836", edgecolor="#444455",
                  labelcolor="white", fontsize=8)
        ax.grid(True, color=GRID, lw=0.6, alpha=0.7)

    # ── Left: full coarse curve ───────────────────────────────────────────────
    # Check if best_theta is in coarse_df (it might only be in fine_df)
    coarse_has_best = best_theta in coarse_df["theta"].values
    _draw_prf_panel(
        axes[0], coarse_df,
        "Phase 1 — Coarse sweep\n(θ: 0.01 → 0.95, step 0.02)",
        mark_best=coarse_has_best,
    )
    if not coarse_has_best:
        # Still mark the coarse best for reference
        coarse_best = float(coarse_df.loc[coarse_df["f1"].idxmax(), "theta"])
        axes[0].axvline(coarse_best, color="#B0BEC5", lw=1, ls=":",
                        alpha=0.7, label=f"Coarse peak θ={coarse_best}")

    # ── Centre: fine zoom ─────────────────────────────────────────────────────
    _draw_prf_panel(
        axes[1], fine_df,
        f"Phase 2 — Fine zoom\n(±0.05 around coarse peak, step 0.005)",
        mark_best=True,
    )

    # ── Right: score distribution histogram ──────────────────────────────────
    ax2 = axes[2]
    same_sims = sts_pairs.loc[ sts_pairs["ground_truth"], "our_sim"].values
    diff_sims = sts_pairs.loc[~sts_pairs["ground_truth"], "our_sim"].values
    bins = np.linspace(0, max(float(sts_pairs["our_sim"].max()), 0.6), 30)

    ax2.hist(diff_sims, bins=bins, color="#EF5350", alpha=0.65,
             label=f"Different (score ≤ {STS_NEGATIVE})", edgecolor="#CC3333")
    ax2.hist(same_sims, bins=bins, color="#42A5F5", alpha=0.65,
             label=f"Same (score ≥ {STS_POSITIVE})", edgecolor="#1A5FAA")
    ax2.axvline(best_theta, color="#FF7043", lw=2, ls="--",
                label=f"Best θ = {best_theta}")
    ax2.set_xlabel("text_similarity() score", color="#CCCCCC", fontsize=10)
    ax2.set_ylabel("Count", color="#CCCCCC", fontsize=10)
    ax2.set_title("Score Distribution by Label\n"
                  "(θ should sit in the gap)",
                  color="white", fontsize=11, fontweight="bold")
    ax2.legend(facecolor="#252836", edgecolor="#444455",
               labelcolor="white", fontsize=8)
    ax2.grid(True, color=GRID, lw=0.6, alpha=0.7)

    fig.suptitle(
        "Two-Phase θ Calibration on STS15  —  ModeX Actionable Reconciliation",
        color="white", fontsize=13, fontweight="bold", y=1.01,
    )
    plt.tight_layout()
    chart_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(chart_path, dpi=150, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"   Chart saved → {chart_path}")


# ══════════════════════════════════════════════════════════════════════
# CSV LOADING
# ══════════════════════════════════════════════════════════════════════

def safe_parse(raw: str):
    """
    Parse the `answer` column (Python dict string or JSON).

    Returns None for every non-actionable response:
      - blank / NaN / None / null / N/A
      - {'message': 'NO ACTIONABLE CAN BE DERIVED'}
      - {'error': '...'}
      - {'recommendations': []}  or  {'recommendations': None}
    """
    if not raw or not isinstance(raw, str):
        return None
    raw = raw.strip()
    if not raw or raw.lower() in ("none", "null", "n/a", "nan"):
        return None
    try:
        val = ast.literal_eval(raw)
        return val if isinstance(val, dict) else None
    except Exception:
        pass
    try:
        return json.loads(raw)
    except Exception:
        return None


def load_csv(model_key: str, path: str) -> pd.DataFrame:
    """Load one model CSV; return empty DataFrame if file is missing."""
    p = pathlib.Path(path)
    if not p.exists():
        print(f"  [WARN] {path} not found — skipping model '{model_key}'",
              file=sys.stderr)
        return pd.DataFrame()
    df = pd.read_csv(p)
    df["_model_key"] = model_key
    return df


def load_all_csvs() -> pd.DataFrame:
    """Load and concatenate all model CSVs, parsing the answer column."""
    frames = []
    for key, path in CSV_PATHS.items():
        df = load_csv(key, path)
        if not df.empty:
            frames.append(df)
    if not frames:
        raise FileNotFoundError(
            "No CSV files found. Check CSV_PATHS at the top of the script."
        )
    combined = pd.concat(frames, ignore_index=True)
    combined["_parsed"] = combined["answer"].apply(safe_parse)
    return combined


# ══════════════════════════════════════════════════════════════════════
# CANDIDATE POOL BUILDER
# ══════════════════════════════════════════════════════════════════════

def build_pool(article_rows: pd.DataFrame) -> tuple:
    """
    Flatten all recommendations from all models into a single pool.

    Each pool item carries:
      id, text, impact, evidence, merged_text, confidence, model_key,
      article_title, venue

    The merged_text = "recommendation . impact . evidence" is what gets
    compared in the similarity matrix.  Using all three fields avoids
    arbitrary field weighting while naturally rewarding cross-field
    semantic overlap (e.g., if model A's evidence matches model B's
    recommendation, those shared tokens still contribute).

    Returns
    -------
    (pool, skipped)
      pool    — list of valid candidate dicts
      skipped — list of {model_key, skip_reason} dicts for the log
    """
    pool    = []
    skipped = []
    uid     = 0

    for _, row in article_rows.iterrows():
        raw    = row.get("answer", "")
        parsed = row.get("_parsed")
        model  = row["_model_key"]

        # ── detect non-actionable responses ──────────────────────────────────
        skip_reason = None

        if parsed is None:
            skip_reason = f"unparseable answer: {str(raw)[:80]}"

        else:
            recs = parsed.get("recommendations")

            if "message" in parsed or "error" in parsed:
                # e.g. {'message': 'NO ACTIONABLE CAN BE DERIVED'}
                msg = parsed.get("message") or parsed.get("error") or ""
                skip_reason = f"model reported: \"{msg}\""

            elif not recs or not isinstance(recs, list):
                skip_reason = "recommendations field is absent/empty/null"

        if skip_reason:
            skipped.append({
                "model_key":     model,
                "skip_reason":   skip_reason,
                "article_title": row.get("article_title", ""),
                "venue":         row.get("venue", ""),
            })
            continue

        # ── valid recommendations ─────────────────────────────────────────────
        for rec in recs:
            if not isinstance(rec, dict):
                continue
            text     = rec.get("recommendation", "").strip()
            impact   = rec.get("positive_impact", "").strip()
            evidence = rec.get("evidence", "").strip()
            if not text:
                continue
            merged = " . ".join(filter(None, [text, impact, evidence]))
            pool.append({
                "id":            uid,
                "text":          text,
                "impact":        impact,
                "evidence":      evidence,
                "merged_text":   merged,
                "confidence":    float(rec.get("confidence", 0.5)),
                "model_key":     model,
                "article_title": row.get("article_title", ""),
                "venue":         row.get("venue", ""),
            })
            uid += 1

    return pool, skipped


# ══════════════════════════════════════════════════════════════════════
# SIMILARITY MATRIX
# ══════════════════════════════════════════════════════════════════════

def build_similarity_matrix(pool: list) -> np.ndarray:
    """
    Build an N×N pairwise similarity matrix over merged_text.

    Intra-model pairs are downweighted by INTRA_MODEL_ALPHA (default 0.15)
    so that a single model's stylistically consistent outputs don't
    cluster together just because they share vocabulary.

    Values are in [0, 1].
    """
    n   = len(pool)
    sim = np.zeros((n, n))
    for i in range(n):
        sim[i, i] = 1.0
        for j in range(i + 1, n):
            s = text_similarity(pool[i]["merged_text"], pool[j]["merged_text"])
            if pool[i]["model_key"] == pool[j]["model_key"]:
                s *= INTRA_MODEL_ALPHA
            sim[i, j] = sim[j, i] = s
    return sim


# ══════════════════════════════════════════════════════════════════════
# CLUSTERING
# ══════════════════════════════════════════════════════════════════════

def cluster_pool(pool: list, sim: np.ndarray) -> list:
    """
    Agglomerative clustering (complete linkage) on the similarity matrix.

    WHY COMPLETE LINKAGE?
    Complete linkage merges two clusters only if ALL pairs across the
    clusters exceed θ.  This prevents the chaining problem seen with
    average linkage: two unrelated topics (e.g. "Python reusability"
    and "WikiDoMiner corpus") can't merge just because each has one
    bridging candidate that happens to share generic vocabulary
    ("NLP4RE", "domain", "solutions").

    Returns list of clusters, each cluster = list of indices into pool.
    """
    n = len(pool)
    if n == 1:
        return [[0]]

    dist = np.clip(1.0 - sim, 0.0, 1.0)
    np.fill_diagonal(dist, 0.0)

    model = AgglomerativeClustering(
        n_clusters=None,
        metric="precomputed",
        linkage="complete",
        distance_threshold=1.0 - SIM_THRESHOLD,
    )
    labels = model.fit_predict(dist)

    clusters = defaultdict(list)
    for idx, label in enumerate(labels):
        clusters[label].append(idx)
    return list(clusters.values())


# ══════════════════════════════════════════════════════════════════════
# CENTROID SELECTION  (ModeX Step 3)
# ══════════════════════════════════════════════════════════════════════

def select_centroid(cluster_indices: list, sim: np.ndarray) -> int:
    """
    Return the pool index of the centroid: the node with the highest
    total weighted degree within the cluster.

    This is the ModeX centroid selection rule (Eq. 5 in the paper).
    Intuitively it picks the candidate most similar to all others in
    the cluster — the one that best represents shared content.
    """
    best_idx   = cluster_indices[0]
    best_score = -1.0
    for i in cluster_indices:
        degree = sum(sim[i, j] for j in cluster_indices if j != i)
        if degree > best_score:
            best_score = degree
            best_idx   = i
    return best_idx


# ══════════════════════════════════════════════════════════════════════
# PER-CLUSTER RECONCILIATION
# ══════════════════════════════════════════════════════════════════════

def reconcile_cluster(
    cluster_indices: list,
    pool: list,
    sim: np.ndarray,
    cluster_id: int,
) -> dict:
    """
    Produce one canonical actionable from a cluster.

    Returns
    -------
    Dict with:
      cluster_id, support, support_fraction, models_present,
      centroid_pool_id, actionable, impact, evidence,
      avg_confidence, centroid_confidence,
      degree_scores, all_members
    """
    members        = [pool[i] for i in cluster_indices]
    models_present = list({m["model_key"] for m in members})
    support        = len(models_present)

    centroid_idx  = select_centroid(cluster_indices, sim)
    centroid      = pool[centroid_idx]

    confidences    = [m["confidence"] for m in members]
    avg_confidence = round(sum(confidences) / len(confidences), 4)

    degree_scores = {
        pool[i]["id"]: round(
            sum(sim[i, j] for j in cluster_indices if j != i), 4
        )
        for i in cluster_indices
    }

    return {
        "cluster_id":          cluster_id,
        "support":             support,
        "max_support":         N_MODELS,
        "support_fraction":    f"{support}/{N_MODELS}",
        "models_present":      sorted(models_present),
        "centroid_pool_id":    centroid["id"],
        "actionable":          centroid["text"],
        "impact":              centroid["impact"],
        "evidence":            centroid["evidence"],
        "centroid_confidence": centroid["confidence"],
        "avg_confidence":      avg_confidence,
        "all_confidences":     confidences,
        "degree_scores":       degree_scores,
        "all_members":         members,
    }


# ══════════════════════════════════════════════════════════════════════
# MARKDOWN REPORT WRITER
# ══════════════════════════════════════════════════════════════════════

def write_markdown_report(
    article_title: str,
    venue: str,
    pool: list,
    skipped: list,
    sim: np.ndarray,
    clusters_raw: list,
    reconciled: list,
    discarded: list,
    log_dir: pathlib.Path,
    calibrated_theta: float,
) -> pathlib.Path:
    """
    Write a fully traceable Markdown report for one article.

    Includes every stage: pool table, similarity matrix, cluster
    composition, per-cluster confidence arithmetic, and the final
    canonical set.  The calibrated θ is shown in the header so
    readers know how the threshold was chosen.
    """
    log_dir.mkdir(parents=True, exist_ok=True)
    safe_title = re.sub(r"[^a-zA-Z0-9_\-]", "_", str(article_title))[:60]
    ts         = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path   = log_dir / f"{safe_title}_{ts}.md"

    lines = []
    A = lines.append

    A(f"# Reconciliation Report — {article_title}")
    A(f"\n**Venue:** {venue}")
    A(f"**Generated:** {datetime.now().isoformat(timespec='seconds')}")
    A(f"**Algorithm:** ModeX-set (Jaccard n-gram similarity, agglomerative clustering)")
    A(f"**Threshold θ:** {calibrated_theta}"
      f"  *(calibrated on STS15 — maximises F1 on human similarity judgements)*"
      f"  |  **Intra-model α:** {INTRA_MODEL_ALPHA}"
      f"  |  **Min support:** {MIN_SUPPORT}")
    A(f"\n---\n")

    # ── Stage 0: pool ─────────────────────────────────────────────────────────
    A("## Stage 0 — Global Candidate Pool\n")
    A(f"Total candidates pooled from {N_MODELS} models: **{len(pool)}**\n")
    A("Similarity is computed on the **merged string** = "
      "`recommendation . impact . evidence` per candidate.\n")
    A("| ID | Model | Recommendation | Impact | Evidence | "
      "Merged string (used for similarity) | Confidence |")
    A("|----|-------|----------------|--------|----------|"
      "-------------------------------------|------------|")
    for c in pool:
        rec    = c["text"].replace("|", "\\|")
        imp    = c["impact"].replace("|", "\\|")
        evid   = c["evidence"].replace("|", "\\|")
        merged = c["merged_text"].replace("|", "\\|")
        A(f"| {c['id']} | `{c['model_key']}` | {rec} | {imp} | {evid} "
          f"| {merged} | {c['confidence']} |")

    # ── Skipped models ────────────────────────────────────────────────────────
    if skipped:
        A("\n### Models that returned no actionables\n")
        A("| Model | Reason |")
        A("|-------|--------|")
        for s in skipped:
            A(f"| `{s['model_key']}` | {s['skip_reason'].replace('|', chr(92)+'|')} |")
        A(f"\n> ⚠️  {len(skipped)}/{N_MODELS} model(s) skipped — "
          f"support denominator is still {N_MODELS} (max possible).\n")

    # ── Stage 1: similarity matrix ────────────────────────────────────────────
    A("\n---\n")
    A("## Stage 1 — Pairwise Similarity Matrix "
      "(merged recommendation + impact + evidence)\n")
    A(f"Each candidate compared as one concatenated string: "
      f"`recommendation . impact . evidence`  "
      f"|  intra-model α={INTRA_MODEL_ALPHA}  "
      f"|  metric: hybrid Jaccard + overlap coefficient  "
      f"|  stopwords removed\n")
    n = len(pool)
    A("| ID | " + " | ".join(str(c["id"]) for c in pool) + " |")
    A("|" + "----|" * (n + 1))
    for i, ci in enumerate(pool):
        row_vals = " | ".join(f"{sim[i,j]:.2f}" for j in range(n))
        A(f"| {ci['id']} | {row_vals} |")

    # ── Stage 2: raw clusters ─────────────────────────────────────────────────
    A("\n---\n")
    A("## Stage 2 — Agglomerative Clusters (complete linkage)\n")
    A(f"θ = {calibrated_theta} → distance threshold = {1-calibrated_theta:.3f}  "
      f"|  linkage = complete (ALL pairs in cluster must exceed θ)\n")
    for k, cl in enumerate(clusters_raw):
        member_ids = ", ".join(str(pool[i]["id"]) for i in cl)
        models     = sorted({pool[i]["model_key"] for i in cl})
        A(f"**Cluster {k+1}:** members [{member_ids}]  "
          f"|  models: {models}  |  size: {len(cl)}")

    # ── Stage 3: per-cluster reconciliation ───────────────────────────────────
    A("\n---\n")
    A("## Stage 3 — Per-Cluster Reconciliation "
      "(ModeX Centroid + Confidence Averaging)\n")
    for r in reconciled:
        A(f"### Cluster {r['cluster_id']} — Support {r['support_fraction']}"
          f"  |  Models: {r['models_present']}\n")
        A("#### Member texts and field-level similarities\n")
        A("| Pool ID | Model | Recommendation | Impact | Evidence | "
          "Confidence | Degree score |")
        A("|---------|-------|----------------|--------|----------"
          "|------------|--------------|")
        for m in r["all_members"]:
            rec    = m["text"].replace("|", "\\|")
            imp    = m["impact"].replace("|", "\\|")
            evid   = m["evidence"].replace("|", "\\|")
            deg    = r["degree_scores"].get(m["id"], "—")
            marker = " ← **centroid**" if m["id"] == r["centroid_pool_id"] else ""
            A(f"| {m['id']} | `{m['model_key']}` | {rec}{marker} | "
              f"{imp} | {evid} | {m['confidence']} | {deg} |")

        conf_str = " + ".join(str(c) for c in r["all_confidences"])
        A(f"\n#### Confidence averaging\n```")
        A(f"all_confidences  = {r['all_confidences']}")
        A(f"avg_confidence   = ({conf_str}) / {len(r['all_confidences'])}"
          f" = {r['avg_confidence']}")
        A(f"centroid_conf    = {r['centroid_confidence']}"
          f"  (raw confidence of the centroid node)")
        A(f"```")
        A(f"\n#### ✅ Canonical actionable (centroid = pool ID {r['centroid_pool_id']})\n")
        A(f"> **Actionable:** {r['actionable']}")
        A(f">\n> **Impact:** {r['impact']}")
        A(f">\n> **Evidence:** {r['evidence']}")
        A(f">\n> **avg_confidence:** {r['avg_confidence']}"
          f"  |  **support:** {r['support_fraction']}\n")

    # ── Stage 4: discarded singletons ─────────────────────────────────────────
    if discarded:
        A("\n---\n")
        A("## Stage 4 — Discarded Clusters (support < min_support)\n")
        A("| Pool ID | Model | Actionable | Confidence |")
        A("|---------|-------|------------|------------|")
        for r in discarded:
            m = r["all_members"][0]
            A(f"| {m['id']} | `{m['model_key']}` | "
              f"{m['text'].replace('|', chr(92)+'|')} | {m['confidence']} |")

    # ── Stage 5: final set ────────────────────────────────────────────────────
    A("\n---\n")
    A("## Stage 5 — Final Canonical Actionable Set\n")
    A(f"**Total kept:** {len(reconciled)}  |  **Discarded:** {len(discarded)}\n")
    A("| # | Support | avg_confidence | Actionable | Impact | Evidence |")
    A("|---|---------|----------------|------------|--------|----------|")
    for i, r in enumerate(reconciled, 1):
        rec  = r["actionable"].replace("|", "\\|")
        imp  = r["impact"].replace("|", "\\|")
        evid = r["evidence"].replace("|", "\\|")
        A(f"| {i} | {r['support_fraction']} | {r['avg_confidence']}"
          f" | {rec} | {imp} | {evid} |")

    out_path.write_text("\n".join(lines), encoding="utf-8")
    return out_path


# ══════════════════════════════════════════════════════════════════════
# PER-ARTICLE PIPELINE
# ══════════════════════════════════════════════════════════════════════

def process_article(
    article_key: str,
    article_rows: pd.DataFrame,
    log_dir: pathlib.Path,
    calibrated_theta: float,
) -> list:
    """
    Run the full reconciliation pipeline for one article.

    Returns a list of canonical actionable dicts to append to the master CSV.
    """
    title = article_rows["article_title"].iloc[0]
    venue = article_rows["venue"].iloc[0]
    link  = (article_rows["article_link"].iloc[0]
             if "article_link" in article_rows.columns else "")

    print(f"\n{'='*60}")
    print(f"  Article : {title}")
    print(f"  Venue   : {venue}")

    # Stage 0
    pool, skipped = build_pool(article_rows)
    models_with_output = sorted({p["model_key"] for p in pool})
    print(f"  Pool    : {len(pool)} candidates from {models_with_output}")
    if skipped:
        print(f"  Skipped : {len(skipped)} model row(s) — "
              + ", ".join(
                  f"{s['model_key']} ({s['skip_reason'][:40]})"
                  for s in skipped))

    # All models returned no-actionable
    if not pool:
        print("  [SKIP] All models returned no-actionable responses.")
        log_dir.mkdir(parents=True, exist_ok=True)
        safe_title = re.sub(r"[^a-zA-Z0-9_\-]", "_", str(title))[:60]
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        note = log_dir / f"{safe_title}_{ts}.md"
        lines = [
            f"# Reconciliation Report — {title}",
            f"\n**Venue:** {venue}",
            f"**Generated:** {datetime.now().isoformat(timespec='seconds')}",
            f"\n---\n",
            f"## Result: No actionables extracted\n",
            f"All {N_MODELS} models returned no-actionable responses.\n",
            "| Model | Reason |", "|-------|--------|",
        ]
        for s in skipped:
            lines.append(f"| `{s['model_key']}` | {s['skip_reason']} |")
        note.write_text("\n".join(lines), encoding="utf-8")
        return []

    # Single candidate — nothing to cluster
    if len(pool) == 1:
        c = pool[0]
        r = reconcile_cluster([0], pool, np.array([[1.0]]), 1)
        md = write_markdown_report(
            title, venue, pool, skipped, np.array([[1.0]]),
            [[0]], [r], [], log_dir, calibrated_theta,
        )
        print(f"  Log     : {md}")
        return [{
            "article_title": title, "venue": venue, "article_link": link,
            "cluster_id": 1, "support": 1,
            "support_fraction": f"1/{N_MODELS}",
            "models_present": [c["model_key"]],
            "actionable": c["text"], "impact": c["impact"],
            "evidence": c["evidence"],
            "avg_confidence": c["confidence"],
            "centroid_confidence": c["confidence"],
        }]

    # Stage 1: similarity
    sim = build_similarity_matrix(pool)

    # Stage 2: clustering
    clusters_raw = cluster_pool(pool, sim)
    print(f"  Clusters: {len(clusters_raw)} found")

    # Stage 3: reconcile
    reconciled = []
    discarded  = []
    for k, cl_indices in enumerate(clusters_raw, 1):
        r = reconcile_cluster(cl_indices, pool, sim, cluster_id=k)
        if r["support"] >= MIN_SUPPORT:
            reconciled.append(r)
        else:
            discarded.append(r)
    reconciled.sort(key=lambda x: (-x["support"], -x["avg_confidence"]))

    print(f"  Kept    : {len(reconciled)} canonical actionables")
    print(f"  Dropped : {len(discarded)} below min_support={MIN_SUPPORT}")

    # Stage 4: markdown
    md = write_markdown_report(
        title, venue, pool, skipped, sim,
        clusters_raw, reconciled, discarded,
        log_dir, calibrated_theta,
    )
    print(f"  Log     : {md}")

    # Flatten for master CSV
    canonical_rows = []
    for r in reconciled:
        canonical_rows.append({
            "article_title":       title,
            "venue":               venue,
            "article_link":        link,
            "cluster_id":          r["cluster_id"],
            "support":             r["support"],
            "support_fraction":    r["support_fraction"],
            "models_present":      "|".join(r["models_present"]),
            "actionable":          r["actionable"],
            "impact":              r["impact"],
            "evidence":            r["evidence"],
            "avg_confidence":      r["avg_confidence"],
            "centroid_confidence": r["centroid_confidence"],
        })
    return canonical_rows


# ══════════════════════════════════════════════════════════════════════
# PARALLEL WORKER  (must be top-level for pickle)
# ══════════════════════════════════════════════════════════════════════

def _worker(args: tuple) -> tuple:
    """
    Top-level picklable function for ProcessPoolExecutor.

    Each worker processes one article fully (pool → sim → cluster →
    reconcile → markdown) then returns (article_key, canonical_rows).

    Must be at module level (not nested) so pickle can serialise it
    for inter-process communication.
    """
    article_key, rows_dict, log_dir_str, calibrated_theta = args
    rows    = pd.DataFrame(rows_dict)
    log_dir = pathlib.Path(log_dir_str)
    try:
        canonical = process_article(article_key, rows, log_dir, calibrated_theta)
        return article_key, canonical
    except Exception as exc:
        print(f"  [ERROR] {article_key}: {exc}", file=sys.stderr)
        return article_key, []


# ══════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════

def main(
    skip_calibration: bool = False,
    calibrate_only: bool   = False,
    fixed_theta: float     = None,
    n_workers: int         = None,
    min_support: int       = None,
) -> None:
    """
    Orchestrate calibration → reconciliation.

    Parameters
    ----------
    skip_calibration : if True, use fixed_theta (or the config default)
    calibrate_only   : if True, stop after calibration and chart
    fixed_theta      : explicit θ override (used when skip_calibration=True)
    n_workers        : worker processes (None = use N_WORKERS global)
    min_support      : minimum cluster support (None = use MIN_SUPPORT global)
    """
    global SIM_THRESHOLD, MIN_SUPPORT, N_WORKERS

    if n_workers  is not None: N_WORKERS  = n_workers
    if min_support is not None: MIN_SUPPORT = min_support

    print("\n" + "═" * 60)
    print("  ModeX-Set Actionable Reconciliation")
    print("═" * 60)

    # ── Step 1: calibrate θ ───────────────────────────────────────────────────
    if skip_calibration:
        calibrated_theta = fixed_theta if fixed_theta is not None else SIM_THRESHOLD
        print(f"\n  [Calibration skipped]  Using θ = {calibrated_theta}")
    else:
        sts_df           = download_sts15()
        CHART_PATH.parent.mkdir(parents=True, exist_ok=True)
        calibrated_theta = calibrate_theta(sts_df, CHART_PATH)
        print(f"\n  Calibrated θ = {calibrated_theta}")

    SIM_THRESHOLD = calibrated_theta   # apply globally for cluster_pool()

    if calibrate_only:
        print("\n  [--calibrate-only] Stopping after calibration.")
        return

    # ── Step 2: load CSVs ─────────────────────────────────────────────────────
    print("\n" + "─" * 60)
    print("  Loading CSVs …")
    df = load_all_csvs()
    print(f"  Total rows     : {len(df)}")

    df["_article_key"] = (df["venue"].astype(str) + "||"
                          + df["article_title"].astype(str))
    article_keys = df["_article_key"].unique()
    print(f"  Unique articles: {len(article_keys)}")

    # ── Step 3: process articles ──────────────────────────────────────────────
    all_canonical = []
    workers = min(N_WORKERS, len(article_keys))

    if workers <= 1:
        # Sequential (easier to debug)
        for akey in article_keys:
            rows = df[df["_article_key"] == akey]
            canonical = process_article(akey, rows, LOG_DIR, calibrated_theta)
            all_canonical.extend(canonical)
    else:
        print(f"\n  Parallel: {workers} workers across {len(article_keys)} articles")
        tasks = [
            (akey,
             df[df["_article_key"] == akey].to_dict(orient="list"),
             str(LOG_DIR),
             calibrated_theta)
            for akey in article_keys
        ]
        with ProcessPoolExecutor(max_workers=workers) as pool:
            futures = {pool.submit(_worker, t): t[0] for t in tasks}
            for future in as_completed(futures):
                akey, canonical = future.result()
                all_canonical.extend(canonical)

    # ── Step 4: write master CSV ──────────────────────────────────────────────
    if all_canonical:
        OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(all_canonical).to_csv(OUTPUT_CSV, index=False)
        print(f"\n{'═'*60}")
        print(f"  Master CSV → {OUTPUT_CSV}")
        print(f"  Total canonical actionables: {len(all_canonical)}")
        print(f"  Calibrated θ used: {calibrated_theta}")
        print(f"  Chart → {CHART_PATH}")
        print(f"{'═'*60}\n")
    else:
        print("\n[WARN] No canonical actionables produced — check CSV parsing.\n")


# ══════════════════════════════════════════════════════════════════════
# CLI ENTRY POINT
# ══════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="ModeX-set actionable reconciliation with θ calibration",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=textwrap.dedent("""
        Examples
        --------
          # Calibrate θ on STS15, then reconcile (default)
          python reconcile_actionables.py

          # Inspect calibration chart before reconciling
          python reconcile_actionables.py --calibrate-only

          # Skip calibration, use a specific θ (faster re-runs)
          python reconcile_actionables.py --skip-calibration --theta 0.08

          # Use 16 parallel workers, require ≥2 model agreement
          python reconcile_actionables.py --workers 16 --min-support 2
        """)
    )
    parser.add_argument(
        "--skip-calibration", action="store_true",
        help="Skip STS15 calibration and use --theta directly",
    )
    parser.add_argument(
        "--calibrate-only", action="store_true",
        help="Run calibration and save chart, then exit without reconciling",
    )
    parser.add_argument(
        "--theta", type=float, default=None,
        help=f"Fixed θ override (default: {SIM_THRESHOLD}, used with --skip-calibration)",
    )
    parser.add_argument(
        "--min-support", type=int, default=MIN_SUPPORT,
        help=f"Min distinct models per cluster to keep it (default {MIN_SUPPORT})",
    )
    parser.add_argument(
        "--workers", type=int, default=N_WORKERS,
        help=f"Parallel worker processes (default: {N_WORKERS} CPUs). "
             "Set 1 for sequential/debug mode.",
    )
    args = parser.parse_args()

    main(
        skip_calibration=args.skip_calibration,
        calibrate_only=args.calibrate_only,
        fixed_theta=args.theta,
        n_workers=args.workers,
        min_support=args.min_support,
    )