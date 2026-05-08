"""
reconcile_actionables.py
========================
ModeX-inspired set reconciliation across multiple LLM extraction runs,
with scientifically calibrated similarity threshold (θ) via STS benchmark.

═══════════════════════════════════════════════════════════════════════
OVERVIEW
═══════════════════════════════════════════════════════════════════════

This script does two things:

  1. CALIBRATE θ  — Downloads TWO benchmarks (STS15 + SICK-R) and runs
     a two-phase grid search to find the θ that maximises AVERAGE F1
     across both.  Using two benchmarks prevents overfitting to one
     dataset's sentence distribution.  Similarity is an ensemble of
     lexical (Jaccard + overlap) and semantic (sentence-transformers
     cosine) scores, weighted 40/60 by default.  A four-panel chart
     is saved showing per-benchmark and average F1 curves plus score
     distribution histograms.

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
from tqdm import tqdm
import multiprocessing

# Force 'spawn' start method so child processes don't inherit
# the parent's CUDA context.  'fork' (Linux default) causes
# 'Cannot re-initialize CUDA in forked subprocess' errors when
# sentence-transformers has loaded a GPU model before forking.
# 'spawn' starts a clean Python interpreter each time — safe.
# Must be called before any ProcessPoolExecutor is created.
if multiprocessing.get_start_method(allow_none=True) != 'spawn':
    try:
        multiprocessing.set_start_method('spawn', force=True)
    except RuntimeError:
        pass   # already set (e.g. on macOS where spawn is default)
from datetime import datetime
from itertools import combinations
from statistics import mean

import numpy as np
import pandas as pd
from sklearn.cluster import AgglomerativeClustering

# Lazy-loaded at first use so the script starts fast even when the model
# is not yet cached.  Set to None to force pure-lexical mode.
_SENTENCE_MODEL = None   # populated by _get_sentence_model()
SENTENCE_MODEL_NAME = "all-MiniLM-L6-v2"   # 80 MB, fast, strong STS performance

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
LOG_DIR             = pathlib.Path("/data/Deep_Angiography/ReACT-GPT/log")
OUTPUT_CSV          = pathlib.Path("/data/Deep_Angiography/ReACT-GPT/reconciled_actionables.csv")
CHART_PATH          = pathlib.Path("/data/Deep_Angiography/ReACT-GPT/theta_calibration.png")

# Benchmark datasets are saved here as CSV after the first download.
# On subsequent runs the cached file is loaded instantly — no network needed.
# Delete a file from this directory to force a fresh download.
BENCHMARK_CACHE_DIR = pathlib.Path("/data/Deep_Angiography/ReACT-GPT/benchmark")

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

# θ candidates — built dynamically in calibrate_theta(), kept as fallback.
THETA_GRID = list(round(v, 3) for v in [x * 0.02 + 0.01 for x in range(48)])

# ── Dual-benchmark calibration ────────────────────────────────────────────────
# We calibrate θ on two independent benchmarks and take the θ that maximises
# the AVERAGE F1 across both.  This guards against overfitting to one dataset.
#
#   Benchmark 1 — STS15 (mteb/sts15-sts)
#     General-purpose sentence pairs, annotated 0–5.
#     Good coverage of paraphrase diversity.
#
#   Benchmark 2 — SICK-R (sick)
#     Sentence Involving Compositional Knowledge — Relatedness split.
#     Human scores 1–5 (different scale; we normalise to 0–5).
#     Covers more compositional and negation cases than STS15.
#
# Using two benchmarks is more robust than one because:
#   • They have different sentence types and domains.
#   • A θ that is optimal for both generalises better to unseen data.
#   • If the two optimal θ values differ significantly, it signals that
#     the similarity metric itself needs improvement.
STS_BENCHMARK_CONFIGS = [
    {
        "name":        "STS15",
        "hf_dataset":  "mteb/sts15-sts",
        "hf_split":    "test",
        "col_s1":      "sentence1",
        "col_s2":      "sentence2",
        "col_score":   "score",
        "score_min":   0.0,
        "score_max":   5.0,   # native scale; used as-is
        "positive_threshold": 4.0,   # ≥ this → SAME
        "negative_threshold": 2.0,   # ≤ this → DIFFERENT
    },
    {
        "name":        "SICK-R",
        "hf_dataset":  "sick",
        "hf_split":    "test",
        "col_s1":      "sentence_A",
        "col_s2":      "sentence_B",
        "col_score":   "relatedness_score",
        "score_min":   1.0,
        "score_max":   5.0,   # normalised to 0–5 before thresholding
        "positive_threshold": 4.0,   # ≥ 4/5 → SAME (tight paraphrase)
        "negative_threshold": 2.5,   # ≤ 2.5/5 → DIFFERENT
    },
]

# ── Similarity ensemble weights ───────────────────────────────────────────────
# Final similarity = W_LEXICAL × lexical_score + W_SEMANTIC × semantic_score
# Semantic score: cosine similarity of sentence-transformers embeddings.
# Set W_SEMANTIC = 0.0 to fall back to pure lexical (original behaviour).
W_LEXICAL  = 0.40
W_SEMANTIC = 0.60

# Minimum number of distinct source models a cluster needs to be kept.
# 1 = keep singletons (comprehensive); 2 = require cross-model corroboration.
MIN_SUPPORT = 1

ALL_MODELS = list(CSV_PATHS.keys())
N_MODELS   = len(ALL_MODELS)

# Parallel worker count for article processing.
N_WORKERS = os.cpu_count() or 1


# ══════════════════════════════════════════════════════════════════════
# SENTENCE EMBEDDING UTILITY
# ══════════════════════════════════════════════════════════════════════

def _get_sentence_model():
    """
    Lazily load the sentence-transformers model on first call.

    The model is cached in the module-level _SENTENCE_MODEL variable so
    it is loaded only once per process, even across multiple articles.

    If sentence-transformers is not installed or the model cannot be
    downloaded (e.g. air-gapped environment), this returns None and
    text_similarity() falls back to pure lexical scoring.

    Model choice: all-MiniLM-L6-v2
      • 80 MB — fast to download and encode
      • Trained specifically for semantic textual similarity tasks
      • Strong performance on STS benchmarks (Spearman ρ ~0.88)
      • Drop-in replacement requires no fine-tuning
    """
    global _SENTENCE_MODEL
    if _SENTENCE_MODEL is not None:
        return _SENTENCE_MODEL
    if W_SEMANTIC == 0.0:
        return None   # pure-lexical mode, skip loading
    try:
        from sentence_transformers import SentenceTransformer
        print(f"   Loading sentence model '{SENTENCE_MODEL_NAME}' … ",
              end="", flush=True)
        _SENTENCE_MODEL = SentenceTransformer(SENTENCE_MODEL_NAME)
        print("loaded")
        return _SENTENCE_MODEL
    except Exception as e:
        print(f"\n   [WARN] Could not load sentence model ({e}). "
              "Falling back to pure lexical similarity.")
        _SENTENCE_MODEL = None
        return None


def _semantic_similarity(t1: str, t2: str, model) -> float:
    """
    Cosine similarity of sentence-transformer embeddings.

    Captures paraphrase relationships that lexical overlap misses —
    e.g. 'enhance reusability' vs 'improve maintainability' scores
    near-zero with Jaccard but ~0.75 with embeddings.

    Returns float in [−1, 1] clipped to [0, 1].
    """
    import numpy as np
    e1 = model.encode(t1, normalize_embeddings=True, show_progress_bar=False)
    e2 = model.encode(t2, normalize_embeddings=True, show_progress_bar=False)
    return float(max(0.0, float(np.dot(e1, e2))))


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


def _lexical_similarity(t1: str, t2: str) -> float:
    """
    Pure lexical similarity (original metric).

    Hybrid of:
      • Mean Jaccard over unigram + bigram + trigram sets
      • Overlap coefficient on unigrams

    After stopword removal, captures surface-level paraphrase via
    shared vocabulary.  Fast — no model inference required.
    """
    tok1, tok2 = tokenise(t1), tokenise(t2)
    j_score  = sum(jaccard(ngrams(tok1, n), ngrams(tok2, n)) for n in (1, 2, 3)) / 3.0
    ov_score = overlap_coef(set(tok1), set(tok2))
    return (j_score + ov_score) / 2.0


def text_similarity(t1: str, t2: str) -> float:
    """
    Ensemble similarity in [0, 1]:

        W_LEXICAL  × lexical_score   (Jaccard n-gram + overlap coef)
      + W_SEMANTIC × semantic_score  (sentence-transformers cosine)

    WHY ENSEMBLE?
    ─────────────
    Lexical similarity (Jaccard + overlap) works well when two texts
    share surface vocabulary — common for short, structured actionables
    from the same domain.  But it fails for semantic paraphrases where
    vocabulary differs: "enhance reusability" vs "improve maintainability"
    scores near zero with Jaccard despite meaning similar things.

    Semantic similarity via sentence-transformers captures meaning
    regardless of surface form, but can over-cluster tangentially related
    sentences that happen to share a topic.

    Ensembling (default 40% lexical + 60% semantic) gets the best of
    both: robust to surface variation while remaining discriminative
    about genuine semantic differences.

    FALLBACK
    ────────
    If sentence-transformers is not installed or W_SEMANTIC == 0.0,
    this function is identical to the original _lexical_similarity().
    """
    lex = _lexical_similarity(t1, t2)
    if W_SEMANTIC == 0.0:
        return lex
    model = _get_sentence_model()
    if model is None:
        return lex   # graceful fallback if model unavailable
    sem = _semantic_similarity(t1, t2, model)
    return W_LEXICAL * lex + W_SEMANTIC * sem


# ══════════════════════════════════════════════════════════════════════
# THETA CALIBRATION VIA STS15 BENCHMARK
# ══════════════════════════════════════════════════════════════════════

# ── SE-domain fallback pairs (used when HF is unreachable) ──────────────────
_FALLBACK_PAIRS = [
    # HIGH ≥ 4.0 — paraphrases, SHOULD cluster
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
    # MID 2.0–4.0 — excluded from calibration (ambiguous)
    ("Configure corpus depth parameter for Wikipedia category traversal.",
     "Use WikiDoMiner to generate domain-specific corpora for NLP tasks.", 2.8),
    ("Implement tools in Python for reusability across NLP4RE ecosystem.",
     "Release NLP tools under open-source licenses for community adoption.", 2.2),
    # LOW ≤ 2.0 — different, SHOULD NOT cluster
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


def download_benchmark(cfg: dict) -> pd.DataFrame:
    """
    Load one benchmark dataset, using a local cache when available.

    CACHING BEHAVIOUR
    ─────────────────
    On first run the dataset is fetched from HuggingFace and saved as
    a CSV to BENCHMARK_CACHE_DIR/<name>.csv.  Every subsequent run
    loads the cached file instantly — no network request is made.

    To force a fresh download (e.g. after a dataset update):
        rm /data/Deep_Angiography/ReACT-GPT/benchmark/<name>.csv

    Config keys
    ───────────
    name               : human label; also used as the cache filename
    hf_dataset         : HuggingFace dataset identifier
    hf_split           : split to use ("test", "validation", …)
    col_s1, col_s2     : column names for the two sentences
    col_score          : column name for the similarity score
    score_min/max      : native score range (used for normalisation)
    positive_threshold : normalised score ≥ this → SAME
    negative_threshold : normalised score ≤ this → DIFFERENT

    Scores are normalised to [0, 5] so both STS15 (0–5) and
    SICK-R (1–5) use the same thresholding scale.

    Falls back to the built-in SE-domain pairs on network failure.
    """
    name       = cfg["name"]
    cache_path = BENCHMARK_CACHE_DIR / f"{name}.csv"

    def _normalise(df: pd.DataFrame) -> pd.DataFrame:
        lo, hi = cfg["score_min"], cfg["score_max"]
        df = df.copy()
        df["score"] = ((df["score"] - lo) / (hi - lo) * 5.0).clip(0, 5)
        return df

    # ── Check cache first ─────────────────────────────────────────────────────
    if cache_path.exists():
        df = pd.read_csv(cache_path)
        print(f"\n── Benchmark: {name} — loaded from cache ({cache_path})")
        print(f"   {len(df)} pairs (delete file to force re-download)")
        return df

    # ── Cache miss: download ──────────────────────────────────────────────────
    print(f"\n── Benchmark: {name} — downloading ({cfg['hf_dataset']}) ──")
    df = None

    try:
        from datasets import load_dataset
        ds = load_dataset(cfg["hf_dataset"], split=cfg["hf_split"])
        df = ds.to_pandas()[[cfg["col_s1"], cfg["col_s2"], cfg["col_score"]]]
        df.columns = ["sentence1", "sentence2", "score"]
        df["score"] = df["score"].astype(float)
        df = _normalise(df)
        print(f"   Downloaded {len(df)} pairs from HuggingFace")
    except Exception as e1:
        print(f"   HuggingFace failed ({type(e1).__name__}), trying parquet …")

    if df is None:
        parquet_urls = {
            "mteb/sts15-sts": ("https://huggingface.co/datasets/mteb/sts15-sts"
                               "/resolve/main/data/test-00000-of-00001.parquet"),
            "sick":           ("https://huggingface.co/datasets/sick"
                               "/resolve/main/data/test-00000-of-00001.parquet"),
        }
        if cfg["hf_dataset"] in parquet_urls:
            try:
                df = pd.read_parquet(parquet_urls[cfg["hf_dataset"]])
                df = df[[cfg["col_s1"], cfg["col_s2"], cfg["col_score"]]]
                df.columns = ["sentence1", "sentence2", "score"]
                df["score"] = df["score"].astype(float)
                df = _normalise(df)
                print(f"   Downloaded {len(df)} pairs via parquet")
            except Exception as e2:
                print(f"   Parquet also failed ({type(e2).__name__}).")

    if df is None:
        print(f"   Using built-in SE-domain fallback for {name}.")
        df = pd.DataFrame(_FALLBACK_PAIRS, columns=["sentence1", "sentence2", "score"])

    # ── Save to cache ─────────────────────────────────────────────────────────
    try:
        BENCHMARK_CACHE_DIR.mkdir(parents=True, exist_ok=True)
        df.to_csv(cache_path, index=False)
        print(f"   Cached → {cache_path}")
    except Exception as e:
        print(f"   [WARN] Could not write cache ({e}) — continuing without caching.")

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


def calibrate_theta(chart_path: pathlib.Path) -> float:
    """
    Dual-benchmark θ calibration.

    STRATEGY
    ────────
    1. Download both STS15 and SICK-R from HuggingFace.
    2. Compute text_similarity() on every pair in both benchmarks.
    3. Run a two-phase grid search (coarse 0.01→0.95, fine ±0.05)
       scoring each θ on EACH benchmark independently.
    4. For each θ compute: avg_f1 = (f1_sts15 + f1_sick) / 2
    5. Select the θ that maximises avg_f1.
    6. Save a chart showing all three F1 curves + the chosen θ.

    WHY TWO BENCHMARKS?
    ───────────────────
    A θ calibrated on only one dataset may overfit to that dataset's
    sentence distribution.  STS15 covers general paraphrases; SICK-R
    adds compositional and negation cases.  A θ that works well on
    both is more likely to generalise to your SE/NLP actionable domain.

    Returns
    -------
    Best θ (float, 3 decimal places)
    """
    print("\n── Dual-Benchmark θ Calibration ──")
    print(f"   Similarity: {W_LEXICAL:.0%} lexical + {W_SEMANTIC:.0%} semantic ensemble")

    # ── Step 1: download both benchmarks ─────────────────────────────────────
    benchmark_data = []
    for cfg in STS_BENCHMARK_CONFIGS:
        df = download_benchmark(cfg)
        # Filter to unambiguous pairs
        pos_t = cfg["positive_threshold"]
        neg_t = cfg["negative_threshold"]
        clear = df[(df["score"] >= pos_t) | (df["score"] <= neg_t)].copy()
        clear["ground_truth"] = clear["score"] >= pos_t
        n_same = int(clear["ground_truth"].sum())
        n_diff = len(clear) - n_same
        print(f"   {cfg['name']}: {len(clear)} unambiguous pairs "
              f"(same={n_same}, diff={n_diff}) out of {len(df)}")
        benchmark_data.append((cfg, clear))

    # ── Step 2: compute similarities once per benchmark ───────────────────────
    # Pre-computing avoids redundant model inference during the grid search.
    print("\n   Pre-computing similarities …")
    for cfg, df in benchmark_data:
        sims = [text_similarity(r["sentence1"], r["sentence2"])
                for _, r in df.iterrows()]
        df["our_sim"] = sims
        print(f"   {cfg['name']}: done ({len(sims)} pairs)")

    sims_list = [df["our_sim"]       for _, df in benchmark_data]
    gt_list   = [df["ground_truth"]  for _, df in benchmark_data]
    names     = [cfg["name"]         for cfg, _ in benchmark_data]

    # ── Step 3 & 4: two-phase grid search with avg F1 ────────────────────────
    def _score_all(theta):
        """Score θ on every benchmark and return per-benchmark + avg F1."""
        rows = [_score_theta(theta, sims, gt)
                for sims, gt in zip(sims_list, gt_list)]
        avg_f1 = sum(r["f1"] for r in rows) / len(rows)
        return rows, avg_f1

    def _run_phase(grid, phase_name):
        print(f"\n   {phase_name} ({len(grid)} values: {grid[0]} → {grid[-1]})")
        # Header
        hdr = f"   {'θ':>6}  {'avg_F1':>8}"
        for n in names:
            hdr += f"  {n+' F1':>12}  {n+' P':>8}  {n+' R':>8}"
        print(hdr)
        print("   " + "─" * (6 + 10 + len(names) * 32))

        results = []
        for theta in grid:
            rows, avg_f1 = _score_all(theta)
            entry = {"theta": theta, "avg_f1": avg_f1}
            line = f"   {theta:>6.3f}  {avg_f1:>8.3f}"
            for i, (r, n) in enumerate(zip(rows, names)):
                entry[f"f1_{n}"]        = r["f1"]
                entry[f"precision_{n}"] = r["precision"]
                entry[f"recall_{n}"]    = r["recall"]
                line += f"  {r['f1']:>12.3f}  {r['precision']:>8.3f}  {r['recall']:>8.3f}"
            print(line)
            results.append(entry)
        return pd.DataFrame(results)

    coarse_grid = [round(0.01 + i * 0.02, 3) for i in range(48)]
    coarse_df   = _run_phase(coarse_grid, "Phase 1 — Coarse (step 0.02)")
    coarse_best = float(coarse_df.loc[coarse_df["avg_f1"].idxmax(), "theta"])
    print(f"\n   Phase 1 best: θ={coarse_best}  avg_F1={coarse_df['avg_f1'].max():.3f}")

    fine_lo   = max(0.001, round(coarse_best - 0.05, 3))
    fine_hi   = min(0.999, round(coarse_best + 0.05, 3))
    fine_grid = [round(fine_lo + i * 0.005, 3)
                 for i in range(int((fine_hi - fine_lo) / 0.005) + 1)]
    fine_df   = _run_phase(fine_grid, "Phase 2 — Fine zoom (step 0.005)")
    best_row  = fine_df.loc[fine_df["avg_f1"].idxmax()]
    best_theta = float(best_row["theta"])
    best_avg   = float(best_row["avg_f1"])

    print(f"\n   ★  Best θ = {best_theta}  (avg_F1={best_avg:.3f})")
    for n in names:
        print(f"      {n}: F1={best_row[f'f1_{n}']:.3f}  "
              f"P={best_row[f'precision_{n}']:.3f}  "
              f"R={best_row[f'recall_{n}']:.3f}")
    print(f"   (coarse peak was {coarse_best}, fine search refined to {best_theta})")

    # ── Step 5: chart ─────────────────────────────────────────────────────────
    _plot_calibration(coarse_df, fine_df, best_theta, chart_path,
                      benchmark_data, names)

    return best_theta



def _plot_calibration(
    coarse_df: pd.DataFrame,
    fine_df: pd.DataFrame,
    best_theta: float,
    chart_path: pathlib.Path,
    benchmark_data: list,
    names: list,
) -> None:
    """
    Save a five-panel PNG:
      Panel 1 — Coarse avg_F1 + per-benchmark F1 curves (full range)
      Panel 2 — Fine-zoom avg_F1 curves (±0.05 around best)
      Panel 3 — Score histogram: STS15 same vs different
      Panel 4 — Score histogram: SICK-R same vs different
      Panel 5 — ROC curve (TPR vs FPR) for every θ on both benchmarks

    READING THE ROC CURVE
    ─────────────────────
    Each point on the ROC curve corresponds to one θ value.  Moving
    right along the curve (lower θ) increases both TPR (recall) and FPR
    (false positives).  Moving left (higher θ) increases precision but
    loses recall.

    A random classifier follows the diagonal.  A perfect classifier
    hits the top-left corner (TPR=1, FPR=0).  The chosen θ is marked
    on each curve so you can see where it sits in the TPR/FPR tradeoff.

    AUC (Area Under Curve) is shown in the legend.  Higher = better
    discriminative power regardless of the chosen threshold.
    """
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("   [WARN] matplotlib not installed — skipping chart")
        return

    BG, PANEL, GRID = "#0F1117", "#1A1D27", "#2A2D3A"
    BENCH_COLORS = ["#4FC3F7", "#81C784", "#CE93D8", "#FFCC80"]
    AVG_COLOR    = "#FFD54F"
    BEST_COLOR   = "#FF7043"

    # 5 panels: coarse | fine | hist×n_bench | ROC
    n_bench   = len(names)
    n_panels  = 2 + n_bench + 1          # +1 for ROC
    fig, axes = plt.subplots(1, n_panels, figsize=(6 * n_panels, 5.5))
    fig.patch.set_facecolor(BG)
    for ax in axes:
        ax.set_facecolor(PANEL)
        ax.tick_params(colors="#AAAAAA", labelsize=9)
        for sp in ax.spines.values():
            sp.set_edgecolor("#333344")

    def _draw_panel(ax, df, title, mark=True):
        thetas = df["theta"].values
        for i, n in enumerate(names):
            ax.plot(thetas, df[f"f1_{n}"].values, "o-",
                    color=BENCH_COLORS[i % len(BENCH_COLORS)],
                    lw=1.5, ms=3, alpha=0.75, label=f"F1 {n}")
        ax.plot(thetas, df["avg_f1"].values, "D-",
                color=AVG_COLOR, lw=2.5, ms=5, zorder=5, label="avg F1")
        if mark and best_theta in df["theta"].values:
            bv = float(df.loc[df["theta"] == best_theta, "avg_f1"].values[0])
            ax.axvline(best_theta, color=BEST_COLOR, lw=1.5, ls="--",
                       alpha=0.9, label=f"Best θ={best_theta}")
            ax.scatter([best_theta], [bv], color=BEST_COLOR, s=100,
                       zorder=6, edgecolors="white", lw=1)
            xoff = 0.015 if best_theta < max(thetas) * 0.8 else -0.07
            ax.annotate(f"θ={best_theta}\navg_F1={bv:.3f}",
                        xy=(best_theta, bv),
                        xytext=(best_theta + xoff, max(0.1, bv - 0.15)),
                        color=BEST_COLOR, fontsize=8,
                        arrowprops=dict(arrowstyle="->", color=BEST_COLOR, lw=1))
        ax.set_xlabel("Threshold θ", color="#CCCCCC", fontsize=10)
        ax.set_ylabel("Score", color="#CCCCCC", fontsize=10)
        ax.set_title(title, color="white", fontsize=10, fontweight="bold")
        ax.set_ylim(-0.05, 1.05)
        ax.set_xlim(min(thetas) - 0.01, max(thetas) + 0.01)
        ax.legend(facecolor="#252836", edgecolor="#444455",
                  labelcolor="white", fontsize=8)
        ax.grid(True, color=GRID, lw=0.6, alpha=0.7)

    _draw_panel(axes[0], coarse_df,
                "Phase 1 — Coarse sweep\n(0.01→0.95, step 0.02)", mark=True)
    _draw_panel(axes[1], fine_df,
                "Phase 2 — Fine zoom\n(±0.05 around peak, step 0.005)", mark=True)

    # ── Histograms per benchmark ──────────────────────────────────────────────
    for k, (cfg, df_b) in enumerate(benchmark_data):
        ax = axes[2 + k]
        same = df_b.loc[ df_b["ground_truth"], "our_sim"].values
        diff = df_b.loc[~df_b["ground_truth"], "our_sim"].values
        bins = np.linspace(0, max(float(df_b["our_sim"].max()), 0.6), 30)
        ax.hist(diff, bins=bins, color="#EF5350", alpha=0.65,
                label=f"Different (≤{cfg['negative_threshold']})",
                edgecolor="#CC3333")
        ax.hist(same, bins=bins, color="#42A5F5", alpha=0.65,
                label=f"Same (≥{cfg['positive_threshold']})",
                edgecolor="#1A5FAA")
        ax.axvline(best_theta, color=BEST_COLOR, lw=2, ls="--",
                   label=f"Best θ={best_theta}")
        ax.set_xlabel("text_similarity() score", color="#CCCCCC", fontsize=10)
        ax.set_ylabel("Count", color="#CCCCCC", fontsize=10)
        ax.set_title(f"{cfg['name']} — Score Distribution\n"
                     "(θ should sit in the gap)",
                     color="white", fontsize=10, fontweight="bold")
        ax.legend(facecolor="#252836", edgecolor="#444455",
                  labelcolor="white", fontsize=8)
        ax.grid(True, color=GRID, lw=0.6, alpha=0.7)

    # ── ROC curve panel ───────────────────────────────────────────────────────
    ax_roc = axes[2 + n_bench]

    # Sort θ descending so we sweep from strict (low TPR, low FPR) to
    # permissive (high TPR, high FPR) — standard ROC orientation.
    thetas_sorted = sorted(coarse_df["theta"].values, reverse=True)

    for k, (cfg, df_b) in enumerate(benchmark_data):
        sims = df_b["our_sim"]
        gt   = df_b["ground_truth"]
        n_pos = int(gt.sum())
        n_neg = len(gt) - n_pos

        tprs, fprs = [], []
        for theta in thetas_sorted:
            pred = sims >= theta
            tp = int(( pred &  gt).sum())
            fp = int(( pred & ~gt).sum())
            tpr = tp / n_pos if n_pos else 0.0
            fpr = fp / n_neg if n_neg else 0.0
            tprs.append(tpr)
            fprs.append(fpr)

        # AUC via trapezoidal rule (descending FPR → reverse for np.trapz)
        fprs_arr = np.array(fprs)
        tprs_arr = np.array(tprs)
        order    = np.argsort(fprs_arr)
        # np.trapz was renamed to np.trapezoid in NumPy 2.0
        _trapz = getattr(np, 'trapezoid', None) or getattr(np, 'trapz', None)
        auc    = float(_trapz(tprs_arr[order], fprs_arr[order]))

        color = BENCH_COLORS[k % len(BENCH_COLORS)]
        ax_roc.plot(fprs, tprs, "o-", color=color, lw=2, ms=3,
                    label=f"{cfg['name']}  AUC={auc:.3f}")

        # Mark the chosen θ on this curve
        best_idx = min(range(len(thetas_sorted)),
                       key=lambda i: abs(thetas_sorted[i] - best_theta))
        ax_roc.scatter([fprs[best_idx]], [tprs[best_idx]],
                       color=BEST_COLOR, s=90, zorder=6,
                       edgecolors="white", lw=1)
        ax_roc.annotate(f"θ={best_theta}",
                        xy=(fprs[best_idx], tprs[best_idx]),
                        xytext=(fprs[best_idx] + 0.04,
                                tprs[best_idx] - 0.08),
                        color=BEST_COLOR, fontsize=8,
                        arrowprops=dict(arrowstyle="->",
                                        color=BEST_COLOR, lw=1))

    # Random-classifier diagonal
    ax_roc.plot([0, 1], [0, 1], "--", color="#555566", lw=1,
                alpha=0.6, label="Random classifier")

    ax_roc.set_xlabel("False Positive Rate  (1 − Specificity)",
                      color="#CCCCCC", fontsize=10)
    ax_roc.set_ylabel("True Positive Rate  (Recall / Sensitivity)",
                      color="#CCCCCC", fontsize=10)
    ax_roc.set_title("ROC Curve\n(per benchmark, θ marked)",
                     color="white", fontsize=10, fontweight="bold")
    ax_roc.set_xlim(-0.02, 1.02)
    ax_roc.set_ylim(-0.02, 1.02)
    ax_roc.legend(facecolor="#252836", edgecolor="#444455",
                  labelcolor="white", fontsize=8)
    ax_roc.grid(True, color=GRID, lw=0.6, alpha=0.7)

    ens = f"{W_LEXICAL:.0%} lexical + {W_SEMANTIC:.0%} semantic"
    fig.suptitle(
        f"Dual-Benchmark θ Calibration  ({ens})  —  ModeX Actionable Reconciliation",
        color="white", fontsize=12, fontweight="bold", y=1.01,
    )
    plt.tight_layout()
    chart_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(chart_path, dpi=150, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"   Chart saved → {chart_path}")


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
    limit: int             = None,
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
    limit            : process only the first N articles (None = all).
                       Useful for quick smoke-tests or debugging a subset.
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
        CHART_PATH.parent.mkdir(parents=True, exist_ok=True)
        calibrated_theta = calibrate_theta(CHART_PATH)
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

    if limit is not None and limit > 0:
        article_keys = article_keys[:limit]
        print(f"  Unique articles: {len(df['_article_key'].unique())} total "
              f"→ processing first {len(article_keys)} (--limit {limit})")
    else:
        print(f"  Unique articles: {len(article_keys)}")

    # ── Step 3: process articles ──────────────────────────────────────────────
    all_canonical = []
    workers       = min(N_WORKERS, len(article_keys))
    n_articles    = len(article_keys)

    # tqdm bar shared by both sequential and parallel paths.
    # Shows: progress bar | done/total articles | elapsed | ETA | rate
    bar_fmt = ("{l_bar}{bar}| {n_fmt}/{total_fmt} articles "
               "[{elapsed}<{remaining}, {rate_fmt}]")

    if workers <= 1:
        # ── Sequential ────────────────────────────────────────────────────────
        with tqdm(total=n_articles, desc="  Reconciling",
                  unit="art", bar_format=bar_fmt,
                  dynamic_ncols=True) as pbar:
            for akey in article_keys:
                rows = df[df["_article_key"] == akey]
                canonical = process_article(akey, rows, LOG_DIR, calibrated_theta)
                all_canonical.extend(canonical)
                # Show the short article title in the postfix so you know
                # which article just finished without scrolling the log.
                short = akey.split("||")[-1][:40]
                pbar.set_postfix_str(short, refresh=True)
                pbar.update(1)
    else:
        # ── Parallel ──────────────────────────────────────────────────────────
        print(f"\n  Parallel: {workers} workers across {n_articles} articles")
        tasks = [
            (akey,
             df[df["_article_key"] == akey].to_dict(orient="list"),
             str(LOG_DIR),
             calibrated_theta)
            for akey in article_keys
        ]
        ctx = multiprocessing.get_context('spawn')
        with ProcessPoolExecutor(max_workers=workers, mp_context=ctx) as pool:
            futures = {pool.submit(_worker, t): t[0] for t in tasks}
            # as_completed() yields futures in completion order (fastest first),
            # so the bar advances as soon as any worker finishes — not in
            # submission order.  This gives a more accurate ETA.
            with tqdm(total=n_articles, desc="  Reconciling",
                      unit="art", bar_format=bar_fmt,
                      dynamic_ncols=True) as pbar:
                for future in as_completed(futures):
                    akey, canonical = future.result()
                    all_canonical.extend(canonical)
                    short = akey.split("||")[-1][:40]
                    pbar.set_postfix_str(short, refresh=True)
                    pbar.update(1)

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
    parser.add_argument(
        "--limit", type=int, default=None,
        help="Process only the first N articles (default: all). "
             "Useful for smoke-testing or debugging a small subset.",
    )
    args = parser.parse_args()

    main(
        skip_calibration=args.skip_calibration,
        calibrate_only=args.calibrate_only,
        fixed_theta=args.theta,
        n_workers=args.workers,
        min_support=args.min_support,
        limit=args.limit,
    )