"""
ReACT-GPT SLR — Unified EDA Script
Handles three CSVs:
  1. categorized.csv          – category verdict per article (8 categories × 5 models)
  2. scored.csv               – soundness & precision scores per article (5 models each)
  3. reconciled_actionables.csv – base actionable metadata (confidence, support, etc.)

Run:
    python eda_react_gpt.py \
        --categorized /local_history/categorized.csv \
        --scored      /local_history/scored.csv \
        --reconciled  /local_history/reconciled_actionables.csv \
        --out         ./eda_output
"""

import argparse, os, warnings, textwrap
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec
from math import pi

# ── palette & constants ──────────────────────────────────────────────────────
BG       = "#F7F9FB"
GRID_CLR = "#DEE2E6"
ACCENT   = "#E76F51"
ACCENT2  = "#457B9D"
GREEN_P  = ["#1B4332","#2D6A4F","#40916C","#52B788","#74C69D","#95D5B2","#B7E4C7","#D8F3DC"]
BLUE_P   = ["#03045E","#0077B6","#0096C7","#00B4D8","#48CAE4","#90E0EF","#ADE8F4","#CAF0F8"]
MIX_P    = [ACCENT, ACCENT2, "#2D6A4F", "#E9C46A", "#A8DADC"]

MODELS       = ["qwen3.6:35b","gpt-oss:20b","deepseek-r1:32b","gemma4:31b","mixtral:8x7b"]
MODEL_LABELS = ["Qwen3.6","GPT-oss","DeepSeek-R1","Gemma4","Mixtral"]

CATEGORIES = [
    "New Contributor Onboarding and Involvement",
    "Code Standards and Maintainability",
    "Automated Testing and Quality Assurance",
    "Community Collaboration and Engagement",
    "Documentation Practices",
    "Project Management and Governance",
    "Security Best Practices and Legal Compliance",
    "CI/CD and DevOps Automation",
]
SHORT_CAT = {
    "New Contributor Onboarding and Involvement":  "Onboarding",
    "Code Standards and Maintainability":          "Code Quality",
    "Automated Testing and Quality Assurance":     "Testing & QA",
    "Community Collaboration and Engagement":      "Community",
    "Documentation Practices":                     "Docs",
    "Project Management and Governance":           "Governance",
    "Security Best Practices and Legal Compliance":"Security",
    "CI/CD and DevOps Automation":                 "CI/CD",
}


# ── utilities ────────────────────────────────────────────────────────────────

def mkdir(path):
    os.makedirs(path, exist_ok=True)
    return path

def savefig(fig, path):
    fig.savefig(path, dpi=150, bbox_inches="tight", facecolor=BG)
    plt.close(fig)
    print(f"  ✔  {path}")

def style_ax(ax):
    ax.set_facecolor(BG)
    ax.grid(color=GRID_CLR, linestyle="--", linewidth=0.7)
    ax.spines[["top","right"]].set_visible(False)

def parse_fraction(x):
    try:
        if isinstance(x, str) and "/" in x:
            a, b = x.split("/"); return int(a)/int(b)
        return float(x)
    except:
        return np.nan

def load_csv(path, label):
    if not path or not os.path.exists(path):
        print(f"  ⚠  {label} not found: {path}")
        return None
    df = pd.read_csv(path)
    if "support_fraction" in df.columns:
        df["support_fraction"] = df["support_fraction"].apply(parse_fraction)
    print(f"  ✔  {label}: {len(df)} rows, {len(df.columns)} cols")
    return df

def verdict_bool(df, col):
    return df[col].astype(str).str.strip().str.upper() == "YES"

def section_header(title):
    print(f"\n{'═'*60}")
    print(f"  {title}")
    print(f"{'═'*60}")


# ════════════════════════════════════════════════════════════════════════════
#  A.  RECONCILED ACTIONABLES EDA
# ════════════════════════════════════════════════════════════════════════════

def eda_reconciled(df, out):
    section_header("A. reconciled_actionables.csv")
    d = out

    # console summary
    num_cols = ["avg_confidence","centroid_confidence","support_fraction","support"]
    present  = [c for c in num_cols if c in df.columns]
    print(df[present].describe().round(3).to_string())

    # ── A1. Confidence distribution ──────────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(11,4), facecolor=BG)
    for ax, col, pal in zip(axes,
                             ["avg_confidence","centroid_confidence"],
                             [GREEN_P[1], BLUE_P[2]]):
        if col not in df.columns: continue
        style_ax(ax)
        data = df[col].dropna()
        ax.hist(data, bins=20, color=pal, edgecolor="white", linewidth=0.7)
        ax.axvline(data.mean(), color=ACCENT, linewidth=1.8, linestyle="--",
                   label=f"μ={data.mean():.3f}")
        ax.axvline(data.median(), color=ACCENT2, linewidth=1.8, linestyle=":",
                   label=f"med={data.median():.3f}")
        ax.set_title(col.replace("_"," ").title(), fontsize=11, fontweight="bold")
        ax.set_xlabel("Score"); ax.set_ylabel("Count")
        ax.legend(fontsize=8)
    fig.suptitle("Reconciled Actionables — Confidence Distributions",
                 fontsize=13, fontweight="bold", y=1.01)
    fig.tight_layout()
    savefig(fig, f"{d}/A1_confidence_distributions.png")

    # ── A2. Support fraction histogram ──────────────────────────────────
    if "support_fraction" in df.columns:
        fig, ax = plt.subplots(figsize=(7,4), facecolor=BG)
        style_ax(ax)
        bins = np.linspace(0, 1, 12)
        ax.hist(df["support_fraction"].dropna(), bins=bins,
                color=GREEN_P[2], edgecolor="white", linewidth=0.7)
        ax.set_xlabel("Support Fraction (models agreeing)", fontsize=10)
        ax.set_ylabel("# Actionables", fontsize=10)
        ax.set_title("Model Consensus Distribution\n(reconciled_actionables)", fontsize=12, fontweight="bold")
        ax.set_xticks(bins)
        ax.set_xticklabels([f"{b:.1f}" for b in bins], fontsize=8)
        fig.tight_layout()
        savefig(fig, f"{d}/A2_support_fraction.png")

    # ── A3. Venue breakdown — ICSE vs FSE (normalised) ──────────────────
    if "venue" in df.columns:
        def map_venue(v):
            v = str(v).upper()
            if "ICSE" in v: return "ICSE"
            if "FSE"  in v: return "FSE"
            return "Other"

        df["venue_group"] = df["venue"].apply(map_venue)
        grp = df[df["venue_group"].isin(["ICSE","FSE"])].groupby("venue_group")

        counts = grp.size()                          # raw counts
        norm   = counts / counts.sum()               # fraction of total

        fig, axes = plt.subplots(1, 2, figsize=(11, 5), facecolor=BG)

        # left: raw counts
        ax = axes[0]
        style_ax(ax)
        colors = [GREEN_P[2] if v == "ICSE" else BLUE_P[2] for v in counts.index]
        bars = ax.bar(counts.index, counts.values, color=colors, edgecolor="white",
                      linewidth=0.8, width=0.5)
        for bar, val in zip(bars, counts.values):
            ax.text(bar.get_x()+bar.get_width()/2, val+0.3, str(val),
                    ha="center", va="bottom", fontsize=11, fontweight="bold")
        ax.set_ylabel("# Actionables", fontsize=10)
        ax.set_title("Raw Actionable Count\nper Venue", fontsize=11, fontweight="bold")
        ax.set_ylim(0, counts.max() * 1.15)
        ax.spines["bottom"].set_visible(True)

        # right: normalised (%)
        ax = axes[1]
        style_ax(ax)
        bars = ax.bar(norm.index, norm.values * 100, color=colors, edgecolor="white",
                      linewidth=0.8, width=0.5)
        for bar, val in zip(bars, norm.values):
            ax.text(bar.get_x()+bar.get_width()/2, val*100+0.5, f"{val:.1%}",
                    ha="center", va="bottom", fontsize=11, fontweight="bold")
        ax.set_ylabel("% of Actionables (ICSE + FSE)", fontsize=10)
        ax.set_title("Normalised Share\nper Venue", fontsize=11, fontweight="bold")
        ax.set_ylim(0, norm.max() * 100 * 1.18)
        ax.spines["bottom"].set_visible(True)

        # shared styling
        for ax in axes:
            ax.tick_params(axis="x", labelsize=13)
            ax.spines["left"].set_visible(False)
            patch_icse = mpatches.Patch(color=GREEN_P[2], label="ICSE")
            patch_fse  = mpatches.Patch(color=BLUE_P[2],  label="FSE")
            ax.legend(handles=[patch_icse, patch_fse], fontsize=9, framealpha=0.8)

        fig.suptitle("Actionables per Venue — ICSE vs FSE (reconciled)",
                     fontsize=13, fontweight="bold", y=1.02)
        fig.tight_layout()
        savefig(fig, f"{d}/A3_venue_breakdown.png")

    # ── A4. avg_confidence vs support_fraction scatter ───────────────────
    if {"avg_confidence","support_fraction"}.issubset(df.columns):
        fig, ax = plt.subplots(figsize=(7,5), facecolor=BG)
        style_ax(ax)
        sc = ax.scatter(df["support_fraction"], df["avg_confidence"],
                        alpha=0.6, c=df["avg_confidence"],
                        cmap="YlGn", edgecolors="#666", linewidths=0.4, s=60)
        fig.colorbar(sc, ax=ax, label="avg_confidence")
        ax.set_xlabel("Support Fraction", fontsize=10)
        ax.set_ylabel("Avg Confidence", fontsize=10)
        ax.set_title("Confidence vs Model Consensus\n(reconciled_actionables)",
                     fontsize=12, fontweight="bold")
        fig.tight_layout()
        savefig(fig, f"{d}/A4_confidence_vs_support.png")


# ════════════════════════════════════════════════════════════════════════════
#  B.  SCORED CSV EDA
# ════════════════════════════════════════════════════════════════════════════

def eda_scored(df, out):
    section_header("B. scored.csv")
    d = out

    # derive per-model binary columns
    for prefix in ["sound","precise"]:
        for m in MODELS:
            col = f"{prefix}_{m}"
            if col in df.columns:
                df[col] = verdict_bool(df, col)

    # ── B1. Sound vs Precise overall rate ───────────────────────────────
    sound_cols   = [f"sound_{m}"   for m in MODELS if f"sound_{m}"   in df.columns]
    precise_cols = [f"precise_{m}" for m in MODELS if f"precise_{m}" in df.columns]

    sound_rates   = [df[c].mean() for c in sound_cols]
    precise_rates = [df[c].mean() for c in precise_cols]
    labels        = MODEL_LABELS[:len(sound_rates)]

    x = np.arange(len(labels))
    fig, ax = plt.subplots(figsize=(9,5), facecolor=BG)
    style_ax(ax)
    w = 0.35
    ax.bar(x - w/2, sound_rates,   w, label="Sound",   color=GREEN_P[2], edgecolor="white")
    ax.bar(x + w/2, precise_rates, w, label="Precise",  color=BLUE_P[2],  edgecolor="white")
    ax.set_xticks(x); ax.set_xticklabels(labels, fontsize=10)
    ax.set_ylabel("YES Rate (fraction of articles)", fontsize=10)
    ax.set_ylim(0, 1.05)
    ax.set_title("Soundness vs Precision — Per-Model YES Rate", fontsize=13, fontweight="bold")
    ax.legend(fontsize=10)
    for bar in ax.patches:
        h = bar.get_height()
        ax.text(bar.get_x()+bar.get_width()/2, h+0.01, f"{h:.0%}",
                ha="center", va="bottom", fontsize=8)
    fig.tight_layout()
    savefig(fig, f"{d}/B1_sound_vs_precise_per_model.png")

    # ── B2. Sound verdict distribution ──────────────────────────────────
    if "sound_verdict" in df.columns:
        vc = verdict_bool(df, "sound_verdict")
        fig, ax = plt.subplots(figsize=(5,4), facecolor=BG)
        style_ax(ax)
        ax.bar(["Sound YES","Sound NO"], [vc.sum(), (~vc).sum()],
               color=[GREEN_P[2], "#CCC"], edgecolor="white")
        ax.set_title("Consensus Sound Verdict Distribution", fontsize=11, fontweight="bold")
        ax.set_ylabel("Articles")
        fig.tight_layout()
        savefig(fig, f"{d}/B2_sound_verdict_dist.png")

    # ── B3. Precise verdict distribution ────────────────────────────────
    if "precise_verdict" in df.columns:
        vc = verdict_bool(df, "precise_verdict")
        fig, ax = plt.subplots(figsize=(5,4), facecolor=BG)
        style_ax(ax)
        ax.bar(["Precise YES","Precise NO"], [vc.sum(), (~vc).sum()],
               color=[BLUE_P[2], "#CCC"], edgecolor="white")
        ax.set_title("Consensus Precise Verdict Distribution", fontsize=11, fontweight="bold")
        ax.set_ylabel("Articles")
        fig.tight_layout()
        savefig(fig, f"{d}/B3_precise_verdict_dist.png")

    # ── B4. Sound YES count histogram ───────────────────────────────────
    if "sound_yes_count" in df.columns:
        fig, axes = plt.subplots(1,2, figsize=(11,4), facecolor=BG)
        for ax, col, pal, title in zip(
            axes,
            ["sound_yes_count","precise_yes_count"],
            [GREEN_P[2], BLUE_P[2]],
            ["Sound YES Count","Precise YES Count"]
        ):
            if col not in df.columns: continue
            style_ax(ax)
            bins = np.arange(-0.5, 6.5, 1)
            ax.hist(df[col].dropna(), bins=bins, color=pal, edgecolor="white", linewidth=0.7)
            ax.set_xticks(range(6))
            ax.set_xlabel("# Models voting YES (0–5)", fontsize=9)
            ax.set_ylabel("# Articles", fontsize=9)
            ax.set_title(title, fontsize=11, fontweight="bold")
        fig.suptitle("Model Agreement Depth (scored.csv)",
                     fontsize=13, fontweight="bold", y=1.01)
        fig.tight_layout()
        savefig(fig, f"{d}/B4_yes_count_histograms.png")

    # ── B5. Heatmap: article × (sound|precise) per model ───────────────
    cols_order = [f"sound_{m}" for m in MODELS if f"sound_{m}" in df.columns] + \
                 [f"precise_{m}" for m in MODELS if f"precise_{m}" in df.columns]
    col_labels = [f"S:{MODEL_LABELS[MODELS.index(m)]}" for m in MODELS if f"sound_{m}" in df.columns] + \
                 [f"P:{MODEL_LABELS[MODELS.index(m)]}" for m in MODELS if f"precise_{m}" in df.columns]

    if cols_order:
        mat = df[cols_order].astype(int).values
        n   = mat.shape[0]
        fig, ax = plt.subplots(figsize=(13, max(4, n*0.35+1)), facecolor=BG)
        ax.set_facecolor(BG)
        im = ax.imshow(mat, aspect="auto", cmap="RdYlGn", vmin=0, vmax=1)
        ax.set_xticks(range(len(col_labels)))
        ax.set_xticklabels(col_labels, rotation=40, ha="right", fontsize=8)
        ax.set_yticks(range(n))
        ax.set_yticklabels([f"#{i+1}" for i in range(n)], fontsize=7)
        ax.set_title("Scored CSV — Article × (Sound | Precise) per Model Heatmap",
                     fontsize=12, fontweight="bold", pad=10)

        # vertical separator between sound and precise blocks
        n_sound = sum(1 for m in MODELS if f"sound_{m}" in df.columns)
        ax.axvline(n_sound - 0.5, color="black", linewidth=2)
        ax.text(n_sound/2 - 0.5, -0.8, "SOUND", ha="center", fontsize=9,
                fontweight="bold", color=GREEN_P[1])
        ax.text(n_sound + (len(col_labels)-n_sound)/2 - 0.5, -0.8, "PRECISE",
                ha="center", fontsize=9, fontweight="bold", color=BLUE_P[2])

        cbar = fig.colorbar(im, ax=ax, fraction=0.02, pad=0.02)
        cbar.set_ticks([0,1]); cbar.set_ticklabels(["No","Yes"])
        fig.tight_layout()
        savefig(fig, f"{d}/B5_scored_heatmap.png")

    # ── B6. Spider: sound vs precise per model ───────────────────────────
    if sound_rates and precise_rates:
        N      = len(labels)
        angles = [n / N * 2 * pi for n in range(N)] + [0]
        sv     = sound_rates   + sound_rates[:1]
        pv     = precise_rates + precise_rates[:1]

        fig, ax = plt.subplots(figsize=(7,7), subplot_kw=dict(polar=True), facecolor=BG)
        ax.set_facecolor(BG)
        ax.set_rgrids([0.15,0.30,0.45,0.60], labels=["15%","30%","45%", "50%"],
                      angle=0, fontsize=8, color="#888")
        ax.set_ylim(0,0.6)
        ax.grid(color=GRID_CLR, linewidth=0.8, linestyle="--")
        ax.spines["polar"].set_color(GRID_CLR)
        ax.plot(angles, sv, linewidth=2.2, color=GREEN_P[2], label="Sound")
        ax.fill(angles, sv, color=GREEN_P[2], alpha=0.15)
        ax.plot(angles, pv, linewidth=2.2, color=BLUE_P[2],  label="Precise", linestyle="--")
        ax.fill(angles, pv, color=BLUE_P[2],  alpha=0.12)
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(labels, fontsize=11, fontweight="bold")
        ax.legend(loc="upper right", bbox_to_anchor=(1.35,1.15), fontsize=10)
        ax.set_title("Soundness vs Precision — Per-Model Spider",
                     fontsize=12, fontweight="bold", pad=28)
        fig.tight_layout()
        savefig(fig, f"{d}/B6_sound_precise_spider.png")

    # ── B7. Confidence distributions (same cols as base) ────────────────
    num_cols = [c for c in ["avg_confidence","centroid_confidence","support_fraction"] if c in df.columns]
    if num_cols:
        fig, axes = plt.subplots(1, len(num_cols), figsize=(5*len(num_cols), 4), facecolor=BG)
        if len(num_cols)==1: axes=[axes]
        for ax, col in zip(axes, num_cols):
            style_ax(ax)
            data = df[col].dropna()
            ax.hist(data, bins=15, color=MIX_P[2], edgecolor="white", linewidth=0.7)
            ax.axvline(data.mean(), color=ACCENT, linewidth=1.8, linestyle="--",
                       label=f"μ={data.mean():.3f}")
            ax.set_title(col.replace("_"," ").title(), fontsize=10, fontweight="bold")
            ax.set_xlabel(col); ax.set_ylabel("Count")
            ax.legend(fontsize=8)
        fig.suptitle("Scored CSV — Metric Distributions", fontsize=12, fontweight="bold", y=1.01)
        fig.tight_layout()
        savefig(fig, f"{d}/B7_scored_distributions.png")


# ════════════════════════════════════════════════════════════════════════════
#  C.  CATEGORIZED CSV EDA  (from original script, fully integrated)
# ════════════════════════════════════════════════════════════════════════════

def eda_categorized(df, out):
    section_header("C. categorized.csv")
    d = out

    for cat in CATEGORIES:
        col = f"cat_{cat}__verdict"
        if col in df.columns:
            df[col] = verdict_bool(df, col)

    # ── C1. Category frequency bar ───────────────────────────────────────
    counts = {SHORT_CAT[c]: df[f"cat_{c}__verdict"].sum()
              for c in CATEGORIES if f"cat_{c}__verdict" in df.columns}
    labels  = list(counts.keys())
    values  = list(counts.values())
    order   = np.argsort(values)[::-1]
    labels  = [labels[i] for i in order]
    values  = [values[i] for i in order]

    fig, ax = plt.subplots(figsize=(10,5), facecolor=BG)
    style_ax(ax)
    bars = ax.barh(labels, values, color=GREEN_P[:len(labels)], edgecolor="white")
    for bar, val in zip(bars, values):
        ax.text(val+0.1, bar.get_y()+bar.get_height()/2, str(val),
                va="center", fontsize=9)
    ax.set_xlabel("# Articles (YES verdict)", fontsize=10)
    ax.set_title("Category Frequency Across Articles", fontsize=13, fontweight="bold", pad=12)
    ax.spines["left"].set_visible(False)
    fig.tight_layout()
    savefig(fig, f"{d}/C1_category_frequency.png")

    # ── C2. Stacked model agreement ──────────────────────────────────────
    data_mat = {}
    for cat in CATEGORIES:
        row = []
        for m in MODELS:
            col = f"cat_{cat}__{m}"
            if col in df.columns:
                yes = (df[col].astype(str).str.strip().str.upper()=="YES").sum()
            else:
                yes = 0
            row.append(yes)
        data_mat[SHORT_CAT[cat]] = row

    cats_s = list(data_mat.keys())
    mat    = np.array([data_mat[c] for c in cats_s])

    fig, ax = plt.subplots(figsize=(12,6), facecolor=BG)
    style_ax(ax)
    bottoms = np.zeros(len(cats_s))
    for i, (ml, col) in enumerate(zip(MODEL_LABELS, MIX_P)):
        vals = mat[:, i]
        ax.bar(cats_s, vals, bottom=bottoms, label=ml, color=col, edgecolor="white", linewidth=0.5)
        bottoms += vals
    ax.set_ylabel("YES votes (summed)", fontsize=10)
    ax.set_title("Model Agreement per Category (Stacked)", fontsize=13, fontweight="bold", pad=12)
    ax.set_xticklabels(cats_s, rotation=30, ha="right", fontsize=9)
    ax.legend(fontsize=8, framealpha=0.8)
    fig.tight_layout()
    savefig(fig, f"{d}/C2_model_agreement_stacked.png")

    # ── C3. Heatmap article × category ───────────────────────────────────
    verdict_cols  = [f"cat_{c}__verdict" for c in CATEGORIES if f"cat_{c}__verdict" in df.columns]
    present_cats  = [c for c in CATEGORIES if f"cat_{c}__verdict" in df.columns]
    mat2 = df[verdict_cols].astype(int).values
    n    = mat2.shape[0]

    fig, ax = plt.subplots(figsize=(12, max(4, n*0.35+1)), facecolor=BG)
    ax.set_facecolor(BG)
    im = ax.imshow(mat2, aspect="auto", cmap="YlGn", vmin=0, vmax=1)
    ax.set_xticks(range(len(present_cats)))
    ax.set_xticklabels([SHORT_CAT[c] for c in present_cats], rotation=35, ha="right", fontsize=8)
    ax.set_yticks(range(n)); ax.set_yticklabels([f"#{i+1}" for i in range(n)], fontsize=7)
    ax.set_title("Article × Category Heatmap", fontsize=13, fontweight="bold", pad=10)
    cbar = fig.colorbar(im, ax=ax, fraction=0.02, pad=0.02)
    cbar.set_ticks([0,1]); cbar.set_ticklabels(["No","Yes"])
    fig.tight_layout()
    savefig(fig, f"{d}/C3_category_heatmap.png")

    # ── C4. Distributions ────────────────────────────────────────────────
    num_cols = [c for c in ["support_fraction","avg_confidence","num_categories"] if c in df.columns]
    if num_cols:
        fig, axes = plt.subplots(1, len(num_cols), figsize=(5*len(num_cols),4), facecolor=BG)
        if len(num_cols)==1: axes=[axes]
        for ax, col in zip(axes, num_cols):
            style_ax(ax)
            data = df[col].dropna()
            ax.hist(data, bins=min(15,len(data)), color=GREEN_P[2], edgecolor="white", linewidth=0.7)
            ax.axvline(data.mean(), color=ACCENT, linewidth=1.8, linestyle="--",
                       label=f"μ={data.mean():.2f}")
            ax.set_title(col.replace("_"," ").title(), fontsize=10, fontweight="bold")
            ax.set_xlabel(col); ax.set_ylabel("Count")
            ax.legend(fontsize=8)
        fig.suptitle("Categorized CSV — Metric Distributions", fontsize=12, fontweight="bold", y=1.01)
        fig.tight_layout()
        savefig(fig, f"{d}/C4_categorized_distributions.png")

    # ── C5. Venue breakdown — ICSE vs FSE (normalised) ──────────────────
    if "venue" in df.columns:
        def map_venue(v):
            v = str(v).upper()
            if "ICSE" in v: return "ICSE"
            if "FSE"  in v: return "FSE"
            return "Other"

        df["venue_group"] = df["venue"].apply(map_venue)
        grp    = df[df["venue_group"].isin(["ICSE","FSE"])].groupby("venue_group")
        counts = grp.size()
        norm   = counts / counts.sum()

        fig, axes = plt.subplots(1, 2, figsize=(11, 5), facecolor=BG)
        for ax, vals, ylabel, title, fmt in zip(
            axes,
            [counts.values, norm.values * 100],
            ["# Articles", "% of Articles (ICSE + FSE)"],
            ["Raw Article Count\nper Venue", "Normalised Share\nper Venue"],
            [lambda v: str(v), lambda v: f"{v:.1f}%"],
        ):
            style_ax(ax)
            colors = [GREEN_P[3] if v == "ICSE" else BLUE_P[2] for v in counts.index]
            bars = ax.bar(counts.index, vals, color=colors, edgecolor="white",
                          linewidth=0.8, width=0.5)
            for bar, val in zip(bars, vals):
                ax.text(bar.get_x()+bar.get_width()/2, val + max(vals)*0.01,
                        fmt(val), ha="center", va="bottom", fontsize=11, fontweight="bold")
            ax.set_ylabel(ylabel, fontsize=10)
            ax.set_title(title, fontsize=11, fontweight="bold")
            ax.set_ylim(0, max(vals) * 1.18)
            ax.tick_params(axis="x", labelsize=13)
            ax.spines["left"].set_visible(False)
            patch_icse = mpatches.Patch(color=GREEN_P[3], label="ICSE")
            patch_fse  = mpatches.Patch(color=BLUE_P[2],  label="FSE")
            ax.legend(handles=[patch_icse, patch_fse], fontsize=9, framealpha=0.8)

        fig.suptitle("Articles per Venue — ICSE vs FSE (categorized)",
                     fontsize=13, fontweight="bold", y=1.02)
        fig.tight_layout()
        savefig(fig, f"{d}/C5_venue_distribution.png")

    # ── C6. Spider chart — CONSENSUS ONLY ────────────────────────────────
    verdict_rates = []
    for cat in CATEGORIES:
        col = f"cat_{cat}__verdict"
        verdict_rates.append(df[col].mean() if col in df.columns else 0.0)

    N      = len(CATEGORIES)
    angles = [n / N * 2 * pi for n in range(N)] + [0]
    v_vals = verdict_rates + verdict_rates[:1]

    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True), facecolor=BG)
    ax.set_facecolor(BG)
    ax.set_rgrids(
        [0.15, 0.30, 0.45, 0.60],
        labels=["15%", "30%", "45%", "50%"],
        angle=0, fontsize=8, color="#888",
    )
    ax.set_ylim(0, 0.6)
    ax.grid(color=GRID_CLR, linewidth=0.8, linestyle="--")
    ax.spines["polar"].set_color(GRID_CLR)

    # ── draw filled consensus polygon with markers ──────────────────────
    ax.plot(angles, v_vals, linewidth=2.8, color=ACCENT, label="Consensus")
    ax.fill(angles, v_vals, color=ACCENT, alpha=0.25)
    ax.scatter(angles[:-1], verdict_rates, s=60, color=ACCENT,
               zorder=5, edgecolors="white", linewidths=1.2)

    # ── annotate each vertex with its percentage ────────────────────────
    for angle, rate in zip(angles[:-1], verdict_rates):
        offset = 0.05                        # push label just outside the ring
        ax.text(
            angle, rate + offset,
            f"{rate:.0%}",
            ha="center", va="center",
            fontsize=8, fontweight="bold",
            color=ACCENT,
        )

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(
        [SHORT_CAT[c] for c in CATEGORIES],
        fontsize=10, fontweight="bold",
    )
    ax.legend(loc="upper right", bbox_to_anchor=(1.35, 1.15), fontsize=10)
    ax.set_title(
        "Category Coverage Spider Chart\n(Consensus Verdict)",
        fontsize=13, fontweight="bold", pad=30,
    )
    fig.tight_layout()
    savefig(fig, f"{d}/C6_category_spider.png")

    # ── C7. Bubble chart ─────────────────────────────────────────────────
    n_cats = df[[f"cat_{c}__verdict" for c in CATEGORIES
                 if f"cat_{c}__verdict" in df.columns]].astype(int).sum(axis=1)
    conf   = df["avg_confidence"] if "avg_confidence" in df.columns else pd.Series(np.ones(len(df)))
    supp   = df["support_fraction"] if "support_fraction" in df.columns else pd.Series(np.ones(len(df)))

    fig, ax = plt.subplots(figsize=(9,5), facecolor=BG)
    style_ax(ax)
    sc = ax.scatter(range(len(df)), conf.fillna(0.5),
                    s=supp.fillna(0.5)*400, c=n_cats, cmap="YlGn",
                    edgecolors="#888", linewidths=0.5, alpha=0.85,
                    vmin=0, vmax=max(n_cats.max(),1))
    fig.colorbar(sc, ax=ax, label="# Categories Assigned")
    ax.set_xlabel("Article Index", fontsize=10)
    ax.set_ylabel("Avg Confidence", fontsize=10)
    ax.set_title("Article Bubble Chart (size=support, color=#categories)",
                 fontsize=12, fontweight="bold")
    fig.tight_layout()
    savefig(fig, f"{d}/C7_bubble_chart.png")


# ════════════════════════════════════════════════════════════════════════════
#  D.  CROSS-CSV SUMMARY DASHBOARD
# ════════════════════════════════════════════════════════════════════════════

def cross_dashboard(dfs, out):
    """Single 2×2 summary figure pulling from all three CSVs."""
    section_header("D. Cross-CSV Summary Dashboard")
    df_rec, df_sc, df_cat = dfs

    fig = plt.figure(figsize=(16, 10), facecolor=BG)
    gs  = GridSpec(2, 2, figure=fig, hspace=0.42, wspace=0.32)

    # ── D1. Confidence comparison across datasets ────────────────────────
    ax1 = fig.add_subplot(gs[0, 0])
    style_ax(ax1)
    for df_, label, col in [
        (df_rec, "Reconciled", GREEN_P[1]),
        (df_sc,  "Scored",     BLUE_P[2]),
        (df_cat, "Categorized",ACCENT),
    ]:
        if df_ is not None and "avg_confidence" in df_.columns:
            ax1.hist(df_["avg_confidence"].dropna(), bins=15, alpha=0.55,
                     color=col, edgecolor="white", label=label)
    ax1.set_xlabel("avg_confidence", fontsize=9)
    ax1.set_ylabel("Count", fontsize=9)
    ax1.set_title("Confidence Distribution — All Datasets", fontsize=10, fontweight="bold")
    ax1.legend(fontsize=8)

    # ── D2. Support fraction comparison ─────────────────────────────────
    ax2 = fig.add_subplot(gs[0, 1])
    style_ax(ax2)
    data_sf, labels_sf, colors_sf = [], [], []
    for df_, label, col in [
        (df_rec, "Reconciled", GREEN_P[1]),
        (df_sc,  "Scored",     BLUE_P[2]),
        (df_cat, "Categorized",ACCENT),
    ]:
        if df_ is not None and "support_fraction" in df_.columns:
            data_sf.append(df_["support_fraction"].dropna().values)
            labels_sf.append(label); colors_sf.append(col)

    if data_sf:
        bp = ax2.boxplot(data_sf, patch_artist=True, widths=0.5,
                         medianprops=dict(color="black", linewidth=2))
        for patch, col in zip(bp["boxes"], colors_sf):
            patch.set_facecolor(col); patch.set_alpha(0.7)
        ax2.set_xticklabels(labels_sf, fontsize=9)
        ax2.set_ylabel("Support Fraction", fontsize=9)
        ax2.set_title("Support Fraction Boxplot — All Datasets", fontsize=10, fontweight="bold")

    # ── D3. Scored: sound_yes vs precise_yes counts ──────────────────────
    ax3 = fig.add_subplot(gs[1, 0])
    style_ax(ax3)
    if df_sc is not None:
        for col, pal, label in [
            ("sound_yes_count",   GREEN_P[2], "Sound YES"),
            ("precise_yes_count", BLUE_P[2],  "Precise YES"),
        ]:
            if col in df_sc.columns:
                vc = df_sc[col].value_counts().sort_index()
                ax3.plot(vc.index, vc.values, marker="o", label=label,
                         color=pal, linewidth=2, markersize=6)
        ax3.set_xlabel("# Models voting YES", fontsize=9)
        ax3.set_ylabel("# Articles", fontsize=9)
        ax3.set_title("Agreement Depth: Sound vs Precise", fontsize=10, fontweight="bold")
        ax3.legend(fontsize=9)

    # ── D4. Categorized: top-5 category YES rate bar ─────────────────────
    ax4 = fig.add_subplot(gs[1, 1])
    style_ax(ax4)
    if df_cat is not None:
        rates = {}
        for cat in CATEGORIES:
            col = f"cat_{cat}__verdict"
            if col in df_cat.columns:
                rates[SHORT_CAT[cat]] = df_cat[col].mean()
        if rates:
            sorted_r = sorted(rates.items(), key=lambda x: x[1], reverse=True)[:5]
            labs, vals = zip(*sorted_r)
            bars = ax4.barh(labs, vals, color=GREEN_P[:5], edgecolor="white")
            for bar, val in zip(bars, vals):
                ax4.text(val+0.005, bar.get_y()+bar.get_height()/2,
                         f"{val:.0%}", va="center", fontsize=8)
            ax4.set_xlim(0, 1.1)
            ax4.set_xlabel("YES Rate", fontsize=9)
            ax4.set_title("Top-5 Category YES Rates (categorized)", fontsize=10, fontweight="bold")
            ax4.spines["left"].set_visible(False)

    fig.suptitle("ReACT-GPT SLR — Cross-Dataset Summary Dashboard",
                 fontsize=15, fontweight="bold", y=1.01)
    savefig(fig, f"{out}/D_cross_summary_dashboard.png")


# ════════════════════════════════════════════════════════════════════════════
#  MAIN
# ════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="ReACT-GPT Unified EDA")
    parser.add_argument("--categorized", default="/Users/nafiz43/Documents/GitHub/ReACT-GPT/local_history/categorized.csv")
    parser.add_argument("--scored",      default="/Users/nafiz43/Documents/GitHub/ReACT-GPT/local_history/scored.csv")
    parser.add_argument("--reconciled",  default="/Users/nafiz43/Documents/GitHub/ReACT-GPT/local_history/reconciled_actionables.csv")
    parser.add_argument("--out",         default="/Users/nafiz43/Documents/GitHub/ReACT-GPT/eda_output")
    args = parser.parse_args()

    mkdir(args.out)
    out_rec = mkdir(f"{args.out}/A_reconciled")
    out_sc  = mkdir(f"{args.out}/B_scored")
    out_cat = mkdir(f"{args.out}/C_categorized")

    print("\n📂  Loading CSVs …")
    df_rec = load_csv(args.reconciled,  "reconciled_actionables")
    df_sc  = load_csv(args.scored,      "scored")
    df_cat = load_csv(args.categorized, "categorized")

    if df_rec is not None: eda_reconciled(df_rec, out_rec)
    if df_sc  is not None: eda_scored(df_sc,  out_sc)
    if df_cat is not None: eda_categorized(df_cat, out_cat)

    # cross-CSV only if at least 2 available
    if sum(x is not None for x in [df_rec, df_sc, df_cat]) >= 2:
        cross_dashboard((df_rec, df_sc, df_cat), args.out)

    print(f"\n✅  All figures saved under:  {args.out}/")
    print("""
  A_reconciled/
    A1_confidence_distributions.png
    A2_support_fraction.png
    A3_venue_breakdown.png
    A4_confidence_vs_support.png

  B_scored/
    B1_sound_vs_precise_per_model.png
    B2_sound_verdict_dist.png
    B3_precise_verdict_dist.png
    B4_yes_count_histograms.png
    B5_scored_heatmap.png
    B6_sound_precise_spider.png
    B7_scored_distributions.png

  C_categorized/
    C1_category_frequency.png
    C2_model_agreement_stacked.png
    C3_category_heatmap.png
    C4_categorized_distributions.png
    C5_venue_distribution.png
    C6_category_spider.png
    C7_bubble_chart.png

  D_cross_summary_dashboard.png
""")

if __name__ == "__main__":
    main()