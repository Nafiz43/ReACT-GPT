"""
ReACT-GPT SLR — LaTeX Table Generator
=======================================
Generates two .tex files from final_set.csv:

  Table 1  react_summary_table.tex     — one row per category (all models combined)
  Table 2  react_per_model_table.tex   — one row per model    (all categories combined)

Run:
    python generate_tables.py
"""

import pandas as pd
import numpy as np
import re

# ============================================================
# CONFIGURATION
# ============================================================

INPUT_CSV     = "/Users/nafiz43/Documents/GitHub/ReACT-GPT/local_history/final_set.csv"
OUT_SUMMARY   = "/Users/nafiz43/Documents/GitHub/ReACT-GPT/local_history/react_summary_table.tex"
OUT_PER_MODEL = "/Users/nafiz43/Documents/GitHub/ReACT-GPT/local_history/react_per_model_table.tex"

# ============================================================
# CANONICAL CATEGORIES
# ============================================================

CANONICAL_CATEGORIES = [
    "New Contributor Onboarding and Involvement",
    "Code Standards and Maintainability",
    "Automated Testing and Quality Assurance",
    "Community Collaboration and Engagement",
    "Documentation Practices",
    "Project Management and Governance",
    "Security Best Practices and Legal Compliance",
    "CI/CD and DevOps Automation",
]

# ============================================================
# MODEL REGISTRY  (keyword, display label, latex label)
# ============================================================

MODELS = [
    ("qwen",     "Qwen3.6",     r"\textsc{Qwen3.6}"),
    ("gpt",      "GPT-oss",     r"\textsc{GPT-oss}"),
    ("deepseek", "DeepSeek-R1", r"\textsc{DeepSeek-R1}"),
    ("gemma",    "Gemma4",      r"\textsc{Gemma4}"),
    ("mixtral",  "Mixtral",     r"\textsc{Mixtral}"),
]

# ============================================================
# LATEX CATEGORY SHORT LABELS  (Table 1 only)
# ============================================================

SHORT_CAT = {
    "New Contributor Onboarding and Involvement":
        r"\begin{tabular}[c]{@{}l@{}}New Contributor\\Onboarding \& Involvement\end{tabular}",
    "Code Standards and Maintainability":
        r"\begin{tabular}[c]{@{}l@{}}Code Standards \&\\Maintainability\end{tabular}",
    "Automated Testing and Quality Assurance":
        r"\begin{tabular}[c]{@{}l@{}}Automated Testing \&\\Quality Assurance\end{tabular}",
    "Community Collaboration and Engagement":
        r"\begin{tabular}[c]{@{}l@{}}Community Collaboration\\\& Engagement\end{tabular}",
    "Documentation Practices":
        r"\begin{tabular}[c]{@{}l@{}}Documentation\\Practices\end{tabular}",
    "Project Management and Governance":
        r"\begin{tabular}[c]{@{}l@{}}Project Management\\\& Governance\end{tabular}",
    "Security Best Practices and Legal Compliance":
        r"\begin{tabular}[c]{@{}l@{}}Security Best Practices\\\& Legal Compliance\end{tabular}",
    "CI/CD and DevOps Automation":
        r"\begin{tabular}[c]{@{}l@{}}CI/CD \&\\DevOps Automation\end{tabular}",
}

# ============================================================
# LOAD & CLEAN
# ============================================================

def load_and_clean(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)

    def normalize_bool(x):
        if pd.isna(x):
            return False
        return str(x).strip().lower() in {"1", "true", "yes", "y", "t"}

    def text_exists(x):
        if pd.isna(x):
            return False
        return str(x).strip().lower() not in {"", "none", "nan", "null", "n/a"}

    for col in ["SOUND", "PRECISE"]:
        df[col] = df[col].apply(normalize_bool)
    for col in ["impact", "evidence"]:
        df[col] = df[col].apply(text_exists)

    df["avg_confidence"] = pd.to_numeric(df["avg_confidence"], errors="coerce")
    return df

# ============================================================
# EXPAND MULTI-CATEGORY ROWS  (needed for Table 1 only)
# ============================================================

def expand_categories(df: pd.DataFrame) -> pd.DataFrame:
    expanded = []
    for _, row in df.iterrows():
        raw   = str(row["CATEGORY"])
        parts = re.split(r";|,|\|", raw)
        parts = [p.strip() for p in parts if p.strip()]
        matched = []
        for p in parts:
            for canon in CANONICAL_CATEGORIES:
                if canon.lower() in p.lower() or p.lower() in canon.lower():
                    matched.append(canon)
        for cat in list(set(matched)):
            new_row             = row.copy()
            new_row["CATEGORY"] = cat
            expanded.append(new_row)
    return pd.DataFrame(expanded)

# ============================================================
# HELPERS
# ============================================================

def latex_escape(text: str) -> str:
    return (str(text)
            .replace("&", r"\&")
            .replace("%", r"\%")
            .replace("_", r"\_")
            .replace("#", r"\#"))

def pct(count: int, total: int) -> str:
    frac = (count / total * 100) if total else 0.0
    return rf"{count} ({frac:.2f}\%)"

def model_in_row(cell_value: str, keyword: str) -> bool:
    """Case-insensitive token match; handles list-repr and pipe/comma formats."""
    tokens = re.split(r"[|,\s'\[\]\"]+", str(cell_value).lower())
    return any(keyword in tok for tok in tokens if tok)

def aggregate_sub(sub: pd.DataFrame) -> dict:
    n     = len(sub)
    sound = int(sub["SOUND"].sum())
    prec  = int(sub["PRECISE"].sum())
    imp   = int(sub["impact"].sum())
    evid  = int(sub["evidence"].sum())
    comp  = int((sub["SOUND"] & sub["PRECISE"] & sub["impact"] & sub["evidence"]).sum())
    conf  = sub["avg_confidence"].mean()
    return dict(
        reacts   = n,
        sound    = pct(sound, n),
        precise  = pct(prec,  n),
        impact   = pct(imp,   n),
        evidence = pct(evid,  n),
        complete = pct(comp,  n),
        conf     = f"{conf:.2f}",
    )

# ============================================================
# TABLE 1 — CATEGORY ROWS  (all models combined)
# ============================================================

def generate_summary_table(expanded_df: pd.DataFrame) -> str:
    """One row per canonical category, all models pooled."""

    lines = [
        r"\begin{table*}[ht]",
        r"\centering",
        r"\caption{Summary of ReACT Derivations Across Categories}",
        r"\label{table:ReACTCategories}",
        r"\def\arraystretch{1.2}",
        r"\resizebox{\textwidth}{!}{",
        r"\begin{tabular}{l|cccccc|c}",
        r"\hline",
        r"\textbf{ReACT Category} & \textbf{\# Articles} & \textbf{\# ReACTs} & "
        r"\textbf{SOUND} & \textbf{PRECISE} & \textbf{Impact} & "
        r"\textbf{Evidence} & \textbf{Complete} & \textbf{Confidence} \\",
        r"\hline",
    ]

    cat_rows = []
    for cat in CANONICAL_CATEGORIES:
        sub = expanded_df[expanded_df["CATEGORY"] == cat]
        if len(sub) == 0:
            continue
        agg        = aggregate_sub(sub)
        n_articles = sub["article_title"].nunique()
        cat_rows.append((cat, n_articles, agg))

    for i, (cat, n_articles, agg) in enumerate(cat_rows):
        sep      = r" \\ \hdashline" if i < len(cat_rows) - 1 else r" \\"
        cat_cell = SHORT_CAT.get(cat, latex_escape(cat))
        lines.append(
            r"\textit{" + cat_cell + "}"
            f" & {n_articles}"
            f" & {agg['reacts']}"
            f" & {agg['sound']}"
            f" & {agg['precise']}"
            f" & {agg['impact']}"
            f" & {agg['evidence']}"
            f" & {agg['complete']}"
            f" & {agg['conf']}"
            + sep
        )

    lines += [r"\hline", r"\end{tabular}", r"}", r"\end{table*}"]
    return "\n".join(lines)

# ============================================================
# TABLE 2 — MODEL ROWS  (one row per model, no category/articles column)
# ============================================================

def generate_per_model_table(df: pd.DataFrame) -> str:
    """One row per model, aggregated across all categories and articles."""

    lines = [
        r"\begin{table*}[ht]",
        r"\centering",
        r"\caption{Summary of ReACT Derivations Per Model}",
        r"\label{table:ReACTPerModel}",
        r"\def\arraystretch{1.2}",
        r"\resizebox{\textwidth}{!}{",
        r"\begin{tabular}{l|ccccc|c}",
        r"\hline",
        r"\textbf{Model} & \textbf{\# ReACTs} & \textbf{SOUND} & \textbf{PRECISE} & "
        r"\textbf{Impact} & \textbf{Evidence} & \textbf{Complete} & \textbf{Confidence} \\",
        r"\hline",
    ]

    model_rows = []
    for keyword, display_label, latex_label in MODELS:
        mask = df["models_present"].apply(lambda v: model_in_row(str(v), keyword))
        sub  = df[mask]
        if len(sub) == 0:
            print(f"    ⚠  {display_label}: no matching rows — skipped")
            continue
        agg = aggregate_sub(sub)
        model_rows.append((latex_label, agg))
        print(f"    {display_label:>12s}  →  {agg['reacts']} ReACTs")

    for i, (latex_label, agg) in enumerate(model_rows):
        sep = r" \\ \hdashline" if i < len(model_rows) - 1 else r" \\"
        lines.append(
            f"{latex_label}"
            f" & {agg['reacts']}"
            f" & {agg['sound']}"
            f" & {agg['precise']}"
            f" & {agg['impact']}"
            f" & {agg['evidence']}"
            f" & {agg['complete']}"
            f" & {agg['conf']}"
            + sep
        )

    lines += [r"\hline", r"\end{tabular}", r"}", r"\end{table*}"]
    return "\n".join(lines)

# ============================================================
# MAIN
# ============================================================

def main():
    print("\n📂  Loading and cleaning data …")
    df = load_and_clean(INPUT_CSV)
    print(f"    {len(df)} rows loaded.")

    # Table 1 needs category-expanded rows
    print("\n🔀  Expanding multi-category rows …")
    expanded_df = expand_categories(df)
    print(f"    {len(expanded_df)} rows after expansion.")

    print("\n📝  Generating Table 1 — Category Summary …")
    tex1 = generate_summary_table(expanded_df)
    with open(OUT_SUMMARY, "w") as f:
        f.write(tex1)
    print(f"    ✔  Saved → {OUT_SUMMARY}")

    # Table 2 uses raw (non-expanded) df so each ReACT is counted once per model
    print("\n📝  Generating Table 2 — Per-Model Summary …")
    tex2 = generate_per_model_table(df)
    with open(OUT_PER_MODEL, "w") as f:
        f.write(tex2)
    print(f"    ✔  Saved → {OUT_PER_MODEL}")

    print("\n✅  Done.")
    print(f"       {OUT_SUMMARY}")
    print(f"       {OUT_PER_MODEL}")

if __name__ == "__main__":
    main()