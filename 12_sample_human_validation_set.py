"""Draw the stratified sample used for the human (pilot: agent) validation study.

Selects 200 ReACTs from the master reconciled set (local_history/final_set.csv),
25 per each of the 8 ReACT categories, using each ReACT's first-listed category
as its stratum (CATEGORY is a "|"-joined multi-label field; a ReACT can belong
to more than one category, so the first-listed category is treated as primary
for stratification purposes only -- it does not change the ReACT's actual
category assignments elsewhere in the paper).

ReACTs with CATEGORY in {"", "NONE"} are excluded from the sampling frame.

Output: results/human_validation/sample_200.csv, with one row per sampled
ReACT, including the resolved local path to its source-article PDF so that
annotators (human or agent) can open the original article.
"""

import csv
import os
import random

SEED = 42
PER_CATEGORY = 25
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

SOURCE_CSV = "local_history/final_set.csv"
PDF_BASE = "data/paper-set/infer-set"
OUT_CSV = "results/human_validation/sample_200.csv"


def primary_category(cat_field):
    cat_field = (cat_field or "").strip()
    if not cat_field or cat_field == "NONE":
        return None
    return cat_field.split("|")[0].strip()


def main():
    with open(SOURCE_CSV, newline="", encoding="utf-8", errors="replace") as fh:
        rows = list(csv.DictReader(fh))

    buckets = {c: [] for c in CATEGORIES}
    for idx, row in enumerate(rows):
        cat = primary_category(row["CATEGORY"])
        if cat in buckets:
            row["_row_id"] = idx
            row["_pdf_path"] = os.path.join(PDF_BASE, row["venue"])
            buckets[cat].append(row)

    rng = random.Random(SEED)
    sample = []
    for cat in CATEGORIES:
        pool = buckets[cat]
        if len(pool) < PER_CATEGORY:
            raise SystemExit(f"Category '{cat}' has only {len(pool)} eligible ReACTs (< {PER_CATEGORY})")
        sample.extend(rng.sample(pool, PER_CATEGORY))

    rng.shuffle(sample)

    fieldnames = [
        "item_id", "primary_category", "row_id", "cluster_id",
        "article_title", "venue", "article_link", "pdf_path",
        "actionable", "impact", "evidence",
        "model_sound_label", "model_precise_label",
    ]
    os.makedirs(os.path.dirname(OUT_CSV), exist_ok=True)
    with open(OUT_CSV, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=fieldnames)
        w.writeheader()
        for i, row in enumerate(sample, start=1):
            w.writerow({
                "item_id": f"H{i:03d}",
                "primary_category": primary_category(row["CATEGORY"]),
                "row_id": row["_row_id"],
                "cluster_id": row["cluster_id"],
                "article_title": row["article_title"],
                "venue": row["venue"],
                "article_link": row["article_link"],
                "pdf_path": row["_pdf_path"],
                "actionable": row["actionable"],
                "impact": row["impact"],
                "evidence": row["evidence"],
                "model_sound_label": row["SOUND"],
                "model_precise_label": row["PRECISE"],
            })

    print(f"Wrote {len(sample)} sampled ReACTs to {OUT_CSV}")
    for cat in CATEGORIES:
        n = sum(1 for r in sample if primary_category(r["CATEGORY"]) == cat)
        print(f"  {cat}: {n}")


if __name__ == "__main__":
    main()
