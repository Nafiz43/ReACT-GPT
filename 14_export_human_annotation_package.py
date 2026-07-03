"""Build the hand-off package for the real human-annotator validation study.

Reads the internal sampling artifacts (results/human_validation/sample_200.csv
and sample_200_model_categories.csv, produced by 12_sample_human_validation_set.py)
and produces:

  1. results/human_validation/coordinator_key_200.csv
     -- FULL record per item (all 8 model category flags + model SOUND/PRECISE
        labels + row_id/cluster_id for traceability). NOT for annotators --
        this is what the study coordinator uses later to compute alignment
        metrics against the model consensus. Keeping this out of the
        annotator-facing files prevents anchoring bias.

  2. results/human_validation/task1_react_validity_annotatorN.csv (N=1,2)
     -- Task 1 hand-off: grounding check (actionable/evidence/impact present
        in the source article?). Two identical blank copies, one per
        annotator. No model labels included.

  3. results/human_validation/task2_quality_category_annotatorN.csv (N=1,2)
     -- Task 2 hand-off: SOUND / PRECISE / 8-category judgments made from the
        ReACT text alone (no source PDF needed). Two identical blank copies.

Run after 12_sample_human_validation_set.py.
"""

import csv
import os

DIR = "results/human_validation"
SAMPLE_CSV = os.path.join(DIR, "sample_200.csv")
MODEL_CATS_CSV = os.path.join(DIR, "sample_200_model_categories.csv")

CAT_SLUGS = [
    "cat_onboarding", "cat_code_standards", "cat_testing_qa", "cat_community",
    "cat_documentation", "cat_governance", "cat_security", "cat_cicd",
]


def load_csv(path):
    with open(path, newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def main():
    sample = load_csv(SAMPLE_CSV)
    model_cats = {r["item_id"]: r for r in load_csv(MODEL_CATS_CSV)}

    # 1. Coordinator key (full record, kept private from annotators)
    key_fields = [
        "item_id", "row_id", "cluster_id", "primary_category", "article_title",
        "venue", "pdf_path", "model_sound_label", "model_precise_label",
    ] + CAT_SLUGS
    key_path = os.path.join(DIR, "coordinator_key_200.csv")
    with open(key_path, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=key_fields)
        w.writeheader()
        for row in sample:
            out = {k: row[k] for k in ["item_id", "row_id", "cluster_id", "primary_category",
                                        "article_title", "venue", "pdf_path",
                                        "model_sound_label", "model_precise_label"]}
            out.update({k: model_cats[row["item_id"]]["model_" + k] for k in CAT_SLUGS})
            w.writerow(out)
    print(f"Wrote {key_path} (coordinator-only -- do NOT share with annotators)")

    # 2. Task 1: ReACT validity (actionable / evidence / impact grounding)
    task1_fields = [
        "item_id", "article_title", "source_pdf_path", "doi_or_url",
        "actionable", "impact", "evidence",
        "actionable_valid", "actionable_rationale",
        "evidence_valid", "evidence_rationale",
        "impact_valid", "impact_rationale",
    ]
    for n in (1, 2):
        path = os.path.join(DIR, f"task1_react_validity_annotator{n}.csv")
        with open(path, "w", newline="", encoding="utf-8") as fh:
            w = csv.DictWriter(fh, fieldnames=task1_fields)
            w.writeheader()
            for row in sample:
                w.writerow({
                    "item_id": row["item_id"],
                    "article_title": row["article_title"],
                    "source_pdf_path": row["pdf_path"],
                    "doi_or_url": row["article_link"],
                    "actionable": row["actionable"],
                    "impact": row["impact"],
                    "evidence": row["evidence"],
                    "actionable_valid": "", "actionable_rationale": "",
                    "evidence_valid": "", "evidence_rationale": "",
                    "impact_valid": "", "impact_rationale": "",
                })
        print(f"Wrote {path}")

    # 3. Task 2: SOUND / PRECISE / category (text-only, no PDF needed)
    task2_fields = (
        ["item_id", "actionable", "impact", "evidence",
         "sound_valid", "sound_rationale", "precise_valid", "precise_rationale"]
        + CAT_SLUGS + ["category_rationale"]
    )
    for n in (1, 2):
        path = os.path.join(DIR, f"task2_quality_category_annotator{n}.csv")
        with open(path, "w", newline="", encoding="utf-8") as fh:
            w = csv.DictWriter(fh, fieldnames=task2_fields)
            w.writeheader()
            for row in sample:
                out = {
                    "item_id": row["item_id"], "actionable": row["actionable"],
                    "impact": row["impact"], "evidence": row["evidence"],
                    "sound_valid": "", "sound_rationale": "",
                    "precise_valid": "", "precise_rationale": "",
                    "category_rationale": "",
                }
                out.update({k: "" for k in CAT_SLUGS})
                w.writerow(out)
        print(f"Wrote {path}")


if __name__ == "__main__":
    main()
