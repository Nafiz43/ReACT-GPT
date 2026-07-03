"""Reconcile the two independent annotators' round-1 labels, compute
inter-annotator agreement (Cohen's kappa), identify disagreements for the
round-2 reconciliation ("discussion") step, merge the final ground truth once
round-2 reconciled labels are available, and cross-check that ground truth
against the five-model-consensus pipeline's own SOUND/PRECISE verdicts to
compute precision/recall/F1.

Usage:
    python3 13_reconcile_human_validation.py round1
        -> reads annotator_A_round1.csv / annotator_B_round1.csv
        -> writes kappa report + disagreements_for_round2.csv

    python3 13_reconcile_human_validation.py round2
        -> reads annotator_A_round2.csv / annotator_B_round2.csv (reconciled
           labels for the disagreement subset only, same schema as round1)
        -> merges round1 agreements + round2 reconciled labels into the final
           ground truth, cross-checks against model consensus, writes the
           final report.
"""

import csv
import os
import sys
from collections import Counter

SAMPLE_CSV = "results/human_validation/sample_200.csv"
DIR = "results/human_validation"
FIELDS = ["actionable_valid", "evidence_valid", "impact_valid"]


def load_annotations(path):
    with open(path, newline="", encoding="utf-8") as fh:
        return {row["item_id"]: row for row in csv.DictReader(fh)}


def cohens_kappa(labels_a, labels_b):
    """labels_a/labels_b: parallel lists of category strings (same length, same item order)."""
    n = len(labels_a)
    if n == 0:
        return None
    categories = sorted(set(labels_a) | set(labels_b))
    po = sum(1 for a, b in zip(labels_a, labels_b) if a == b) / n
    count_a = Counter(labels_a)
    count_b = Counter(labels_b)
    pe = sum((count_a[c] / n) * (count_b[c] / n) for c in categories)
    if pe == 1.0:
        return 1.0 if po == 1.0 else 0.0
    return (po - pe) / (1 - pe)


def interpret_kappa(k):
    if k is None:
        return "n/a"
    if k < 0:
        return "poor (worse than chance)"
    if k < 0.20:
        return "slight"
    if k < 0.40:
        return "fair"
    if k < 0.60:
        return "moderate"
    if k < 0.80:
        return "substantial"
    return "almost perfect"


def composite_valid(row):
    """A ReACT is 'valid' overall iff actionable and evidence are YES, and
    impact is YES or N/A (an action-only ReACT with no stated impact is not
    penalized for lacking one)."""
    if row["actionable_valid"] != "YES":
        return "NO"
    if row["evidence_valid"] != "YES":
        return "NO"
    if row["impact_valid"] not in ("YES", "N/A"):
        return "NO"
    return "YES"


def round1():
    ann_a = load_annotations(os.path.join(DIR, "annotator_A_round1.csv"))
    ann_b = load_annotations(os.path.join(DIR, "annotator_B_round1.csv"))

    with open(SAMPLE_CSV, newline="", encoding="utf-8") as fh:
        sample_ids = [row["item_id"] for row in csv.DictReader(fh)]

    missing_a = [i for i in sample_ids if i not in ann_a]
    missing_b = [i for i in sample_ids if i not in ann_b]
    if missing_a or missing_b:
        print(f"WARNING: annotator A missing {len(missing_a)} items, annotator B missing {len(missing_b)} items")

    common_ids = [i for i in sample_ids if i in ann_a and i in ann_b]

    print(f"Items with both annotators' round-1 labels: {len(common_ids)}/{len(sample_ids)}\n")

    kappa_report = []
    for field in FIELDS:
        pairs = [
            (ann_a[i][field], ann_b[i][field])
            for i in common_ids
            if ann_a[i][field] != "N/A" and ann_b[i][field] != "N/A"
        ]
        la = [p[0] for p in pairs]
        lb = [p[1] for p in pairs]
        k = cohens_kappa(la, lb)
        agree = sum(1 for a, b in zip(la, lb) if a == b)
        kappa_report.append((field, len(pairs), agree, k))
        print(f"{field}: n={len(pairs)} raw_agreement={agree}/{len(pairs)} ({100*agree/len(pairs):.1f}%) "
              f"kappa={k:.3f} ({interpret_kappa(k)})")

    # Composite "is this ReACT valid overall" agreement/kappa
    comp_a = {i: composite_valid(ann_a[i]) for i in common_ids}
    comp_b = {i: composite_valid(ann_b[i]) for i in common_ids}
    la = [comp_a[i] for i in common_ids]
    lb = [comp_b[i] for i in common_ids]
    k_comp = cohens_kappa(la, lb)
    agree_comp = sum(1 for i in common_ids if comp_a[i] == comp_b[i])
    print(f"\ncomposite_valid (actionable & evidence & impact all pass): "
          f"n={len(common_ids)} raw_agreement={agree_comp}/{len(common_ids)} "
          f"({100*agree_comp/len(common_ids):.1f}%) kappa={k_comp:.3f} ({interpret_kappa(k_comp)})")

    # Disagreement set for round-2 reconciliation: any of the 4 judgments differ
    disagreement_ids = [
        i for i in common_ids
        if any(ann_a[i][f] != ann_b[i][f] for f in FIELDS) or comp_a[i] != comp_b[i]
    ]
    print(f"\n{len(disagreement_ids)}/{len(common_ids)} items have at least one field disagreement "
          f"-> writing round-2 reconciliation worklist")

    out_path = os.path.join(DIR, "disagreements_for_round2.csv")
    fieldnames = ["item_id"] + [f"A_{f}" for f in FIELDS] + [f"B_{f}" for f in FIELDS] + \
                 [f"A_{f}_rationale" for f in FIELDS] + [f"B_{f}_rationale" for f in FIELDS]
    with open(out_path, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=fieldnames)
        w.writeheader()
        for i in disagreement_ids:
            row = {"item_id": i}
            for f in FIELDS:
                row[f"A_{f}"] = ann_a[i][f]
                row[f"B_{f}"] = ann_b[i][f]
                row[f"A_{f}_rationale"] = ann_a[i][f"{f.replace('_valid','')}_rationale"]
                row[f"B_{f}_rationale"] = ann_b[i][f"{f.replace('_valid','')}_rationale"]
            w.writerow(row)
    print(f"Wrote {out_path}")

    report_path = os.path.join(DIR, "round1_kappa_report.csv")
    with open(report_path, "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["field", "n_items", "raw_agreement", "pct_agreement", "cohens_kappa", "interpretation"])
        for field, n, agree, k in kappa_report:
            w.writerow([field, n, agree, round(100 * agree / n, 1), round(k, 4), interpret_kappa(k)])
        w.writerow(["composite_valid", len(common_ids), agree_comp,
                     round(100 * agree_comp / len(common_ids), 1), round(k_comp, 4), interpret_kappa(k_comp)])
    print(f"Wrote {report_path}")


def round2():
    ann_a1 = load_annotations(os.path.join(DIR, "annotator_A_round1.csv"))
    ann_b1 = load_annotations(os.path.join(DIR, "annotator_B_round1.csv"))
    ann_a2 = load_annotations(os.path.join(DIR, "annotator_A_round2.csv"))
    ann_b2 = load_annotations(os.path.join(DIR, "annotator_B_round2.csv"))

    with open(SAMPLE_CSV, newline="", encoding="utf-8") as fh:
        sample = list(csv.DictReader(fh))

    ground_truth = {}
    for row in sample:
        i = row["item_id"]
        if i in ann_a2 or i in ann_b2:
            # reconciled in round 2: prefer agreement between the two reconciled labels;
            # if they still disagree after discussion, default to the conservative (NO) label
            # and flag it -- this default is a placeholder policy for the pilot and should be
            # revisited (e.g. adjudicated by a third rater) once real humans run this study.
            src_a = ann_a2.get(i, ann_a1[i])
            src_b = ann_b2.get(i, ann_b1[i])
        else:
            src_a = ann_a1[i]
            src_b = ann_b1[i]

        final = {"item_id": i, "flag": ""}
        for f in FIELDS:
            va, vb = src_a[f], src_b[f]
            if va == vb:
                final[f] = va
            else:
                final[f] = "NO" if "NO" in (va, vb) else va
                final["flag"] = "unresolved_disagreement_defaulted_conservative"
        final["composite_valid"] = composite_valid(final)
        ground_truth[i] = final

    gt_path = os.path.join(DIR, "ground_truth_200.csv")
    with open(gt_path, "w", newline="", encoding="utf-8") as fh:
        fieldnames = ["item_id"] + FIELDS + ["composite_valid", "flag"]
        w = csv.DictWriter(fh, fieldnames=fieldnames)
        w.writeheader()
        for i in [r["item_id"] for r in sample]:
            w.writerow(ground_truth[i])
    print(f"Wrote {gt_path}")

    unresolved = sum(1 for v in ground_truth.values() if v["flag"])
    print(f"{unresolved}/{len(ground_truth)} items still disagreed after round 2 "
          f"and were defaulted to the conservative (NO) label")

    # Cross-check against model consensus: an item "passes" model consensus iff
    # SOUND == YES and PRECISE == YES (both quality gates from Section 4.5 of the paper).
    tp = fp = fn = tn = 0
    rows_out = []
    for row in sample:
        i = row["item_id"]
        model_pass = (row["model_sound_label"] == "YES" and row["model_precise_label"] == "YES")
        human_pass = ground_truth[i]["composite_valid"] == "YES"
        if model_pass and human_pass:
            tp += 1
        elif model_pass and not human_pass:
            fp += 1
        elif not model_pass and human_pass:
            fn += 1
        else:
            tn += 1
        rows_out.append({
            "item_id": i, "model_pass": model_pass, "human_pass": human_pass,
        })

    precision = tp / (tp + fp) if (tp + fp) else float("nan")
    recall = tp / (tp + fn) if (tp + fn) else float("nan")
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else float("nan")
    accuracy = (tp + tn) / len(sample)

    print(f"\nModel-consensus vs. human ground truth (n={len(sample)}):")
    print(f"  TP={tp} FP={fp} FN={fn} TN={tn}")
    print(f"  Precision={precision:.3f}  Recall={recall:.3f}  F1={f1:.3f}  Accuracy={accuracy:.3f}")

    metrics_path = os.path.join(DIR, "alignment_metrics.csv")
    with open(metrics_path, "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["metric", "value"])
        for name, val in [("n", len(sample)), ("TP", tp), ("FP", fp), ("FN", fn), ("TN", tn),
                           ("precision", round(precision, 4)), ("recall", round(recall, 4)),
                           ("f1", round(f1, 4)), ("accuracy", round(accuracy, 4)),
                           ("unresolved_after_round2", unresolved)]:
            w.writerow([name, val])
    print(f"Wrote {metrics_path}")


if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else ""
    if mode == "round1":
        round1()
    elif mode == "round2":
        round2()
    else:
        print(__doc__)
        sys.exit(1)
