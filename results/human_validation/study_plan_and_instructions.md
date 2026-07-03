ReACT Human Validation Pilot — Study Plan and Annotator Instructions
=====================================================================

Purpose
-------
The paper's automated pipeline (five open-weight LLMs, ModeX-Set reconciliation,
consensus-based SOUND/PRECISE scoring and categorization) has never been
checked against independent human judgment. This study closes that gap by
having two human annotators independently review a stratified sample of 200
reconciled ReACTs, then reconcile their disagreements, producing a small
expert-annotated ground truth that we compare against the model consensus.

Sampling
--------
- Source: the master reconciled set, `local_history/final_set.csv` (3,837
  canonical ReACTs, after ModeX-Set reconciliation and 5-model SOUND/PRECISE/
  category consensus).
- Method: stratified random sample, 25 ReACTs per each of the 8 ReACT
  categories = 200 total. A ReACT's *first-listed* category (CATEGORY field
  is "|"-joined and non-exclusive) is used as its stratum for sampling
  purposes only; it does not change the ReACT's real, possibly multi-label,
  category assignment.
- Reproducible via: `12_sample_human_validation_set.py` (fixed random seed =
  42). Re-running it reproduces the exact same 200 items.
- Output: `results/human_validation/sample_200.csv` (full internal record,
  includes model labels — coordinator use only, never given to annotators).

Two annotation tasks
---------------------
We split the validation into two independent tasks so that a disagreement in
one does not contaminate the other, and because they require different
inputs (Task 1 needs the source PDF; Task 2 does not).

**Task 1 — ReACT validity (grounding check).**
For each of the three ReACT components (`actionable`, `impact`, `evidence`),
is it actually supported by the source article, or did the extraction
pipeline fabricate/distort it?

**Task 2 — Quality and category (text-only check).**
Judge `SOUND`, `PRECISE`, and category membership (8 yes/no judgments) from
the ReACT's own text alone — exactly the same inputs the 5-LLM consensus
pipeline used (no need to consult the source article).

Each task has its own annotator hand-off files (see "Files to send to
annotators" below) and produces its own row in the paper (Tables X and Y —
see "How this feeds back into the paper").

Annotators and independence protocol
-------------------------------------
- 2 annotators per task (can be the same 2 people for both tasks, or
  different pairs — your choice; keep it consistent per task).
- **Round 1 (independent):** each annotator works through all 200 items
  alone, with no access to the other annotator's answers, and without
  looking at the model's own SOUND/PRECISE/category labels (those are
  deliberately excluded from the hand-off files to avoid anchoring bias).
- **Round 2 (reconciliation):** after both Round-1 files are collected, the
  coordinator identifies every item where the two annotators disagree on any
  field. Both annotators are shown each other's Round-1 answer and rationale
  for *only those disagreement items* and asked to discuss and converge on a
  single final answer per item. If they still cannot agree after discussion,
  default to the more conservative (NO / not-valid) label and flag the item
  as `unresolved_disagreement_defaulted_conservative` — this is a documented
  placeholder policy; a third adjudicator is a better long-run solution if
  unresolved counts turn out to be non-trivial.

Metrics
-------
1. **Inter-annotator reliability (Round 1, before reconciliation):** Cohen's
   kappa, computed separately for each judged field (`actionable_valid`,
   `evidence_valid`, `impact_valid` for Task 1; `sound_valid`, `precise_valid`,
   and each of the 8 category flags for Task 2), plus a composite "overall
   valid" kappa. This measures whether the *task itself* is well-specified
   enough for two independent people to agree, before any discussion.
2. **Alignment with model consensus (after Round 2 reconciliation):** treat
   the reconciled human labels as ground truth and the model consensus's own
   verdict (SOUND=YES & PRECISE=YES for Task 2's quality label; category
   flags directly) as the "prediction." Compute Precision, Recall, and F1.
   For Task 1 there's no direct model-consensus analogue for
   "actionable/evidence/impact grounded in source" (the model pipeline never
   explicitly labels this), so Task 1's headline number is descriptive: the
   percentage of ReACTs where all three components were judged grounded.

Reproducible code (already implemented, ready to run once the annotator
files come back):
- `13_reconcile_human_validation.py round1` — computes Round-1 kappa per
  field + composite, and writes the Round-2 disagreement worklist.
- `13_reconcile_human_validation.py round2` — merges Round-2 reconciled
  labels into final ground truth, cross-checks against model consensus,
  writes Precision/Recall/F1.
(Currently written for Task 1's schema; a `task2_reconcile` variant using the
same `cohens_kappa()` helper should be added once Task 2 files are back — the
category-agreement kappas are just 8 more binary fields through the same
function.)

Files to send to annotators
----------------------------
Do **not** send `sample_200.csv` or `coordinator_key_200.csv` directly to
annotators — both contain the model's own labels, which would bias judgment.
Send instead:

- `task1_react_validity_annotator1.csv` → Annotator 1, Task 1
- `task1_react_validity_annotator2.csv` → Annotator 2, Task 1
- `task2_quality_category_annotator1.csv` → Annotator 1, Task 2
- `task2_quality_category_annotator2.csv` → Annotator 2, Task 2

Each Task 1 row also carries `source_pdf_path` (path relative to the repo
root, under `data/paper-set/infer-set/...`) and `doi_or_url` (best-effort DOI
link — note that for about 15% of sampled rows this DOI string is truncated/
malformed due to a pre-existing data-quality issue in the original
extraction; the local PDF path is the reliable pointer to the source
article, always verified to exist for all 200 sampled rows).

Instructions to paste to each human annotator
------------------------------------------------

> You've been given a CSV of 200 short "ReACTs" — actionable recommendations
> that an AI pipeline extracted from software-engineering research papers,
> for a study validating whether that pipeline's output can be trusted. Work
> through every row independently; do not discuss your answers with the other
> annotator until asked to in a later reconciliation round.
>
> **If you have the Task 1 (`task1_react_validity_...csv`) file:** For each
> row, open the source PDF at `source_pdf_path` (or use `doi_or_url` if the
> PDF isn't available to you) and decide, for each of `actionable`, `impact`,
> and `evidence`: is this actually said/shown/reported in the article, or is
> it fabricated/distorted? Fill in `actionable_valid`, `evidence_valid`,
> `impact_valid` with YES or NO (use N/A for `impact_valid` only if the
> `impact` cell is blank — some ReACTs genuinely have no stated impact). Add
> a one-sentence rationale for each in the matching `*_rationale` column.
>
> **If you have the Task 2 (`task2_quality_category_...csv`) file:** You do
> not need the source PDF for this one — judge purely from the `actionable`
> / `impact` / `evidence` text given.
> - `sound_valid` (YES/NO): is the recommendation logically consistent with
>   its stated impact and evidence — a plausible course of action, not
>   self-contradictory?
> - `precise_valid` (YES/NO): does it name a concrete, schedulable action
>   (who/what, ideally when), rather than just restating a goal without
>   saying how to achieve it?
> - The 8 `cat_*` columns (YES/NO each, not mutually exclusive — a ReACT can
>   fit zero, one, or several): judge against the category definitions
>   provided separately (see `-TOSEM-ReACT-LLM-v02/02_appendix.tex`,
>   "Definition of ReACT Categories," or ask the coordinator for a
>   standalone copy of just those 8 definitions).
>
> Return your completed CSV to the coordinator. You'll hear back with a short
> list of items where you and the other annotator disagreed, and be asked to
> discuss and settle on one final answer for just those items.

How this feeds back into the paper
------------------------------------
The manuscript (`-TOSEM-ReACT-LLM-v02/01-main_no_template.tex`) now describes
this pilot and includes two placeholder tables, to be filled in once real
annotator results are collected:

- **Table (Human Validation — ReACT Validity):** Task 1 results — percentage
  of sampled ReACTs with actionable/evidence/impact judged grounded in the
  source article, Round-1 Cohen's kappa per field, and the reconciled
  ground-truth rate.
- **Table (Human Validation — Quality and Category Agreement):** Task 2
  results — SOUND/PRECISE human-vs-model Precision/Recall/F1, per-category
  agreement kappa, and reconciled human-vs-model category alignment.

Both tables are marked in the manuscript as placeholders pending the actual
human study (see the `\color{red}` placeholder values and the accompanying
"Human Validation Pilot" subsection).
