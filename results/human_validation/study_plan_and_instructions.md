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
- `13_reconcile_human_validation.py round1` — reads both annotators'
  completed `annotatorN_validation.xlsx` workbooks directly, computes
  Round-1 kappa per field + composite for **both** Task 1 and Task 2, and
  writes a single `disagreements_for_round2.xlsx` worklist (both annotators'
  answers side by side, with blank `final_*` columns to fill in during the
  reconciliation discussion).
- `13_reconcile_human_validation.py round2` — reads the filled-in
  `disagreements_for_round2.xlsx`, merges Round-1 agreements + Round-2
  reconciled answers into the final ground truth for both tasks (any
  item left blank in `final_*` after discussion defaults to the
  conservative/NO label and is counted as unresolved), cross-checks Task 2
  against the model consensus in `coordinator_key_200.csv` to compute
  Precision/Recall/F1 per dimension, and **writes the results straight into
  the paper**: it regenerates
  `-TOSEM-ReACT-LLM-v02/tables/human_validation_task1.tex` and
  `human_validation_task2.tex` with the real numbers (replacing the
  `[PENDING]` placeholders) and removes the "Status: pending" red-text note
  from `01-main_no_template.tex`. No manual copy-pasting of numbers into the
  manuscript is needed.
- `15_build_annotator_workbooks.py` — (re)builds the two blank
  `annotatorN_validation.xlsx` hand-off files from the underlying CSVs, if
  the sample ever needs to be regenerated.

Only the validity/category judgments themselves are scored (Cohen's kappa,
and Precision/Recall/F1 against the model consensus) — free-text rationale
is not analyzed anywhere in the pipeline, so the hand-off files do not
collect it.

Both scripts need the `openpyxl` Python package installed
(`pip3 install openpyxl`).

File to send to each annotator
-------------------------------
Do **not** send `sample_200.csv` or `coordinator_key_200.csv` directly to
annotators — both contain the model's own labels, which would bias judgment.

Instead, each annotator gets **one Excel file with a single sheet**
covering both tasks:

- `annotator1_validation.xlsx` → Annotator 1
- `annotator2_validation.xlsx` → Annotator 2

Each workbook has two tabs (visible at the bottom of the Excel window):
`Legend` and `ReACT Validation`. The top row and the `item_id` column are
frozen on the `ReACT Validation` sheet so column headers and the row
identifier stay visible while scrolling. Blue-shaded columns are read-only
context (what the pipeline extracted); yellow-shaded columns are where the
annotator picks their answer, each via a dropdown (click the cell, then the
small arrow, to pick YES/NO/N/A) — no free-text rationale is collected,
only the judgment itself.

The sheet also carries `source_pdf_path` (path relative to the repo root,
under `data/paper-set/infer-set/...`) and `doi_or_url` (DOI link for the
source article; see the "Sample metadata quality" note below — both fields
are now verified valid for all 200 sampled rows).

Instructions to send each human annotator (plain-language, non-technical)
---------------------------------------------------------------------------

> You've been sent one Excel file, `annotatorN_validation.xlsx` (N is your
> annotator number). It contains 200 short "ReACTs" — actionable
> recommendations that an AI pipeline pulled out of software-engineering
> research papers — for a study checking whether that pipeline's output can
> be trusted. Please work through every row on your own; don't discuss your
> answers with the other annotator until asked to, later, in a
> reconciliation round.
>
> **Opening the file.** Open it in Excel (or Google Sheets/Numbers — any
> spreadsheet program that opens `.xlsx` works). At the bottom of the window
> you'll see two tabs: `Legend` and `ReACT Validation`. Start with the
> `Legend` tab — it's a one-glance reminder of what the colors mean:
> - **Blue columns** = context from the pipeline. Read them, but don't edit
>   them.
> - **Yellow columns** = your answers. Fill these in by clicking the cell,
>   then the small dropdown arrow that appears, and picking from the list —
>   you never have to type "YES" or "NO" by hand. No written explanation is
>   needed anywhere; just the picked answer.
>
> Then go to the `ReACT Validation` tab, where all 200 rows live. The header
> row and the leftmost `item_id` column stay pinned in place as you scroll,
> so you can always see the column names and which row you're on. Each row
> has two groups of yellow answer columns:
>
> **Grounding check (needs the source PDF): `actionable_valid`,
> `evidence_valid`, `impact_valid`.** For each row, open the PDF at
> `source_pdf_path` (or use the `doi_or_url` link if you don't have the
> PDF) and decide, for each of `actionable`, `impact`, and `evidence`: is
> this actually said/shown/reported in the article, or was it made
> up/distorted? Pick YES or NO (use N/A for `impact_valid` only if the
> `impact` cell is blank — some ReACTs genuinely have no stated impact).
>
> **Quality and category check (no PDF needed): `sound_valid`,
> `precise_valid`, and the 8 `cat_*` columns.** Judge purely from the
> `actionable` / `impact` / `evidence` text shown in that same row.
> - `sound_valid` (YES/NO): is the recommendation logically consistent with
>   its stated impact and evidence — a plausible course of action, not
>   self-contradictory?
> - `precise_valid` (YES/NO): does it name a concrete, schedulable action
>   (who/what, ideally when), rather than just restating a goal without
>   saying how to achieve it?
> - The 8 `cat_*` columns (YES/NO each — a ReACT can fit zero, one, or
>   several, they're not mutually exclusive): judge against the category
>   definitions provided separately (see `-TOSEM-ReACT-LLM-v02/02_appendix.tex`,
>   "Definition of ReACT Categories," or ask the coordinator for a
>   standalone copy of just those 8 definitions).
>
> **When you're done**, save the file (keep it as `.xlsx`, same filename)
> and send it back to the coordinator. You'll hear back with a short list of
> items where you and the other annotator disagreed, and be asked to discuss
> and settle on one final answer for just those items.

Sample metadata quality
-----------------------
An earlier pass of `article_title` and `doi_or_url` for the 200 sampled
rows was unreliable: both fields were regex-scraped off the PDF-to-markdown
conversion in `00_preprocess_articles.py` (first Markdown heading found, and
first URL-shaped string found), which frequently picked up running headers,
page numbers, or truncated footer text instead of the real title/DOI (e.g.
item H001's title was just the string `"14"`, and its "DOI" was the
dangling fragment `"http://www.seng"`).

Both fields have since been corrected for **all 200/200 sampled rows** by
`16_fix_sample_metadata.py`, which re-derives them directly from each
sampled PDF and cross-checks them against Crossref (the DOI registration
agency's public API):
1. Extract page 1-2 text from the PDF and look for an explicit DOI string
   printed on the page itself (most ACM/IEEE papers print one in the
   header or footer); if found, resolve it against Crossref to get the
   publisher's own canonical title. **141/200 items** resolved this way.
2. For the rest, build a title candidate from the existing title field,
   the PDF's own metadata, or text near the byline, and run a Crossref
   bibliographic search — requiring either a near-exact title match or
   corroboration from an author name appearing in the PDF's byline (this
   catches cases like a generic title such as "Release Planning" that
   would otherwise collide with an unrelated paper of the same name).
   **59/200 items** resolved this way.

Final result: 200/200 resolved (199 at high confidence, 1 at medium
confidence — manually spot-checked and confirmed correct: that PDF is
itself a workshop's front-matter/table-of-contents page, not a
research paper, and Crossref's own record for it matches). Full detail
per item (old vs. new title/DOI, method, confidence) is in
`metadata_fix_report.csv`; the pre-fix version of the sample is preserved
as `sample_200.csv.prefix_backup`.

**Known pre-existing corpus issue (unrelated to this fix, left as-is):**
`data/paper-set/infer-set/ICSE/ICSE 21/43.pdf` and `.../44.pdf` are
byte-for-byte duplicate downloads of the same paper ("An Empirical
Assessment of Global COVID-19 Contact Tracing Applications") — both
correctly resolve to the same DOI, they are simply the same source article
counted as two separate PDFs in the corpus. This means 4 of the 200
sampled rows (H015, H029, H050, H093) are ReACTs drawn from what is
actually a single article. Flagged here for awareness; not corrected since
fixing it would change the sample composition (would need re-sampling a
replacement item from `12_sample_human_validation_set.py`'s pool).

After re-running this script, `14_export_human_annotation_package.py` and
`15_build_annotator_workbooks.py` were re-run to propagate the corrected
title/DOI into `coordinator_key_200.csv` and both annotators' `.xlsx`
hand-off files.

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
