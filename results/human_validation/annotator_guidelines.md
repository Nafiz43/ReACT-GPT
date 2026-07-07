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
