"""
ReACT Scorer — Soundness, Preciseness & Category Assignment
=============================================================
Optimizations applied (single-GPU, one model at a time):
  1. keep_alive      — model stays hot in VRAM for the entire run
  2. Model-first batching — process ALL rows for one model before loading next
                            (eliminates repeated model swaps — biggest speedup)
  3. Per-model num_predict — deepseek-r1 gets more tokens for <think> blocks;
                             others are capped tight
  4. Streaming + early exit — stop generating the moment YES/NO is seen

Two modes:

  TASK 1 — Sound & Precise scoring  (--mode score)
  -------------------------------------------------
  Input : CSV with columns: actionable, impact, evidence
  Output: same CSV + per-model binary decisions + majority verdicts

  TASK 2 — Category assignment  (--mode categorize)
  --------------------------------------------------
  Input : CSV with column: actionable
  Output: new CSV + per-model per-category decisions + assigned_categories

Majority rule: >= 3 / 5 models must agree.
An actionable can belong to multiple categories.

Usage:
  python react_scorer.py --mode score \\
      --input  /data/.../reconciled_actionables.csv \\
      --output ./reconciled_actionables_scored.csv

  python react_scorer.py --mode categorize \\
      --input  /data/.../reconciled_actionables_scored.csv \\
      --output ./reconciled_actionables_categorized.csv

Flags:
  --ollama_url   http://localhost:11434
  --keep_alive   120m          how long to keep model in VRAM (default 120m)
  --timeout      600           seconds per Ollama call
  --delay        0.0           sleep between calls (usually 0 with streaming)
  --resume                     skip already-processed rows
  --limit        N             process only first N rows (testing)
"""

import argparse
import json
import logging
import re
import time
from pathlib import Path

import pandas as pd
import requests
from tqdm import tqdm
from tqdm.contrib.logging import logging_redirect_tqdm

# ─────────────────────────────────────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────────────────────────────────────

MODELS = [
    "qwen3.6:35b",
    "gpt-oss:20b",
    "deepseek-r1:32b",
    "gemma4:31b",
    "mixtral:8x7b",
]

# Per-model token budgets.
# deepseek-r1 emits <think>...</think> blocks before YES/NO — needs more room.
# Others are well-behaved with a tight cap.
MODEL_NUM_PREDICT = {
    "deepseek-r1:32b": 512,
    "qwen3.6:35b":      64,
    "gpt-oss:20b":      32,
    "gemma4:31b":       32,
    "mixtral:8x7b":     16,
}

# ── Category definitions (verbatim from the paper appendix) ──────────────────
CATEGORIES = {
    "New Contributor Onboarding and Involvement": (
        "This category focuses on ensuring that new contributors can easily join, understand, "
        "and meaningfully contribute to the project. "
        "Criteria: (a) Actionable facilitates the integration of new contributors by providing "
        "mentorship, onboarding materials, or simplifying the contribution process; "
        "(b) Actionable relates to improving project documentation or offering better support "
        "mechanisms for first-time contributors; "
        "(c) Actionable helps build a welcoming, inclusive, and open culture for new participants."
    ),
    "Code Standards and Maintainability": (
        "This category deals with ensuring that the codebase adheres to established standards, "
        "making it easier to maintain and scale. It includes efforts to ensure code readability, "
        "modularity, and compliance with coding best practices. "
        "Criteria: (a) Actionable relates to improving the quality, readability, or structure of "
        "the codebase; (b) Actionable includes efforts to enforce coding guidelines, refactor code "
        "for better maintainability, or reduce technical debt; "
        "(c) Actionable includes the use of linters, formatters, or static code analysis tools."
    ),
    "Automated Testing and Quality Assurance": (
        "This category focuses on ensuring the project's robustness and reliability through "
        "automated testing practices, such as unit, integration, and end-to-end tests. It also "
        "includes broader quality assurance activities. "
        "Criteria: (a) Actionable involves the implementation or improvement of automated testing "
        "frameworks and testing strategies; (b) Actionable includes practices that ensure the "
        "detection of bugs early in the development cycle and ensure high-quality releases."
    ),
    "Community Collaboration and Engagement": (
        "This category deals with activities that foster collaboration, communication, and "
        "engagement within the OSS community. It includes practices for keeping the community "
        "active and involved. "
        "Criteria: (a) Actionable aims to improve communication between contributors, maintainers, "
        "and users; (b) Actionable involves organizing community-driven events, discussions, or "
        "collaborations, as well as platforms to enhance transparency and teamwork; "
        "(c) Actionable relates to tools and processes for better community governance and "
        "decision-making."
    ),
    "Documentation Practices": (
        "This category focuses on ensuring that the project's documentation is thorough, "
        "up-to-date, and easily accessible. Documentation practices are crucial for both current "
        "and future contributors. "
        "Criteria: (a) Actionable focuses on improving the quality, clarity, or accessibility of "
        "project documentation, such as user guides, API references, or contributor guides; "
        "(b) Actionable includes practices for keeping documentation synchronized with the "
        "codebase and ensuring it meets the needs of different stakeholders; "
        "(c) Actionable involves translation efforts or making documentation more accessible to "
        "non-expert audiences."
    ),
    "Project Management and Governance": (
        "This category deals with the governance structure and project management practices that "
        "keep the project organized, transparent, and sustainable over the long term. "
        "Criteria: (a) Actionable enhances the governance model, clarifies roles and "
        "responsibilities, or improves the decision-making process; (b) Actionable involves "
        "defining or refining processes for issue triaging, release management, or conflict "
        "resolution; (c) Actionable includes efforts to improve the transparency of project "
        "goals, progress, and decision-making."
    ),
    "Security Best Practices and Legal Compliance": (
        "This category addresses efforts to secure the project and ensure compliance with relevant "
        "legal standards, such as licenses, data privacy laws, and security protocols. "
        "Criteria: (a) Actionable focuses on improving the security posture of the project by "
        "following best practices, addressing vulnerabilities, or conducting audits; "
        "(b) Actionable involves ensuring compliance with open-source licenses, setting up "
        "contributor license agreements (CLAs), or aligning with data privacy regulations; "
        "(c) Actionable includes security measures such as dependency management, security audits, "
        "and secure coding practices."
    ),
    "CI/CD and DevOps Automation": (
        "This category deals with continuous integration and continuous deployment (CI/CD) "
        "processes that automate building, testing, and deployment pipelines. It also includes "
        "broader DevOps automation tasks. "
        "Criteria: (a) Actionable involves the setup or enhancement of CI/CD pipelines to ensure "
        "faster, reliable, and automated releases; (b) Actionable relates to automating "
        "infrastructure provisioning, containerization, or deployment to cloud environments; "
        "(c) Actionable includes the integration of DevOps practices that ensure smooth, "
        "automated, and repeatable processes for software development, testing, and deployment."
    ),
}

CATEGORY_KEYS = list(CATEGORIES.keys())

SOUND_DEFINITION = (
    "A ReACT is SOUND if it makes logical sense, has no contradictions, "
    "and all parts of the recommendation work together consistently. "
    "For instance, 'Project Managers should use peer reviews and automated tools "
    "for code review' is SOUND because peer reviews and tools work together to "
    "improve code quality. On the other hand, 'To improve user satisfaction, add "
    "new features without onboarding newcomers' is UNSOUND because adding features "
    "without testing can hurt user satisfaction, and not onboarding newcomers can "
    "reduce support quality."
)

PRECISE_DEFINITION = (
    "A ReACT is PRECISE if it is clear, specific, easy to follow, and leaves no "
    "room for confusion. For example, 'To attract newcomers, help them make their "
    "first contribution' is PRECISE as it gives a clear action to take. On the "
    "contrary, 'To attract core developers, ensure high code quality' is IMPRECISE "
    "as it does not explain HOW to ensure high code quality."
)

# ─────────────────────────────────────────────────────────────────────────────
# Logging
# ─────────────────────────────────────────────────────────────────────────────

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

def tqdm_log(msg: str):
    tqdm.write(msg)


# ─────────────────────────────────────────────────────────────────────────────
# Ollama — streaming with early exit
# ─────────────────────────────────────────────────────────────────────────────

def ollama_generate(model: str, prompt: str, base_url: str,
                    timeout: int = 600, keep_alive: str = "120m") -> str:
    """
    Stream tokens from Ollama and return as soon as YES or NO is found.

    Speed optimizations active:
      - stream=True      tokens arrive immediately; we stop on first YES/NO
      - keep_alive       model stays in VRAM — no reload between rows
      - num_predict      per-model cap via MODEL_NUM_PREDICT
      - num_ctx=512      small KV-cache → fast prefill for short prompts
      - temperature=0    deterministic
    """
    url = f"{base_url.rstrip('/')}/api/generate"
    payload = {
        "model":      model,
        "prompt":     prompt,
        "stream":     True,
        "keep_alive": keep_alive,
        "options": {
            "temperature": 0.0,
            "num_predict": MODEL_NUM_PREDICT.get(model, 64),
            "num_ctx":     512,
        },
    }

    collected = ""
    try:
        with requests.post(url, json=payload, stream=True, timeout=timeout) as resp:
            resp.raise_for_status()
            for raw_line in resp.iter_lines():
                if not raw_line:
                    continue
                try:
                    chunk = json.loads(raw_line)
                except json.JSONDecodeError:
                    continue

                collected += chunk.get("response", "")

                # Early exit — stop the moment YES or NO appears
                if re.search(r"\b(YES|NO)\b", collected, re.IGNORECASE):
                    return collected.strip()

                if chunk.get("done", False):
                    break

        return collected.strip()

    except requests.exceptions.Timeout:
        log.warning("Timeout — model=%s  collected=%r", model, collected[:80])
        return collected.strip() if collected else "TIMEOUT"
    except Exception as exc:
        log.warning("Error — model=%s  %s", model, exc)
        return f"ERROR: {exc}"


def unload_model(model: str, base_url: str) -> None:
    """Evict model from VRAM immediately so the next model can load cleanly."""
    url = f"{base_url.rstrip('/')}/api/generate"
    try:
        requests.post(
            url,
            json={"model": model, "prompt": "", "keep_alive": "0"},
            timeout=30,
        )
        tqdm_log(f"  ↓ unloaded {model} from VRAM")
    except Exception as exc:
        log.warning("Could not unload model %s: %s", model, exc)


# ─────────────────────────────────────────────────────────────────────────────
# Parsing & voting
# ─────────────────────────────────────────────────────────────────────────────

def parse_yes_no(text: str) -> str:
    match = re.search(r"\b(YES|NO)\b", text.strip(), re.IGNORECASE)
    return match.group(1).upper() if match else "UNCLEAR"


def majority_vote(votes: list, threshold: int = 3) -> str:
    return "YES" if sum(1 for v in votes if v == "YES") >= threshold else "NO"


# ─────────────────────────────────────────────────────────────────────────────
# Prompt builders
# ─────────────────────────────────────────────────────────────────────────────

def _clean(val: str) -> str:
    v = str(val).strip()
    return v if v and v.lower() != "nan" else "Not explicitly stated"


def build_sound_prompt(actionable: str, impact: str, evidence: str) -> str:
    return (
        f"Definition of SOUND:\n{SOUND_DEFINITION}\n\n"
        f"ReACT to evaluate:\n"
        f"  Actionable : {actionable}\n"
        f"  Impact     : {_clean(impact)}\n"
        f"  Evidence   : {_clean(evidence)}\n\n"
        f"Given the definition of SOUND, is this ReACT SOUND?\n\n"
        f"STRICT INSTRUCTION: Reply with a single word only — either YES or NO. "
        f"Do not write anything else. No explanation, no punctuation, no extra words."
    )


def build_precise_prompt(actionable: str, impact: str, evidence: str) -> str:
    return (
        f"Definition of PRECISE:\n{PRECISE_DEFINITION}\n\n"
        f"ReACT to evaluate:\n"
        f"  Actionable : {actionable}\n"
        f"  Impact     : {_clean(impact)}\n"
        f"  Evidence   : {_clean(evidence)}\n\n"
        f"Given the definition of PRECISE, is this ReACT PRECISE?\n\n"
        f"STRICT INSTRUCTION: Reply with a single word only — either YES or NO. "
        f"Do not write anything else. No explanation, no punctuation, no extra words."
    )


def build_category_prompt(actionable: str, category_name: str, category_def: str) -> str:
    return (
        f"You are classifying a software engineering ReACT (Researched Actionable) "
        f"into a predefined category.\n\n"
        f"Category       : {category_name}\n"
        f"Category Definition: {category_def}\n\n"
        f"ReACT to classify:\n"
        f"  {actionable}\n\n"
        f"Does this ReACT belong to the category '{category_name}'?\n\n"
        f"STRICT INSTRUCTION: Reply with a single word only — either YES or NO. "
        f"Do not write anything else. No explanation, no punctuation, no extra words."
    )


# ─────────────────────────────────────────────────────────────────────────────
# Column detection & safe field access
# ─────────────────────────────────────────────────────────────────────────────

def detect_columns(df: pd.DataFrame) -> dict:
    col_map    = {}
    lower_cols = {c.lower(): c for c in df.columns}
    for key, candidates in {
        "actionable": ["actionable", "react", "action", "recommendation"],
        "impact":     ["impact", "stated_impact", "impacts"],
        "evidence":   ["evidence", "empirical_evidence", "emp_evidence"],
    }.items():
        for cand in candidates:
            if cand in lower_cols:
                col_map[key] = lower_cols[cand]
                break
        if key not in col_map:
            col_map[key] = None
    return col_map


def get_field(row: pd.Series, col_map: dict, key: str) -> str:
    col = col_map.get(key)
    if col and col in row.index:
        val = row[col]
        return "" if pd.isna(val) else str(val)
    return ""


# ─────────────────────────────────────────────────────────────────────────────
# Resume helper
# ─────────────────────────────────────────────────────────────────────────────

def load_existing(output_path: str, df: pd.DataFrame,
                  sentinel_col: str) -> tuple:
    """Load existing output CSV for resume. Returns (out_df, done_indices)."""
    p = Path(output_path)
    if not p.exists():
        return df.copy(), []
    existing = pd.read_csv(output_path)
    if len(existing) != len(df) or sentinel_col not in existing.columns:
        log.warning("Output file mismatch — starting fresh.")
        return df.copy(), []
    done_mask    = existing[sentinel_col].notna() & (existing[sentinel_col] != "")
    done_indices = existing.index[done_mask].tolist()
    log.info("Resuming: %d / %d rows already done.", len(done_indices), len(df))
    return existing.copy(), done_indices


# ─────────────────────────────────────────────────────────────────────────────
# TASK 1 — Sound & Precise scoring  (model-first batching)
#
# Loop order:
#   for each model:          ← loads ONCE, stays hot for all rows
#       for each row:
#           sound call
#           precise call
#   unload model → load next
#   after all models: compute verdicts
# ─────────────────────────────────────────────────────────────────────────────

def run_scoring(args):
    log.info("=== MODE: score ===")
    df = pd.read_csv(args.input)
    log.info("Loaded %d rows × %d cols  |  %s", len(df), len(df.columns), args.input)

    col_map = detect_columns(df)
    log.info("Detected columns: %s", col_map)

    new_cols = (
        [f"sound_{m}"   for m in MODELS] +
        [f"precise_{m}" for m in MODELS] +
        ["sound_yes_count", "sound_verdict", "precise_yes_count", "precise_verdict"]
    )

    if args.resume:
        out_df, done_indices = load_existing(args.output, df, "sound_verdict")
    else:
        out_df, done_indices = df.copy(), []

    for col in new_cols:
        if col not in out_df.columns:
            out_df[col] = None

    limit   = args.limit or len(df)
    indices = [i for i in range(min(limit, len(df))) if i not in done_indices]
    log.info("Processing %d rows × %d models = %d × 2 calls total.",
             len(indices), len(MODELS), len(indices) * len(MODELS))

    # ── Outer loop: model ─────────────────────────────────────────────────────
    model_bar = tqdm(MODELS, desc="Models", unit="model",
                     colour="yellow", dynamic_ncols=True)

    with logging_redirect_tqdm():
        for model in model_bar:
            model_bar.set_postfix_str(model, refresh=True)
            tqdm_log(f"\n{'='*62}")
            tqdm_log(f"  ▶ Model: {model}  (keep_alive={args.keep_alive})")
            tqdm_log(f"{'='*62}")

            # ── Inner loop: rows ──────────────────────────────────────────────
            row_bar = tqdm(indices, desc=f"  {model[:22]}", unit="row",
                           colour="green", dynamic_ncols=True, leave=False)

            for idx in row_bar:
                row        = df.iloc[idx]
                actionable = get_field(row, col_map, "actionable")
                impact     = get_field(row, col_map, "impact")
                evidence   = get_field(row, col_map, "evidence")
                preview    = actionable[:48]
                row_bar.set_postfix_str(f"{preview}…", refresh=True)

                # Soundness
                raw   = ollama_generate(
                    model,
                    build_sound_prompt(actionable, impact, evidence),
                    args.ollama_url, args.timeout, args.keep_alive,
                )
                s_dec = parse_yes_no(raw)
                out_df.at[idx, f"sound_{model}"] = s_dec
                if args.delay > 0:
                    time.sleep(args.delay)

                # Preciseness
                raw   = ollama_generate(
                    model,
                    build_precise_prompt(actionable, impact, evidence),
                    args.ollama_url, args.timeout, args.keep_alive,
                )
                p_dec = parse_yes_no(raw)
                out_df.at[idx, f"precise_{model}"] = p_dec
                if args.delay > 0:
                    time.sleep(args.delay)

                tqdm_log(
                    f"    row {idx+1:>5}  "
                    f"sound={s_dec:<7} precise={p_dec:<7}  | {preview}…"
                )

            # Free VRAM before next model loads
            unload_model(model, args.ollama_url)

            # Checkpoint — saved after every complete model pass
            out_df.to_csv(args.output, index=False)
            tqdm_log(f"  ✓ checkpoint saved after {model}")

    # ── Compute majority verdicts ─────────────────────────────────────────────
    tqdm_log("\nComputing majority verdicts...")
    for idx in indices:
        s_votes = [out_df.at[idx, f"sound_{m}"]   for m in MODELS]
        p_votes = [out_df.at[idx, f"precise_{m}"] for m in MODELS]
        out_df.at[idx, "sound_yes_count"]   = sum(1 for v in s_votes if v == "YES")
        out_df.at[idx, "precise_yes_count"] = sum(1 for v in p_votes if v == "YES")
        out_df.at[idx, "sound_verdict"]     = majority_vote(s_votes)
        out_df.at[idx, "precise_verdict"]   = majority_vote(p_votes)

    out_df.to_csv(args.output, index=False)
    log.info("Done. Output: %s", args.output)

    total       = out_df["sound_verdict"].notna().sum()
    sound_yes   = (out_df["sound_verdict"]   == "YES").sum()
    precise_yes = (out_df["precise_verdict"] == "YES").sum()
    print("\n" + "=" * 55)
    print(f"  Scoring summary ({total} rows evaluated)")
    if total:
        print(f"  SOUND    YES : {sound_yes:>5}  ({100 * sound_yes   / total:.1f}%)")
        print(f"  PRECISE  YES : {precise_yes:>5}  ({100 * precise_yes / total:.1f}%)")
    print("=" * 55)


# ─────────────────────────────────────────────────────────────────────────────
# TASK 2 — Category assignment  (model-first batching)
#
# Loop order:
#   for each model:              ← loads ONCE
#       for each row:
#           for each category:
#               YES/NO call
#   unload model → load next
#   after all models: compute verdicts + assigned_categories
# ─────────────────────────────────────────────────────────────────────────────

def run_categorization(args):
    log.info("=== MODE: categorize ===")
    df = pd.read_csv(args.input)
    log.info("Loaded %d rows × %d cols  |  %s", len(df), len(df.columns), args.input)

    col_map = detect_columns(df)
    log.info("Actionable column: %s", col_map.get("actionable"))

    cat_cols = []
    for cat_name in CATEGORY_KEYS:
        for model in MODELS:
            cat_cols.append(f"cat_{cat_name}__{model}")
        cat_cols.append(f"cat_{cat_name}__yes_count")
        cat_cols.append(f"cat_{cat_name}__verdict")
    cat_cols += ["assigned_categories", "num_categories"]

    if args.resume:
        out_df, done_indices = load_existing(args.output, df, "assigned_categories")
    else:
        out_df, done_indices = df.copy(), []

    for col in cat_cols:
        if col not in out_df.columns:
            out_df[col] = None

    limit   = args.limit or len(df)
    indices = [i for i in range(min(limit, len(df))) if i not in done_indices]
    n_calls = len(indices) * len(MODELS) * len(CATEGORIES)
    log.info("Processing %d rows × %d models × %d categories = %d calls.",
             len(indices), len(MODELS), len(CATEGORIES), n_calls)

    # ── Outer loop: model ─────────────────────────────────────────────────────
    model_bar = tqdm(MODELS, desc="Models", unit="model",
                     colour="yellow", dynamic_ncols=True)

    with logging_redirect_tqdm():
        for model in model_bar:
            model_bar.set_postfix_str(model, refresh=True)
            tqdm_log(f"\n{'='*62}")
            tqdm_log(f"  ▶ Model: {model}  (keep_alive={args.keep_alive})")
            tqdm_log(f"{'='*62}")

            # ── Inner loop: rows ──────────────────────────────────────────────
            row_bar = tqdm(indices, desc=f"  {model[:22]}", unit="row",
                           colour="cyan", dynamic_ncols=True, leave=False)

            for idx in row_bar:
                row        = df.iloc[idx]
                actionable = get_field(row, col_map, "actionable")
                preview    = actionable[:45]
                row_bar.set_postfix_str(f"{preview}…", refresh=True)

                # ── Innermost loop: categories ────────────────────────────────
                cat_bar = tqdm(CATEGORIES.items(), desc="    cats",
                               unit="cat", leave=False, dynamic_ncols=True)

                for cat_name, cat_def in cat_bar:
                    cat_bar.set_postfix_str(cat_name[:32], refresh=True)
                    raw      = ollama_generate(
                        model,
                        build_category_prompt(actionable, cat_name, cat_def),
                        args.ollama_url, args.timeout, args.keep_alive,
                    )
                    decision = parse_yes_no(raw)
                    out_df.at[idx, f"cat_{cat_name}__{model}"] = decision
                    if args.delay > 0:
                        time.sleep(args.delay)

                tqdm_log(f"    row {idx+1:>5} done  | {preview}…")

            # Free VRAM before next model loads
            unload_model(model, args.ollama_url)

            # Checkpoint after every complete model pass
            out_df.to_csv(args.output, index=False)
            tqdm_log(f"  ✓ checkpoint saved after {model}")

    # ── Compute per-category verdicts and assigned_categories ─────────────────
    tqdm_log("\nComputing category verdicts...")
    for idx in indices:
        assigned = []
        for cat_name in CATEGORY_KEYS:
            votes     = [out_df.at[idx, f"cat_{cat_name}__{m}"] for m in MODELS]
            yes_count = sum(1 for v in votes if v == "YES")
            verdict   = "YES" if yes_count >= 3 else "NO"
            out_df.at[idx, f"cat_{cat_name}__yes_count"] = yes_count
            out_df.at[idx, f"cat_{cat_name}__verdict"]   = verdict
            if verdict == "YES":
                assigned.append(cat_name)

        out_df.at[idx, "assigned_categories"] = (
            " | ".join(assigned) if assigned else "NONE"
        )
        out_df.at[idx, "num_categories"] = len(assigned)

    out_df.to_csv(args.output, index=False)
    log.info("Done. Output: %s", args.output)

    total = out_df["assigned_categories"].notna().sum()
    print("\n" + "=" * 65)
    print(f"  Categorization summary ({total} rows evaluated)")
    avg = pd.to_numeric(out_df["num_categories"], errors="coerce").mean()
    print(f"  Avg categories per actionable : {avg:.2f}")
    print(f"\n  Category distribution:")
    for cat in CATEGORY_KEYS:
        col = f"cat_{cat}__verdict"
        if col in out_df.columns:
            n   = (out_df[col] == "YES").sum()
            pct = 100 * n / total if total else 0
            print(f"    {cat:<50} {n:>5}  ({pct:.1f}%)")
    print("=" * 65)


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="ReACT Scorer — soundness/preciseness + category assignment via Ollama"
    )
    parser.add_argument("--mode",       choices=["score", "categorize"], required=True)
    parser.add_argument("--input",  "-i", required=True, help="Path to input CSV")
    parser.add_argument("--output", "-o", required=True, help="Path for output CSV")
    parser.add_argument("--ollama_url",  default="http://localhost:11434")
    parser.add_argument("--keep_alive",  default="120m",
                        help="How long to keep model in VRAM between calls (default: 120m)")
    parser.add_argument("--timeout",     type=int,   default=600,
                        help="Seconds per Ollama call (default: 600)")
    parser.add_argument("--delay",       type=float, default=0.0,
                        help="Extra sleep between calls — 0 is fine with streaming (default: 0)")
    parser.add_argument("--resume",      action="store_true",
                        help="Skip rows already completed in an existing output file")
    parser.add_argument("--limit",       type=int,   default=None,
                        help="Process only first N rows (for testing)")
    args = parser.parse_args()

    if args.mode == "score":
        run_scoring(args)
    else:
        run_categorization(args)


if __name__ == "__main__":
    main()