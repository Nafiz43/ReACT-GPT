"""
ReACT Soundness & Preciseness Scorer
=====================================
Evaluates each ReACT (actionable + impact + evidence) for:
  - Soundness  : Is the recommendation logically consistent and free of contradictions?
  - Preciseness: Is the recommendation clear, specific, and easy to follow?

Models used (via Ollama):
  1. qwen3.6:35b
  2. gpt-oss:20b          (mapped as gpt-oss:20b in Ollama)
  3. deepseek-r1:32b
  4. gemma4:31b
  5. mixtral:8x7b

Final verdict: majority vote (>= 3 out of 5 YES => YES).

Usage:
    python react_sound_precise_scorer.py \
        --input  /data/Deep_Angiography/ReACT-GPT/local_history/reconciled_actionables.csv \
        --output ./reconciled_actionables_scored.csv \
        [--ollama_url http://localhost:11434] \
        [--delay 1.0]                          # seconds between Ollama calls
        [--resume]                             # skip rows already processed

Output columns added (12 new columns):
    sound_qwen3.6:35b, sound_gpt-oss:20b, sound_deepseek-r1:32b,
    sound_gemma4:31b,  sound_mixtral:8x7b,
    precise_qwen3.6:35b, precise_gpt-oss:20b, precise_deepseek-r1:32b,
    precise_gemma4:31b,  precise_mixtral:8x7b,
    sound_yes_count, sound_verdict,
    precise_yes_count, precise_verdict
"""

import argparse
import logging
import re
import time
from pathlib import Path

import pandas as pd
import requests
from tqdm import tqdm
from tqdm.contrib.logging import logging_redirect_tqdm

# ─────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────
MODELS = [
    "qwen3.6:35b",
    "gpt-oss:20b",
    "deepseek-r1:32b",
    "gemma4:31b",
    "mixtral:8x7b",
]

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

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

# tqdm helper — writes above the progress bar without breaking it
def tqdm_log(msg: str):
    tqdm.write(msg)


# ─────────────────────────────────────────────
# Ollama helpers
# ─────────────────────────────────────────────

def ollama_generate(model: str, prompt: str, base_url: str, timeout: int = 600) -> str:
    """Call Ollama /api/generate and return the full response text."""
    url = f"{base_url.rstrip('/')}/api/generate"
    payload = {
        "model": model,
        "prompt": prompt,
        "stream": False,
        "options": {
            "temperature": 0.0,   # deterministic — matches paper's temp=0
            "num_predict": 256,
        },
    }
    try:
        resp = requests.post(url, json=payload, timeout=timeout)
        resp.raise_for_status()
        return resp.json().get("response", "").strip()
    except requests.exceptions.Timeout:
        log.warning("Timeout for model %s", model)
        return "TIMEOUT"
    except Exception as exc:
        log.warning("Error for model %s: %s", model, exc)
        return f"ERROR: {exc}"


def parse_yes_no(text: str) -> str:
    """Extract YES or NO from model output. Returns 'UNCLEAR' if ambiguous."""
    text = text.strip()
    # Look for leading YES/NO (answers typically start with it)
    match = re.search(r"\b(YES|NO)\b", text[:80], re.IGNORECASE)
    if match:
        return match.group(1).upper()
    # Fallback: search whole response
    match = re.search(r"\b(YES|NO)\b", text, re.IGNORECASE)
    if match:
        return match.group(1).upper()
    return "UNCLEAR"


def build_sound_prompt(actionable: str, impact: str, evidence: str) -> str:
    return f"""Definition of SOUND:
{SOUND_DEFINITION}

You are evaluating the following ReACT (Researched Actionable):

  Actionable : {actionable}
  Impact     : {impact if pd.notna(impact) and str(impact).strip() else "Not explicitly stated"}
  Evidence   : {evidence if pd.notna(evidence) and str(evidence).strip() else "Not explicitly stated"}

Given the definition of SOUND, can this ReACT be considered SOUND?

STRICT INSTRUCTION: Reply with a single word only — either YES or NO. Do not write anything else. No explanation, no punctuation, no extra words."""


def build_precise_prompt(actionable: str, impact: str, evidence: str) -> str:
    return f"""Definition of PRECISE:
{PRECISE_DEFINITION}

You are evaluating the following ReACT (Researched Actionable):

  Actionable : {actionable}
  Impact     : {impact if pd.notna(impact) and str(impact).strip() else "Not explicitly stated"}
  Evidence   : {evidence if pd.notna(evidence) and str(evidence).strip() else "Not explicitly stated"}

Given the definition of PRECISE, can this ReACT be considered PRECISE?

STRICT INSTRUCTION: Reply with a single word only — either YES or NO. Do not write anything else. No explanation, no punctuation, no extra words."""


# ─────────────────────────────────────────────
# Core evaluation logic
# ─────────────────────────────────────────────

def majority_vote(votes: list) -> str:
    """Return YES if >= 3 out of 5 models say YES, else NO."""
    yes_count = sum(1 for v in votes if v == "YES")
    return "YES" if yes_count >= 3 else "NO"


def evaluate_row(row: pd.Series, models: list, base_url: str, delay: float, timeout: int) -> dict:
    """Run all models on one ReACT row; return dict of new column values."""
    actionable = str(row.get("actionable", row.get("Actionable", row.get("react", ""))))
    impact     = str(row.get("impact",     row.get("Impact",     "")))
    evidence   = str(row.get("evidence",   row.get("Evidence",   "")))

    sound_prompt   = build_sound_prompt(actionable, impact, evidence)
    precise_prompt = build_precise_prompt(actionable, impact, evidence)

    results = {}
    sound_votes   = []
    precise_votes = []

    for model in tqdm(models, desc="  models", leave=False, unit="model"):
        # ── Soundness ──
        tqdm_log(f"    ├─ [{model}] soundness ...")
        raw_sound = ollama_generate(model, sound_prompt, base_url, timeout)
        sound_decision = parse_yes_no(raw_sound)
        sound_votes.append(sound_decision)
        results[f"sound_{model}"] = sound_decision
        time.sleep(delay)

        # ── Preciseness ──
        tqdm_log(f"    └─ [{model}] preciseness ... → sound={sound_decision}")
        raw_precise = ollama_generate(model, precise_prompt, base_url, timeout)
        precise_decision = parse_yes_no(raw_precise)
        precise_votes.append(precise_decision)
        results[f"precise_{model}"] = precise_decision
        time.sleep(delay)

    results["sound_verdict"]   = majority_vote(sound_votes)
    results["precise_verdict"] = majority_vote(precise_votes)
    results["sound_yes_count"]   = sum(1 for v in sound_votes   if v == "YES")
    results["precise_yes_count"] = sum(1 for v in precise_votes if v == "YES")

    return results


# ─────────────────────────────────────────────
# Column name detection helpers
# ─────────────────────────────────────────────

def detect_columns(df: pd.DataFrame) -> tuple:
    """Detect actual column names for actionable, impact, evidence."""
    col_map = {}
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
            col_map[key] = None  # will use empty string fallback

    return col_map


# ─────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Score ReACTs for soundness & preciseness via Ollama")
    parser.add_argument(
        "--input", "-i",
        default="/data/Deep_Angiography/ReACT-GPT/local_history/reconciled_actionables.csv",
        help="Path to input CSV",
    )
    parser.add_argument(
        "--output", "-o",
        default="./reconciled_actionables_scored.csv",
        help="Path for output CSV",
    )
    parser.add_argument(
        "--ollama_url",
        default="http://localhost:11434",
        help="Ollama base URL",
    )
    parser.add_argument(
        "--delay", type=float, default=0.5,
        help="Seconds to sleep between Ollama calls (avoid overloading GPU)",
    )
    parser.add_argument(
        "--resume", action="store_true",
        help="If output CSV exists, skip rows that already have 'sound_verdict' filled",
    )
    parser.add_argument(
        "--timeout", type=int, default=600,
        help="Seconds to wait for a single Ollama call before timing out (default 600)",
    )
    parser.add_argument(
        "--limit", type=int, default=None,
        help="Process only first N rows (for testing)",
    )
    args = parser.parse_args()

    # ── Load input ──
    log.info("Loading: %s", args.input)
    df = pd.read_csv(args.input)
    log.info("Loaded %d rows × %d cols", len(df), len(df.columns))
    log.info("Columns: %s", df.columns.tolist())

    col_map = detect_columns(df)
    log.info("Detected columns: %s", col_map)

    # ── Resume support ──
    if args.resume and Path(args.output).exists():
        existing = pd.read_csv(args.output)
        # Find rows already processed (sound_verdict present and not NaN)
        if "sound_verdict" in existing.columns:
            done_mask = existing["sound_verdict"].notna() & (existing["sound_verdict"] != "")
            done_indices = existing.index[done_mask].tolist()
            log.info("Resuming: %d rows already done, skipping them.", len(done_indices))
        else:
            existing = None
            done_indices = []
    else:
        existing = None
        done_indices = []

    # ── Build new columns list ──
    new_cols_order = []
    for model in MODELS:
        new_cols_order.append(f"sound_{model}")
    for model in MODELS:
        new_cols_order.append(f"precise_{model}")
    new_cols_order += [
        "sound_yes_count",
        "sound_verdict",
        "precise_yes_count",
        "precise_verdict",
    ]

    # Pre-fill output df
    if existing is not None and len(existing) == len(df):
        out_df = existing.copy()
    else:
        out_df = df.copy()
        for col in new_cols_order:
            if col not in out_df.columns:
                out_df[col] = None

    # ── Process rows ──
    limit = args.limit if args.limit else len(df)
    process_indices = [i for i in range(min(limit, len(df))) if i not in done_indices]
    log.info("Will process %d rows.", len(process_indices))

    outer_bar = tqdm(
        process_indices,
        desc="ReACTs",
        unit="row",
        colour="green",
        dynamic_ncols=True,
    )

    with logging_redirect_tqdm():
        for idx in outer_bar:
            row = df.iloc[idx]
            actionable_preview = str(
                row.get(col_map.get("actionable") or "actionable", "")
            )[:55]
            outer_bar.set_postfix_str(f"{actionable_preview}…", refresh=True)
            tqdm_log(f"\n[Row {idx + 1}/{len(df)}] {actionable_preview}...")

            try:
                result = evaluate_row(row, MODELS, args.ollama_url, args.delay, args.timeout)
            except Exception as exc:
                tqdm_log(f"  ✗ Row {idx} failed: {exc}")
                result = {col: "ERROR" for col in new_cols_order}

            for col, val in result.items():
                out_df.at[idx, col] = val

            # Save checkpoint after every row
            out_df.to_csv(args.output, index=False)

            sound_v   = result.get("sound_verdict",   "?")
            precise_v = result.get("precise_verdict", "?")
            tqdm_log(
                f"  ✓ saved  |  sound={sound_v} ({result.get('sound_yes_count','?')}/5)"
                f"  precise={precise_v} ({result.get('precise_yes_count','?')}/5)"
            )

    log.info("Done. Final output: %s", args.output)

    # ── Summary statistics ──
    if "sound_verdict" in out_df.columns and "precise_verdict" in out_df.columns:
        sound_yes   = (out_df["sound_verdict"]   == "YES").sum()
        precise_yes = (out_df["precise_verdict"] == "YES").sum()
        total = out_df["sound_verdict"].notna().sum()
        print("\n" + "=" * 55)
        print(f"  Summary ({total} evaluated rows)")
        print(f"  SOUND    YES : {sound_yes:>5}  ({100 * sound_yes   / total:.1f}%)" if total else "")
        print(f"  PRECISE  YES : {precise_yes:>5}  ({100 * precise_yes / total:.1f}%)" if total else "")
        print("=" * 55)


if __name__ == "__main__":
    main()