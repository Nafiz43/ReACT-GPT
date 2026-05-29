"""
build_final_set.py
==================
Merges scored.csv and categorized.csv into a clean final_set.csv with columns:

  article_title, venue, article_link, cluster_id, support, support_fraction,
  models_present, actionable, impact, evidence, avg_confidence,
  centroid_confidence, SOUND, PRECISE, CATEGORY

SOUND    — majority verdict from scoring   (YES / NO)
PRECISE  — majority verdict from scoring   (YES / NO)
CATEGORY — assigned_categories from categorize (e.g. "CI/CD and DevOps Automation | Documentation Practices")

Usage:
  python build_final_set.py

Optional overrides:
  python build_final_set.py \\
      --scored      /path/to/scored.csv \\
      --categorized /path/to/categorized.csv \\
      --output      /path/to/final_set.csv
"""

import argparse
import logging
from pathlib import Path

import pandas as pd

# ─────────────────────────────────────────────────────────────────────────────
# Default paths
# ─────────────────────────────────────────────────────────────────────────────

BASE_DIR          = "/Users/nafiz43/Documents/GitHub/ReACT-GPT/local_history"
DEFAULT_SCORED      = f"{BASE_DIR}/scored.csv"
DEFAULT_CATEGORIZED = f"{BASE_DIR}/categorized.csv"
DEFAULT_OUTPUT      = f"{BASE_DIR}/final_set.csv"

# ─────────────────────────────────────────────────────────────────────────────
# Logging
# ─────────────────────────────────────────────────────────────────────────────

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# Core columns to keep
# ─────────────────────────────────────────────────────────────────────────────

BASE_COLS = [
    "article_title",
    "venue",
    "article_link",
    "cluster_id",
    "support",
    "support_fraction",
    "models_present",
    "actionable",
    "impact",
    "evidence",
    "avg_confidence",
    "centroid_confidence",
]


def main():
    parser = argparse.ArgumentParser(description="Build final_set.csv from scored + categorized CSVs")
    parser.add_argument("--scored",      default=DEFAULT_SCORED)
    parser.add_argument("--categorized", default=DEFAULT_CATEGORIZED)
    parser.add_argument("--output",      default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    # ── Load ──────────────────────────────────────────────────────────────────
    for path in [args.scored, args.categorized]:
        if not Path(path).exists():
            log.error("File not found: %s", path)
            raise SystemExit(1)

    scored      = pd.read_csv(args.scored)
    categorized = pd.read_csv(args.categorized)
    log.info("scored.csv      : %d rows × %d cols", len(scored),      len(scored.columns))
    log.info("categorized.csv : %d rows × %d cols", len(categorized), len(categorized.columns))

    if len(scored) != len(categorized):
        log.error("Row count mismatch: scored=%d, categorized=%d", len(scored), len(categorized))
        raise SystemExit(1)

    # ── Pull the columns we need ──────────────────────────────────────────────
    final = pd.DataFrame()

    # Base metadata columns — take from scored (they're identical in both)
    for col in BASE_COLS:
        if col in scored.columns:
            final[col] = scored[col]
        else:
            log.warning("Column '%s' not found in scored.csv — filling with None", col)
            final[col] = None

    # SOUND and PRECISE — majority verdicts from scored.csv
    final["SOUND"]   = scored["sound_verdict"]   if "sound_verdict"   in scored.columns else None
    final["PRECISE"] = scored["precise_verdict"] if "precise_verdict" in scored.columns else None

    # CATEGORY — assigned_categories from categorized.csv
    final["CATEGORY"] = categorized["assigned_categories"] if "assigned_categories" in categorized.columns else None

    # ── Save ──────────────────────────────────────────────────────────────────
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    final.to_csv(args.output, index=False)
    log.info("✓ final_set.csv saved → %s  (%d rows × %d cols)",
             args.output, len(final), len(final.columns))

    # ── Summary ───────────────────────────────────────────────────────────────
    total = len(final)
    print("\n" + "=" * 55)
    print(f"  final_set.csv summary ({total} rows)")
    print(f"  Columns : {list(final.columns)}")
    if "SOUND" in final.columns:
        print(f"  SOUND    YES : {(final['SOUND']   == 'YES').sum():>5} / {total}")
    if "PRECISE" in final.columns:
        print(f"  PRECISE  YES : {(final['PRECISE'] == 'YES').sum():>5} / {total}")
    if "CATEGORY" in final.columns:
        none_count = (final["CATEGORY"] == "NONE").sum()
        print(f"  CATEGORY NONE: {none_count:>5} / {total}")
    print("=" * 55)


if __name__ == "__main__":
    main()