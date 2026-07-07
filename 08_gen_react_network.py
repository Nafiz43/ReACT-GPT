#!/usr/bin/env python3
"""
gen_react_network.py

Generates an interactive Category → Actionable network from:
    local_history/final_set.csv

Columns used (final_set.csv schema):
    article_title, CATEGORY, actionable, impact,
    evidence, avg_confidence, SOUND, PRECISE, support

CATEGORY may contain pipe-separated values, e.g.:
    "Automated Testing | CI/CD and DevOps Automation"
Only the 8 canonical categories are kept; one edge is created
per unique (category, actionable) pair so actionables can fan
out to multiple category hubs.

Architecture:
  • ALL_NODES       — vis-only props, sanitized through VIS_NODE_KEYS whitelist
  • ALL_NODE_META   — application data, never touches vis DataSet
  • BAKED_EMBEDDINGS — node vectors pre-computed in Python, baked into HTML
  • Transformers.js — loads only for query embedding (~2–3 s, browser-cached)
  • Physics         — Barnes-Hut, runs once then freezes
  • Search          — Enter / button only, semantic + keyword modes
  • NOTE: search matches ONLY against the "impact" field of actionable nodes
  • Filters         — SOUND and PRECISE quality filters in side panel

Usage
─────
    pip install sentence-transformers numpy tqdm
    python gen_react_network.py
    python gen_react_network.py --skip-embed
    python gen_react_network.py --workers 4
"""

from __future__ import annotations

import argparse
import base64
import concurrent.futures
import csv
import json
import math
import os
import re
import subprocess
import sys
from collections import OrderedDict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

try:
    from tqdm import tqdm
    HAS_TQDM = True
except ImportError:
    HAS_TQDM = False
    class tqdm:
        def __init__(self, iterable=None, **kw): self._it = iterable
        def __iter__(self): return iter(self._it)
        def __enter__(self): return self
        def __exit__(self, *a): pass
        def update(self, n=1): pass
        def set_description(self, s): pass
        @staticmethod
        def write(s): print(s)

try:
    import numpy as np
    HAS_NP = True
except ImportError:
    HAS_NP = False

try:
    from sentence_transformers import SentenceTransformer
    HAS_ST = True
except ImportError:
    HAS_ST = False


# ── defaults ───────────────────────────────────────────────────────────────────

DEFAULT_INPUT_CSV   = Path("local_history/final_set.csv")
DEFAULT_OUTPUT_HTML = Path("results/index.html")

# Canonical categories — only these 8 are kept; anything else is discarded
CANONICAL_CATEGORIES = {
    "New Contributor Onboarding and Involvement",
    "Code Standards and Maintainability",
    "Automated Testing and Quality Assurance",
    "Community Collaboration and Engagement",
    "Documentation Practices",
    "Project Management and Governance",
    "Security Best Practices and Legal Compliance",
    "CI/CD and DevOps Automation",
}

REQUIRED_COLUMNS = ["actionable", "CATEGORY", "article_title"]
OPTIONAL_COLUMNS = ["support", "impact", "evidence", "avg_confidence", "SOUND", "PRECISE"]

# The category whose actionables are baked into the main HTML on first load;
# every other category is shipped as a side-car JSON file fetched only when
# the user picks it in the browser (see partition_by_category / loadCategoryData).
DEFAULT_CATEGORY = "New Contributor Onboarding and Involvement"

# Public, read-only Google Sheets view of the full underlying dataset — linked
# from the topbar and the About panel so visitors can inspect/download the
# raw data behind the tool.
DEFAULT_DATASET_URL = (
    "https://docs.google.com/spreadsheets/d/e/2PACX-1vSnHNS0fcQtn0F5IWbNcTB7HS_nbtVMtQ8vBYf0Ebqt3pbfKf3Bvd_dp3tLNtdmnQiSm44HGZ3bOrJA/pubhtml?gid=1298707173&single=true"
)


def slugify(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", name.lower()).strip("_") or "category"


# ── dependency check ───────────────────────────────────────────────────────────

def ensure_embedding_deps() -> bool:
    """Return True if embedding deps are present (installing if needed)."""
    global HAS_ST, HAS_NP
    if HAS_ST and HAS_NP:
        return True
    missing = []
    if not HAS_NP:
        missing.append("numpy")
    if not HAS_ST:
        missing.append("sentence-transformers")
    print(f"\n  ⚠  Missing packages for baked embeddings: {', '.join(missing)}")
    print(f"     Attempting: {sys.executable} -m pip install {' '.join(missing)}")
    try:
        subprocess.check_call(
            [sys.executable, "-m", "pip", "install", "--quiet"] + missing
        )
    except subprocess.CalledProcessError:
        print("  ✗  Auto-install failed. Run manually:")
        print(f"       pip install {' '.join(missing)}")
        return False
    # Re-import after install
    try:
        import numpy as np_
        import sentence_transformers as st_
        HAS_NP = True
        HAS_ST = True
        globals()["np"] = np_
        globals()["SentenceTransformer"] = st_.SentenceTransformer
        print("  ✓  Packages installed successfully.")
        return True
    except ImportError:
        return False


# ── text helpers ───────────────────────────────────────────────────────────────

def normalize_text(value: str) -> str:
    return " ".join((value or "").strip().split())

def safe_col(row: Dict[str, str], col: str) -> str:
    return normalize_text(row.get(col, "") or "")

def split_categories(raw: str) -> List[str]:
    """Split a pipe-delimited CATEGORY string and filter to canonical set."""
    parts = [p.strip() for p in raw.split("|") if p.strip()]
    return [p for p in parts if p in CANONICAL_CATEGORIES]


# ── CSV reader ─────────────────────────────────────────────────────────────────

def read_csv_rows(csv_path: Path) -> List[Dict]:
    if not csv_path.exists():
        raise FileNotFoundError(f"Input CSV not found: {csv_path}")

    with csv_path.open("r", encoding="utf-8-sig") as f:
        total = sum(1 for _ in f) - 1

    rows: List[Dict] = []
    skipped_no_action = 0
    skipped_no_cat    = 0

    with csv_path.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None:
            raise ValueError("CSV appears empty or has no header row.")
        missing = [c for c in REQUIRED_COLUMNS if c not in reader.fieldnames]
        if missing:
            raise ValueError("Missing required column(s): " + ", ".join(missing))

        with tqdm(reader, total=total, desc="  Reading CSV", unit="row") as bar:
            for row in bar:
                cleaned = {k: normalize_text(v or "") for k, v in row.items()}

                if not cleaned.get("actionable"):
                    skipped_no_action += 1
                    continue

                cats = split_categories(cleaned.get("CATEGORY", ""))
                if not cats:
                    skipped_no_cat += 1
                    continue

                cleaned["_categories"] = cats
                rows.append(cleaned)

    if skipped_no_action:
        tqdm.write(f"  [info] Skipped {skipped_no_action} rows with empty actionable.")
    if skipped_no_cat:
        tqdm.write(f"  [info] Skipped {skipped_no_cat} rows with no canonical category.")

    if not rows:
        raise ValueError("No usable rows found in CSV.")
    return rows


# ── graph payload builder ──────────────────────────────────────────────────────

def build_graph_payload(
    rows: List[Dict],
) -> Tuple[List[dict], List[dict], List[dict], dict, List[str], List[str]]:

    category_to_id: OrderedDict = OrderedDict()
    actionable_to_id: OrderedDict = OrderedDict()
    actionable_to_meta: Dict[str, dict] = {}

    edges_seen: set = set()
    edges: List[dict] = []

    # ── Pass 1: collect unique categories & actionables ────────────────────────
    with tqdm(rows, desc="  Building nodes", unit="row") as bar:
        for row in bar:
            cats = row["_categories"]
            act  = safe_col(row, "actionable")

            for cat in cats:
                if cat not in category_to_id:
                    category_to_id[cat] = f"C_{len(category_to_id) + 1}"

            if act not in actionable_to_id:
                visid = f"A_{len(actionable_to_id) + 1}"
                actionable_to_id[act] = visid
                actionable_to_meta[act] = {
                    "id":               visid,
                    "short_label":      f"a{len(actionable_to_id)}",
                    "node_type":        "actionable",
                    "actionable":       act,
                    "categories":       list(cats),
                    "article_title":    safe_col(row, "article_title"),
                    "impact":           safe_col(row, "impact"),
                    "empirical_evidence": safe_col(row, "evidence"),
                    "confidence":       safe_col(row, "avg_confidence"),
                    "sound":            safe_col(row, "SOUND"),
                    "precise":          safe_col(row, "PRECISE"),
                    "count":            safe_col(row, "support"),
                }
            else:
                existing = set(actionable_to_meta[act]["categories"])
                for cat in cats:
                    if cat not in existing:
                        actionable_to_meta[act]["categories"].append(cat)
                        existing.add(cat)

    # ── Pass 2: one edge per unique (category_id, actionable_id) pair ──────────
    edge_counter = 1
    with tqdm(rows, desc="  Building edges", unit="row") as bar:
        for row in bar:
            act = safe_col(row, "actionable")
            aid = actionable_to_id[act]
            for cat in row["_categories"]:
                cid = category_to_id[cat]
                ek  = (cid, aid)
                if ek in edges_seen:
                    continue
                edges_seen.add(ek)
                edges.append({"id": f"E_{edge_counter}", "from": cid, "to": aid, "width": 1.6})
                edge_counter += 1

    # ── Serialise actionable vis-nodes + node_meta ─────────────────────────────
    all_ids_ordered:   List[str] = []
    all_texts_ordered: List[str] = []
    nodes:     List[dict] = []
    node_meta: List[dict] = []

    with tqdm(actionable_to_id.items(), desc="  Serialising actionables",
              unit="node", total=len(actionable_to_id)) as bar:
        for act, visid in bar:
            meta = actionable_to_meta[act]

            # ── SEARCH TARGET: only the "impact" field ─────────────────────────
            semantic_text = meta["impact"] or ""
            search_blob   = (meta["impact"] or "").lower()
            # ──────────────────────────────────────────────────────────────────

            all_ids_ordered.append(visid)
            all_texts_ordered.append(semantic_text)
            nodes.append({
                "id": visid,
                "label": meta["short_label"],
                "shape": "dot", "size": 16,
                "color": {
                    "background": "#fb6a4a", "border": "#c2410c",
                    "highlight": {"background": "#fb923c", "border": "#9a3412"},
                    "hover":     {"background": "#fb923c", "border": "#9a3412"},
                },
                "borderWidth": 1.5,
                "font": {"color": "#1e293b", "size": 13, "face": "IBM Plex Sans, sans-serif"},
                "shadow": True,
            })
            node_meta.append({
                "id":               visid,
                "node_type":        "actionable",
                "actionable":       meta["actionable"],
                "categories":       meta["categories"],
                "article_title":    meta["article_title"],
                "impact":           meta["impact"],
                "empirical_evidence": meta["empirical_evidence"],
                "confidence":       meta["confidence"],
                "sound":            meta["sound"],
                "precise":          meta["precise"],
                "count":            meta["count"],
                "search_blob":      search_blob,
                "semantic_text":    semantic_text,
            })

    # ── Serialise category vis-nodes ───────────────────────────────────────────
    with tqdm(category_to_id.items(), desc="  Serialising categories",
              unit="node", total=len(category_to_id)) as bar:
        for cat, visid in bar:
            semantic_text = ""
            search_blob   = ""
            all_ids_ordered.append(visid)
            all_texts_ordered.append(semantic_text)
            nodes.append({
                "id": visid,
                "label": cat if len(cat) <= 22 else cat[:20] + "…",
                "shape": "dot", "size": 26,
                "color": {
                    "background": "#6baed6", "border": "#2563eb",
                    "highlight": {"background": "#93c5fd", "border": "#1d4ed8"},
                    "hover":     {"background": "#93c5fd", "border": "#1d4ed8"},
                },
                "borderWidth": 2,
                "font": {"color": "#1e293b", "size": 14, "face": "IBM Plex Sans, sans-serif"},
                "shadow": True,
            })
            node_meta.append({
                "id":          visid,
                "node_type":   "category",
                "category":    cat,
                "search_blob": search_blob,
                "semantic_text": semantic_text,
            })

    stats = {
        "row_count":        len(rows),
        "actionable_count": len(actionable_to_id),
        "category_count":   len(category_to_id),
        "edge_count":       len(edges),
    }
    return nodes, node_meta, edges, stats, all_ids_ordered, all_texts_ordered


# ── per-category partitioning (for lazy front-end loading) ─────────────────────

def partition_by_category(
    nodes: List[dict],
    node_meta: List[dict],
    edges: List[dict],
    embeddings_by_id: Dict[str, str],
    default_category: str,
) -> Tuple[List[dict], List[dict], List[dict], Dict[str, str], Dict[str, dict], str]:
    """Split the full graph into a 'main' bundle (all 8 category hub nodes +
    the default category's actionables/edges/embeddings) and one side-car
    chunk per remaining category (nodes/node_meta/edges/embeddings for the
    actionables newly introduced by that category — an actionable shared
    with the default category is not duplicated). The front-end fetches a
    chunk only when the user selects that category, so first paint only
    ever has to download/parse/lay out one category's worth of data.

    Returns (main_nodes, main_meta, main_edges, main_embeddings, chunks,
    resolved_default_category_id).
    """
    meta_by_id = {m["id"]: m for m in node_meta}
    node_by_id = {n["id"]: n for n in nodes}
    category_meta = [m for m in node_meta if m["node_type"] == "category"]
    category_ids = [m["id"] for m in category_meta]

    default_cat_id = next(
        (m["id"] for m in category_meta if m["category"] == default_category), None
    )

    actionables_by_cat: Dict[str, List[str]] = {cid: [] for cid in category_ids}
    edges_by_cat: Dict[str, List[dict]] = {cid: [] for cid in category_ids}
    for e in edges:
        cid = e["from"]
        if cid in actionables_by_cat:
            actionables_by_cat[cid].append(e["to"])
            edges_by_cat[cid].append(e)

    default_action_ids = set(actionables_by_cat.get(default_cat_id, []))

    main_nodes = [n for n in nodes
                  if meta_by_id[n["id"]]["node_type"] == "category" or n["id"] in default_action_ids]
    main_meta = [m for m in node_meta
                 if m["node_type"] == "category" or m["id"] in default_action_ids]
    main_edges = edges_by_cat.get(default_cat_id, [])
    main_embeddings = {nid: b64 for nid, b64 in embeddings_by_id.items() if nid in default_action_ids}

    chunks: Dict[str, dict] = {}
    for cid in category_ids:
        if cid == default_cat_id:
            continue
        new_action_ids = [aid for aid in actionables_by_cat[cid] if aid not in default_action_ids]
        chunks[cid] = {
            "nodes":      [node_by_id[aid] for aid in new_action_ids],
            "node_meta":  [meta_by_id[aid] for aid in new_action_ids],
            "edges":      edges_by_cat[cid],
            "embeddings": {aid: embeddings_by_id[aid] for aid in new_action_ids
                           if aid in embeddings_by_id},
        }

    return main_nodes, main_meta, main_edges, main_embeddings, chunks, default_cat_id


# ── embedding ──────────────────────────────────────────────────────────────────

def _worker_embed(args: Tuple[List[str], int]) -> Tuple[int, List[str]]:
    texts, chunk_idx = args
    from sentence_transformers import SentenceTransformer
    import base64 as _b64, numpy as _np
    model = SentenceTransformer("all-MiniLM-L6-v2")
    vecs = model.encode(
        texts, batch_size=32, normalize_embeddings=True,
        show_progress_bar=False, convert_to_numpy=True,
    ).astype("float32")
    return chunk_idx, [_b64.b64encode(v.tobytes()).decode("ascii") for v in vecs]


def embed_texts(texts: List[str], n_workers: int = 1) -> Optional[List[str]]:
    """Encode texts with sentence-transformers; returns base64 float32 vectors."""
    if not (HAS_ST and HAS_NP):
        return None

    n = len(texts)
    n_workers  = max(1, min(n_workers, os.cpu_count() or 1, math.ceil(n / 50)))
    chunk_size = math.ceil(n / n_workers)
    chunks     = [(texts[i:i + chunk_size], i // chunk_size)
                  for i in range(0, n, chunk_size)]

    tqdm.write(f"  Embedding {n} texts across {n_workers} worker(s) …")
    results: List[Optional[List[str]]] = [None] * len(chunks)

    if n_workers == 1:
        _, encoded = _worker_embed((texts, 0))
        results[0] = encoded
    else:
        with concurrent.futures.ProcessPoolExecutor(max_workers=n_workers) as ex:
            futures = {ex.submit(_worker_embed, c): c[1] for c in chunks}
            with tqdm(total=len(chunks), desc="  Embedding chunks", unit="chunk") as bar:
                for fut in concurrent.futures.as_completed(futures):
                    idx, enc = fut.result()
                    results[idx] = enc
                    bar.update(1)

    flat: List[str] = []
    for r in results:
        flat.extend(r)
    kb = sum(len(e) for e in flat) // 1024
    tqdm.write(f"  ✓  Embedded {len(flat)} vectors → {kb} KB (baking into HTML)")
    return flat


# ── HTML template ──────────────────────────────────────────────────────────────

HTML_TEMPLATE = r"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8"/>
  <meta name="viewport" content="width=device-width,initial-scale=1"/>
  <title>ReACTive · Intelligent Actionable Search Tool</title>

  <script src="https://unpkg.com/vis-network/standalone/umd/vis-network.min.js"></script>

  <script type="module">
    import { pipeline, env } from 'https://cdn.jsdelivr.net/npm/@huggingface/transformers@3/dist/transformers.min.js';
    env.allowLocalModels = false;
    window.__transformers = { pipeline, env };
    window.__transformersReady = true;
    window.dispatchEvent(new Event('transformers-ready'));
  </script>

  <link rel="preconnect" href="https://fonts.googleapis.com">
  <link href="https://fonts.googleapis.com/css2?family=IBM+Plex+Sans:wght@400;600;700&family=IBM+Plex+Mono:wght@400;500&display=swap" rel="stylesheet">

  <style>
    :root,[data-theme="light"] {
      --bg:#f2f6fc; --surface:#fff; --surface2:#f8fafd; --border:#dbe4f0;
      --text:#0f172a; --muted:#64748b; --accent:#2563eb;
      --cat:#6baed6; --action:#fb6a4a;
      --shadow:0 8px 28px rgba(15,23,42,.09);
      --net-bg:linear-gradient(160deg,#f8fbff 0%,#eef4fb 100%);
      --net-glass:rgba(255,255,255,.88);
      --stab-bg:rgba(242,246,252,.88);
    }
    [data-theme="dark"] {
      --bg:#0d1117; --surface:#161b22; --surface2:#1c2128; --border:#30363d;
      --text:#e6edf3; --muted:#7d8590; --accent:#58a6ff;
      --cat:#5b9bd5; --action:#f97316;
      --shadow:0 8px 28px rgba(0,0,0,.55);
      --net-bg:radial-gradient(ellipse at 50% 40%,#1a2035 0%,#0d1117 100%);
      --net-glass:rgba(22,27,34,.88);
      --stab-bg:rgba(13,17,23,.88);
    }
    *,*::before,*::after{box-sizing:border-box;margin:0;padding:0}
    html{transition:background .25s,color .25s}
    html,body{height:100%;font-family:'IBM Plex Sans',system-ui,sans-serif;
              background:var(--bg);color:var(--text)}
    div.vis-tooltip{
      position:absolute;visibility:hidden;padding:8px 11px;max-width:340px;
      white-space:pre-wrap;word-break:break-word;font-size:12px;line-height:1.5;
      color:var(--text);background:var(--surface);border:1px solid var(--border);
      border-radius:10px;box-shadow:var(--shadow);z-index:10;pointer-events:none
    }
    .app{display:flex;height:100vh;overflow:hidden}
    .main{display:flex;flex-direction:column;flex:1 1 0;min-width:0;overflow:hidden}
    /* topbar */
    .topbar{background:var(--surface);border-bottom:1px solid var(--border);
            padding:10px 18px;display:flex;align-items:center;gap:14px;flex-wrap:wrap;flex-shrink:0}
    .topbar-brand{flex:1;min-width:0}
    .eyebrow{font-size:10px;font-weight:700;letter-spacing:.1em;text-transform:uppercase;
             color:var(--accent);margin-bottom:2px}
    .topbar h1{font-size:15px;font-weight:700;line-height:1.2}
    .topbar-actions{display:flex;gap:8px;flex-shrink:0;flex-wrap:wrap}
    .btn-topbar{display:inline-flex;align-items:center;gap:6px;padding:7px 13px;
                border-radius:8px;font-size:12px;font-weight:700;border:1px solid var(--border);
                background:var(--surface2);color:var(--text);cursor:pointer;text-decoration:none;
                font-family:'IBM Plex Sans',sans-serif;transition:all .15s;white-space:nowrap}
    .btn-topbar:hover{border-color:var(--accent);color:var(--accent)}
    .btn-topbar-accent{background:var(--accent);color:#fff;border-color:var(--accent)}
    .btn-topbar-accent:hover{opacity:.88;color:#fff}
    /* about modal */
    .modal-overlay{position:fixed;inset:0;background:rgba(15,23,42,.6);backdrop-filter:blur(3px);
                   display:flex;align-items:center;justify-content:center;z-index:50;
                   padding:24px;opacity:1;transition:opacity .2s ease}
    .modal-overlay.hidden{opacity:0;pointer-events:none}
    .modal-card{background:var(--surface);border:1px solid var(--border);border-radius:16px;
                max-width:580px;width:100%;max-height:88vh;overflow-y:auto;padding:30px 32px;
                box-shadow:0 24px 70px rgba(15,23,42,.4);position:relative}
    .modal-close{position:absolute;top:14px;right:14px;width:30px;height:30px;
                 background:var(--surface2);border:1px solid var(--border);border-radius:8px;
                 font-size:18px;line-height:1;cursor:pointer;color:var(--muted);
                 display:flex;align-items:center;justify-content:center;transition:all .15s}
    .modal-close:hover{color:var(--text);border-color:var(--accent)}
    .modal-eyebrow{font-size:11px;font-weight:700;letter-spacing:.1em;text-transform:uppercase;
                   color:var(--accent);margin-bottom:8px}
    .modal-title{font-size:23px;font-weight:800;line-height:1.32;margin-bottom:16px;color:var(--text)}
    .modal-text{font-size:13.5px;line-height:1.75;color:var(--text);margin-bottom:13px}
    .modal-cta{padding:13px 15px;margin-top:4px;border-radius:9px;font-size:13px;color:var(--text);
               background:linear-gradient(135deg,rgba(37,99,235,.09) 0%,rgba(37,99,235,.02) 100%);
               border-left:4px solid var(--accent)}
    [data-theme="dark"] .modal-cta{background:linear-gradient(135deg,rgba(88,166,255,.13) 0%,rgba(88,166,255,.03) 100%)}
    .modal-stats{display:grid;grid-template-columns:repeat(4,1fr);gap:8px;margin:18px 0}
    .modal-stats .stat-box{padding:11px 6px;text-align:center}
    .modal-stats .stat-label{font-size:9px}
    .modal-stats .stat-val{font-size:19px}
    .modal-btn-row{display:flex;gap:8px;flex-wrap:wrap;margin-top:16px}
    /* ai pill */
    .ai-pill{display:flex;align-items:center;gap:6px;flex-shrink:0;min-width:200px;
             padding:5px 11px;border-radius:99px;font-size:11px;font-weight:700;
             font-family:'IBM Plex Mono',monospace;border:1px solid var(--border);
             background:var(--surface2);color:var(--muted);transition:all .3s}
    .ai-pill.loading{border-color:#f59e0b;color:#b45309;background:#fffbeb}
    .ai-pill.ready  {border-color:#16a34a;color:#15803d;background:#f0fdf4}
    .ai-pill.error  {border-color:#dc2626;color:#b91c1c;background:#fef2f2}
    .ai-dot{width:7px;height:7px;border-radius:50%;flex-shrink:0;background:currentColor}
    .ai-pill-body{display:flex;flex-direction:column;gap:3px;flex:1;min-width:0}
    .ai-pill-text{white-space:nowrap;overflow:hidden;text-overflow:ellipsis}
    .ai-progress-track{height:3px;border-radius:99px;background:rgba(0,0,0,.10);
                       overflow:hidden;display:none}
    .ai-progress-track.visible{display:block}
    .ai-progress-fill{height:100%;border-radius:99px;background:currentColor;
                      width:0%;transition:width .25s ease}
    .ai-pill.loading .ai-dot{animation:pulse .8s ease-in-out infinite alternate}
    @keyframes pulse{from{opacity:.3}to{opacity:1}}
    /* stabilisation overlay */
    .stab-overlay{position:absolute;inset:0;display:flex;flex-direction:column;
                  align-items:center;justify-content:center;gap:12px;z-index:6;
                  background:var(--stab-bg);backdrop-filter:blur(4px);transition:opacity .4s}
    .stab-overlay.hidden{opacity:0;pointer-events:none}
    .stab-spinner{width:36px;height:36px;border:3px solid var(--border);
                  border-top-color:var(--accent);border-radius:50%;
                  animation:spin .7s linear infinite}
    @keyframes spin{to{transform:rotate(360deg)}}
    .stab-text{font-size:12px;color:var(--muted);font-family:'IBM Plex Mono',monospace}
    .stab-bar-track{width:220px;height:4px;border-radius:99px;background:var(--border)}
    .stab-bar-fill{height:100%;border-radius:99px;background:var(--accent);
                   width:0%;transition:width .2s}
    /* network */
    .net-wrap{flex:1 1 0;min-height:0;position:relative;background:var(--net-bg)}
    #mynetwork{width:100%;height:100%}
    .net-toolbar{position:absolute;top:12px;left:12px;right:12px;display:flex;
                 justify-content:space-between;align-items:center;gap:10px;
                 pointer-events:none;z-index:4}
    .net-summary{background:var(--net-glass);backdrop-filter:blur(6px);
                 border:1px solid var(--border);border-radius:10px;
                 padding:6px 12px;font-size:11px;color:var(--muted)}
    .legend{display:flex;gap:10px;flex-wrap:wrap;font-size:11px;color:var(--muted);
            background:var(--net-glass);backdrop-filter:blur(6px);
            border:1px solid var(--border);border-radius:10px;padding:6px 12px}
    .legend-item{display:flex;align-items:center;gap:5px}
    .leg-dot{width:9px;height:9px;border-radius:50%}
    /* side panel */
    .side{flex:0 0 420px;width:420px;height:100vh;overflow-y:auto;
          background:var(--surface);border-left:1px solid var(--border);
          display:flex;flex-direction:column}
    .side-section{padding:13px 15px;border-bottom:1px solid var(--border)}
    .side-section:last-child{border-bottom:none;flex:1}
    .section-head{display:flex;justify-content:space-between;align-items:center;
                  font-size:10px;font-weight:700;text-transform:uppercase;
                  letter-spacing:.07em;color:var(--muted);margin-bottom:8px}
    .badge{background:#eff6ff;color:#1d4ed8;border-radius:99px;
           padding:2px 7px;font-size:10px;font-weight:800}
    /* search */
    .search-row{display:flex;gap:6px;align-items:center}
    .search-wrap{position:relative;flex:1}
    .search-icon{position:absolute;left:10px;top:50%;transform:translateY(-50%);
                 color:var(--muted);font-size:14px;pointer-events:none}
    .search-input{width:100%;padding:9px 30px 9px 30px;border:1px solid var(--border);
                  border-radius:10px;background:var(--surface2);color:var(--text);
                  font-size:12px;font-family:'IBM Plex Sans',sans-serif;outline:none;
                  transition:border-color .15s,box-shadow .15s}
    .search-input::placeholder{color:var(--muted)}
    .search-input:focus{border-color:var(--accent);box-shadow:0 0 0 3px rgba(37,99,235,.1)}
    .search-clear{position:absolute;right:8px;top:50%;transform:translateY(-50%);
                  background:none;border:none;color:var(--muted);cursor:pointer;
                  font-size:14px;display:none;padding:2px}
    .search-clear.visible{display:block}
    .search-btn{flex-shrink:0;padding:9px 14px;background:var(--accent);color:#fff;
                border:none;border-radius:10px;font-size:12px;font-weight:700;
                cursor:pointer;font-family:'IBM Plex Sans',sans-serif;
                transition:opacity .15s;white-space:nowrap}
    .search-btn:hover{opacity:.85}
    /* mode banner */
    .search-mode-banner{display:flex;align-items:center;gap:6px;padding:5px 9px;
                        border-radius:7px;font-size:10px;font-weight:700;
                        font-family:'IBM Plex Mono',monospace;margin-top:6px;
                        border:1px solid transparent;transition:all .25s}
    .smb-dot{width:6px;height:6px;border-radius:50%;flex-shrink:0}
    .search-mode-banner.smb-semantic{background:#f0fdf4;border-color:#86efac;color:#15803d}
    .search-mode-banner.smb-keyword {background:#eff6ff;border-color:#93c5fd;color:#1d4ed8}
    .search-mode-banner.smb-fallback{background:#fffbeb;border-color:#fcd34d;color:#92400e}
    .search-mode-banner.smb-error   {background:#fef2f2;border-color:#fca5a5;color:#991b1b}
    /* threshold / mode toggle */
    .threshold-row{display:flex;align-items:center;gap:8px;margin-top:6px;
                   font-size:11px;color:var(--muted)}
    .threshold-row input[type=range]{flex:1;accent-color:var(--accent)}
    .threshold-val{font-family:'IBM Plex Mono',monospace;font-size:11px;
                   color:var(--accent);min-width:28px;text-align:right}
    .mode-toggle{display:flex;gap:6px;margin-top:6px}
    .mode-pill{padding:5px 10px;border-radius:99px;font-size:11px;font-weight:700;
               border:1px solid var(--border);background:var(--surface2);color:var(--muted);
               cursor:pointer;transition:all .15s}
    .mode-pill.active{background:var(--accent);color:#fff;border-color:var(--accent)}
    /* syntax hint */
    .syntax-hint{background:var(--surface2);border:1px solid var(--border);
                 border-radius:8px;padding:8px 10px;margin-top:6px;
                 font-size:10px;color:var(--muted);display:none;line-height:1.7}
    .syntax-hint.visible{display:block}
    .syntax-hint code{font-family:'IBM Plex Mono',monospace;font-size:10px;
                      background:var(--border);border-radius:3px;padding:1px 4px;color:var(--text)}
    .hint-row{display:flex;justify-content:space-between;gap:6px}
    /* quality filters */
    .filter-group{margin-bottom:10px}
    .filter-group:last-child{margin-bottom:0}
    .filter-label{font-size:10px;font-weight:700;text-transform:uppercase;
                  letter-spacing:.07em;color:var(--muted);margin-bottom:5px}
    .filter-pills{display:flex;gap:5px}
    .filter-pill{flex:1;padding:6px 4px;border-radius:8px;font-size:11px;font-weight:700;
                 border:1px solid var(--border);background:var(--surface2);color:var(--muted);
                 cursor:pointer;font-family:'IBM Plex Sans',sans-serif;transition:all .15s;
                 text-align:center}
    .filter-pill:hover{border-color:var(--accent);color:var(--text)}
    .filter-pill.fp-any{background:var(--accent);color:#fff;border-color:var(--accent)}
    .filter-pill.fp-yes{background:#16a34a;color:#fff;border-color:#16a34a}
    .filter-pill.fp-no {background:#dc2626;color:#fff;border-color:#dc2626}
    /* explore hint */
    .explore-hint{font-size:11.5px;color:var(--text);line-height:1.6;
                  padding:10px 12px 10px 14px;
                  background:linear-gradient(135deg,rgba(37,99,235,.08) 0%,rgba(37,99,235,.03) 100%);
                  border-radius:9px;border-left:4px solid var(--accent);
                  margin-bottom:12px;box-shadow:0 1px 4px rgba(37,99,235,.08)}
    .explore-hint strong{color:var(--accent);font-weight:700}
    [data-theme="dark"] .explore-hint{
      background:linear-gradient(135deg,rgba(88,166,255,.12) 0%,rgba(88,166,255,.04) 100%);
      box-shadow:0 1px 4px rgba(88,166,255,.1)}
    /* buttons */
    .btn-row{display:flex;gap:6px;margin-top:8px;flex-wrap:wrap}
    .btn{padding:7px 12px;border-radius:8px;font-size:11px;font-weight:700;
         border:1px solid transparent;cursor:pointer;
         font-family:'IBM Plex Sans',sans-serif;transition:opacity .15s,transform .1s}
    .btn:hover{opacity:.85;transform:translateY(-1px)}
    .btn-primary  {background:var(--accent);color:#fff}
    .btn-secondary{background:var(--surface2);color:var(--text);border-color:var(--border)}
    /* stats */
    .stats-grid{display:grid;grid-template-columns:1fr 1fr;gap:6px}
    .stat-box{background:var(--surface2);border:1px solid var(--border);
              border-radius:10px;padding:9px}
    .stat-label{font-size:10px;color:var(--muted);margin-bottom:3px;
                text-transform:uppercase;letter-spacing:.05em}
    .stat-val{font-size:20px;font-weight:800;font-family:'IBM Plex Mono',monospace}
    /* category list */
    .group-list{display:flex;flex-direction:column;gap:5px;max-height:220px;overflow-y:auto}
    .group-item{width:100%;text-align:left;border:1px solid var(--border);
                background:var(--surface2);color:var(--text);padding:8px 10px;
                border-radius:9px;font-size:11px;font-weight:600;cursor:pointer;
                font-family:'IBM Plex Sans',sans-serif;transition:border-color .12s;
                display:flex;align-items:center;justify-content:space-between;gap:6px}
    .group-item:hover{border-color:var(--accent)}
    .group-item.selected{background:#eff6ff;border-color:#93c5fd;color:#1e3a8a}
    .group-item.matched {border-color:var(--accent)}
    .sim-score{font-family:'IBM Plex Mono',monospace;font-size:10px;
               color:var(--accent);flex-shrink:0}
    .empty-note{border:1px dashed var(--border);border-radius:9px;padding:9px;
                color:var(--muted);font-size:11px;text-align:center}
    /* detail card */
    .detail-card{background:var(--surface2);border:1px solid var(--border);
                 border-radius:10px;padding:12px;font-size:11px;color:var(--muted);
                 line-height:1.6;transition:opacity .15s,transform .15s}
    .detail-card.animating{opacity:0;transform:translateY(5px)}
    .detail-title{font-size:13px;font-weight:800;color:var(--text);
                  margin-bottom:8px;line-height:1.35}
    .dlabel{font-size:9px;font-weight:700;letter-spacing:.07em;text-transform:uppercase;
            color:var(--muted);margin-top:9px;margin-bottom:3px}
    .dval{color:var(--text);font-size:11px;white-space:pre-wrap;line-height:1.5;
          word-break:break-word}
    .tag{display:inline-block;padding:2px 6px;border-radius:4px;
         font-size:10px;font-weight:700;margin:1px}
    .tag-a{background:rgba(251,106,74,.15);color:#c2410c}
    .tag-c{background:rgba(107,174,214,.15);color:#1d4ed8}
    /* multi-category chips in detail panel */
    .cat-chips{display:flex;flex-wrap:wrap;gap:4px;margin-top:4px}
    .cat-chip{display:inline-block;padding:3px 8px;border-radius:6px;font-size:10px;
              font-weight:600;background:rgba(107,174,214,.15);color:#1d4ed8;
              border:1px solid rgba(107,174,214,.4)}
    /* quality badges */
    .badge-row{display:flex;flex-wrap:wrap;gap:5px;margin-top:4px}
    .qbadge{display:inline-flex;align-items:center;gap:4px;padding:3px 8px;
            border-radius:6px;font-size:10px;font-weight:700;
            background:var(--surface);border:1px solid var(--border);color:var(--muted)}
    .qbadge.yes{background:#f0fdf4;border-color:#86efac;color:#15803d}
    .qbadge.no {background:#fef2f2;border-color:#fca5a5;color:#991b1b}
    /* sim bar */
    .sim-bar-wrap{margin-top:8px}
    .sim-bar-label{font-size:9px;color:var(--muted);margin-bottom:4px;
                   text-transform:uppercase;letter-spacing:.06em}
    .sim-bar-track{height:5px;border-radius:99px;background:var(--border);overflow:hidden}
    .sim-bar-fill{height:100%;border-radius:99px;background:var(--accent);transition:width .4s ease}
    .theme-toggle{flex-shrink:0;padding:6px 11px;border-radius:8px;font-size:15px;
                  border:1px solid var(--border);background:var(--surface2);cursor:pointer;
                  line-height:1;transition:border-color .15s,background .2s;
                  display:flex;align-items:center;justify-content:center}
    .theme-toggle:hover{border-color:var(--accent)}
    [data-theme="dark"] .group-item.selected{background:#1c2d4a;border-color:#3b82f6;color:#93c5fd}
    [data-theme="dark"] .badge{background:rgba(59,130,246,.2);color:#93c5fd}
    [data-theme="dark"] .qbadge.yes{background:#0f2b1a;border-color:#22c55e;color:#4ade80}
    [data-theme="dark"] .qbadge.no {background:#2b0f0f;border-color:#ef4444;color:#f87171}
    [data-theme="dark"] .search-mode-banner.smb-semantic{background:#0f2b1a;border-color:#22c55e;color:#4ade80}
    [data-theme="dark"] .search-mode-banner.smb-keyword {background:#0f1e3a;border-color:#3b82f6;color:#93c5fd}
    [data-theme="dark"] .search-mode-banner.smb-fallback{background:#2b200a;border-color:#f59e0b;color:#fbbf24}
    [data-theme="dark"] .search-mode-banner.smb-error   {background:#2b0f0f;border-color:#ef4444;color:#f87171}
    [data-theme="dark"] .ai-pill.loading{border-color:#f59e0b;color:#fbbf24;background:#2b200a}
    [data-theme="dark"] .ai-pill.ready  {border-color:#22c55e;color:#4ade80;background:#0f2b1a}
    [data-theme="dark"] .ai-pill.error  {border-color:#ef4444;color:#f87171;background:#2b0f0f}
    [data-theme="dark"] .tag-a{background:rgba(249,115,22,.2);color:#fb923c}
    [data-theme="dark"] .tag-c{background:rgba(91,155,213,.2);color:#93c5fd}
    [data-theme="dark"] .cat-chip{background:rgba(91,155,213,.2);color:#93c5fd;border-color:rgba(91,155,213,.4)}
    [data-theme="dark"] .filter-pill.fp-yes{background:#166534;color:#4ade80;border-color:#16a34a}
    [data-theme="dark"] .filter-pill.fp-no {background:#7f1d1d;color:#f87171;border-color:#dc2626}
    /* ── right-panel text scale ─────────────────────────────────────────── */
    .side .section-head{font-size:12px}
    .side .badge{font-size:12px}
    .side .search-input{font-size:13px}
    .side .search-btn{font-size:13px}
    .side .search-mode-banner{font-size:12px}
    .side .threshold-row{font-size:12px}
    .side .threshold-val{font-size:12px}
    .side .mode-pill{font-size:13px}
    .side .syntax-hint{font-size:12px}
    .side .syntax-hint code{font-size:12px}
    .side .filter-label{font-size:12px}
    .side .filter-pill{font-size:13px}
    .side .explore-hint{font-size:13px}
    .side .stat-label{font-size:12px}
    .side .stat-val{font-size:24px}
    .side .group-item{font-size:13px}
    .side .sim-score{font-size:12px}
    .side .empty-note{font-size:12px}
    .side .detail-card{font-size:13px}
    .side .detail-title{font-size:15px}
    .side .dlabel{font-size:11px}
    .side .dval{font-size:13px}
    .side .tag{font-size:12px}
    .side .cat-chip{font-size:12px}
    .side .qbadge{font-size:12px}
    .side .sim-bar-label{font-size:11px}
    .side .btn{font-size:13px}
    @media(max-width:960px){
      .app{flex-direction:column;height:auto}
      .main,.net-wrap{min-height:520px}
      .side{height:auto;width:100%;flex:none}
    }
  </style>
</head>
<body>
<div class="modal-overlay" id="aboutOverlay" onclick="if(event.target===this) closeAbout()">
  <div class="modal-card">
    <button class="modal-close" onclick="closeAbout()" aria-label="Close">&times;</button>
    <div class="modal-eyebrow">About ReACTive</div>
    <div class="modal-title">From software engineering research to practitioner action &mdash; instantly.</div>
    <p class="modal-text">Every year, thousands of software engineering research papers uncover what actually works &mdash; better onboarding practices, sturdier CI/CD pipelines, safer security habits, healthier open-source communities. Almost none of it ever reaches the people building real software. <strong>ReACTive closes that gap.</strong></p>
    <p class="modal-text">ReACTive is an intelligent discovery engine that turns the collective knowledge of the software engineering research community into a single, searchable catalog of <strong>ready-to-adopt, evidence-backed actions</strong>. Instead of reading hundreds of papers, practitioners, maintainers, and engineering leaders can search, browse, and filter by category to find concrete recommendations &mdash; each paired with its real-world impact and the empirical evidence behind it &mdash; and put them to work immediately.</p>
    <p class="modal-text">Behind the scenes, ReACTive is powered by a comprehensive, AI-assisted mining pipeline that has systematically analyzed papers from top-tier software engineering venues to extract, validate, and organize this catalog of actionable recommendations &mdash; so the hard work of finding what works has already been done for you.</p>
    <div class="modal-stats">
      <div class="stat-box"><div class="stat-label">Categories</div><div class="stat-val" style="color:#6baed6" id="aboutCats">0</div></div>
      <div class="stat-box"><div class="stat-label">Actionables</div><div class="stat-val" style="color:#fb6a4a" id="aboutActions">0</div></div>
      <div class="stat-box"><div class="stat-label">Source Papers</div><div class="stat-val" id="aboutRows">0</div></div>
      <div class="stat-box"><div class="stat-label">Connections</div><div class="stat-val" id="aboutEdges">0</div></div>
    </div>
    <p class="modal-text modal-cta">ReACTive is a University of California, Davis technology, offered here as supplementary material. Organizations interested in licensing, integrating, or collaborating around this technology are warmly invited to reach out &mdash; <a href="mailto:nikhan@ucdavis.edu" style="color:var(--accent);font-weight:700">nikhan@ucdavis.edu</a>.</p>
    <div class="modal-btn-row">
      <a class="btn btn-primary" href="__DATASET_URL__" target="_blank" rel="noopener">View the full dataset &rarr;</a>
      <button class="btn btn-secondary" onclick="closeAbout()">Start exploring</button>
    </div>
  </div>
</div>
<div class="app">
  <main class="main">
    <div class="topbar">
      <div class="topbar-brand">
        <div class="eyebrow">ReACTive</div>
        <h1>Intelligent Actionable Search Tool</h1>
      </div>
      <div class="topbar-actions">
        <button class="btn-topbar" id="aboutBtn" onclick="openAbout()">&#8505;&#65039; About</button>
        <a class="btn-topbar btn-topbar-accent" href="__DATASET_URL__" target="_blank" rel="noopener">&#128202; View Dataset</a>
      </div>
      <button class="theme-toggle" id="themeToggle" onclick="toggleTheme()"
              title="Toggle dark / light mode" aria-label="Toggle theme">
        <span id="themeIcon">☀️</span>
      </button>
      <div class="ai-pill" id="aiPill">
        <span class="ai-dot"></span>
        <div class="ai-pill-body">
          <span class="ai-pill-text" id="aiPillText">Initialising&hellip;</span>
          <div class="ai-progress-track" id="aiProgressTrack">
            <div class="ai-progress-fill" id="aiProgressFill"></div>
          </div>
        </div>
      </div>
    </div>

    <div class="net-wrap">
      <div class="net-toolbar">
        <div class="net-summary" id="netSummary">Loading&hellip;</div>
        <div class="legend">
          <span class="legend-item"><span class="leg-dot" style="background:#6baed6"></span>Category</span>
          <span class="legend-item"><span class="leg-dot" style="background:#fb6a4a"></span>Actionable</span>
        </div>
      </div>
      <div class="stab-overlay" id="stabOverlay">
        <div class="stab-spinner"></div>
        <div class="stab-text" id="stabText">Laying out graph&hellip;</div>
        <div class="stab-bar-track"><div class="stab-bar-fill" id="stabBarFill"></div></div>
      </div>
      <div id="mynetwork"></div>
    </div>
  </main>

  <aside class="side">
    <!-- ── Search ─────────────────────────────────────────────────────── -->
    <div class="side-section">
      <div class="section-head">Search <span style="font-weight:400;text-transform:none;letter-spacing:0;font-size:9px;color:var(--accent)">&nbsp;(matches Impact field)</span></div>
      <div class="search-row">
        <div class="search-wrap">
          <span class="search-icon">&#128269;</span>
          <input id="searchBox" class="search-input" type="text"
                 placeholder="Search by impact description&hellip;"/>
          <button class="search-clear" id="searchClear" title="Clear">&#10005;</button>
        </div>
        <button class="search-btn" onclick="commitSearch()">Search</button>
      </div>
      <div class="search-mode-banner smb-fallback" id="searchModeBanner">
        <span class="smb-dot"></span>
        <span id="smbText">Keyword search active &mdash; semantic model loading&hellip;</span>
      </div>
      <div class="threshold-row" id="thresholdRow">
        <span>Threshold</span>
        <input type="range" id="thresholdSlider" min="5" max="80" value="30" step="1"/>
        <span class="threshold-val" id="thresholdVal">0.30</span>
      </div>
      <div class="mode-toggle">
        <button class="mode-pill active" id="pillSemantic" onclick="setSearchMode('semantic')">&#129504; Semantic</button>
        <button class="mode-pill"        id="pillKeyword"  onclick="setSearchMode('keyword')">&#128190; Keyword</button>
      </div>
      <div class="syntax-hint" id="syntaxHint">
        <div style="font-weight:700;color:var(--text);margin-bottom:4px">Keyword syntax</div>
        <div class="hint-row"><span><code>reduce mortality</code></span><span>AND</span></div>
        <div class="hint-row"><span><code>screening OR prevention</code></span><span>either</span></div>
        <div class="hint-row"><span><code>"early detection"</code></span><span>exact phrase</span></div>
        <div class="hint-row"><span><code>screen* -bias</code></span><span>wildcard / exclude</span></div>
      </div>
      <div class="btn-row">
        <button class="btn btn-secondary" onclick="fitView()">Fit view</button>
        <button class="btn btn-primary"   onclick="showAll()">Show all</button>
      </div>
    </div>

    <!-- ── Quality Filters ───────────────────────────────────────────── -->
    <div class="side-section">
      <div class="section-head">
        Quality Filters
        <span class="badge" id="filterActiveBadge" style="display:none">0 active</span>
      </div>

      <div class="filter-group">
        <div class="filter-label">&#128264; Sound</div>
        <div class="filter-pills">
          <button class="filter-pill fp-any" id="soundAny" onclick="setSoundFilter('any')">Any</button>
          <button class="filter-pill"        id="soundYes" onclick="setSoundFilter('yes')">&#10003; Yes</button>
          <button class="filter-pill"        id="soundNo"  onclick="setSoundFilter('no')">&#10007; No</button>
        </div>
      </div>

      <div class="filter-group">
        <div class="filter-label">&#127919; Precise</div>
        <div class="filter-pills">
          <button class="filter-pill fp-any" id="preciseAny" onclick="setPreciseFilter('any')">Any</button>
          <button class="filter-pill"        id="preciseYes" onclick="setPreciseFilter('yes')">&#10003; Yes</button>
          <button class="filter-pill"        id="preciseNo"  onclick="setPreciseFilter('no')">&#10007; No</button>
        </div>
      </div>

      <div class="btn-row">
        <button class="btn btn-secondary" onclick="resetQualityFilters()">Reset filters</button>
      </div>
    </div>

    <!-- ── Stats ─────────────────────────────────────────────────────── -->
    <div class="side-section">
      <p class="explore-hint"><strong>Browse by category</strong> to focus the graph &mdash; only one category loads by default for speed, others load the first time you pick them. Click any actionable node to inspect its insight, evidence, and quality indicators.</p>
      <div class="stats-grid">
        <div class="stat-box">
          <div class="stat-label">Categories</div>
          <div class="stat-val" style="color:#6baed6" id="statCats">0</div>
        </div>
        <div class="stat-box">
          <div class="stat-label">Actionables</div>
          <div class="stat-val" style="color:#fb6a4a" id="statActions">0</div>
        </div>
        <div class="stat-box">
          <div class="stat-label">Selected</div>
          <div class="stat-val" style="color:var(--accent)" id="statSelected">0</div>
        </div>
        <div class="stat-box">
          <div class="stat-label">Visible edges</div>
          <div class="stat-val" id="statEdges">0</div>
        </div>
      </div>
    </div>

    <!-- ── Categories ────────────────────────────────────────────────── -->
    <div class="side-section">
      <div class="section-head">
        <span>Categories</span>
        <span class="badge" id="catCountBadge">0</span>
      </div>
      <div id="groupList" class="group-list"></div>
      <div class="btn-row" style="margin-top:8px">
        <button class="btn btn-secondary" onclick="selectVisible()">Select visible</button>
        <button class="btn btn-secondary" onclick="clearSelection()">Clear</button>
      </div>
    </div>

    <!-- ── Details ───────────────────────────────────────────────────── -->
    <div class="side-section">
      <div class="section-head">Details</div>
      <div id="detailPanel" class="detail-card">Click any node to inspect details.</div>
    </div>
  </aside>
</div>

<script>
// ── injected data ──────────────────────────────────────────────────────────────
// ALL_NODES / ALL_NODE_META / ALL_EDGES start out scoped to DEFAULT_CATEGORY only
// (plus the 8 category hub nodes) -- every other category's data lives in a
// side-car JSON file (CATEGORY_DATA_FILES) fetched lazily by loadCategoryData()
// the first time the user selects it. This keeps first paint fast: only one
// category's actionables are downloaded, parsed, and physics-laid-out up front.
let ALL_NODES           = __NODES_JSON__;
let ALL_NODE_META       = __NODE_META_JSON__;
let ALL_EDGES           = __EDGES_JSON__;
const GRAPH_STATS       = __STATS_JSON__;
const BAKED_EMBEDDINGS  = __BAKED_EMBEDDINGS__;
const CATEGORY_DATA_FILES = __CATEGORY_FILES_JSON__;   // category node id -> relative JSON path
const DEFAULT_CATEGORY    = __DEFAULT_CATEGORY_JSON__;
// ──────────────────────────────────────────────────────────────────────────────

// ── lookup structures (mutated in place as more categories are loaded) ─────────
const nodeMap     = new Map(ALL_NODE_META.map(m => [String(m.id), m]));
let categoryIds = ALL_NODE_META
  .filter(m => m.node_type === "category")
  .map(m => String(m.id))
  .sort((a,b) => (nodeMap.get(a)?.category||"").localeCompare(nodeMap.get(b)?.category||""));
let actionableIds = ALL_NODE_META
  .filter(m => m.node_type === "actionable")
  .map(m => String(m.id));

// adjacency — category → actionables, actionable → categories
const catToActions = new Map(categoryIds.map(id => [id, new Set()]));
const actionToCats = new Map(actionableIds.map(id => [id, new Set()]));
for (const e of ALL_EDGES) {
  const f = String(e.from), t = String(e.to);
  if (!catToActions.has(f)) catToActions.set(f, new Set());
  if (!actionToCats.has(t)) actionToCats.set(t, new Set());
  catToActions.get(f).add(t);
  actionToCats.get(t).add(f);
}

// ── state ──────────────────────────────────────────────────────────────────────
let network            = null;
let nodesDS            = null;
let edgesDS            = null;
let selectedCats       = new Set();
let committedSearch    = "";
let searchMode         = "semantic";
let threshold          = 0.30;
let lastActiveMode     = "fallback";
let embedder           = null;
let nodeEmbeddings     = null;
let embedReady         = false;
let simScores          = new Map();
let loadedCategoryIds  = new Set();   // categories whose actionable data is in memory
let loadingCategoryIds = new Set();   // categories with an in-flight fetch

// quality filter state
let soundFilter     = "any";
let preciseFilter   = "any";

// ── quality filter helpers ─────────────────────────────────────────────────────
function isYesVal(v) { return /^(yes|true|1|high|strong)$/i.test((v||"").trim()); }
function isNoVal(v)  { return /^(no|false|0|low|weak)$/i.test((v||"").trim()); }

/**
 * Returns true if the node passes the current SOUND + PRECISE filters.
 * Category nodes always pass (they are filtered indirectly via their actionables).
 */
function passesQualityFilter(m) {
  if (!m || m.node_type === "category") return true;
  if (soundFilter !== "any") {
    if (soundFilter === "yes" && !isYesVal(m.sound))  return false;
    if (soundFilter === "no"  && !isNoVal(m.sound))   return false;
  }
  if (preciseFilter !== "any") {
    if (preciseFilter === "yes" && !isYesVal(m.precise)) return false;
    if (preciseFilter === "no"  && !isNoVal(m.precise))  return false;
  }
  return true;
}

function updateFilterBadge() {
  const n = (soundFilter !== "any" ? 1 : 0) + (preciseFilter !== "any" ? 1 : 0);
  const badge = document.getElementById("filterActiveBadge");
  badge.textContent = n + " active";
  badge.style.display = n > 0 ? "" : "none";
}

function setSoundFilter(val) {
  soundFilter = val;
  for (const v of ["any","yes","no"]) {
    const btn = document.getElementById("sound" + v.charAt(0).toUpperCase() + v.slice(1));
    if (btn) btn.className = "filter-pill" + (v === val ? " fp-" + val : "");
  }
  updateFilterBadge(); applyStyles(); renderCategoryPanel();
}

function setPreciseFilter(val) {
  preciseFilter = val;
  for (const v of ["any","yes","no"]) {
    const btn = document.getElementById("precise" + v.charAt(0).toUpperCase() + v.slice(1));
    if (btn) btn.className = "filter-pill" + (v === val ? " fp-" + val : "");
  }
  updateFilterBadge(); applyStyles(); renderCategoryPanel();
}

function resetQualityFilters() {
  soundFilter = "any"; preciseFilter = "any";
  ["sound","precise"].forEach(k => {
    for (const v of ["any","yes","no"]) {
      const btn = document.getElementById(k + v.charAt(0).toUpperCase() + v.slice(1));
      if (btn) btn.className = "filter-pill" + (v === "any" ? " fp-any" : "");
    }
  });
  updateFilterBadge(); applyStyles(); renderCategoryPanel();
}

// ── AI pill ────────────────────────────────────────────────────────────────────
function setAIPill(state, text, pct) {
  const pill  = document.getElementById("aiPill");
  const track = document.getElementById("aiProgressTrack");
  const fill  = document.getElementById("aiProgressFill");
  pill.className = "ai-pill " + state;
  document.getElementById("aiPillText").textContent = text;
  if (pct != null) { track.classList.add("visible"); fill.style.width = pct + "%"; }
  else             { track.classList.remove("visible"); fill.style.width = "0%"; }
}
function setModeBanner(state) {
  const el = document.getElementById("searchModeBanner");
  el.className = "search-mode-banner smb-" +
    (state==="semantic"?"semantic":state==="keyword"?"keyword":state==="error"?"error":"fallback");
  const labels = {
    semantic: "🧠 Semantic search active — matching Impact field",
    keyword:  "🔤 Keyword search active — matching Impact field",
    fallback: "⚠ Keyword fallback — semantic model loading…",
    error:    "⚠ Keyword fallback — semantic model unavailable",
  };
  document.getElementById("smbText").textContent = labels[state] || labels.fallback;
}

// ── baked embeddings ───────────────────────────────────────────────────────────
function loadBakedEmbeddings() {
  if (!BAKED_EMBEDDINGS) return null;
  const map = new Map();
  for (const [id, b64] of Object.entries(BAKED_EMBEDDINGS)) {
    const bin = atob(b64);
    const buf = new Uint8Array(bin.length);
    for (let i = 0; i < bin.length; i++) buf[i] = bin.charCodeAt(i);
    map.set(id, new Float32Array(buf.buffer));
  }
  return map;
}

async function bootstrapEmbedder() {
  const baked = loadBakedEmbeddings();
  if (!baked) {
    setAIPill("error", "No baked embeddings — keyword only");
    lastActiveMode = "error"; setModeBanner("error"); return;
  }
  nodeEmbeddings = baked;
  setAIPill("loading", `${baked.size} nodes — loading query model…`, 0);
  if (!window.__transformersReady)
    await new Promise(r => window.addEventListener('transformers-ready', r, {once:true}));
  try {
    const { pipeline } = window.__transformers;
    embedder = await pipeline("feature-extraction", "Xenova/all-MiniLM-L6-v2", {
      progress_callback: (p) => {
        if (p.status === "progress" && p.total) {
          const pct = Math.round(p.loaded / p.total * 100);
          setAIPill("loading", `Loading query model… ${pct}%`, pct);
        }
      }
    });
    embedReady = true;
    setAIPill("ready", `Semantic search ready ✓  (${baked.size} nodes)`);
    lastActiveMode = "semantic"; setModeBanner("semantic");
    if (committedSearch) runSearch();
  } catch(err) {
    console.error("Transformers.js:", err);
    setAIPill("error", "Query model failed — keyword only");
    lastActiveMode = "error"; setModeBanner("error");
  }
}

function cosineSim(a, b) {
  let dot = 0;
  for (let i = 0; i < a.length; i++) dot += a[i] * b[i];
  return dot;
}
async function embedQuery(text) {
  const out = await embedder([text], {pooling:"mean", normalize:true});
  return out.data.slice(0, out.data.length);
}

// ── lazy category loading ───────────────────────────────────────────────────────
/** Fetch one category's side-car JSON (nodes/node_meta/edges/embeddings). */
async function fetchCategoryChunk(id) {
  const path = CATEGORY_DATA_FILES[id];
  if (!path) return null;
  const resp = await fetch(path);
  if (!resp.ok) throw new Error(`HTTP ${resp.status} loading ${path}`);
  return resp.json();
}

/** Merge a fetched chunk into the in-memory graph. Returns true if anything new was added. */
function mergeCategoryChunk(chunk) {
  if (!chunk) return false;
  let changed = false;

  const newVisNodes = [];
  for (const n of (chunk.nodes || [])) {
    const id = String(n.id);
    if (nodeMap.has(id)) continue;
    ALL_NODES.push(n);
    newVisNodes.push(n);
  }
  for (const m of (chunk.node_meta || [])) {
    const id = String(m.id);
    if (nodeMap.has(id)) continue;
    ALL_NODE_META.push(m);
    nodeMap.set(id, m);
    if (m.node_type === "actionable") actionableIds.push(id);
    changed = true;
  }

  const newVisEdges = [];
  const existingEdgeIds = edgesDS ? new Set(edgesDS.getIds().map(String)) : new Set();
  for (const e of (chunk.edges || [])) {
    const f = String(e.from), t = String(e.to);
    if (!catToActions.has(f)) catToActions.set(f, new Set());
    if (!actionToCats.has(t)) actionToCats.set(t, new Set());
    catToActions.get(f).add(t);
    actionToCats.get(t).add(f);
    if (existingEdgeIds.has(String(e.id))) continue;
    ALL_EDGES.push(e);
    newVisEdges.push(e);
    changed = true;
  }

  if (nodeEmbeddings) {
    for (const [id, b64] of Object.entries(chunk.embeddings || {})) {
      if (nodeEmbeddings.has(id)) continue;
      const bin = atob(b64);
      const buf = new Uint8Array(bin.length);
      for (let i = 0; i < bin.length; i++) buf[i] = bin.charCodeAt(i);
      nodeEmbeddings.set(id, new Float32Array(buf.buffer));
    }
  }

  if (newVisNodes.length) nodesDS.update(newVisNodes.map(n => visNode(n)));
  if (newVisEdges.length) edgesDS.update(newVisEdges.map(e => ({id:e.id, from:e.from, to:e.to, width:e.width})));
  return changed;
}

/** Re-run physics briefly so newly-added nodes settle, then freeze again. */
function restabilize() {
  if (!network) return;
  const overlay  = document.getElementById("stabOverlay");
  const stabText = document.getElementById("stabText");
  const stabBar  = document.getElementById("stabBarFill");
  overlay.classList.remove("hidden");
  stabText.textContent = "Laying out new actionables…";
  stabBar.style.width = "0%";
  network.setOptions({physics:{enabled:true, stabilization:{enabled:true, iterations:200, fit:true}}});
  network.once("stabilizationIterationsDone", () => {
    network.setOptions({physics:{enabled:false}});
    network.fit({animation:{duration:500, easingFunction:"easeInOutQuad"}});
    overlay.classList.add("hidden");
    applyStyles(); updateSummary(); renderCategoryPanel();
  });
  network.stabilize(200);
}

/** Fetch + merge a batch of not-yet-loaded categories as one unit (one restabilize). */
async function loadCategories(ids) {
  const targets = [...new Set(ids)].filter(
    id => !loadedCategoryIds.has(id) && !loadingCategoryIds.has(id) && CATEGORY_DATA_FILES[id]
  );
  if (!targets.length) return;
  targets.forEach(id => loadingCategoryIds.add(id));
  renderCategoryPanel();

  const chunks = await Promise.all(targets.map(id =>
    fetchCategoryChunk(id).catch(err => { console.error("Failed to load category", id, err); return null; })
  ));

  let changed = false;
  chunks.forEach((chunk, i) => {
    if (mergeCategoryChunk(chunk)) changed = true;
    loadedCategoryIds.add(targets[i]);
    loadingCategoryIds.delete(targets[i]);
  });

  if (changed) restabilize();
  else { applyStyles(); renderCategoryPanel(); }
}

/** Used by "Show all" / a committed search — pull in every remaining category. */
async function ensureAllCategoriesLoaded() {
  const toLoad = categoryIds.filter(id => !loadedCategoryIds.has(id) && CATEGORY_DATA_FILES[id]);
  if (!toLoad.length) return;
  const prevAiState = document.getElementById("aiPill").className;
  setAIPill("loading", `Loading ${toLoad.length} more categor${toLoad.length===1?"y":"ies"}…`, 0);
  await loadCategories(toLoad);
  if (embedReady) setAIPill("ready", `Semantic search ready ✓  (${nodeEmbeddings?nodeEmbeddings.size:0} nodes)`);
  else if (!prevAiState.includes("loading")) setAIPill("loading", "Initialising…");
}

// ── search ─────────────────────────────────────────────────────────────────────
async function commitSearch() {
  committedSearch = document.getElementById("searchBox").value.trim();
  if (committedSearch) await ensureAllCategoriesLoaded();
  runSearch();
}
async function runSearch() {
  const term = committedSearch;
  if (!term) {
    simScores = new Map();
    setModeBanner(embedReady ? (searchMode==="keyword"?"keyword":"semantic") : lastActiveMode);
    applyStyles(); renderCategoryPanel(); return;
  }
  if (searchMode === "semantic" && embedReady) {
    try {
      const qVec = await embedQuery(term);
      simScores = new Map();
      for (const [id, vec] of nodeEmbeddings)
        simScores.set(id, Math.max(0, cosineSim(qVec, vec)));
      lastActiveMode = "semantic"; setModeBanner("semantic");
    } catch(e) { buildKeywordScores(term); lastActiveMode="error"; setModeBanner("error"); }
  } else if (searchMode === "semantic" && !embedReady) {
    buildKeywordScores(term); lastActiveMode="fallback"; setModeBanner("fallback");
  } else {
    buildKeywordScores(term); lastActiveMode="keyword"; setModeBanner("keyword");
  }
  applyStyles(); renderCategoryPanel();
}

// ── keyword parser ─────────────────────────────────────────────────────────────
function parseQuery(raw) {
  const phrases = [];
  const phraseRe = /"([^"]+)"/g; let pm;
  while ((pm = phraseRe.exec(raw)) !== null) phrases.push(pm[1].trim().toLowerCase());
  let s = raw.replace(phraseRe, "\u0000").toLowerCase();
  const tokens = []; let pi = 0;
  for (const part of s.split(/\s+/)) {
    if (!part) continue;
    if (part === "\u0000") { if (pi < phrases.length) tokens.push({type:"phrase",value:phrases[pi++]}); }
    else tokens.push({type:"raw",value:part});
  }
  const orGroups = [[]], excludes = [];
  for (const tok of tokens) {
    if (tok.type==="phrase") { orGroups[orGroups.length-1].push({type:"phrase",value:tok.value}); continue; }
    const v = tok.value;
    if (v==="or") { orGroups.push([]); continue; }
    if (v.startsWith("-")&&v.length>1) { excludes.push(v.slice(1)); continue; }
    orGroups[orGroups.length-1].push({type:v.endsWith("*")?"wildcard":"term",value:v.endsWith("*")?v.slice(0,-1):v});
  }
  return {orGroups, excludes};
}
function scoreKeyword(blob, parsed) {
  const {orGroups, excludes} = parsed;
  for (const ex of excludes) if (blob.includes(ex)) return 0;
  let best = 0;
  for (const group of orGroups) {
    if (!group.length) continue;
    const matched = group.filter(t => blob.includes(t.value)).length;
    const s = matched / group.length;
    if (s > best) best = s;
  }
  return best;
}

function buildKeywordScores(term) {
  const parsed = parseQuery(term); simScores = new Map();
  for (const [id, m] of nodeMap)
    simScores.set(id, scoreKeyword((m.search_blob||"").toLowerCase(), parsed));
}

// ── tooltip ────────────────────────────────────────────────────────────────────
function buildTooltip(n) {
  const m = nodeMap.get(String(n.id));
  if (!m) return n.label || String(n.id);
  if (m.node_type === "actionable") return m.actionable;
  return m.category;
}

// ── vis-network ────────────────────────────────────────────────────────────────
const VIS_NODE_KEYS = new Set([
  "id","label","shape","size","color","borderWidth","font","shadow","opacity","title"
]);
function sanitizeVisNode(input) {
  const out = {};
  for (const k of VIS_NODE_KEYS) if (input[k] !== undefined) out[k] = input[k];
  return out;
}
function visNode(n) {
  return { ...sanitizeVisNode(n), opacity: 1, title: buildTooltip(n) };
}

function initNetwork() {
  const container = document.getElementById("mynetwork");
  nodesDS = new vis.DataSet(ALL_NODES.map(n => visNode(n)));
  edgesDS = new vis.DataSet(ALL_EDGES.map(e => ({id:e.id, from:e.from, to:e.to, width:e.width})));

  network = new vis.Network(container, {nodes: nodesDS, edges: edgesDS}, {
    autoResize: true,
    physics: {
      enabled: true,
      stabilization: { enabled:true, iterations:500, fit:true },
      barnesHut: {
        gravitationalConstant: -8000, centralGravity: 0.3,
        springLength: 150, springConstant: 0.04,
        damping: 0.2, avoidOverlap: 0.5,
      },
    },
    interaction: {hover:true, tooltipDelay:100, navigationButtons:true, keyboard:true},
    nodes: {font:{face:"IBM Plex Sans, sans-serif", color:"#1e293b"}},
    edges: {selectionWidth:3, hoverWidth:2, smooth:{enabled:true, type:"dynamic"}},
  });

  const overlay  = document.getElementById("stabOverlay");
  const stabText = document.getElementById("stabText");
  const stabBar  = document.getElementById("stabBarFill");
  overlay.classList.remove("hidden");

  network.on("stabilizationProgress", e => {
    const pct = Math.round(e.iterations / e.total * 100);
    stabText.textContent = `Laying out graph… ${pct}%`;
    stabBar.style.width = pct + "%";
  });
  network.once("stabilizationIterationsDone", () => {
    network.setOptions({physics:{enabled:false}});
    network.fit({animation:{duration:500, easingFunction:"easeInOutQuad"}});
    overlay.classList.add("hidden");
    applyStyles(); updateSummary();
  });
  network.on("click", params => {
    if (params.nodes && params.nodes.length) {
      const node = nodeMap.get(String(params.nodes[0]));
      if (node) {
        showDetail(node);
        // After the detail panel is populated, scroll it into view inside the side panel
        requestAnimationFrame(() => {
          const panel = document.getElementById("detailPanel");
          if (panel) panel.scrollIntoView({ behavior: "smooth", block: "nearest" });
        });
      }
    } else {
      resetDetail();
    }
  });
}

// ── core visibility logic ──────────────────────────────────────────────────────
/**
 * Compute visibility helpers for the current state of all filters.
 * Returns { isActionableVisible, isCategoryVisible, isVisible, isDirect }.
 * Called from both applyStyles() and renderCategoryPanel() to keep logic DRY.
 */
function buildVisibility() {
  const term      = committedSearch;
  const hasSearch = term.length > 0;
  const hasSel    = selectedCats.size > 0;
  const isKW      = searchMode === "keyword" || !embedReady;

  // Actionables that pass the search filter (quality filter applied below)
  let searchMatchSet = null;   // Set of actionable IDs
  if (hasSearch) {
    searchMatchSet = new Set();
    for (const id of actionableIds) {
      const m = nodeMap.get(id);
      if (!passesQualityFilter(m)) continue;
      const s = simScores.get(id) || 0;
      if (isKW ? s > 0 : s >= threshold) searchMatchSet.add(id);
    }
  }

  // Actionables that belong to a selected category (quality filter applied)
  let selMatchSet = null;      // Set of actionable IDs
  if (hasSel) {
    selMatchSet = new Set();
    for (const c of selectedCats) {
      for (const a of (catToActions.get(c)||[])) {
        if (passesQualityFilter(nodeMap.get(a))) selMatchSet.add(a);
      }
    }
  }

  function isActionableVisible(id) {
    if (!passesQualityFilter(nodeMap.get(id))) return false;
    if (searchMatchSet !== null && !searchMatchSet.has(id))  return false;
    if (selMatchSet    !== null && !selMatchSet.has(id))     return false;
    return true;
  }

  // A category is visible when at least one of its actionables is visible.
  // A category whose data hasn't been fetched yet has no actionables in
  // memory at all -- keep its hub visible/clickable by default (so the user
  // can select it to trigger a load) unless an active search or a *different*
  // category selection would otherwise hide it.
  function isCategoryVisible(id) {
    if (!loadedCategoryIds.has(id)) {
      if (hasSearch) return false;
      if (hasSel && !selectedCats.has(id)) return false;
      return true;
    }
    return [...(catToActions.get(id)||[])].some(a => isActionableVisible(a));
  }

  function isVisible(id) {
    const m = nodeMap.get(id);
    if (!m) return false;
    return m.node_type === "actionable"
      ? isActionableVisible(id)
      : isCategoryVisible(id);
  }

  // An actionable is a "direct" match when search is active and it's in the match set
  function isDirect(id) {
    if (!hasSearch || searchMatchSet === null) return false;
    return searchMatchSet.has(id);
  }

  // Best search score among quality-passing actionables in a category
  function catBestScore(catId) {
    let best = 0;
    for (const a of (catToActions.get(catId)||[])) {
      if (!passesQualityFilter(nodeMap.get(a))) continue;
      const s = simScores.get(a) || 0;
      if (s > best) best = s;
    }
    return best;
  }

  return { isActionableVisible, isCategoryVisible, isVisible, isDirect, catBestScore, isKW };
}

// ── style-only update ──────────────────────────────────────────────────────────
function applyStyles() {
  if (!nodesDS || !edgesDS) return;
  const hasSearch = committedSearch.length > 0;
  const { isVisible, isDirect } = buildVisibility();

  nodesDS.update(ALL_NODES.map(n => {
    const id   = String(n.id);
    const meta = nodeMap.get(id);
    const vis_ = isVisible(id);
    const dir  = isDirect(id);
    if (!vis_) return sanitizeVisNode({id:n.id, opacity:0.08, borderWidth:n.borderWidth||1.5, size:n.size, color:n.color});
    if (dir && hasSearch) {
      const isCat = meta?.node_type === "category";
      return sanitizeVisNode({id:n.id, opacity:1, size:n.size*1.35, borderWidth:5, color:{
        background: isCat?"#6baed6":"#fb6a4a", border: isCat?"#1d4ed8":"#9a3412",
        highlight: isCat?{background:"#93c5fd",border:"#1e40af"}:{background:"#fb923c",border:"#7c2d12"},
        hover:     isCat?{background:"#93c5fd",border:"#1e40af"}:{background:"#fb923c",border:"#7c2d12"},
      }});
    }
    return sanitizeVisNode({id:n.id, opacity:1, size:n.size, borderWidth:n.borderWidth||1.5, color:n.color});
  }));

  edgesDS.update(ALL_EDGES.map(e => {
    const f=String(e.from), t=String(e.to);
    const vis_=(isVisible(f)&&isVisible(t));
    const hi=((isDirect(f)||isDirect(t))&&hasSearch);
    if (!vis_) return {id:e.id, color:{color:"#e2e8f0",opacity:.08}, width:.8};
    if (hi)    return {id:e.id, color:{color:"#334155",highlight:"#0f172a",hover:"#0f172a",opacity:.9}, width:2.8};
    return          {id:e.id, color:{color:"#94a3b8",highlight:"#64748b",hover:"#64748b",opacity:.55}, width:1.4};
  }));

  updateSummary();
}

// ── category panel ─────────────────────────────────────────────────────────────
function renderCategoryPanel() {
  const el   = document.getElementById("groupList");
  const term = committedSearch;
  const { isActionableVisible, isCategoryVisible, catBestScore, isKW } = buildVisibility();

  const visible = categoryIds.filter(id => isCategoryVisible(id));

  // Count visible actionables for stats
  const visibleActionCount = actionableIds.filter(id => isActionableVisible(id)).length;

  document.getElementById("catCountBadge").textContent = visible.length;
  // Show filtered/total when a filter reduces the count
  document.getElementById("statCats").textContent =
    visible.length < categoryIds.length
      ? visible.length + "/" + categoryIds.length
      : categoryIds.length;
  document.getElementById("statActions").textContent =
    visibleActionCount < actionableIds.length
      ? visibleActionCount + "/" + actionableIds.length
      : actionableIds.length;
  document.getElementById("statSelected").textContent = selectedCats.size;

  if (!visible.length) {
    el.innerHTML = '<div class="empty-note">No categories match current filters.</div>'; return;
  }

  const sorted = term
    ? [...visible].sort((a,b) => catBestScore(b) - catBestScore(a))
    : visible;

  el.innerHTML = sorted.map(id => {
    const m       = nodeMap.get(id);
    const s       = catBestScore(id);
    const scoreStr = (term && !isKW && s>0) ? (s*100).toFixed(0)+"%" : "";
    const isSel   = selectedCats.has(id);
    const isDir   = term && (isKW ? s>0 : s>=threshold);
    const isLoading = loadingCategoryIds.has(id);
    const notLoaded = !loadedCategoryIds.has(id) && !isLoading;
    const tag = isLoading
      ? ' <span style="opacity:.7;font-weight:400">(loading…)</span>'
      : notLoaded ? ' <span style="opacity:.55;font-weight:400">(click to load)</span>' : '';
    return `<button class="group-item${isSel?" selected":""}${isDir?" matched":""}"
              onclick="toggleCat('${id}')"${isLoading?" disabled":""}>
      <span>${esc(m?.category||id)}${tag}</span>
      ${scoreStr?`<span class="sim-score">${scoreStr}</span>`:""}
    </button>`;
  }).join("");
}

async function toggleCat(id) {
  if (selectedCats.has(id)) {
    selectedCats.delete(id);
    document.getElementById("statSelected").textContent = selectedCats.size;
    applyStyles(); renderCategoryPanel();
    return;
  }
  selectedCats.add(id);
  document.getElementById("statSelected").textContent = selectedCats.size;
  applyStyles(); renderCategoryPanel();
  await loadCategories([id]);
  applyStyles(); renderCategoryPanel();
}
async function selectVisible() {
  const { isCategoryVisible } = buildVisibility();
  const toLoad = [];
  for (const id of categoryIds) {
    if (isCategoryVisible(id)) {
      selectedCats.add(id);
      if (!loadedCategoryIds.has(id)) toLoad.push(id);
    }
  }
  document.getElementById("statSelected").textContent = selectedCats.size;
  applyStyles(); renderCategoryPanel();
  await loadCategories(toLoad);
  applyStyles(); renderCategoryPanel();
}
function clearSelection() {
  selectedCats.clear();
  document.getElementById("statSelected").textContent = 0;
  applyStyles(); renderCategoryPanel(); resetDetail();
}

// ── detail panel ───────────────────────────────────────────────────────────────
function qBadge(label, val) {
  const v = (val||"").trim();
  const isYes = /^(yes|true|1|high|strong)$/i.test(v);
  const isNo  = /^(no|false|0|low|weak)$/i.test(v);
  const cls   = isYes ? "yes" : isNo ? "no" : "";
  return `<span class="qbadge ${cls}">${esc(label)}: ${esc(v||"N/A")}</span>`;
}

function showDetail(m) {
  const el = document.getElementById("detailPanel");
  el.classList.add("animating");
  const score = simScores.get(String(m.id));
  requestAnimationFrame(() => {
    el.innerHTML = buildDetailHTML(m, score);
    requestAnimationFrame(() => el.classList.remove("animating"));
  });
}

function buildDetailHTML(m, score) {
  const simBar = (score != null && committedSearch && searchMode==="semantic" && embedReady)
    ? `<div class="sim-bar-wrap">
         <div class="sim-bar-label">Semantic similarity (Impact field)</div>
         <div class="sim-bar-track"><div class="sim-bar-fill" style="width:${(score*100).toFixed(1)}%"></div></div>
         <div style="font-size:10px;color:var(--accent);margin-top:3px;font-family:'IBM Plex Mono',monospace">${(score*100).toFixed(1)}%</div>
       </div>` : "";

  if (m.node_type === "actionable") {
    const catChips = (m.categories||[])
      .map(c => `<span class="cat-chip">${esc(c)}</span>`).join("");
    return `
      <div class="detail-title">${esc(m.short_label||m.id)} <span class="tag tag-a">actionable</span></div>
      <div class="dlabel">Actionable</div>
      <div class="dval">${esc(m.actionable)}</div>
      <div class="dlabel">Impact</div>
      <div class="dval">${esc(m.impact||"N/A")}</div>
      <div class="dlabel">Evidence</div>
      <div class="dval">${esc(m.empirical_evidence||"N/A")}</div>
      <div class="dlabel">Quality indicators</div>
      <div class="badge-row">
        ${qBadge("Confidence", m.confidence)}
        ${qBadge("Sound",      m.sound)}
        ${qBadge("Precise",    m.precise)}
      </div>
      <div class="dlabel">Categories (${(m.categories||[]).length})</div>
      <div class="cat-chips">${catChips || "N/A"}</div>
      <div class="dlabel">Article</div>
      <div class="dval">${esc(m.article_title||"N/A")}</div>
      ${m.count ? `<div class="dlabel">Support</div><div class="dval">${esc(m.count)}</div>` : ""}
      ${simBar}`;
  }

  const connected = [...(catToActions.get(String(m.id))||[])];
  return `
    <div class="detail-title">${esc(m.category)} <span class="tag tag-c">category</span></div>
    <div class="dlabel">Connected actionables</div>
    <div class="dval">${connected.length}</div>
    ${simBar}`;
}

function resetDetail() {
  const el = document.getElementById("detailPanel");
  el.classList.add("animating");
  requestAnimationFrame(() => {
    el.innerHTML = "Click any node to inspect details.";
    requestAnimationFrame(() => el.classList.remove("animating"));
  });
}

// ── controls ───────────────────────────────────────────────────────────────────
document.getElementById("searchBox").addEventListener("keydown", e => {
  if (e.key === "Enter") commitSearch();
});
document.getElementById("searchBox").addEventListener("input", e => {
  document.getElementById("searchClear").classList.toggle("visible", e.target.value.length > 0);
});
document.getElementById("searchClear").addEventListener("click", () => {
  document.getElementById("searchBox").value = "";
  document.getElementById("searchClear").classList.remove("visible");
  committedSearch = ""; simScores = new Map();
  applyStyles(); renderCategoryPanel(); resetDetail();
});
document.getElementById("thresholdSlider").addEventListener("input", e => {
  threshold = parseInt(e.target.value) / 100;
  document.getElementById("thresholdVal").textContent = threshold.toFixed(2);
  if (committedSearch) { applyStyles(); renderCategoryPanel(); }
});
function setSearchMode(mode) {
  searchMode = mode;
  document.getElementById("pillSemantic").classList.toggle("active", mode==="semantic");
  document.getElementById("pillKeyword").classList.toggle("active",  mode==="keyword");
  document.getElementById("thresholdRow").style.display = mode==="semantic"?"flex":"none";
  document.getElementById("syntaxHint").classList.toggle("visible",  mode==="keyword");
  if (mode==="keyword") setModeBanner("keyword");
  else if (embedReady) setModeBanner("semantic");
  else if (lastActiveMode==="error") setModeBanner("error");
  else setModeBanner("fallback");
  if (committedSearch) runSearch();
}
function fitView() { if(network) network.fit({animation:{duration:400,easingFunction:"easeInOutQuad"}}); }

async function showAll() {
  // Reset search
  document.getElementById("searchBox").value = "";
  document.getElementById("searchClear").classList.remove("visible");
  committedSearch = "";
  selectedCats.clear();
  simScores = new Map();
  // Reset quality filters without triggering redundant redraws
  soundFilter = "any"; preciseFilter = "any";
  ["sound","precise"].forEach(k => {
    for (const v of ["any","yes","no"]) {
      const btn = document.getElementById(k + v.charAt(0).toUpperCase() + v.slice(1));
      if (btn) btn.className = "filter-pill" + (v === "any" ? " fp-any" : "");
    }
  });
  updateFilterBadge();
  // Single redraw
  applyStyles(); renderCategoryPanel(); resetDetail(); fitView();
  // "Show all" means all -- pull in any category not yet loaded.
  await ensureAllCategoriesLoaded();
  fitView();
}

// ── summary ────────────────────────────────────────────────────────────────────
function updateSummary() {
  const vEdges = edgesDS
    ? edgesDS.get().filter(e=>(e.color?.opacity||1)>0.2).length
    : ALL_EDGES.length;
  const { isActionableVisible, isCategoryVisible } = buildVisibility();
  const vCats    = categoryIds.filter(id => isCategoryVisible(id)).length;
  const vActions = actionableIds.filter(id => isActionableVisible(id)).length;
  document.getElementById("netSummary").textContent =
    `${vCats} categories · ${vActions} actionables · ${vEdges} edges visible`;
  document.getElementById("statEdges").textContent = vEdges;
}

function esc(v) {
  return String(v??"").replace(/&/g,"&amp;").replace(/</g,"&lt;")
         .replace(/>/g,"&gt;").replace(/"/g,"&quot;").replace(/'/g,"&#39;");
}

// ── theme ──────────────────────────────────────────────────────────────────────
function applyTheme(dark) {
  document.documentElement.setAttribute("data-theme", dark ? "dark" : "light");
  var icon = document.getElementById("themeIcon");
  if (icon) icon.textContent = dark ? "🌙" : "☀️";
  if (network) network.setOptions({nodes:{font:{color: dark ? "#e6edf3" : "#1e293b"}}});
}
function toggleTheme() {
  var next = document.documentElement.getAttribute("data-theme") !== "dark";
  applyTheme(next);
  try { localStorage.setItem("react_network_theme", next ? "dark" : "light"); } catch(e){}
}

// ── about modal ──────────────────────────────────────────────────────────────
function openAbout() { document.getElementById("aboutOverlay").classList.remove("hidden"); }
function closeAbout() { document.getElementById("aboutOverlay").classList.add("hidden"); }
document.addEventListener("keydown", e => {
  if (e.key === "Escape") closeAbout();
});

// ── boot ───────────────────────────────────────────────────────────────────────
(function boot() {
  try {
    var saved      = localStorage.getItem("react_network_theme");
    var prefersDark = window.matchMedia("(prefers-color-scheme: dark)").matches;
    applyTheme(saved ? saved === "dark" : prefersDark);
  } catch(e) { applyTheme(false); }

  document.getElementById("aboutRows").textContent    = GRAPH_STATS.row_count;
  document.getElementById("aboutCats").textContent    = GRAPH_STATS.category_count;
  document.getElementById("aboutActions").textContent = GRAPH_STATS.actionable_count;
  document.getElementById("aboutEdges").textContent   = GRAPH_STATS.edge_count;
  openAbout();

  // Pre-select the default category -- its actionables already shipped in the
  // main bundle, so it's marked loaded immediately with no fetch needed.
  for (const m of ALL_NODE_META) {
    if (m.node_type === "category" && m.category === DEFAULT_CATEGORY) {
      selectedCats.add(String(m.id));
      loadedCategoryIds.add(String(m.id));
      break;
    }
  }

  initNetwork();
  renderCategoryPanel();
  bootstrapEmbedder().catch(err => {
    console.error("Embedder bootstrap:", err);
    setAIPill("error", "Semantic search unavailable");
    setModeBanner("error");
  });
})();
</script>
</body>
</html>
"""


# ── generate_html ──────────────────────────────────────────────────────────────

def generate_html(
    nodes: List[dict],
    node_meta: List[dict],
    edges: List[dict],
    stats: dict,
    default_category: str,
    category_files: Dict[str, str],
    dataset_url: str,
    node_ids: Optional[List[str]] = None,
    embeddings_b64: Optional[List[str]] = None,
) -> str:
    if node_ids and embeddings_b64 and len(node_ids) == len(embeddings_b64):
        baked = {nid: b64 for nid, b64 in zip(node_ids, embeddings_b64)}
        baked_json = json.dumps(baked, ensure_ascii=False)
        tqdm.write(f"  ✓  Baking {len(baked)} node embeddings into HTML (default category only).")
    else:
        baked_json = "null"
        tqdm.write("  ✗  No embeddings baked — HTML will use keyword search only.")
    return (
        HTML_TEMPLATE
        .replace("__NODES_JSON__",          json.dumps(nodes,          ensure_ascii=False))
        .replace("__NODE_META_JSON__",      json.dumps(node_meta,      ensure_ascii=False))
        .replace("__EDGES_JSON__",          json.dumps(edges,          ensure_ascii=False))
        .replace("__STATS_JSON__",          json.dumps(stats,          ensure_ascii=False))
        .replace("__BAKED_EMBEDDINGS__",    baked_json)
        .replace("__CATEGORY_FILES_JSON__", json.dumps(category_files, ensure_ascii=False))
        .replace("__DEFAULT_CATEGORY_JSON__", json.dumps(default_category, ensure_ascii=False))
        .replace("__DATASET_URL__",         dataset_url)
    )


def write_html(output_path: Path, html_text: str) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with tqdm(total=1, desc="  Writing HTML", unit="file") as bar:
        output_path.write_text(html_text, encoding="utf-8")
        bar.update(1)


def write_category_chunks(
    chunks: Dict[str, dict],
    category_names: Dict[str, str],
    output_html: Path,
) -> Dict[str, str]:
    """Write one JSON side-car file per non-default category next to the
    output HTML, in a `<stem>_data/` folder. Returns {category_id: relative
    path} for baking into the main HTML as CATEGORY_DATA_FILES."""
    data_dir = output_html.parent / f"{output_html.stem}_data"
    data_dir.mkdir(parents=True, exist_ok=True)
    manifest: Dict[str, str] = {}
    with tqdm(chunks.items(), desc="  Writing category chunks", unit="file", total=len(chunks)) as bar:
        for cid, chunk in bar:
            slug = slugify(category_names[cid])
            fname = f"{slug}.json"
            (data_dir / fname).write_text(
                json.dumps(chunk, ensure_ascii=False), encoding="utf-8"
            )
            manifest[cid] = f"{data_dir.name}/{fname}"
    return manifest


# ── main ───────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate ReACT-LLM Category→Actionable network HTML."
    )
    parser.add_argument("--input_csv",   type=Path, default=DEFAULT_INPUT_CSV)
    parser.add_argument("--output_html", type=Path, default=DEFAULT_OUTPUT_HTML)
    parser.add_argument("--skip-embed",  action="store_true",
                        help="Skip embedding — keyword search only in browser")
    parser.add_argument("--workers",     type=int, default=1,
                        help="Parallel embedding workers (default 1)")
    parser.add_argument("--default-category", type=str, default=DEFAULT_CATEGORY,
                        help="Category baked into the main HTML on first load; "
                             "every other category is fetched on demand in the browser "
                             f"(default: {DEFAULT_CATEGORY!r})")
    parser.add_argument("--dataset-url", type=str, default=DEFAULT_DATASET_URL,
                        help="Link to the published dataset, shown in the topbar and About panel")
    args = parser.parse_args()

    if not args.skip_embed:
        if not (HAS_ST and HAS_NP):
            ok = ensure_embedding_deps()
            if not ok:
                print("\n  Falling back to keyword-only mode (--skip-embed behaviour).")
                args.skip_embed = True
        else:
            print("  ✓  sentence-transformers and numpy found.")

    print(f"\n[1/6] Reading {args.input_csv}")
    rows = read_csv_rows(args.input_csv)
    print(f"      {len(rows)} usable row(s) loaded")

    print("\n[2/6] Building graph payload …")
    nodes, node_meta, edges, stats, node_ids, node_texts = build_graph_payload(rows)
    print(f"      categories={stats['category_count']}  "
          f"actionables={stats['actionable_count']}  "
          f"edges={stats['edge_count']}")

    embeddings_b64 = None
    if not args.skip_embed:
        print(f"\n[3/6] Embedding {len(node_texts)} node texts (workers={args.workers}) …")
        embeddings_b64 = embed_texts(node_texts, n_workers=args.workers)
        if embeddings_b64 is None:
            print("  ✗  Embedding returned None — HTML will be keyword-only.")
    else:
        print("\n[3/6] Skipping embedding (--skip-embed).")

    print("\n[4/6] Partitioning graph by category (default category loads eagerly;"
          " the rest load on demand in the browser) …")
    category_names = sorted(m["category"] for m in node_meta if m["node_type"] == "category")
    default_category = args.default_category
    if default_category not in category_names:
        tqdm.write(f"  ⚠  Default category {default_category!r} not found in the data; "
                    f"falling back to {category_names[0]!r}.")
        default_category = category_names[0]

    embeddings_by_id = dict(zip(node_ids, embeddings_b64)) if embeddings_b64 else {}
    main_nodes, main_meta, main_edges, main_embeddings, chunks, default_cat_id = (
        partition_by_category(nodes, node_meta, edges, embeddings_by_id, default_category)
    )
    cat_name_by_id = {m["id"]: m["category"] for m in node_meta if m["node_type"] == "category"}
    default_actionable_count = sum(1 for m in main_meta if m["node_type"] == "actionable")
    print(f"      default category {default_category!r}: "
          f"{default_actionable_count}/{stats['actionable_count']} actionables baked into main HTML")
    print(f"      {len(chunks)} other categories deferred to on-demand side-car files")

    print("\n[5/6] Writing category side-car files + rendering HTML …")
    category_files = write_category_chunks(chunks, cat_name_by_id, args.output_html)
    main_node_ids = list(main_embeddings.keys()) if main_embeddings else None
    main_embeddings_b64 = list(main_embeddings.values()) if main_embeddings else None
    with tqdm(total=1, desc="  Serialising", unit="step") as bar:
        html_text = generate_html(
            main_nodes, main_meta, main_edges, stats,
            default_category, category_files, args.dataset_url,
            main_node_ids, main_embeddings_b64,
        )
        bar.update(1)

    print(f"\n[6/6] Writing {args.output_html}")
    write_html(args.output_html, html_text)

    kb = len(html_text.encode()) // 1024
    print(f"\n  Done!  {kb} KB → {args.output_html}  (was a single ~12 MB file before "
          f"per-category lazy loading)")
    if embeddings_b64:
        print("  🧠  Semantic search: baked embeddings + Transformers.js query model")
    else:
        print("  🔤  Semantic search: DISABLED (keyword only).")
        print("      To enable: pip install sentence-transformers numpy")
        print("      Then re-run without --skip-embed")
    print(f"  {len(category_files)} category file(s) written under "
          f"{args.output_html.parent / (args.output_html.stem + '_data')}/")
    print("  Open the HTML in a browser served over http(s) (e.g. GitHub Pages) so the "
          "on-demand category fetches work — file:// may block them in some browsers.\n")


if __name__ == "__main__":
    main()