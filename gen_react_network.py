#!/usr/bin/env python3
"""
gen_react_network.py

Generates an interactive Category → Actionable network from:
    results/Final ReACT-LLM Set.csv

Columns used:
    COUNT, AricleTile, ACTIONABLE, CATEGORY, IMPACT,
    EMPIRICAL EVIDENCE, CONFIDENCE, SOUND, PRECISE

Graph: Category nodes (left hub) ↔ Actionable nodes (right detail)
       One edge per unique (CATEGORY, ACTIONABLE) pair.

Detail panel fields shown on actionable click:
    ACTIONABLE, IMPACT, EMPIRICAL EVIDENCE, CONFIDENCE,
    SOUND, PRECISE, CATEGORY, AricleTile

Tooltip on hover: plain text — ACTIONABLE text only (no HTML).

Architecture (identical to OVC gen_net_graph_all.py v12):
  • ALL_NODES  — vis-only props, sanitized through VIS_NODE_KEYS whitelist
  • ALL_NODE_META — application data, never touches vis DataSet
  • BAKED_EMBEDDINGS — node vectors pre-computed in Python, baked into HTML
  • Transformers.js — loads only for query embedding (~2–3 s, browser-cached)
  • Physics — Barnes-Hut, runs once on first load then freezes
  • Search — Enter / button only, semantic + keyword modes

Usage
─────
    pip install sentence-transformers tqdm numpy
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
from collections import OrderedDict, defaultdict
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

DEFAULT_INPUT_CSV = Path("results/Final ReACT-LLM Set.csv")
DEFAULT_OUTPUT_HTML = Path("results/react_llm_category_network.html")

REQUIRED_COLUMNS = [
    "ACTIONABLE", "CATEGORY", "AricleTile",
]

# Optional columns — present if available, shown as N/A otherwise
OPTIONAL_COLUMNS = ["COUNT", "IMPACT", "EMPIRICAL EVIDENCE", "CONFIDENCE", "SOUND", "PRECISE"]


# ── text helpers ───────────────────────────────────────────────────────────────

def normalize_text(value: str) -> str:
    return " ".join((value or "").strip().split())

def safe_col(row: Dict[str, str], col: str) -> str:
    return normalize_text(row.get(col, "") or "")


# ── CSV reader ─────────────────────────────────────────────────────────────────

def read_csv_rows(csv_path: Path) -> List[Dict[str, str]]:
    if not csv_path.exists():
        raise FileNotFoundError(f"Input CSV not found: {csv_path}")

    with csv_path.open("r", encoding="utf-8-sig") as f:
        total = sum(1 for _ in f) - 1

    rows: List[Dict[str, str]] = []
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
                if not cleaned.get("ACTIONABLE") or not cleaned.get("CATEGORY"):
                    continue
                rows.append(cleaned)

    if not rows:
        raise ValueError("No usable rows found in CSV.")
    return rows


# ── graph payload builder ──────────────────────────────────────────────────────

def build_graph_payload(
    rows: List[Dict[str, str]],
) -> Tuple[List[dict], List[dict], List[dict], dict, List[str], List[str]]:

    # category → vis id
    category_to_id: OrderedDict = OrderedDict()
    # actionable key → vis id  (key = full text to deduplicate)
    actionable_to_id: OrderedDict = OrderedDict()
    actionable_to_meta: Dict[str, dict] = {}

    edges_seen: set = set()
    edges: List[dict] = []

    with tqdm(rows, desc="  Building nodes/edges", unit="row") as bar:
        for row in bar:
            cat  = safe_col(row, "CATEGORY")
            act  = safe_col(row, "ACTIONABLE")

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
                    "category":         cat,
                    "article_title":    safe_col(row, "AricleTile"),
                    "impact":           safe_col(row, "IMPACT"),
                    "empirical_evidence": safe_col(row, "EMPIRICAL EVIDENCE"),
                    "confidence":       safe_col(row, "CONFIDENCE"),
                    "sound":            safe_col(row, "SOUND"),
                    "precise":          safe_col(row, "PRECISE"),
                    "count":            safe_col(row, "COUNT"),
                }

    # edges — one per unique (category, actionable) pair
    edge_counter = 1
    with tqdm(rows, desc="  Building edges", unit="row") as bar:
        for row in bar:
            cat = safe_col(row, "CATEGORY")
            act = safe_col(row, "ACTIONABLE")
            cid = category_to_id[cat]
            aid = actionable_to_id[act]
            ek  = (cid, aid)
            if ek in edges_seen:
                continue
            edges_seen.add(ek)
            edges.append({"id": f"E_{edge_counter}", "from": cid, "to": aid, "width": 1.6})
            edge_counter += 1

    # ── build vis nodes + node_meta (split cleanly) ────────────────────────────
    all_ids_ordered:   List[str] = []
    all_texts_ordered: List[str] = []
    nodes:     List[dict] = []   # vis-only — safe for DataSet
    node_meta: List[dict] = []   # application data — never touches DataSet

    with tqdm(actionable_to_id.items(), desc="  Serialising actionables",
              unit="node", total=len(actionable_to_id)) as bar:
        for act, visid in bar:
            meta = actionable_to_meta[act]
            semantic_text = " ".join(filter(None, [
                meta["actionable"], meta["category"],
                meta["impact"], meta["article_title"],
            ]))
            search_blob = " ".join([
                meta["short_label"], meta["actionable"], meta["category"],
                meta["impact"], meta["article_title"],
                meta["empirical_evidence"], meta["confidence"],
                meta["sound"], meta["precise"],
            ]).lower()
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
                "category":         meta["category"],
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

    with tqdm(category_to_id.items(), desc="  Serialising categories",
              unit="node", total=len(category_to_id)) as bar:
        for cat, visid in bar:
            semantic_text = cat
            search_blob   = cat.lower()
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
        "row_count":       len(rows),
        "actionable_count": len(actionable_to_id),
        "category_count":   len(category_to_id),
        "edge_count":       len(edges),
    }
    return nodes, node_meta, edges, stats, all_ids_ordered, all_texts_ordered


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
    if not (HAS_ST and HAS_NP):
        tqdm.write("  [SKIP] sentence-transformers or numpy not installed.")
        tqdm.write("         pip install sentence-transformers numpy")
        return None

    n = len(texts)
    n_workers  = max(1, min(n_workers, os.cpu_count() or 1, math.ceil(n / 50)))
    chunk_size = math.ceil(n / n_workers)
    chunks     = [(texts[i:i+chunk_size], i//chunk_size) for i in range(0, n, chunk_size)]

    tqdm.write(f"  Embedding {n} texts across {n_workers} worker(s) ({chunk_size}/worker) …")
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
    tqdm.write(f"  Embedded {len(flat)} vectors → {kb} KB (baked into HTML)")
    return flat


# ── HTML template ──────────────────────────────────────────────────────────────

HTML_TEMPLATE = r"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8"/>
  <meta name="viewport" content="width=device-width,initial-scale=1"/>
  <title>ReACT-LLM Category Network Explorer</title>

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
    :root {
      --bg:#f2f6fc; --surface:#fff; --surface2:#f8fafd; --border:#dbe4f0;
      --text:#0f172a; --muted:#64748b; --accent:#2563eb;
      --cat:#6baed6; --action:#fb6a4a;
      --shadow:0 8px 28px rgba(15,23,42,.09);
    }
    *,*::before,*::after{box-sizing:border-box;margin:0;padding:0}
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
    .stat-chips{display:flex;gap:8px;flex-wrap:wrap}
    .stat-chip{font-size:11px;font-family:'IBM Plex Mono',monospace;padding:4px 9px;
               border-radius:99px;border:1px solid var(--border);
               background:var(--surface2);color:var(--muted)}
    .stat-chip b{color:var(--text)}

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
                  background:rgba(242,246,252,.88);backdrop-filter:blur(4px);
                  transition:opacity .4s}
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
    .net-wrap{flex:1 1 0;min-height:0;position:relative;
              background:linear-gradient(160deg,#f8fbff 0%,#eef4fb 100%)}
    #mynetwork{width:100%;height:100%}
    .net-toolbar{position:absolute;top:12px;left:12px;right:12px;display:flex;
                 justify-content:space-between;align-items:center;gap:10px;
                 pointer-events:none;z-index:4}
    .net-summary{background:rgba(255,255,255,.88);backdrop-filter:blur(6px);
                 border:1px solid var(--border);border-radius:10px;
                 padding:6px 12px;font-size:11px;color:var(--muted)}
    .legend{display:flex;gap:10px;flex-wrap:wrap;font-size:11px;color:var(--muted);
            background:rgba(255,255,255,.88);backdrop-filter:blur(6px);
            border:1px solid var(--border);border-radius:10px;padding:6px 12px}
    .legend-item{display:flex;align-items:center;gap:5px}
    .leg-dot{width:9px;height:9px;border-radius:50%}

    /* side panel */
    .side{flex:0 0 340px;width:340px;height:100vh;overflow-y:auto;
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
    .group-list{display:flex;flex-direction:column;gap:5px;
                max-height:220px;overflow-y:auto}
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

    /* confidence / quality badges */
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
    .sim-bar-fill{height:100%;border-radius:99px;background:var(--accent);
                  transition:width .4s ease}

    @media(max-width:960px){
      .app{flex-direction:column;height:auto}
      .main,.net-wrap{min-height:520px}
      .side{height:auto;width:100%;flex:none}
    }
  </style>
</head>
<body>
<div class="app">
  <main class="main">

    <div class="topbar">
      <div class="topbar-brand">
        <div class="eyebrow">ReACT-LLM &middot; Actionable Intelligence</div>
        <h1>Category &ndash; Actionable Network Explorer</h1>
      </div>
      <div class="stat-chips">
        <div class="stat-chip"><b id="scRows">0</b> rows</div>
        <div class="stat-chip"><b id="scCats">0</b> categories</div>
        <div class="stat-chip"><b id="scActions">0</b> actionables</div>
        <div class="stat-chip"><b id="scEdges">0</b> edges</div>
      </div>
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

    <div class="side-section">
      <div class="section-head">Search</div>
      <div class="search-row">
        <div class="search-wrap">
          <span class="search-icon">&#128269;</span>
          <input id="searchBox" class="search-input" type="text"
                 placeholder="Type then press Enter or Search&hellip;"/>
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

    <div class="side-section">
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

    <div class="side-section">
      <div class="section-head">Details</div>
      <div id="detailPanel" class="detail-card">Click any node to inspect details.</div>
    </div>

  </aside>
</div>

<script>
// ── injected data ──────────────────────────────────────────────────────────────
const ALL_NODES        = __NODES_JSON__;
const ALL_NODE_META    = __NODE_META_JSON__;
const ALL_EDGES        = __EDGES_JSON__;
const GRAPH_STATS      = __STATS_JSON__;
const BAKED_EMBEDDINGS = __BAKED_EMBEDDINGS__;
// ──────────────────────────────────────────────────────────────────────────────

Object.freeze(ALL_NODES);
Object.freeze(ALL_NODE_META);

// ── lookup structures ──────────────────────────────────────────────────────────
const nodeMap    = new Map(ALL_NODE_META.map(m => [String(m.id), m]));
const categoryIds = ALL_NODE_META
  .filter(m => m.node_type === "category")
  .map(m => String(m.id))
  .sort((a,b) => (nodeMap.get(a)?.category||"").localeCompare(nodeMap.get(b)?.category||""));
const actionableIds = ALL_NODE_META
  .filter(m => m.node_type === "actionable")
  .map(m => String(m.id));

// adjacency: category → actionables, actionable → categories
const catToActions  = new Map(categoryIds.map(id => [id, new Set()]));
const actionToCats  = new Map(actionableIds.map(id => [id, new Set()]));
for (const e of ALL_EDGES) {
  const f = String(e.from), t = String(e.to);
  if (!catToActions.has(f)) catToActions.set(f, new Set());
  if (!actionToCats.has(t)) actionToCats.set(t, new Set());
  catToActions.get(f).add(t);
  actionToCats.get(t).add(f);
}

// ── state ──────────────────────────────────────────────────────────────────────
let network         = null;
let nodesDS         = null;
let edgesDS         = null;
let selectedCats    = new Set();
let committedSearch = "";
let searchMode      = "semantic";
let threshold       = 0.30;
let lastActiveMode  = "fallback";
let embedder        = null;
let nodeEmbeddings  = null;
let embedReady      = false;
let simScores       = new Map();

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
    semantic: "🧠 Semantic search active",
    keyword:  "🔤 Keyword search active",
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

// ── search ─────────────────────────────────────────────────────────────────────
function commitSearch() {
  committedSearch = document.getElementById("searchBox").value.trim();
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

// ── tooltip: plain text only ───────────────────────────────────────────────────
function buildTooltip(n) {
  const m = nodeMap.get(String(n.id));
  if (!m) return n.label || String(n.id);
  if (m.node_type === "actionable") return m.actionable;
  return m.category;
}

// ── vis-network ────────────────────────────────────────────────────────────────
const VIS_NODE_KEYS = new Set([
  "id","label","shape","size","color",
  "borderWidth","font","shadow","opacity","title"
]);
function sanitizeVisNode(input) {
  const out = {};
  for (const k of VIS_NODE_KEYS) if (input[k] !== undefined) out[k] = input[k];
  return out;
}
function visNode(n) {
  const base = sanitizeVisNode(n);
  return { ...base, opacity: 1, title: buildTooltip(n) };
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

  const overlay = document.getElementById("stabOverlay");
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
      if (node) showDetail(node);
    } else {
      resetDetail();
    }
  });
}

// ── style-only update ──────────────────────────────────────────────────────────
function applyStyles() {
  if (!nodesDS || !edgesDS) return;
  const term      = committedSearch;
  const hasSearch = term.length > 0;
  const hasSel    = selectedCats.size > 0;
  const isKW      = searchMode === "keyword" || !embedReady;

  let searchMatchNodes = null;
  if (hasSearch) {
    searchMatchNodes = new Set();
    for (const [id, score] of simScores)
      if (isKW ? score > 0 : score >= threshold) searchMatchNodes.add(id);
    const expanded = new Set(searchMatchNodes);
    for (const id of searchMatchNodes) {
      const m = nodeMap.get(id);
      if (!m) continue;
      if (m.node_type === "category")   { for (const a of (catToActions.get(id)||[])) expanded.add(a); }
      else                              { for (const c of (actionToCats.get(id)||[])) expanded.add(c); }
    }
    searchMatchNodes = expanded;
  }

  let selMatchNodes = null;
  if (hasSel) {
    selMatchNodes = new Set(selectedCats);
    for (const c of selectedCats) { for (const a of (catToActions.get(c)||[])) selMatchNodes.add(a); }
  }

  function isVisible(id) {
    if (!hasSearch && !hasSel) return true;
    return (!searchMatchNodes || searchMatchNodes.has(id)) &&
           (!selMatchNodes    || selMatchNodes.has(id));
  }
  function isDirect(id) {
    if (!hasSearch) return false;
    const s = simScores.get(id) || 0;
    return isKW ? s > 0 : s >= threshold;
  }

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
  const isKW = searchMode === "keyword" || !embedReady;

  const visible = categoryIds.filter(id => {
    if (!term) return true;
    const s = simScores.get(id)||0;
    if (isKW?s>0:s>=threshold) return true;
    for (const a of (catToActions.get(id)||[])) {
      const as = simScores.get(a)||0;
      if (isKW?as>0:as>=threshold) return true;
    }
    return false;
  });

  document.getElementById("catCountBadge").textContent = visible.length;
  document.getElementById("statCats").textContent    = categoryIds.length;
  document.getElementById("statActions").textContent = actionableIds.length;
  document.getElementById("statSelected").textContent = selectedCats.size;

  if (!visible.length) { el.innerHTML='<div class="empty-note">No categories match.</div>'; return; }

  const sorted = term
    ? [...visible].sort((a,b)=>(simScores.get(b)||0)-(simScores.get(a)||0))
    : visible;

  el.innerHTML = sorted.map(id => {
    const m = nodeMap.get(id);
    const s = simScores.get(id)||0;
    const scoreStr = (term && !isKW && s>0) ? (s*100).toFixed(0)+"%" : "";
    const isSel = selectedCats.has(id);
    const isDir = term && (isKW?s>=1:s>=threshold);
    return `<button class="group-item${isSel?" selected":""}${isDir?" matched":""}"
              onclick="toggleCat('${id}')">
      <span>${esc(m?.category||id)}</span>
      ${scoreStr?`<span class="sim-score">${scoreStr}</span>`:""}
    </button>`;
  }).join("");
}

function toggleCat(id) {
  if (selectedCats.has(id)) selectedCats.delete(id); else selectedCats.add(id);
  document.getElementById("statSelected").textContent = selectedCats.size;
  applyStyles(); renderCategoryPanel();
}
function selectVisible() {
  const term=committedSearch, isKW=searchMode==="keyword"||!embedReady;
  for (const id of categoryIds) {
    const s=simScores.get(id)||0;
    if (!term||(isKW?s>0:s>=threshold)) selectedCats.add(id);
  }
  document.getElementById("statSelected").textContent = selectedCats.size;
  applyStyles(); renderCategoryPanel();
}
function clearSelection() {
  selectedCats.clear();
  document.getElementById("statSelected").textContent = 0;
  applyStyles(); renderCategoryPanel(); resetDetail();
}

// ── detail panel ───────────────────────────────────────────────────────────────
function qBadge(label, val) {
  // For SOUND / PRECISE / CONFIDENCE — show as coloured badge when Yes/No/numeric
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
         <div class="sim-bar-label">Semantic similarity</div>
         <div class="sim-bar-track"><div class="sim-bar-fill" style="width:${(score*100).toFixed(1)}%"></div></div>
         <div style="font-size:10px;color:var(--accent);margin-top:3px;font-family:'IBM Plex Mono',monospace">${(score*100).toFixed(1)}%</div>
       </div>` : "";

  if (m.node_type === "actionable") {
    return `
      <div class="detail-title">${esc(m.short_label||m.id)} <span class="tag tag-a">actionable</span></div>
      <div class="dlabel">Actionable</div>
      <div class="dval">${esc(m.actionable)}</div>
      <div class="dlabel">Impact</div>
      <div class="dval">${esc(m.impact||"N/A")}</div>
      <div class="dlabel">Empirical Evidence</div>
      <div class="dval">${esc(m.empirical_evidence||"N/A")}</div>
      <div class="dlabel">Quality indicators</div>
      <div class="badge-row">
        ${qBadge("Confidence", m.confidence)}
        ${qBadge("Sound",      m.sound)}
        ${qBadge("Precise",    m.precise)}
      </div>
      <div class="dlabel">Category</div>
      <div class="dval">${esc(m.category||"N/A")}</div>
      <div class="dlabel">Article</div>
      <div class="dval">${esc(m.article_title||"N/A")}</div>
      ${m.count ? `<div class="dlabel">Count</div><div class="dval">${esc(m.count)}</div>` : ""}
      ${simBar}`;
  }

  // category node
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
function showAll() {
  document.getElementById("searchBox").value = "";
  document.getElementById("searchClear").classList.remove("visible");
  committedSearch = ""; selectedCats.clear(); simScores = new Map();
  applyStyles(); renderCategoryPanel(); resetDetail(); fitView();
}

// ── summary ────────────────────────────────────────────────────────────────────
function updateSummary() {
  const vEdges = edgesDS ? edgesDS.get().filter(e=>(e.color?.opacity||1)>0.2).length : ALL_EDGES.length;
  document.getElementById("netSummary").textContent =
    `${categoryIds.length} categories · ${actionableIds.length} actionables · ${vEdges} edges visible`;
  document.getElementById("statEdges").textContent = vEdges;
}

function esc(v) {
  return String(v??"").replace(/&/g,"&amp;").replace(/</g,"&lt;")
         .replace(/>/g,"&gt;").replace(/"/g,"&quot;").replace(/'/g,"&#39;");
}

// ── boot ───────────────────────────────────────────────────────────────────────
(function boot() {
  document.getElementById("scRows").textContent    = GRAPH_STATS.row_count;
  document.getElementById("scCats").textContent    = GRAPH_STATS.category_count;
  document.getElementById("scActions").textContent = GRAPH_STATS.actionable_count;
  document.getElementById("scEdges").textContent   = GRAPH_STATS.edge_count;
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
    node_ids: Optional[List[str]] = None,
    embeddings_b64: Optional[List[str]] = None,
) -> str:
    if node_ids and embeddings_b64 and len(node_ids) == len(embeddings_b64):
        baked = {nid: b64 for nid, b64 in zip(node_ids, embeddings_b64)}
        baked_json = json.dumps(baked, ensure_ascii=False)
        tqdm.write(f"  Baking {len(baked)} node embeddings into HTML.")
    else:
        baked_json = "null"
        tqdm.write("  No embeddings baked — browser will use keyword search.")
    return (
        HTML_TEMPLATE
        .replace("__NODES_JSON__",       json.dumps(nodes,     ensure_ascii=False))
        .replace("__NODE_META_JSON__",   json.dumps(node_meta, ensure_ascii=False))
        .replace("__EDGES_JSON__",       json.dumps(edges,     ensure_ascii=False))
        .replace("__STATS_JSON__",       json.dumps(stats,     ensure_ascii=False))
        .replace("__BAKED_EMBEDDINGS__", baked_json)
    )


def write_html(output_path: Path, html_text: str) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with tqdm(total=1, desc="  Writing HTML", unit="file") as bar:
        output_path.write_text(html_text, encoding="utf-8")
        bar.update(1)


# ── main ───────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate ReACT-LLM Category→Actionable network HTML."
    )
    parser.add_argument("--input_csv",   type=Path, default=DEFAULT_INPUT_CSV)
    parser.add_argument("--output_html", type=Path, default=DEFAULT_OUTPUT_HTML)
    parser.add_argument("--skip-embed",  action="store_true",
                        help="Skip embedding — keyword search only")
    parser.add_argument("--workers",     type=int, default=1,
                        help="Parallel embedding workers (default 1)")
    args = parser.parse_args()

    print(f"\n[1/5] Reading {args.input_csv}")
    rows = read_csv_rows(args.input_csv)
    print(f"      {len(rows)} row(s) loaded")

    print("\n[2/5] Building graph payload …")
    nodes, node_meta, edges, stats, node_ids, node_texts = build_graph_payload(rows)
    print(f"      categories={stats['category_count']}  "
          f"actionables={stats['actionable_count']}  "
          f"edges={stats['edge_count']}")

    embeddings_b64 = None
    if not args.skip_embed:
        print(f"\n[3/5] Embedding {len(node_texts)} node texts (workers={args.workers}) …")
        embeddings_b64 = embed_texts(node_texts, n_workers=args.workers)
    else:
        print("\n[3/5] Skipping embedding (--skip-embed).")

    print("\n[4/5] Rendering HTML …")
    with tqdm(total=1, desc="  Serialising", unit="step") as bar:
        html_text = generate_html(nodes, node_meta, edges, stats, node_ids, embeddings_b64)
        bar.update(1)

    print(f"\n[5/5] Writing {args.output_html}")
    write_html(args.output_html, html_text)

    kb = len(html_text.encode()) // 1024
    print(f"\n  Done!  {kb} KB → {args.output_html}")
    if embeddings_b64:
        print("  Semantic search: baked embeddings + Transformers.js query model")
    else:
        print("  Semantic search: disabled. Run without --skip-embed to enable.")
    print("  Open the HTML directly in any browser — no server needed.")


if __name__ == "__main__":
    main()