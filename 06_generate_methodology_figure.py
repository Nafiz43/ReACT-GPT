"""
generate_methodology_figure.py — clear gaps between all boxes
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

ACC = dict(
    p1="#6B63C4",
    p2="#B07D3A",
    p3="#3A7D65",
    p4t="#2E6E5E",
    p4n="#8A8A85",
    p5="#9E5035",
)
BG = dict(p1="#F6F6FA", p2="#FAF7EE", p3="#F3F8F5", p4="#F2F8F5", p5="#FAF4F1")
BAND_BD = "#DEDEDE"

FW, FH = 6.8, 8.8
fig, ax = plt.subplots(figsize=(FW, FH))
ax.set_xlim(0, FW); ax.set_ylim(0, FH)
ax.axis("off"); fig.patch.set_facecolor("white")

ML, MR = 0.15, 0.15
BW  = FW - ML - MR        # total band width
BH  = 0.38                # box height (standard)
BHS = 0.32                # box height (small LLM row)
CX  = ML + BW / 2
FF  = "DejaVu Sans"

# Gap between adjacent boxes within a row
GAP_H = 0.22              # horizontal gap between sibling boxes
# Inner band padding (boxes don't touch band edge)
BP   = 0.18               # band inner horizontal padding

def _tint(hex_color, alpha):
    h = hex_color.lstrip("#")
    r, g, b = int(h[0:2],16), int(h[2:4],16), int(h[4:6],16)
    return "#{:02x}{:02x}{:02x}".format(
        int(r*alpha+255*(1-alpha)), int(g*alpha+255*(1-alpha)), int(b*alpha+255*(1-alpha)))


def band(y_top, height, bg, label, accent):
    r = FancyBboxPatch((ML, y_top-height), BW, height,
                       boxstyle="round,pad=0", facecolor=bg,
                       edgecolor=BAND_BD, linewidth=0.6, zorder=1)
    ax.add_patch(r)
    ax.text(ML+0.08, y_top-0.10, label,
            fontsize=6.5, fontweight="bold", color=accent,
            va="top", zorder=10, fontfamily=FF,
            bbox=dict(facecolor=bg, edgecolor="none", pad=1.2, alpha=0.95))


def node(cx, cy, w, h, accent, title, *subs, tsz=8.0, ssz=6.5):
    r = FancyBboxPatch((cx-w/2, cy-h/2), w, h,
                       boxstyle="round,pad=0.07",
                       facecolor="#FFFFFF", edgecolor=accent,
                       linewidth=0.9, zorder=3)
    ax.add_patch(r)
    n   = len(subs)
    lh  = 0.092
    top = cy + n * lh / 2
    ax.text(cx, top, title, ha="center", va="center",
            fontsize=tsz, fontweight="bold", color="#1C1C1C",
            zorder=4, fontfamily=FF)
    for i, s in enumerate(subs):
        ax.text(cx, top-(i+1)*lh, s, ha="center", va="center",
                fontsize=ssz, color="#555555", zorder=4, fontfamily=FF)


def arr(x1, y1, x2, y2, color="#BBBBBB", lw=0.8, ls="-", hw=0.07):
    ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(
                    arrowstyle=f"->,head_width={hw},head_length={hw*1.1}",
                    color=color, lw=lw, linestyle=ls),
                zorder=6)


def harr(x1, x2, y, accent, lw=1.0):
    arr(x1, y, x2, y, color=accent, lw=lw, hw=0.07)


# Compute evenly-spaced box centres & widths for n boxes in the band,
# with BP padding on left/right and GAP_H between each box.
def row_geom(n):
    usable = BW - 2*BP                      # inner band width
    w = (usable - (n-1)*GAP_H) / n         # each box width
    x0 = ML + BP + w/2                     # centre of first box
    centres = [x0 + i*(w + GAP_H) for i in range(n)]
    return centres, w


# ── Phase geometry ────────────────────────────────────────────────────────────
H = [0, 1.22, 0.72, 0.90, 0.63, 1.54]
G = 0.06
Y = [0]*6
Y[1] = FH - 0.10
for i in range(2, 6):
    Y[i] = Y[i-1] - H[i-1] - G

# ══ PHASE 1 ═══════════════════════════════════════════════════════════════════
band(Y[1], H[1], BG["p1"], "PHASE 1 — EXTRACTION", ACC["p1"])

cy_corp = Y[1] - 0.29
node(CX, cy_corp, 1.65, BH, ACC["p1"],
     "Article Corpus", "SE / NLP research papers")

cx5, W5 = row_geom(5)
cy_llm  = cy_corp - 0.52
llms = [("Qwen3","35b"),("GPT-oss","20b"),("DeepSeek","R1:32b"),
        ("Gemma4","31b"),("Mixtral","8×7b")]
for i, (nm, sb) in enumerate(llms):
    node(cx5[i], cy_llm, W5, BHS, ACC["p1"], nm, sb, tsz=7.5, ssz=6.2)
    arr(CX, cy_corp-BH/2, cx5[i], cy_llm+BHS/2,
        color=_tint(ACC["p1"], 0.4), lw=0.7, hw=0.06)

cy_note1 = cy_llm - 0.25
ax.text(CX, cy_note1,
        "Each model extracts: actionable · impact · evidence · confidence  →  saved to CSV",
        ha="center", va="center", fontsize=6.2, style="italic",
        color="#888888", zorder=5, fontfamily=FF)

# ══ PHASE 2 ═══════════════════════════════════════════════════════════════════
band(Y[2], H[2], BG["p2"], "PHASE 2 — θ CALIBRATION  (STS15 BENCHMARK)", ACC["p2"])

cx3, W3 = row_geom(3)
cy2 = Y[2] - H[2]/2 - 0.02

node(cx3[0], cy2, W3,    BH, ACC["p2"],
     "STS15 Benchmark", "~1,400 sentence pairs", "human scores 0–5")
node(cx3[1], cy2, W3,    BH, ACC["p2"],
     "Two-Phase Grid Search", "Coarse: 0.01→0.95, step 0.02", "Fine: ±0.05, step 0.005")
node(cx3[2], cy2, W3,    BH, ACC["p2"],
     "Optimal θ", "maximises F1 on", "same / different pairs")

harr(cx3[0]+W3/2+0.01, cx3[1]-W3/2-0.01, cy2, ACC["p2"])
harr(cx3[1]+W3/2+0.01, cx3[2]-W3/2-0.01, cy2, ACC["p2"])

arr(CX, cy_note1-0.06, CX, Y[2]+0.01, color="#CCCCCC", lw=0.75, ls="dashed")

# ══ PHASE 3 ═══════════════════════════════════════════════════════════════════
band(Y[3], H[3], BG["p3"], "PHASE 3 — ModeX-SET RECONCILIATION  (PER ARTICLE)", ACC["p3"])

cx4, W4 = row_geom(4)
cy3 = Y[3] - 0.38

stages = [
    ("Stage 0","Global candidate pool","tag source model per item"),
    ("Stage 1","Similarity matrix","Jaccard + overlap, stopwords off"),
    ("Stage 2","Agglomerative clustering","complete linkage, θ"),
    ("Stage 3","ModeX centroid selection","best candidate · avg. confidence"),
]
for i, (t, s1, s2) in enumerate(stages):
    node(cx4[i], cy3, W4, BH, ACC["p3"], t, s1, s2)
    if i < 3:
        harr(cx4[i]+W4/2+0.01, cx4[i+1]-W4/2-0.01, cy3, ACC["p3"])

# θ dashed arrow
arr(cx3[2], cy2-BH/2, cx4[0], cy3+BH/2,
    color=ACC["p2"], lw=0.9, ls="dashed", hw=0.07)
ax.text((cx3[2]+cx4[0])/2+0.09, (cy2-BH/2+cy3+BH/2)/2+0.05,
        "θ", fontsize=7.0, fontweight="bold", color=ACC["p2"],
        zorder=7, fontfamily=FF)

arr(cx3[0], cy2-BH/2, cx3[0], Y[3]+0.01, color="#CCCCCC", lw=0.6, ls="dashed")

# note box
cy3n = Y[3] - H[3] + 0.17
nb = FancyBboxPatch((ML+BP, cy3n-0.115), BW-2*BP, 0.235,
                    boxstyle="round,pad=0.03", facecolor="#F8FAF9",
                    edgecolor="#CCCCCC", linewidth=0.45,
                    linestyle="dashed", zorder=3)
ax.add_patch(nb)
ax.text(CX, cy3n+0.025,
        "merged string = rec · impact · evidence  |  intra-model α = 0.15  |"
        "  support = # distinct models per cluster",
        ha="center", va="center", fontsize=5.8, color="#555555",
        zorder=4, fontfamily=FF)
ax.text(CX, cy3n-0.065,
        "no-actionable responses handled  ·  singletons below min-support"
        " discarded  ·  parallel via ProcessPoolExecutor",
        ha="center", va="center", fontsize=5.4, color="#777777",
        zorder=4, fontfamily=FF)

bus_y = cy3 - BH/2 - 0.04
for cx in cx4:
    ax.plot([cx, cx], [cy3-BH/2, bus_y], color="#CCCCCC", lw=0.55, zorder=5)
ax.plot([cx4[0], cx4[-1]], [bus_y]*2, color="#CCCCCC", lw=0.55, zorder=5)
arr(CX, bus_y, CX, cy3n+0.115, color="#BBBBBB", lw=0.75, hw=0.07)

# ══ PHASE 4 ═══════════════════════════════════════════════════════════════════
band(Y[4], H[4], BG["p4"], "PHASE 4 — OUTPUTS", ACC["p4t"])
cy4 = Y[4] - H[4]/2

# Reuse cx3, W3 geometry (same 3-column layout)
outs = [
    (cx3[0], ACC["p4t"], "Canonical Actionable Set",
     "actionable · impact · evidence", "support · avg confidence"),
    (cx3[1], ACC["p4n"], "Per-Article MD Trace",
     "similarity matrix · clusters",   "confidence arithmetic"),
    (cx3[2], ACC["p4n"], "Master CSV",
     "reconciled_actionables.csv",     "all articles · all clusters"),
]
for cx, acc, t, s1, s2 in outs:
    node(cx, cy4, W3, BH, acc, t, s1, s2)
    clr = ACC["p3"] if acc == ACC["p4t"] else "#BBBBBB"
    arr(cx, cy3n-0.115, cx, Y[4]+0.01, color=clr, lw=0.9, hw=0.07)

# ══ PHASE 5 ═══════════════════════════════════════════════════════════════════
band(Y[5], H[5], BG["p5"], "PHASE 5 — ReACT QUALITY SCORING", ACC["p5"])
cy5_intro = Y[5] - 0.19
ax.text(CX, cy5_intro,
        "Is this actionable  sound?  ·  Is it  precise?  →  each model answers YES / NO independently",
        ha="center", va="center", fontsize=6.2, color="#555555",
        zorder=5, fontfamily=FF)

arr(cx3[0], cy4-BH/2, cx3[0], Y[5]+0.01, color=ACC["p4t"], lw=1.1, hw=0.08)

cy5_llm = cy5_intro - 0.43
for i, (nm, sb) in enumerate(llms):
    node(cx5[i], cy5_llm, W5, BHS, ACC["p5"], nm, sb, tsz=7.5, ssz=6.2)

cy5_maj = cy5_llm - 0.57
node(CX, cy5_maj, 2.45, BH, ACC["p5"],
     "Majority Voting",
     "sound: YES / NO  ·  precise: YES / NO",
     "per-model decisions stored for audit")
for cx in cx5:
    arr(cx, cy5_llm-BHS/2, CX, cy5_maj+BH/2,
        color=_tint(ACC["p5"], 0.4), lw=0.7, hw=0.06)

cy5_sc = cy5_maj - 0.57
node(CX, cy5_sc, 2.75, BH, ACC["p4t"],
     "Scored Actionable Set",
     "sound ✓/✗  ·  precise ✓/✗  ·  per-model verdicts",
     "scored.csv  →  categorized.csv")
arr(CX, cy5_maj-BH/2, CX, cy5_sc+BH/2, color=ACC["p5"], lw=1.1, hw=0.08)

# ── Crop & save ───────────────────────────────────────────────────────────────
ax.set_ylim(cy5_sc - BH/2 - 0.18, FH)
plt.subplots_adjust(0, 0, 1, 1)
for fmt in ("png", "pdf"):
    fig.savefig(f"methodology_figure.{fmt}",
                dpi=300 if fmt=="png" else None,
                bbox_inches="tight", facecolor="white", pad_inches=0.06)
    print(f"✓  Saved methodology_figure.{fmt}")
plt.close()
