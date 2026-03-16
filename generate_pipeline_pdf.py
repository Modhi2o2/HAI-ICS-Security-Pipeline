"""
HAI ICS Anomaly Detection Pipeline — PDF Flowchart Generator
Generates a detailed, professional, multi-page PDF using matplotlib only.
"""

import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.gridspec as gridspec
from matplotlib.lines import Line2D

# ─── COLOR PALETTE ────────────────────────────────────────────────────────────
C_BG        = "#0d1b2a"   # page background
C_PRIMARY   = "#1e3a5f"   # dark blue
C_ACCENT    = "#4fc3f7"   # light blue
C_SUCCESS   = "#00cc44"   # green
C_WARNING   = "#ff8800"   # orange
C_DANGER    = "#ff4444"   # red
C_TEAL      = "#00b4d8"   # teal
C_PURPLE    = "#9b59b6"   # purple
C_GOLD      = "#f4d03f"   # gold
C_TEXT      = "#e8f4fd"   # light text
C_SUBTEXT   = "#a8d8ea"   # muted text
C_BORDER    = "#2e6da4"   # box border
C_DARK_BOX  = "#122840"   # dark box fill

OUTPUT_PATH = "C:/Users/PC GAMING/Desktop/AI/HAI_Pipeline/outputs/HAI_Pipeline_Flowchart.pdf"

# ─── HELPERS ──────────────────────────────────────────────────────────────────

def set_page_bg(fig):
    fig.patch.set_facecolor(C_BG)

def draw_box(ax, x, y, w, h, title, subtitle="", metric="",
             facecolor=C_PRIMARY, edgecolor=C_ACCENT, title_color=C_TEXT,
             sub_color=C_SUBTEXT, metric_color=C_GOLD,
             fontsize_title=9, fontsize_sub=7, fontsize_metric=8,
             radius=0.015, alpha=0.92):
    """Draw a rounded box with title, optional subtitle and metric."""
    box = FancyBboxPatch((x, y), w, h,
                         boxstyle=f"round,pad=0.005,rounding_size={radius}",
                         facecolor=facecolor, edgecolor=edgecolor,
                         linewidth=1.4, alpha=alpha, zorder=3)
    ax.add_patch(box)
    cx = x + w / 2
    # vertical layout
    if metric and subtitle:
        ty = y + h * 0.72
        sy = y + h * 0.45
        my = y + h * 0.18
    elif subtitle:
        ty = y + h * 0.65
        sy = y + h * 0.28
        my = None
    elif metric:
        ty = y + h * 0.65
        sy = None
        my = y + h * 0.28
    else:
        ty = y + h * 0.5
        sy = None
        my = None

    ax.text(cx, ty, title, ha='center', va='center',
            color=title_color, fontsize=fontsize_title,
            fontweight='bold', zorder=4, wrap=True,
            transform=ax.transData)
    if subtitle and sy is not None:
        ax.text(cx, sy, subtitle, ha='center', va='center',
                color=sub_color, fontsize=fontsize_sub, zorder=4,
                transform=ax.transData)
    if metric and my is not None:
        ax.text(cx, my, metric, ha='center', va='center',
                color=metric_color, fontsize=fontsize_metric,
                fontweight='bold', zorder=4, transform=ax.transData)


def arrow(ax, x1, y1, x2, y2, color=C_ACCENT, lw=1.5, style="->", zorder=2):
    ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle=style, color=color,
                                lw=lw, connectionstyle="arc3,rad=0.0"),
                zorder=zorder)


def arrow_curve(ax, x1, y1, x2, y2, rad=0.2, color=C_ACCENT, lw=1.5):
    ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle="->", color=color, lw=lw,
                                connectionstyle=f"arc3,rad={rad}"),
                zorder=2)


def page_title(ax, text, sub=""):
    ax.text(0.5, 0.97, text, transform=ax.transAxes,
            ha='center', va='top', color=C_ACCENT,
            fontsize=14, fontweight='bold')
    if sub:
        ax.text(0.5, 0.93, sub, transform=ax.transAxes,
                ha='center', va='top', color=C_SUBTEXT, fontsize=9)


def divider(ax, y, color=C_BORDER, lw=0.8):
    ax.axhline(y, color=color, lw=lw, alpha=0.5, zorder=1)


def new_ax(fig, rect=None):
    if rect is None:
        rect = [0, 0, 1, 1]
    ax = fig.add_axes(rect)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis('off')
    ax.set_facecolor(C_BG)
    return ax


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 1 — COVER PAGE
# ══════════════════════════════════════════════════════════════════════════════

def make_page1():
    fig = plt.figure(figsize=(16, 10))
    set_page_bg(fig)
    ax = new_ax(fig)

    # gradient-ish background panel
    bg = FancyBboxPatch((0.03, 0.05), 0.94, 0.90,
                        boxstyle="round,pad=0.01,rounding_size=0.02",
                        facecolor=C_PRIMARY, edgecolor=C_ACCENT,
                        linewidth=2, alpha=0.6, zorder=1)
    ax.add_patch(bg)

    # decorative accent bars
    ax.add_patch(mpatches.Rectangle((0.03, 0.91), 0.94, 0.008,
                 facecolor=C_ACCENT, zorder=5, alpha=0.9))
    ax.add_patch(mpatches.Rectangle((0.03, 0.05), 0.94, 0.008,
                 facecolor=C_ACCENT, zorder=5, alpha=0.9))

    # TITLE
    ax.text(0.5, 0.80, "HAI ICS Security Pipeline",
            ha='center', va='center', color=C_TEXT,
            fontsize=32, fontweight='bold', zorder=6)
    ax.text(0.5, 0.72, "Technical Architecture Report",
            ha='center', va='center', color=C_ACCENT,
            fontsize=22, fontweight='bold', zorder=6)
    ax.text(0.5, 0.64, "Industrial Control System Anomaly Detection using LSTM Autoencoder",
            ha='center', va='center', color=C_SUBTEXT,
            fontsize=13, zorder=6)

    # horizontal divider
    ax.add_patch(mpatches.Rectangle((0.15, 0.60), 0.70, 0.002,
                 facecolor=C_TEAL, zorder=5))

    # Key stats boxes
    stats = [
        ("F1 Score",       "0.6886",  C_SUCCESS),
        ("Precision",      "0.7381",  C_TEAL),
        ("Recall",         "0.6454",  C_WARNING),
        ("ROC-AUC",        "0.8650",  C_ACCENT),
        ("Sensors",        "225",     C_PURPLE),
        ("Test Samples",   "284,400", C_GOLD),
    ]
    bw, bh = 0.12, 0.10
    total_w = len(stats) * bw + (len(stats) - 1) * 0.02
    start_x = (1.0 - total_w) / 2
    for i, (label, val, col) in enumerate(stats):
        bx = start_x + i * (bw + 0.02)
        by = 0.43
        box = FancyBboxPatch((bx, by), bw, bh,
                             boxstyle="round,pad=0.005,rounding_size=0.012",
                             facecolor=C_DARK_BOX, edgecolor=col,
                             linewidth=2, alpha=0.95, zorder=6)
        ax.add_patch(box)
        ax.text(bx + bw/2, by + bh*0.68, val,
                ha='center', va='center', color=col,
                fontsize=16, fontweight='bold', zorder=7)
        ax.text(bx + bw/2, by + bh*0.25, label,
                ha='center', va='center', color=C_SUBTEXT,
                fontsize=8, zorder=7)

    # Best model line
    ax.text(0.5, 0.36, "Best Model: LSTM Autoencoder (haiend-23.05, 225 DCS sensors, window=30s, 150K training windows)",
            ha='center', va='center', color=C_TEXT,
            fontsize=10, zorder=6)
    ax.text(0.5, 0.29, "Dataset: haiend-23.05  •  896,400 normal training samples  •  11,384 attack timesteps in test",
            ha='center', va='center', color=C_SUBTEXT,
            fontsize=9, zorder=6)

    # tech tags
    tags = ["PyTorch LSTM-AE", "225 DCS Sensors", "5-Layer Digital Twin",
            "Physics Graph (44 edges)", "Real-time Streaming"]
    tag_colors = [C_ACCENT, C_TEAL, C_SUCCESS, C_WARNING, C_PURPLE]
    tw = 0.13
    start_x2 = (1.0 - (len(tags) * tw + (len(tags)-1)*0.01)) / 2
    for i, (tag, tc) in enumerate(zip(tags, tag_colors)):
        tx = start_x2 + i*(tw+0.01)
        tp = FancyBboxPatch((tx, 0.19), tw, 0.05,
                            boxstyle="round,pad=0.003,rounding_size=0.01",
                            facecolor=C_BG, edgecolor=tc,
                            linewidth=1.5, alpha=0.9, zorder=6)
        ax.add_patch(tp)
        ax.text(tx + tw/2, 0.215, tag,
                ha='center', va='center', color=tc,
                fontsize=8, fontweight='bold', zorder=7)

    ax.text(0.5, 0.11, "Generated: 2026-03-16",
            ha='center', va='center', color=C_SUBTEXT,
            fontsize=9, zorder=6)
    ax.text(0.5, 0.075, "HAI ICS Anomaly Detection Research Project",
            ha='center', va='center', color=C_BORDER,
            fontsize=8, zorder=6)

    return fig


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 2 — SYSTEM OVERVIEW FLOWCHART
# ══════════════════════════════════════════════════════════════════════════════

def make_page2():
    fig = plt.figure(figsize=(16, 10))
    set_page_bg(fig)
    ax = new_ax(fig)

    page_title(ax, "System Overview — End-to-End Pipeline",
               "Full architecture from raw ICS data to real-time anomaly detection")

    # Main vertical flow — left column
    boxes_left = [
        (0.06, 0.70, "HAI Dataset\nhaiend-23.05",
         "end-train1..4.csv\n4 CSV files, 1Hz sampling",
         "896K rows × 225 sensors", C_PRIMARY, C_TEAL),
        (0.06, 0.52, "Data Loading\n& Validation",
         "Read CSVs, parse timestamps\nSelect 225 sensor columns",
         "896,400 normal samples", C_PRIMARY, C_ACCENT),
        (0.06, 0.34, "Preprocessing\n& Normalization",
         "Compute mean/std per sensor\nstd = max(std, 1.0)  ← CRITICAL FIX",
         "160 binary sensors handled", C_DARK_BOX, C_WARNING),
        (0.06, 0.16, "Window Sampling",
         "Random 30s windows\n150,000 windows total",
         "Shape: (150K, 30, 225)", C_PRIMARY, C_ACCENT),
    ]

    # Middle column
    boxes_mid = [
        (0.36, 0.70, "LSTM Autoencoder\nTraining",
         "Encoder→Bottleneck→Decoder\nAdam, lr=3e-4, 60 epochs",
         "batch=512, CosineAnnealingLR", C_DARK_BOX, C_SUCCESS),
        (0.36, 0.52, "Threshold\nCalibration",
         "Percentile sweep 70–99.9%\nMaximize F1 on val set",
         "θ = 0.008075", C_PRIMARY, C_WARNING),
        (0.36, 0.34, "Test Set\nEvaluation",
         "284,400 timesteps\n11,384 attacks (4.0%)",
         "F1=0.6886 / AUC=0.8650", C_DARK_BOX, C_SUCCESS),
        (0.36, 0.16, "Model\nArtifacts",
         "Weights + norm params\nlstm_ae_haiend.pt",
         "Saved to disk", C_PRIMARY, C_ACCENT),
    ]

    # Right column
    boxes_right = [
        (0.66, 0.70, "Digital Twin\n5-Layer Ensemble",
         "Layer A: LSTM-haiend (primary)\nLayers B-E: fallback + context",
         "Decision: MSE ≥ 0.008075", C_DARK_BOX, C_TEAL),
        (0.66, 0.52, "Root Cause\nAnalysis",
         "Rank 225 sensors by MSE\nPhysics edges violated",
         "Attack type classification", C_PRIMARY, C_PURPLE),
        (0.66, 0.34, "Health Score\nDynamics",
         "Starts 100, decays on attack\n+5% recovery per timestep",
         "Acceleration factor", C_DARK_BOX, C_WARNING),
        (0.66, 0.16, "Dashboard\n& Alerting",
         "Real-time visualization\nStreaming F1=0.6874",
         "REST API / live UI", C_PRIMARY, C_DANGER),
    ]

    bw, bh = 0.24, 0.14

    def draw_col(boxes):
        for (bx, by, title, sub, metric, fc, ec) in boxes:
            draw_box(ax, bx, by, bw, bh, title, sub, metric,
                     facecolor=fc, edgecolor=ec,
                     fontsize_title=9, fontsize_sub=7, fontsize_metric=7.5)

    draw_col(boxes_left)
    draw_col(boxes_mid)
    draw_col(boxes_right)

    # Vertical arrows within columns
    def col_arrows(boxes, ec_color):
        for i in range(len(boxes)-1):
            _, by1, *_ = boxes[i]
            _, by2, *_ = boxes[i+1]
            bx0 = boxes[i][0]
            cx = bx0 + bw/2
            arrow(ax, cx, by1, cx, by2+bh, color=ec_color, lw=1.8)

    col_arrows(boxes_left,  C_TEAL)
    col_arrows(boxes_mid,   C_SUCCESS)
    col_arrows(boxes_right, C_PURPLE)

    # Horizontal arrows connecting columns at training level
    arrow(ax, 0.06+bw, 0.70+bh/2, 0.36, 0.70+bh/2, color=C_ACCENT, lw=1.8)
    arrow(ax, 0.36+bw, 0.70+bh/2, 0.66, 0.70+bh/2, color=C_TEAL, lw=1.8)

    # Column headers
    for cx, label, col in [(0.06+bw/2, "DATA PIPELINE", C_TEAL),
                           (0.36+bw/2, "MODEL PIPELINE", C_SUCCESS),
                           (0.66+bw/2, "DEPLOYMENT", C_PURPLE)]:
        ax.text(cx, 0.88, label, ha='center', va='center',
                color=col, fontsize=10, fontweight='bold')
        ax.add_patch(mpatches.Rectangle((cx-0.10, 0.865), 0.20, 0.003,
                     facecolor=col, alpha=0.7, zorder=3))

    # Legend
    legend_items = [
        mpatches.Patch(facecolor=C_PRIMARY, edgecolor=C_TEAL, label='Data Component'),
        mpatches.Patch(facecolor=C_DARK_BOX, edgecolor=C_SUCCESS, label='Model Component'),
        mpatches.Patch(facecolor=C_DARK_BOX, edgecolor=C_PURPLE, label='Deployment Component'),
        mpatches.Patch(facecolor=C_DARK_BOX, edgecolor=C_WARNING, label='Critical Fix'),
    ]
    ax.legend(handles=legend_items, loc='lower right', fontsize=8,
              facecolor=C_DARK_BOX, edgecolor=C_BORDER,
              labelcolor=C_TEXT, framealpha=0.9)

    return fig


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 3 — DATA PIPELINE
# ══════════════════════════════════════════════════════════════════════════════

def make_page3():
    fig = plt.figure(figsize=(16, 10))
    set_page_bg(fig)
    ax = new_ax(fig)

    page_title(ax, "Data Pipeline — From Raw CSV to Training Windows",
               "haiend-23.05 dataset preprocessing and normalization strategy")

    # ── Row 1: Source files
    files = [
        ("end-train1.csv", "~224K rows"),
        ("end-train2.csv", "~224K rows"),
        ("end-train3.csv", "~224K rows"),
        ("end-train4.csv", "~224K rows"),
    ]
    fw, fh = 0.17, 0.09
    gap = 0.03
    total = len(files)*(fw+gap)-gap
    sx = (1-total)/2
    for i, (fname, detail) in enumerate(files):
        fx = sx + i*(fw+gap)
        draw_box(ax, fx, 0.79, fw, fh, fname, detail,
                 facecolor=C_DARK_BOX, edgecolor=C_TEAL,
                 fontsize_title=9, fontsize_sub=7.5)
        if i < len(files)-1:
            arrow(ax, fx+fw, 0.79+fh/2, fx+fw+gap, 0.79+fh/2,
                  color=C_TEAL, lw=1.2)

    # merge arrow down
    merge_cx = 0.5
    arrow(ax, merge_cx, 0.79, merge_cx, 0.72, color=C_ACCENT, lw=2)

    # ── Row 2: Concat box
    draw_box(ax, 0.35, 0.62, 0.30, 0.09,
             "Concatenate All CSV Files",
             "pandas.concat  |  sort by timestamp\nDrop NaN rows",
             "896,400 rows × 226 cols",
             facecolor=C_PRIMARY, edgecolor=C_ACCENT,
             fontsize_title=10, fontsize_sub=7.5, fontsize_metric=8.5)

    arrow(ax, 0.5, 0.62, 0.5, 0.55, color=C_ACCENT, lw=2)

    # ── Row 3: Column selection
    draw_box(ax, 0.10, 0.46, 0.35, 0.08,
             "Select Sensor Columns",
             "Drop 'timestamp', 'attack' label\n225 DCS boiler sensor columns remain",
             "X_train: (896400, 225)",
             facecolor=C_DARK_BOX, edgecolor=C_ACCENT,
             fontsize_title=9, fontsize_sub=7, fontsize_metric=8)
    draw_box(ax, 0.55, 0.46, 0.35, 0.08,
             "Sensor Type Analysis",
             "160/225 sensors: binary DCS constants\n65/225 sensors: continuous analog values",
             "std ≈ 0 for 160 sensors",
             facecolor=C_DARK_BOX, edgecolor=C_WARNING,
             fontsize_title=9, fontsize_sub=7, fontsize_metric=8)

    arrow(ax, 0.45, 0.50, 0.55, 0.50, color=C_ACCENT, lw=1.5)
    arrow(ax, 0.5, 0.46, 0.5, 0.39, color=C_ACCENT, lw=2)

    # ── Row 4: Normalization (CRITICAL)
    norm_box_x = 0.25
    draw_box(ax, norm_box_x, 0.28, 0.50, 0.10,
             "Normalization  ← CRITICAL FIX",
             "mean = X_train.mean(axis=0)    std = X_train.std(axis=0)\nstd = np.maximum(std, 1.0)   # prevents ÷0 for 160 constant sensors\nX_norm = (X - mean) / std",
             "F1 impact: 0.558 → 0.684 (+0.126)",
             facecolor="#1a0000", edgecolor=C_WARNING,
             title_color=C_WARNING, fontsize_title=10,
             fontsize_sub=7.5, fontsize_metric=8.5,
             metric_color=C_SUCCESS)

    arrow(ax, 0.5, 0.28, 0.5, 0.21, color=C_SUCCESS, lw=2)

    # ── Row 5: Window sampling
    draw_box(ax, 0.10, 0.10, 0.35, 0.10,
             "Random Window Sampling",
             "For each sample: pick random start index t\nExtract X_norm[t : t+30]  →  shape (30, 225)",
             "150,000 windows sampled",
             facecolor=C_PRIMARY, edgecolor=C_SUCCESS,
             fontsize_title=9, fontsize_sub=7, fontsize_metric=8)
    draw_box(ax, 0.55, 0.10, 0.35, 0.10,
             "Training Tensor",
             "Shape: (150000, 30, 225)\nStored in memory (float32 ≈ 3.2 GB)\nShuffle + batch to DataLoader",
             "batch_size = 512",
             facecolor=C_PRIMARY, edgecolor=C_ACCENT,
             fontsize_title=9, fontsize_sub=7, fontsize_metric=8)

    arrow(ax, 0.45, 0.15, 0.55, 0.15, color=C_SUCCESS, lw=1.8)

    # Annotation box for why max(std,1.0)
    note_box = FancyBboxPatch((0.01, 0.28), 0.22, 0.11,
                              boxstyle="round,pad=0.005,rounding_size=0.01",
                              facecolor="#1a1a00", edgecolor=C_WARNING,
                              linewidth=1.2, alpha=0.9, zorder=3)
    ax.add_patch(note_box)
    ax.text(0.12, 0.365, "WHY?", ha='center', color=C_WARNING,
            fontsize=9, fontweight='bold', zorder=4)
    ax.text(0.12, 0.335, "If std < 1.0, dividing by\nnear-zero amplifies noise\nby factor of 10^16",
            ha='center', color=C_SUBTEXT, fontsize=7.5, zorder=4)

    arrow(ax, 0.23, 0.335, 0.25, 0.335, color=C_WARNING, lw=1.5)

    return fig


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 4 — LSTM-AE ARCHITECTURE
# ══════════════════════════════════════════════════════════════════════════════

def make_page4():
    fig = plt.figure(figsize=(16, 10))
    set_page_bg(fig)
    ax = new_ax(fig)

    page_title(ax, "LSTM Autoencoder Architecture",
               "Encoder–Bottleneck–Decoder with tensor shapes at each stage")

    # ── Main architecture diagram (horizontal flow)
    arch_y = 0.45  # center y
    bh_arch = 0.22

    components = [
        (0.02,  0.14, "INPUT\nWINDOW",
         "(Batch, 30, 225)\n30s of 225 sensors",
         "", C_DARK_BOX, C_TEAL, C_TEAL),
        (0.18,  0.14, "LSTM\nEncoder 1",
         "Input size: 225\nHidden size: 128\nReturn sequences: Yes",
         "(B, 30, 128)", C_PRIMARY, C_ACCENT, C_GOLD),
        (0.34,  0.14, "LSTM\nEncoder 2",
         "Input size: 128\nHidden size: 48\nReturn sequences: No",
         "(B, 48)", C_PRIMARY, C_ACCENT, C_GOLD),
        (0.50,  0.14, "BOTTLENECK\nLatent z",
         "Compressed repr.\n48-dim vector\nMaximum compression",
         "(B, 48)", C_DARK_BOX, C_SUCCESS, C_SUCCESS),
        (0.66,  0.14, "LSTM\nDecoder 1",
         "Repeat z → (B,30,48)\nHidden size: 128\nReturn sequences: Yes",
         "(B, 30, 128)", C_PRIMARY, C_ACCENT, C_GOLD),
        (0.82,  0.14, "LSTM\nDecoder 2",
         "Input size: 128\nHidden size: 128\nReturn sequences: Yes",
         "(B, 30, 128)", C_PRIMARY, C_ACCENT, C_GOLD),
    ]
    # Output box (separate since it sits right-most)
    bw_arch = 0.14
    for (bx, by, title, sub, metric, fc, ec, mc) in components:
        draw_box(ax, bx, by, bw_arch, bh_arch, title, sub, metric,
                 facecolor=fc, edgecolor=ec, metric_color=mc,
                 fontsize_title=9, fontsize_sub=7, fontsize_metric=8)

    # Final output (linear projection)
    draw_box(ax, 0.82, by, bw_arch, bh_arch,
             "Linear\nProjection",
             "128 → 225\nper timestep",
             "(B, 30, 225)", C_DARK_BOX, C_TEAL, metric_color=C_TEAL,
             fontsize_title=9, fontsize_sub=7, fontsize_metric=8)

    # Arrows between components
    for i in range(len(components)-1):
        bx1 = components[i][0]
        bx2 = components[i+1][0]
        arrow(ax, bx1+bw_arch, by+bh_arch/2, bx2, by+bh_arch/2,
              color=C_ACCENT, lw=2)

    # dropout markers
    for dx in [0.32, 0.80]:
        ax.text(dx, by+bh_arch+0.03, "Dropout(0.1)", ha='center',
                color=C_WARNING, fontsize=7.5, style='italic',
                bbox=dict(boxstyle='round,pad=0.3', facecolor=C_BG,
                          edgecolor=C_WARNING, alpha=0.8))

    # Encoder / Decoder span brackets
    ax.add_patch(mpatches.FancyArrow(0.18, 0.89, 0.30, 0,
                 width=0.003, head_width=0, head_length=0,
                 facecolor=C_ACCENT, alpha=0.7))
    ax.text(0.33, 0.905, "ENCODER", ha='center', color=C_ACCENT,
            fontsize=11, fontweight='bold')
    ax.add_patch(mpatches.FancyArrow(0.66, 0.89, 0.30, 0,
                 width=0.003, head_width=0, head_length=0,
                 facecolor=C_WARNING, alpha=0.7))
    ax.text(0.81, 0.905, "DECODER", ha='center', color=C_WARNING,
            fontsize=11, fontweight='bold')
    ax.text(0.50+bw_arch/2, 0.905, "BOTTLENECK",
            ha='center', color=C_SUCCESS, fontsize=11, fontweight='bold')

    # ── Loss & threshold section
    ax.add_patch(mpatches.Rectangle((0.0, 0.0), 1.0, 0.003,
                 facecolor=C_BORDER, alpha=0.4))

    # Loss box
    draw_box(ax, 0.05, 0.02, 0.27, 0.10,
             "Reconstruction Loss (Training)",
             "MSE = mean((X_input - X_recon)²)\naveraged over (Batch, Time=30, Sensors=225)",
             "Optimizer: Adam, lr=3×10⁻⁴",
             facecolor=C_DARK_BOX, edgecolor=C_SUCCESS,
             fontsize_title=9, fontsize_sub=7.5, fontsize_metric=8)

    # Threshold box
    draw_box(ax, 0.38, 0.02, 0.27, 0.10,
             "Anomaly Threshold (Inference)",
             "score = MSE(window)\nATTACK if score ≥ 0.008075",
             "Pre-calibrated: F1 = 0.6886",
             facecolor=C_DARK_BOX, edgecolor=C_DANGER,
             fontsize_title=9, fontsize_sub=7.5, fontsize_metric=8,
             metric_color=C_SUCCESS)

    # Training config
    draw_box(ax, 0.70, 0.02, 0.27, 0.10,
             "Training Configuration",
             "Epochs: 60  |  Batch: 512\nScheduler: CosineAnnealingLR\nWindows: 150,000",
             "Latent dim: 48",
             facecolor=C_DARK_BOX, edgecolor=C_ACCENT,
             fontsize_title=9, fontsize_sub=7.5, fontsize_metric=8)

    # Compression ratio annotation
    ax.text(0.5, 0.75, "Compression ratio: 225×30 = 6,750 → 48  (×140 compression)",
            ha='center', color=C_GOLD, fontsize=10, fontweight='bold',
            bbox=dict(boxstyle='round,pad=0.4', facecolor=C_BG,
                      edgecolor=C_GOLD, alpha=0.8))

    return fig


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 5 — MODEL EVOLUTION CHART
# ══════════════════════════════════════════════════════════════════════════════

def make_page5():
    fig = plt.figure(figsize=(16, 10))
    set_page_bg(fig)

    # Use a normal axes for the bar chart + an overlay axes for annotations
    ax_main = fig.add_axes([0.07, 0.14, 0.88, 0.72])
    ax_main.set_facecolor(C_DARK_BOX)
    for spine in ax_main.spines.values():
        spine.set_edgecolor(C_BORDER)

    models = [
        "1. Supervised\nXGB/LGB",
        "2. MLP-AE\n(38 feat)",
        "3. MLP-AE\n+EWM",
        "4. GDN\n(38 feat)",
        "5. LSTM-AE\nsmall",
        "6. LSTM-AE\nlarge",
        "7. GDN+LSTM\nEnsemble",
        "8. haiend LSTM\n(bug)",
        "9. haiend\nw=60s",
        "10. haiend\nw=30s 100K",
        "11. haiend\nw=30s 150K\n(FINAL)",
    ]
    f1_scores = [0.12, 0.37, 0.396, 0.417, 0.434, 0.391,
                 0.414, 0.558, 0.496, 0.6874, 0.6886]

    bar_colors = []
    for i, f in enumerate(f1_scores):
        if i == len(f1_scores) - 1:
            bar_colors.append(C_SUCCESS)
        elif f >= 0.65:
            bar_colors.append(C_TEAL)
        elif f >= 0.50:
            bar_colors.append(C_WARNING)
        elif f >= 0.40:
            bar_colors.append(C_ACCENT)
        else:
            bar_colors.append(C_DANGER)

    x = np.arange(len(models))
    bars = ax_main.bar(x, f1_scores, color=bar_colors, width=0.62,
                       edgecolor=C_BORDER, linewidth=0.8, zorder=3)

    # value labels on bars
    for bar, val in zip(bars, f1_scores):
        ax_main.text(bar.get_x() + bar.get_width()/2,
                     bar.get_height() + 0.012,
                     f'{val:.3f}', ha='center', va='bottom',
                     color=C_TEXT, fontsize=8, fontweight='bold')

    ax_main.set_xticks(x)
    ax_main.set_xticklabels(models, fontsize=7.8, color=C_TEXT,
                            ha='center')
    ax_main.set_yticks(np.arange(0, 0.85, 0.1))
    ax_main.set_yticklabels([f'{v:.1f}' for v in np.arange(0, 0.85, 0.1)],
                             color=C_TEXT, fontsize=9)
    ax_main.set_ylabel("F1 Score", color=C_TEXT, fontsize=11)
    ax_main.set_ylim(0, 0.82)
    ax_main.yaxis.grid(True, color=C_BORDER, alpha=0.4, lw=0.7, zorder=1)
    ax_main.tick_params(colors=C_TEXT)

    # Baseline comparison line
    ax_main.axhline(0.5, color=C_WARNING, lw=1, ls='--', alpha=0.6)
    ax_main.text(10.6, 0.505, '0.5', color=C_WARNING, fontsize=8, va='bottom')

    # Breakthrough annotation 1: 225 sensors (model 5→10)
    ax_main.annotate("",
        xy=(9, 0.6874), xytext=(4, 0.434),
        arrowprops=dict(arrowstyle="-|>", color=C_SUCCESS, lw=2,
                        connectionstyle="arc3,rad=-0.35"))
    ax_main.text(6.0, 0.60,
                 "225 sensors\n+0.254 F1",
                 ha='center', color=C_SUCCESS, fontsize=9,
                 fontweight='bold',
                 bbox=dict(boxstyle='round,pad=0.35', facecolor=C_BG,
                           edgecolor=C_SUCCESS, alpha=0.9))

    # Breakthrough annotation 2: norm fix (model 8→10)
    ax_main.annotate("",
        xy=(9, 0.6874), xytext=(7, 0.558),
        arrowprops=dict(arrowstyle="-|>", color=C_WARNING, lw=2,
                        connectionstyle="arc3,rad=-0.25"))
    ax_main.text(8.15, 0.53,
                 "Norm fix\n+0.129 F1",
                 ha='center', color=C_WARNING, fontsize=9,
                 fontweight='bold',
                 bbox=dict(boxstyle='round,pad=0.35', facecolor=C_BG,
                           edgecolor=C_WARNING, alpha=0.9))

    # Best model callout
    ax_main.bar([10], [0.6886], color=C_SUCCESS, width=0.62,
                edgecolor=C_GOLD, linewidth=2.5, zorder=4)
    ax_main.text(10, 0.72, "BEST MODEL\nF1 = 0.6886",
                 ha='center', color=C_GOLD, fontsize=9,
                 fontweight='bold',
                 bbox=dict(boxstyle='round,pad=0.4', facecolor=C_BG,
                           edgecolor=C_GOLD, alpha=0.9))

    # Legend
    legend_items = [
        mpatches.Patch(color=C_DANGER,  label='F1 < 0.40'),
        mpatches.Patch(color=C_ACCENT,  label='F1 0.40–0.49'),
        mpatches.Patch(color=C_WARNING, label='F1 0.50–0.64'),
        mpatches.Patch(color=C_SUCCESS, label='F1 ≥ 0.65 (Best)'),
    ]
    ax_main.legend(handles=legend_items, loc='upper left', fontsize=8,
                   facecolor=C_DARK_BOX, edgecolor=C_BORDER,
                   labelcolor=C_TEXT, framealpha=0.9)

    # Title
    fig.text(0.5, 0.96, "Model Evolution — F1 Score Across All 11 Models",
             ha='center', color=C_ACCENT, fontsize=14, fontweight='bold')
    fig.text(0.5, 0.91, "Key breakthroughs: full sensor set (225) and normalization fix drove +0.26 F1 improvement",
             ha='center', color=C_SUBTEXT, fontsize=9)
    fig.patch.set_facecolor(C_BG)

    return fig


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 6 — DETECTION INFERENCE FLOW
# ══════════════════════════════════════════════════════════════════════════════

def make_page6():
    fig = plt.figure(figsize=(16, 10))
    set_page_bg(fig)
    ax = new_ax(fig)

    page_title(ax, "Real-Time Detection Inference Flow",
               "Step-by-step: sensor reading → buffer → normalize → LSTM → MSE → threshold decision")

    # ── Steps (horizontal pipeline)
    steps = [
        ("STEP 1\nSensor Reading",
         "225 raw sensor\nvalues at time t",
         "1Hz, float32", C_DARK_BOX, C_TEAL),
        ("STEP 2\nSliding Buffer",
         "FIFO buffer of last\n30 timesteps",
         "Shape: (30, 225)", C_PRIMARY, C_ACCENT),
        ("STEP 3\nNormalize",
         "(window − mean)\n÷ max(std, 1.0)",
         "Z-score normalized", C_DARK_BOX, C_WARNING),
        ("STEP 4\nLSTM-AE\nForward Pass",
         "Encode → z (48)\nDecode → recon",
         "~2ms inference", C_PRIMARY, C_ACCENT),
        ("STEP 5\nMSE Score",
         "mean((X − X̂)²)\nover (30, 225)",
         "Scalar score", C_DARK_BOX, C_GOLD),
        ("STEP 6\nThreshold\nDecision",
         "score ≥ 0.008075?",
         "Binary decision", C_PRIMARY, C_DANGER),
    ]

    bw, bh = 0.13, 0.18
    total_w = len(steps)*bw + (len(steps)-1)*0.02
    sx = (1-total_w)/2
    step_centers = []
    for i, (title, sub, metric, fc, ec) in enumerate(steps):
        bx = sx + i*(bw+0.02)
        by = 0.55
        draw_box(ax, bx, by, bw, bh, title, sub, metric,
                 facecolor=fc, edgecolor=ec,
                 fontsize_title=8.5, fontsize_sub=7, fontsize_metric=7.5)
        cx = bx + bw/2
        step_centers.append((cx, by))
        if i < len(steps)-1:
            arrow(ax, bx+bw, by+bh/2,
                  bx+bw+0.02, by+bh/2,
                  color=ec, lw=2)

    # ── Outcome branches
    # ATTACK branch (red)
    attack_x = 0.63
    draw_box(ax, attack_x+0.02, 0.30, 0.18, 0.12,
             "ATTACK DETECTED",
             "MSE ≥ threshold\nTrigger alert",
             "Increment FP/TP counter", C_DARK_BOX, C_DANGER,
             title_color=C_DANGER, fontsize_title=10, fontsize_sub=7.5)
    arrow(ax, attack_x+0.065+0.075, 0.55,
              attack_x+0.065+0.075, 0.42,
              color=C_DANGER, lw=2)

    # NORMAL branch (green)
    draw_box(ax, 0.05, 0.30, 0.18, 0.12,
             "NORMAL OPERATION",
             "MSE < threshold\nUpdate buffer only",
             "Increment TN counter", C_DARK_BOX, C_SUCCESS,
             title_color=C_SUCCESS, fontsize_title=10, fontsize_sub=7.5)
    # Arrow from step 6 box bottom-left to normal box
    arrow_curve(ax, attack_x+0.065+0.075, 0.55,
                    0.14, 0.42,
                    rad=0.2, color=C_SUCCESS, lw=2)

    # ── Digital twin update
    draw_box(ax, 0.38, 0.30, 0.20, 0.12,
             "Digital Twin Update",
             "Health score adjust\nLayer scores update\nRoot cause triggered if attack",
             "", C_DARK_BOX, C_TEAL,
             fontsize_title=9, fontsize_sub=7)

    # Arrows to digital twin
    arrow(ax, 0.14+0.18, 0.36, 0.38, 0.36, color=C_TEAL, lw=1.5)
    arrow(ax, 0.63+0.02+0.18, 0.36, 0.58, 0.36, color=C_TEAL, lw=1.5)

    # ── Performance callout
    draw_box(ax, 0.20, 0.09, 0.60, 0.12,
             "Streaming Performance  (same as batch — no degradation)",
             "Sliding window inference  •  F1=0.6874  •  284,400 timesteps  •  11,384 attacks",
             "Latency: ~2ms per window  |  Throughput: > 500 timesteps/sec",
             facecolor=C_DARK_BOX, edgecolor=C_SUCCESS,
             title_color=C_SUCCESS, fontsize_title=10,
             fontsize_sub=8, fontsize_metric=8.5)

    return fig


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 7 — DIGITAL TWIN ARCHITECTURE
# ══════════════════════════════════════════════════════════════════════════════

def make_page7():
    fig = plt.figure(figsize=(16, 10))
    set_page_bg(fig)
    ax = new_ax(fig)

    page_title(ax, "Digital Twin — 5-Layer Ensemble Architecture",
               "Multi-layer detection with decision fusion, health score, and root cause analysis")

    # ── Five layers (left side)
    layers = [
        ("Layer A — PRIMARY",
         "LSTM-haiend  (225 sensors, w=30s)\nMSE ≥ 0.008075 → ATTACK\nF1=0.6886  |  Weight: 60%",
         C_SUCCESS, C_SUCCESS),
        ("Layer B — FALLBACK",
         "38-feat LSTM-AE  (if haiend unavailable)\nUses subset of continuous sensors\nF1=0.3907  |  Weight: 0%*",
         C_WARNING, C_DARK_BOX),
        ("Layer C — PHYSICS",
         "Physics Residual (44 causal edges)\nRidge regression per edge: tgt~src(t,t-1,…)\nF1=0.0806  |  Weight: 10%",
         C_PURPLE, C_DARK_BOX),
        ("Layer D — BASELINE",
         "Z-score Deviation (225 sensors)\nPer-sensor rolling mean/std\nConfidence indicator  |  Weight: 20%",
         C_ACCENT, C_DARK_BOX),
        ("Layer E — UNSUPERVISED",
         "Isolation Forest  (unsupervised)\nCatch-all for novel anomaly types\nWeight: 10%",
         C_TEAL, C_DARK_BOX),
    ]

    lbw, lbh = 0.33, 0.10
    ly_start = 0.73
    ly_gap = 0.13
    layer_centers = []
    for i, (title, sub, ec, fc) in enumerate(layers):
        by = ly_start - i * ly_gap
        fc_actual = C_PRIMARY if fc == C_DARK_BOX else fc
        if i == 0:
            fc_actual = "#002200"
        draw_box(ax, 0.02, by, lbw, lbh, title, sub,
                 facecolor=fc_actual, edgecolor=ec,
                 title_color=ec, fontsize_title=9,
                 fontsize_sub=7.5)
        layer_centers.append((0.02 + lbw, by + lbh/2))

    # ── Decision logic (center)
    draw_box(ax, 0.42, 0.60, 0.22, 0.22,
             "DECISION LOGIC",
             "is_anomalous =\n(LSTM_MSE ≥ 0.008075)\n\nCombined score [0–1]:\nweighted avg all layers\n(display only)",
             "", C_DARK_BOX, C_WARNING,
             title_color=C_WARNING, fontsize_title=10, fontsize_sub=8)

    # Arrows from layers to decision
    for cx, cy in layer_centers:
        arrow(ax, cx, cy, 0.42, 0.71, color=C_BORDER, lw=1.2)

    # ── Confidence level
    conf_items = [
        (0.42, 0.44, "HIGH Confidence",  "2+ layers agree", C_SUCCESS),
        (0.52, 0.44, "MEDIUM Confidence","1 layer agrees",   C_WARNING),
        (0.62, 0.44, "LOW Confidence",   "0 layers agree",   C_DANGER),
    ]
    cw, ch = 0.12, 0.09
    for bx, by, title, sub, col in conf_items:
        draw_box(ax, bx-0.06, by, cw, ch, title, sub,
                 facecolor=C_DARK_BOX, edgecolor=col,
                 title_color=col, fontsize_title=8.5, fontsize_sub=7)

    arrow(ax, 0.53, 0.60, 0.53, 0.53, color=C_WARNING, lw=1.8)

    # ── Health score
    draw_box(ax, 0.42, 0.26, 0.22, 0.15,
             "Health Score Dynamics",
             "Start: 100\nAttack: score −= anomaly×8×accel\nRecovery: score += 0.05×(100−score)",
             "Range: [0–100]",
             facecolor=C_DARK_BOX, edgecolor=C_ACCENT,
             fontsize_title=9, fontsize_sub=7.5, fontsize_metric=8)

    arrow(ax, 0.53, 0.44, 0.53, 0.41, color=C_ACCENT, lw=1.5)

    # ── Root cause analysis
    draw_box(ax, 0.70, 0.60, 0.28, 0.22,
             "Root Cause Analysis",
             "1. Rank 225 sensors by reconstruction MSE\n   → Top offenders identified\n2. Check 44 physics edges\n   → Violated causal relationships\n3. Classify attack type:\n   sensor_spike | cyberattack\n   sensor_drift | replay_attack",
             "", C_DARK_BOX, C_PURPLE,
             title_color=C_PURPLE, fontsize_title=10, fontsize_sub=7.5)

    arrow(ax, 0.64, 0.71, 0.70, 0.71, color=C_PURPLE, lw=1.5)

    # ── Physics graph note
    draw_box(ax, 0.70, 0.26, 0.28, 0.30,
             "Physics Graph",
             "41 boiler components\n67 physical links → 44 sensor edges\n\nEach edge: Ridge regression\ntgt(t) ~ src(t), src(t-1), src(t-2)\n         src(t-5), src(t-10), src(t-30)\n\nResidual = |actual − predicted|\nHigh residual → causality violated",
             "", C_DARK_BOX, C_PURPLE,
             title_color=C_SUBTEXT, fontsize_title=9, fontsize_sub=7.5)

    arrow(ax, 0.84, 0.60, 0.84, 0.56, color=C_PURPLE, lw=1.5)

    # Health score mini chart (decorative)
    ax_hs = fig.add_axes([0.03, 0.03, 0.35, 0.12])
    ax_hs.set_facecolor(C_DARK_BOX)
    t = np.linspace(0, 100, 400)
    health = 100 * np.ones(400)
    # simulate attack period 30-60
    for i in range(400):
        if 120 <= i <= 200:
            health[i] = health[i-1] - 0.8*8*1.2
            health[i] = max(0, health[i])
        elif i > 200:
            health[i] = health[i-1] + 0.05*(100-health[i-1])
    ax_hs.plot(t, health, color=C_SUCCESS, lw=2)
    ax_hs.axvspan(30, 50, alpha=0.2, color=C_DANGER, label='Attack')
    ax_hs.set_ylim(0, 110)
    ax_hs.set_facecolor(C_DARK_BOX)
    ax_hs.tick_params(colors=C_TEXT, labelsize=7)
    ax_hs.set_title("Health Score Over Time", color=C_ACCENT,
                    fontsize=8, pad=3)
    ax_hs.set_xlabel("Time", color=C_SUBTEXT, fontsize=7)
    for sp in ax_hs.spines.values():
        sp.set_edgecolor(C_BORDER)
    ax_hs.legend(fontsize=7, facecolor=C_BG, labelcolor=C_TEXT,
                 edgecolor=C_BORDER)

    return fig


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 8 — FINAL RESULTS DASHBOARD
# ══════════════════════════════════════════════════════════════════════════════

def make_page8():
    fig = plt.figure(figsize=(16, 10))
    set_page_bg(fig)
    fig.suptitle("Final Results Dashboard — LSTM Autoencoder Performance",
                 color=C_ACCENT, fontsize=14, fontweight='bold', y=0.97)

    # ── Layout: 3 columns
    # Col 1: metric cards + confusion matrix
    # Col 2: performance comparison bar
    # Col 3: summary text

    # ─ Performance metric cards ─────────────────────────────────────────────
    ax_cards = fig.add_axes([0.02, 0.55, 0.30, 0.38])
    ax_cards.set_facecolor(C_BG)
    ax_cards.axis('off')
    ax_cards.set_xlim(0, 1)
    ax_cards.set_ylim(0, 1)

    cards = [
        ("F1 Score",  "0.6886", C_SUCCESS),
        ("Precision", "0.7381", C_TEAL),
        ("Recall",    "0.6454", C_WARNING),
        ("ROC-AUC",   "0.8650", C_ACCENT),
    ]
    cw, ch = 0.44, 0.38
    for i, (label, val, col) in enumerate(cards):
        cx = (i % 2) * (cw + 0.10)
        cy = 0.52 - (i // 2) * (ch + 0.10)
        box = FancyBboxPatch((cx, cy), cw, ch,
                             boxstyle="round,pad=0.01,rounding_size=0.05",
                             facecolor=C_DARK_BOX, edgecolor=col,
                             linewidth=2.5, alpha=0.95)
        ax_cards.add_patch(box)
        ax_cards.text(cx+cw/2, cy+ch*0.68, val,
                      ha='center', va='center', color=col,
                      fontsize=22, fontweight='bold')
        ax_cards.text(cx+cw/2, cy+ch*0.25, label,
                      ha='center', va='center', color=C_SUBTEXT,
                      fontsize=9)

    ax_cards.text(0.5, 0.97, "Key Metrics — Test Set",
                  ha='center', color=C_TEXT, fontsize=10, fontweight='bold')

    # ─ Confusion matrix ──────────────────────────────────────────────────────
    ax_cm = fig.add_axes([0.02, 0.08, 0.30, 0.43])
    ax_cm.set_facecolor(C_DARK_BOX)
    for sp in ax_cm.spines.values():
        sp.set_edgecolor(C_BORDER)

    cm = np.array([[270409, 2607],
                   [4037,   7347]])
    labels = [["TN\n270,409", "FP\n2,607"],
              ["FN\n4,037",   "TP\n7,347"]]
    colors_cm = [[C_SUCCESS, C_DANGER],
                 [C_WARNING, C_SUCCESS]]

    for r in range(2):
        for c in range(2):
            fc = "#002200" if (r==0 and c==0) else \
                 "#220000" if (r==0 and c==1) else \
                 "#221100" if (r==1 and c==0) else \
                 "#003300"
            rect = FancyBboxPatch((c*0.49+0.01, r*0.38+0.06), 0.47, 0.36,
                                  boxstyle="round,pad=0.01,rounding_size=0.03",
                                  facecolor=fc, edgecolor=colors_cm[r][c],
                                  linewidth=2.5, transform=ax_cm.transAxes,
                                  zorder=2)
            ax_cm.add_patch(rect)
            ax_cm.text(c*0.49+0.245, r*0.38+0.265, labels[r][c],
                       ha='center', va='center',
                       color=colors_cm[r][c], fontsize=11,
                       fontweight='bold', transform=ax_cm.transAxes, zorder=3)

    ax_cm.set_xticks([])
    ax_cm.set_yticks([])
    ax_cm.set_title("Confusion Matrix  (284,400 test timesteps)",
                    color=C_TEXT, fontsize=9, pad=5)

    # column/row labels
    ax_cm.text(0.25, -0.04, "Predicted Normal",
               ha='center', color=C_SUBTEXT, fontsize=8,
               transform=ax_cm.transAxes)
    ax_cm.text(0.75, -0.04, "Predicted Attack",
               ha='center', color=C_SUBTEXT, fontsize=8,
               transform=ax_cm.transAxes)
    ax_cm.text(-0.06, 0.27, "Actual\nNormal", ha='center', color=C_SUBTEXT,
               fontsize=8, rotation=90, transform=ax_cm.transAxes)
    ax_cm.text(-0.06, 0.65, "Actual\nAttack", ha='center', color=C_SUBTEXT,
               fontsize=8, rotation=90, transform=ax_cm.transAxes)

    # ─ Model comparison bar ──────────────────────────────────────────────────
    ax_bar = fig.add_axes([0.37, 0.08, 0.38, 0.83])
    ax_bar.set_facecolor(C_DARK_BOX)
    for sp in ax_bar.spines.values():
        sp.set_edgecolor(C_BORDER)

    model_names = [
        "XGB/LGB\n(supervised)",
        "MLP-AE\n(38 feat)",
        "MLP-AE\n+EWM",
        "GDN\n(38 feat)",
        "LSTM-AE\nsmall",
        "LSTM-AE\nlarge",
        "GDN+LSTM\nEnsemble",
        "haiend LSTM\n(bug)",
        "haiend w=60s",
        "haiend\nw=30s 100K",
        "FINAL\nBEST",
    ]
    f1_vals = [0.12, 0.37, 0.396, 0.417, 0.434, 0.391,
               0.414, 0.558, 0.496, 0.6874, 0.6886]

    colors_bar = [C_DANGER if v < 0.40 else
                  C_ACCENT if v < 0.50 else
                  C_WARNING if v < 0.65 else
                  C_SUCCESS
                  for v in f1_vals]
    colors_bar[-1] = C_GOLD

    y_pos = np.arange(len(model_names))
    h_bars = ax_bar.barh(y_pos, f1_vals, color=colors_bar,
                         edgecolor=C_BORDER, height=0.65, linewidth=0.7)

    for bar, val in zip(h_bars, f1_vals):
        ax_bar.text(val + 0.008, bar.get_y() + bar.get_height()/2,
                    f'{val:.3f}', va='center', color=C_TEXT, fontsize=8.5,
                    fontweight='bold')

    ax_bar.set_yticks(y_pos)
    ax_bar.set_yticklabels(model_names, color=C_TEXT, fontsize=8)
    ax_bar.set_xlim(0, 0.85)
    ax_bar.set_xlabel("F1 Score", color=C_TEXT, fontsize=10)
    ax_bar.tick_params(colors=C_TEXT)
    ax_bar.axvline(0.5, color=C_WARNING, lw=1, ls='--', alpha=0.6)
    ax_bar.xaxis.grid(True, color=C_BORDER, alpha=0.3, lw=0.7)
    ax_bar.set_title("All Models Compared — F1 Score",
                     color=C_ACCENT, fontsize=11, pad=8)

    # Best model arrow
    ax_bar.annotate("BEST\nF1=0.6886",
                    xy=(0.6886, 10), xytext=(0.62, 9.2),
                    arrowprops=dict(arrowstyle="->", color=C_GOLD, lw=1.5),
                    color=C_GOLD, fontsize=9, fontweight='bold',
                    ha='center')

    # ─ Summary panel ─────────────────────────────────────────────────────────
    ax_sum = fig.add_axes([0.78, 0.08, 0.20, 0.83])
    ax_sum.set_facecolor(C_DARK_BOX)
    ax_sum.set_xlim(0, 1)
    ax_sum.set_ylim(0, 1)
    ax_sum.axis('off')
    for sp in ax_sum.spines.values():
        sp.set_edgecolor(C_BORDER)

    summary_lines = [
        ("FINAL MODEL", C_GOLD, 12, True),
        ("", C_TEXT, 8, False),
        ("LSTM Autoencoder", C_ACCENT, 10, True),
        ("haiend-23.05 dataset", C_SUBTEXT, 8, False),
        ("225 DCS sensors", C_SUBTEXT, 8, False),
        ("Window = 30 seconds", C_SUBTEXT, 8, False),
        ("150K training windows", C_SUBTEXT, 8, False),
        ("", C_TEXT, 8, False),
        ("PERFORMANCE", C_GOLD, 10, True),
        ("", C_TEXT, 8, False),
        ("F1     = 0.6886", C_SUCCESS, 9, True),
        ("Prec.  = 0.7381", C_TEAL, 9, False),
        ("Recall = 0.6454", C_WARNING, 9, False),
        ("AUC    = 0.8650", C_ACCENT, 9, False),
        ("", C_TEXT, 8, False),
        ("TEST SET", C_GOLD, 10, True),
        ("", C_TEXT, 8, False),
        ("284,400 timesteps", C_SUBTEXT, 8, False),
        ("11,384 attacks (4%)", C_SUBTEXT, 8, False),
        ("TP = 7,347", C_SUCCESS, 8, False),
        ("FP = 2,607", C_WARNING, 8, False),
        ("TN = 270,409", C_SUCCESS, 8, False),
        ("FN = 4,037", C_DANGER, 8, False),
        ("", C_TEXT, 8, False),
        ("THRESHOLD", C_GOLD, 10, True),
        ("", C_TEXT, 8, False),
        ("θ = 0.008075", C_ACCENT, 9, True),
        ("", C_TEXT, 8, False),
        ("STREAMING", C_GOLD, 10, True),
        ("", C_TEXT, 8, False),
        ("F1 = 0.6874", C_SUCCESS, 9, True),
        ("(no degradation)", C_SUBTEXT, 8, False),
    ]

    y_pos_text = 0.97
    dy = 0.028
    for text, col, fs, bold in summary_lines:
        if text:
            ax_sum.text(0.5, y_pos_text, text,
                        ha='center', va='top', color=col,
                        fontsize=fs, fontweight='bold' if bold else 'normal')
        y_pos_text -= dy

    return fig


# ══════════════════════════════════════════════════════════════════════════════
# MAIN — ASSEMBLE PDF
# ══════════════════════════════════════════════════════════════════════════════

def main():
    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    print(f"Generating PDF: {OUTPUT_PATH}")

    page_funcs = [
        ("Cover Page",                    make_page1),
        ("System Overview",               make_page2),
        ("Data Pipeline",                 make_page3),
        ("LSTM-AE Architecture",          make_page4),
        ("Model Evolution Chart",         make_page5),
        ("Detection Inference Flow",      make_page6),
        ("Digital Twin Architecture",     make_page7),
        ("Final Results Dashboard",       make_page8),
    ]

    with PdfPages(OUTPUT_PATH) as pdf:
        for i, (name, func) in enumerate(page_funcs, 1):
            print(f"  Rendering page {i}: {name} ...", end=" ", flush=True)
            fig = func()
            pdf.savefig(fig, facecolor=C_BG, bbox_inches='tight', dpi=150)
            plt.close(fig)
            print("done")

        # PDF metadata
        d = pdf.infodict()
        d['Title']   = 'HAI ICS Security Pipeline — Technical Architecture'
        d['Author']  = 'HAI ICS Anomaly Detection Research'
        d['Subject'] = 'LSTM Autoencoder Anomaly Detection for Industrial Control Systems'
        d['Keywords'] = 'LSTM Autoencoder ICS anomaly detection haiend digital twin'
        d['CreationDate'] = '2026-03-16'

    print(f"\nPDF successfully saved to:\n  {OUTPUT_PATH}")
    size_kb = os.path.getsize(OUTPUT_PATH) / 1024
    print(f"  File size: {size_kb:.1f} KB")


if __name__ == "__main__":
    main()
