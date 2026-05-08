"""
visualizer.py
=============
AIDRA – Adaptive Intelligent Disaster Response Agent
AIC-201 CCP | Module 7: Visualization

Generates all publication-quality figures used in the IEEE report:

  1. plot_environment_grid   – 10×10 grid with victims, paths, base, centres
  2. plot_search_comparison  – nodes expanded & path length bar charts
  3. plot_ml_metrics         – grouped bar chart of accuracy/precision/recall/F1
  4. plot_confusion_matrices – 3 side-by-side confusion matrices
  5. plot_fuzzy_scores       – victim priority bar chart with threshold lines
  6. plot_kpi_dashboard      – KPI summary + CSP backtrack comparison
  7. plot_rescue_timeline    – rescue sequence Gantt-style chart

All plots save to *figures/* directory as high-resolution PDFs.
"""

import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.gridspec as gridspec
from matplotlib.colors import ListedColormap

from environment       import GRID_SIZE, RESCUE_BASE, MEDICAL_CENTERS
from search_algorithms import astar

FIGURES_DIR = 'figures'
os.makedirs(FIGURES_DIR, exist_ok=True)

PALETTE = {
    'critical': '#e63946',
    'moderate': '#f4a261',
    'minor':    '#e9c46a',
    'free':     '#f8f9fa',
    'blocked':  '#495057',
    'risk':     '#ff6b6b',
}


# ─────────────────────────────────────────────────────────────────────────────
# 1. ENVIRONMENT GRID
# ─────────────────────────────────────────────────────────────────────────────

def plot_environment_grid(victims: list, grid: list,
                          filename: str = 'grid.pdf') -> str:
    """
    Render the disaster grid with victims, rescue base, medical centres,
    and A* planned routes for all victims.
    """
    fig, ax = plt.subplots(figsize=(6.5, 6.5))

    cmap = ListedColormap([PALETTE['free'], PALETTE['blocked'], PALETTE['risk']])
    ax.imshow(np.array(grid), cmap=cmap, vmin=0, vmax=2, aspect='equal')
    ax.set_xticks(range(GRID_SIZE))
    ax.set_yticks(range(GRID_SIZE))
    ax.set_xticklabels(range(GRID_SIZE), fontsize=7)
    ax.set_yticklabels(range(GRID_SIZE), fontsize=7)
    ax.grid(True, color='gray', linewidth=0.4, alpha=0.5)

    # Plot A* routes for all victims
    route_colors = ['#1d3557', '#457b9d', '#2a9d8f', '#e76f51', '#8338ec']
    for idx, v in enumerate(victims):
        path, _ = astar(RESCUE_BASE, v['pos'], grid, allow_risk=True)
        if path:
            xs = [p[1] for p in path]
            ys = [p[0] for p in path]
            ax.plot(xs, ys, '-', lw=2, alpha=0.6,
                    color=route_colors[idx % len(route_colors)],
                    label=f"Route V{v['id']}")

    # Plot victims
    for v in victims:
        r, c   = v['pos']
        color  = PALETTE.get(v['severity'], 'gray')
        ax.plot(c, r, 's', ms=16, color=color,
                markeredgecolor='black', markeredgewidth=1.2, zorder=6)
        ax.text(c, r, f"V{v['id']}", ha='center', va='center',
                fontsize=8, fontweight='bold', zorder=7)

    # Rescue base
    ax.plot(RESCUE_BASE[1], RESCUE_BASE[0], '^', ms=16,
            color='#2dc653', markeredgecolor='black',
            markeredgewidth=1.5, zorder=6, label='Rescue Base')

    # Medical centres
    for i, mc in enumerate(MEDICAL_CENTERS):
        ax.plot(mc[1], mc[0], 'P', ms=16,
                color='#4361ee', markeredgecolor='black',
                markeredgewidth=1.5, zorder=6,
                label=f'Med Centre {i+1}')

    # Legend
    patches = [
        mpatches.Patch(color=PALETTE['free'],     label='Free cell'),
        mpatches.Patch(color=PALETTE['blocked'],  label='Blocked'),
        mpatches.Patch(color=PALETTE['risk'],     label='High-Risk'),
        mpatches.Patch(color=PALETTE['critical'], label='Critical victim'),
        mpatches.Patch(color=PALETTE['moderate'], label='Moderate victim'),
        mpatches.Patch(color=PALETTE['minor'],    label='Minor victim'),
    ]
    ax.legend(handles=patches, loc='upper right', fontsize=6.5,
              framealpha=0.92, edgecolor='gray')
    ax.set_title('AIDRA Environment Grid with A* Planned Routes',
                 fontsize=10, fontweight='bold', pad=8)

    path = os.path.join(FIGURES_DIR, filename)
    plt.tight_layout()
    plt.savefig(path, dpi=180, bbox_inches='tight')
    plt.close()
    print(f"[VIZ] Saved → {path}")
    return path


# ─────────────────────────────────────────────────────────────────────────────
# 2. SEARCH ALGORITHM COMPARISON
# ─────────────────────────────────────────────────────────────────────────────

def plot_search_comparison(search_results: dict,
                           filename: str = 'search_compare.pdf') -> str:
    """
    Side-by-side bar charts: nodes expanded and path length per victim,
    grouped by algorithm.
    """
    victim_ids = sorted(search_results.keys())
    alg_names  = ['BFS', 'DFS', 'Greedy', 'A*']
    colors     = ['#4cc9f0', '#f72585', '#7209b7', '#3a86ff']
    x          = np.arange(len(victim_ids))
    width      = 0.18

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))

    for metric, ax, ylabel, title in [
        ('expanded', axes[0], 'Nodes Expanded',    'Nodes Expanded per Victim'),
        ('path_len', axes[1], 'Path Length (steps)', 'Path Length per Victim'),
    ]:
        for i, (alg, col) in enumerate(zip(alg_names, colors)):
            vals = [search_results[vid].get(alg, {}).get(metric, 0)
                    for vid in victim_ids]
            bars = ax.bar(x + i * width, vals, width,
                          label=alg, color=col, edgecolor='white', linewidth=0.5)
        ax.set_xlabel('Victim ID', fontsize=9)
        ax.set_ylabel(ylabel, fontsize=9)
        ax.set_title(title, fontsize=9, fontweight='bold')
        ax.set_xticks(x + width * 1.5)
        ax.set_xticklabels([f'V{v}' for v in victim_ids])
        ax.legend(fontsize=8, framealpha=0.85)
        ax.spines[['top', 'right']].set_visible(False)

    plt.suptitle('Search Algorithm Comparison', fontsize=11,
                 fontweight='bold', y=1.01)
    plt.tight_layout()
    path = os.path.join(FIGURES_DIR, filename)
    plt.savefig(path, dpi=180, bbox_inches='tight')
    plt.close()
    print(f"[VIZ] Saved → {path}")
    return path


# ─────────────────────────────────────────────────────────────────────────────
# 3. ML METRIC COMPARISON
# ─────────────────────────────────────────────────────────────────────────────

def plot_ml_metrics(ml_results: dict,
                    filename: str = 'ml_compare.pdf') -> str:
    """
    Grouped bar chart comparing Accuracy, Precision, Recall, F1 across models.
    """
    metrics     = ['accuracy', 'precision', 'recall', 'f1']
    metric_lbls = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
    model_names = list(ml_results.keys())
    colors      = ['#4361ee', '#7209b7', '#f72585']
    x           = np.arange(len(metrics))
    width       = 0.25

    fig, ax = plt.subplots(figsize=(8.5, 4.5))
    for i, (name, col) in enumerate(zip(model_names, colors)):
        vals = [ml_results[name][m] for m in metrics]
        bars = ax.bar(x + i * width, vals, width, label=name,
                      color=col, edgecolor='white', linewidth=0.5)
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.01,
                    f'{v:.2f}', ha='center', va='bottom', fontsize=7)

    ax.set_ylim(0, 1.15)
    ax.set_xticks(x + width)
    ax.set_xticklabels(metric_lbls, fontsize=9)
    ax.set_ylabel('Score', fontsize=9)
    ax.set_title('ML Model Performance Comparison', fontsize=10,
                 fontweight='bold')
    ax.legend(fontsize=8, framealpha=0.85)
    ax.spines[['top', 'right']].set_visible(False)
    ax.axhline(1.0, color='gray', linestyle='--', linewidth=0.7, alpha=0.6)

    plt.tight_layout()
    path = os.path.join(FIGURES_DIR, filename)
    plt.savefig(path, dpi=180, bbox_inches='tight')
    plt.close()
    print(f"[VIZ] Saved → {path}")
    return path


# ─────────────────────────────────────────────────────────────────────────────
# 4. CONFUSION MATRICES
# ─────────────────────────────────────────────────────────────────────────────

def plot_confusion_matrices(ml_results: dict,
                            filename: str = 'confusion.pdf') -> str:
    """Three side-by-side annotated confusion matrices."""
    fig, axes = plt.subplots(1, 3, figsize=(11, 3.8))

    for ax, (name, res) in zip(axes, ml_results.items()):
        cm = np.array(res['cm'])
        im = ax.imshow(cm, cmap='Blues', vmin=0)
        ax.set_title(name, fontsize=9, fontweight='bold', pad=6)
        ax.set_xticks([0, 1])
        ax.set_yticks([0, 1])
        ax.set_xticklabels(['Pred: 0', 'Pred: 1'], fontsize=8)
        ax.set_yticklabels(['True: 0', 'True: 1'], fontsize=8)
        ax.set_xlabel('Predicted', fontsize=8)
        ax.set_ylabel('Actual',    fontsize=8)
        for i in range(2):
            for j in range(2):
                color = 'white' if cm[i, j] > cm.max() / 2 else 'black'
                ax.text(j, i, str(cm[i, j]),
                        ha='center', va='center',
                        fontsize=14, fontweight='bold', color=color)

    plt.suptitle('Confusion Matrices – All ML Models', fontsize=10,
                 fontweight='bold', y=1.02)
    plt.tight_layout()
    path = os.path.join(FIGURES_DIR, filename)
    plt.savefig(path, dpi=180, bbox_inches='tight')
    plt.close()
    print(f"[VIZ] Saved → {path}")
    return path


# ─────────────────────────────────────────────────────────────────────────────
# 5. FUZZY PRIORITY SCORES
# ─────────────────────────────────────────────────────────────────────────────

def plot_fuzzy_scores(fuzzy_scores: dict,
                      filename: str = 'fuzzy.pdf') -> str:
    """
    Bar chart of fuzzy priority score per victim, with threshold lines.
    """
    ids  = [f'V{k}' for k in sorted(fuzzy_scores)]
    vals = [fuzzy_scores[k] for k in sorted(fuzzy_scores)]
    cols = [
        PALETTE['critical'] if v > 0.70 else
        PALETTE['moderate'] if v > 0.50 else
        '#6bcb77'
        for v in vals
    ]

    fig, ax = plt.subplots(figsize=(6.5, 4.0))
    bars = ax.bar(ids, vals, color=cols, edgecolor='#333333',
                  linewidth=0.8, width=0.5)

    ax.axhline(0.70, color=PALETTE['critical'], linestyle='--',
               linewidth=1.2, label='High-priority threshold (0.70)')
    ax.axhline(0.50, color=PALETTE['moderate'], linestyle='--',
               linewidth=1.2, label='Medium threshold (0.50)')

    for bar, v in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.015, f'{v:.2f}',
                ha='center', va='bottom', fontsize=10, fontweight='bold')

    ax.set_ylim(0, 1.05)
    ax.set_xlabel('Victim', fontsize=9)
    ax.set_ylabel('Fuzzy Priority Score', fontsize=9)
    ax.set_title('Fuzzy Inference Priority Scores per Victim',
                 fontsize=10, fontweight='bold')
    ax.legend(fontsize=8, framealpha=0.85)
    ax.spines[['top', 'right']].set_visible(False)

    plt.tight_layout()
    path = os.path.join(FIGURES_DIR, filename)
    plt.savefig(path, dpi=180, bbox_inches='tight')
    plt.close()
    print(f"[VIZ] Saved → {path}")
    return path


# ─────────────────────────────────────────────────────────────────────────────
# 6. KPI DASHBOARD
# ─────────────────────────────────────────────────────────────────────────────

def plot_kpi_dashboard(kpis: dict, csp_data: dict,
                       filename: str = 'kpis.pdf') -> str:
    """
    Two-panel figure:
      Left  – selected KPI bar chart
      Right – CSP backtrack comparison (with vs without MRV heuristic)
    """
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))

    # ── Left: KPIs ──────────────────────────────────────────────────────────
    ax1   = axes[0]
    kpi_labels = ['Victims\nSaved', 'Avg Rescue\nTime', 'Path\nOptimality×10',
                  'Resource\nUtilisation', 'Risk\nExposure']
    kpi_vals   = [
        kpis['victims_saved'],
        kpis['avg_rescue_time'],
        kpis['path_optimality_ratio'] * 10,
        kpis['resource_utilization'] * 5,
        kpis['risk_exposure_score'],
    ]
    kpi_colors = ['#2dc653', '#3a86ff', '#8338ec', '#f4a261', '#e63946']

    bars = ax1.bar(kpi_labels, kpi_vals, color=kpi_colors,
                   edgecolor='white', linewidth=0.5)
    for bar, v in zip(bars, kpi_vals):
        ax1.text(bar.get_x() + bar.get_width() / 2,
                 bar.get_height() + 0.05,
                 f'{v:.1f}', ha='center', va='bottom', fontsize=9)
    ax1.set_title('Key Performance Indicators', fontsize=9, fontweight='bold')
    ax1.set_ylabel('Value (scaled for display)', fontsize=8)
    ax1.spines[['top', 'right']].set_visible(False)

    # ── Right: CSP backtrack comparison ─────────────────────────────────────
    ax2 = axes[1]
    bt_labels = ['Plain\nBacktracking', 'MRV +\nForward Checking']
    bt_vals   = [csp_data['bt_no_heuristic'], csp_data['bt_heuristic']]
    bt_colors = ['#e63946', '#2dc653']

    bars2 = ax2.bar(bt_labels, bt_vals, color=bt_colors,
                    edgecolor='white', linewidth=0.5, width=0.4)
    for bar, v in zip(bars2, bt_vals):
        ax2.text(bar.get_x() + bar.get_width() / 2,
                 bar.get_height() + 0.2,
                 str(v), ha='center', va='bottom', fontsize=11, fontweight='bold')
    ax2.set_title('CSP Backtrack Count Comparison', fontsize=9, fontweight='bold')
    ax2.set_ylabel('Number of Backtracks', fontsize=8)
    ax2.spines[['top', 'right']].set_visible(False)

    plt.suptitle('AIDRA Performance Summary', fontsize=11,
                 fontweight='bold', y=1.02)
    plt.tight_layout()
    path = os.path.join(FIGURES_DIR, filename)
    plt.savefig(path, dpi=180, bbox_inches='tight')
    plt.close()
    print(f"[VIZ] Saved → {path}")
    return path


# ─────────────────────────────────────────────────────────────────────────────
# 7. RESCUE TIMELINE (Gantt-style)
# ─────────────────────────────────────────────────────────────────────────────

def plot_rescue_timeline(victims: list,
                         filename: str = 'rescue_timeline.pdf') -> str:
    """
    Horizontal Gantt-style chart showing each victim's rescue window:
    from step 0 to their rescue_time, coloured by severity.
    """
    rescued = [v for v in victims if v.get('rescue_time') is not None]
    if not rescued:
        print("[VIZ] No rescued victims – skipping timeline")
        return ''

    fig, ax = plt.subplots(figsize=(8, 3.5))

    y_labels = []
    for i, v in enumerate(sorted(rescued, key=lambda x: x['rescue_time'])):
        col  = PALETTE.get(v['severity'], 'steelblue')
        ax.barh(i, v['rescue_time'], left=0,
                color=col, edgecolor='white', linewidth=0.5, height=0.6)
        ax.text(v['rescue_time'] + 0.2, i,
                f"  {v['rescue_time']} steps", va='center', fontsize=8)
        y_labels.append(f"V{v['id']} ({v['severity'][:3].upper()})")

    ax.set_yticks(range(len(rescued)))
    ax.set_yticklabels(y_labels, fontsize=8)
    ax.set_xlabel('Simulation Steps', fontsize=9)
    ax.set_title('Victim Rescue Timeline', fontsize=10, fontweight='bold')
    ax.spines[['top', 'right']].set_visible(False)

    patches = [
        mpatches.Patch(color=PALETTE['critical'], label='Critical'),
        mpatches.Patch(color=PALETTE['moderate'], label='Moderate'),
        mpatches.Patch(color=PALETTE['minor'],    label='Minor'),
    ]
    ax.legend(handles=patches, fontsize=8, loc='lower right', framealpha=0.85)

    plt.tight_layout()
    path = os.path.join(FIGURES_DIR, filename)
    plt.savefig(path, dpi=180, bbox_inches='tight')
    plt.close()
    print(f"[VIZ] Saved → {path}")
    return path


# ─────────────────────────────────────────────────────────────────────────────
# CONVENIENCE: run all visualizations at once
# ─────────────────────────────────────────────────────────────────────────────

def generate_all_figures(results: dict) -> list[str]:
    """
    Generate every figure from a simulation results dict.

    Parameters
    ----------
    results : dict returned by Controller.run_simulation()

    Returns
    -------
    List of saved file paths
    """
    saved = []
    saved.append(plot_environment_grid(results['victims'],
                                       results['grid']))
    saved.append(plot_search_comparison(results['search_results']))
    saved.append(plot_ml_metrics(results['ml_results']))
    saved.append(plot_confusion_matrices(results['ml_results']))
    saved.append(plot_fuzzy_scores(results['fuzzy']))
    saved.append(plot_kpi_dashboard(results['kpis'], results['csp']))
    saved.append(plot_rescue_timeline(results['victims']))
    return [s for s in saved if s]