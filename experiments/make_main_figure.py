"""Generate the headline figure for the poster:
5 models × 3 datasets × 4 horizons grouped bar chart.

All numbers are hard-coded from the final experiments (data as of poster
deadline). Running this script produces:
    result/poster_figures/main_results_5models.png   (300 DPI)
    result/poster_figures/main_results_5models.pdf   (vector)

Usage:
    python make_main_figure.py
"""

from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


# ─── Data: 5 models × 3 datasets × 4 horizons (Test MSE) ────────────────
HORIZONS = [96, 192, 336, 720]

DATA = {
    'Weather':     {'M': 21,
                    'PatchTST/64':  [0.146, 0.189, 0.242, 0.317],
                    'PatchTST/42':  [0.155, 0.196, 0.250, 0.318],
                    'iTransformer': [0.176, 0.225, 0.281, 0.358],
                    'DLinear-I':    [0.156, 0.200, 0.251, 0.326],
                    'DLinear':      [0.180, 0.220, 0.265, 0.329]},
    'Electricity': {'M': 321,
                    'PatchTST/64':  [0.152, 0.167, 0.180, 0.224],
                    'PatchTST/42':  [0.133, 0.149, 0.166, 0.204],
                    'iTransformer': [0.148, 0.166, 0.179, 0.209],
                    'DLinear-I':    [0.140, 0.154, 0.167, 0.212],
                    'DLinear':      [0.144, 0.156, 0.168, 0.205]},
    'Traffic':     {'M': 862,
                    'PatchTST/64':  [0.396, 0.412, 0.416, 0.447],
                    'PatchTST/42':  [0.388, 0.404, 0.415, 0.444],
                    'iTransformer': [0.393, 0.412, 0.425, 0.459],
                    'DLinear-I':    [0.504, 0.520, 0.535, 0.604],
                    'DLinear':      [0.432, 0.451, 0.465, 0.489]},
}

MODELS = ['PatchTST/64', 'PatchTST/42', 'iTransformer', 'DLinear-I', 'DLinear']

# Color scheme:
#   blues        = PatchTST family (winners)
#   warm red     = iTransformer (the antagonist)
#   greens       = DLinear baselines (linear models)
COLORS = {
    'PatchTST/64':  '#1B3A5C',   # deep navy
    'PatchTST/42':  '#2E86AB',   # medium blue
    'iTransformer': '#E63946',   # warm red
    'DLinear-I':    '#386641',   # dark green
    'DLinear':      '#A7C957',   # light green
}


def plot_grouped_bars(ax, dataset_name, dataset_data, show_legend=False):
    """One panel: 4 horizons × 5 model bars."""
    x = np.arange(len(HORIZONS))
    width = 0.16   # 5 bars per group → group width = 0.80

    bar_handles = []
    for i, model in enumerate(MODELS):
        values = dataset_data[model]
        offset = (i - 2) * width   # center the 5-bar group on the tick
        bars = ax.bar(x + offset, values, width,
                      label=model, color=COLORS[model],
                      edgecolor='white', linewidth=0.5)
        bar_handles.append(bars[0])

    # ★ on the winner of each horizon
    for j in range(len(HORIZONS)):
        col_values = [dataset_data[m][j] for m in MODELS]
        winner_idx = int(np.argmin(col_values))
        winner_value = col_values[winner_idx]
        offset = (winner_idx - 2) * width
        ax.scatter([j + offset], [winner_value * 1.04],
                   marker='*', s=170, color='gold',
                   edgecolor='black', linewidth=0.7, zorder=5)

    ax.set_xticks(x)
    ax.set_xticklabels([f'h={h}' for h in HORIZONS], fontsize=11)
    ax.set_ylabel('Test MSE', fontsize=12)
    ax.set_title(f'{dataset_name}  (M = {dataset_data["M"]})',
                 fontsize=14, fontweight='bold', pad=10)
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    ax.set_axisbelow(True)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    # add a touch of headroom for the gold stars
    ymin, ymax = ax.get_ylim()
    ax.set_ylim(ymin, ymax * 1.05)

    return bar_handles


def main():
    out_dir = Path('result/poster_figures')
    out_dir.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(1, 3, figsize=(20, 6.5))
    fig.subplots_adjust(top=0.82, bottom=0.12, left=0.05, right=0.98, wspace=0.20)

    handles = None
    for ax, (dataset_name, dataset_data) in zip(axes, DATA.items()):
        h = plot_grouped_bars(ax, dataset_name, dataset_data)
        if handles is None:
            handles = h

    # Shared legend on top (outside panels).
    fig.legend(handles, MODELS,
               loc='upper center', ncol=5,
               bbox_to_anchor=(0.5, 0.96),
               fontsize=12, frameon=False,
               handletextpad=0.5, columnspacing=1.6)

    # Figure-level title (above legend).
    fig.suptitle(
        'PatchTST achieves the lowest MSE on 12/12 (dataset, horizon) cells '
        '— under matched 20-epoch compute',
        fontsize=15, fontweight='bold', y=0.995,
    )

    # Subtitle / caption row at bottom (outside the panels).
    fig.text(0.5, 0.02,
             '★ marks the best (lowest MSE) in each horizon group.  '
             'Datasets ordered by variable count M.',
             ha='center', fontsize=11, style='italic', color='#444')

    fig.savefig(out_dir / 'main_results_5models.png',
                dpi=300, bbox_inches='tight')
    fig.savefig(out_dir / 'main_results_5models.pdf',
                bbox_inches='tight')
    plt.close(fig)
    print(f"Saved → {out_dir / 'main_results_5models.png'}")
    print(f"Saved → {out_dir / 'main_results_5models.pdf'}")

    # --- Bonus: aggregate per-dataset bar chart (mean across horizons) -------
    fig2, ax2 = plt.subplots(figsize=(10, 5.5))
    datasets = list(DATA.keys())
    x2 = np.arange(len(datasets))
    width2 = 0.16

    for i, model in enumerate(MODELS):
        avgs = [np.mean(DATA[d][model]) for d in datasets]
        offset = (i - 2) * width2
        ax2.bar(x2 + offset, avgs, width2,
                label=model, color=COLORS[model],
                edgecolor='white', linewidth=0.5)

    # Mark winner per dataset (lowest avg MSE)
    for j, d in enumerate(datasets):
        col = [np.mean(DATA[d][m]) for m in MODELS]
        winner_idx = int(np.argmin(col))
        offset = (winner_idx - 2) * width2
        ax2.scatter([j + offset], [col[winner_idx] * 1.03],
                    marker='*', s=200, color='gold',
                    edgecolor='black', linewidth=0.7, zorder=5)

    ax2.set_xticks(x2)
    ax2.set_xticklabels([f'{d}\n(M={DATA[d]["M"]})' for d in datasets], fontsize=12)
    ax2.set_ylabel('Mean Test MSE  (averaged over horizons)', fontsize=12)
    ax2.set_title('Aggregate Performance per Dataset',
                  fontsize=14, fontweight='bold')
    ax2.legend(loc='upper left', fontsize=11, frameon=False, ncol=2)
    ax2.grid(axis='y', alpha=0.3, linestyle='--')
    ax2.set_axisbelow(True)
    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)

    fig2.tight_layout()
    fig2.savefig(out_dir / 'main_results_aggregate.png',
                 dpi=300, bbox_inches='tight')
    fig2.savefig(out_dir / 'main_results_aggregate.pdf',
                 bbox_inches='tight')
    plt.close(fig2)
    print(f"Saved → {out_dir / 'main_results_aggregate.png'}")
    print(f"Saved → {out_dir / 'main_results_aggregate.pdf'}")


if __name__ == '__main__':
    main()
