"""Reproduction quality figure for the poster.

Compares our reproduction vs paper-published numbers, computing
relative deviation Δ = (ours − paper) / paper for each (model,
dataset, horizon) cell.

Outputs:
    result/poster_figures/reproduction_heatmap.{png,pdf}
        Color-coded heatmap of Δ% per cell
    result/poster_figures/reproduction_summary.{png,pdf}
        Aggregate mean |Δ|% per model (bar chart)
    result/poster_figures/reproduction_table.png
        Table-style figure with numeric Δ% per cell

Usage:
    python make_reproduction_figure.py
"""

from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


# ─── Numbers ────────────────────────────────────────────────────────────
HORIZONS = [96, 192, 336, 720]
DATASETS = ['Weather', 'Electricity', 'Traffic']

# Our reproduction (Test MSE)
OURS = {
    'PatchTST/64': {
        'Weather':     [0.146, 0.189, 0.242, 0.317],
        'Electricity': [0.152, 0.167, 0.180, 0.224],
        'Traffic':     [0.396, 0.412, 0.416, 0.447],
    },
    'PatchTST/42': {
        'Weather':     [0.155, 0.196, 0.250, 0.318],
        'Electricity': [0.133, 0.149, 0.166, 0.204],
        'Traffic':     [0.388, 0.404, 0.415, 0.444],
    },
    'iTransformer': {
        'Weather':     [0.176, 0.225, 0.281, 0.358],
        'Electricity': [0.148, 0.166, 0.179, 0.209],
        'Traffic':     [0.393, 0.412, 0.425, 0.459],
    },
    'DLinear-I': {
        'Weather':     [0.156, 0.200, 0.251, 0.326],
        'Electricity': [0.140, 0.154, 0.167, 0.212],
        'Traffic':     [0.504, 0.520, 0.535, 0.604],
    },
    'DLinear': {
        'Weather':     [0.180, 0.220, 0.265, 0.329],
        'Electricity': [0.144, 0.156, 0.168, 0.205],
        'Traffic':     [0.432, 0.451, 0.465, 0.489],
    },
}

# Published / expected numbers from respective papers
PAPER = {
    'PatchTST/64': {
        'Weather':     [0.149, 0.194, 0.245, 0.314],
        'Electricity': [0.129, 0.147, 0.163, 0.197],
        'Traffic':     [0.360, 0.379, 0.392, 0.432],
    },
    'PatchTST/42': {
        'Weather':     [0.152, 0.197, 0.249, 0.320],
        'Electricity': [0.130, 0.148, 0.167, 0.202],
        'Traffic':     [0.367, 0.385, 0.398, 0.434],
    },
    'iTransformer': {
        'Weather':     [0.174, 0.221, 0.278, 0.358],
        'Electricity': [0.148, 0.162, 0.178, 0.225],
        'Traffic':     [0.395, 0.417, 0.433, 0.467],
    },
    'DLinear-I': {
        'Weather':     [0.176, 0.220, 0.265, 0.323],
        'Electricity': [0.140, 0.153, 0.169, 0.203],
        'Traffic':     [0.410, 0.423, 0.436, 0.466],
    },
    'DLinear': {
        'Weather':     [0.176, 0.220, 0.265, 0.323],
        'Electricity': [0.140, 0.153, 0.169, 0.203],
        'Traffic':     [0.410, 0.423, 0.436, 0.466],
    },
}

MODELS = ['PatchTST/64', 'PatchTST/42', 'iTransformer', 'DLinear-I', 'DLinear']


def compute_delta_matrix():
    """Returns matrix of Δ% with shape (n_models, 12)."""
    delta = np.zeros((len(MODELS), 12))
    for i, model in enumerate(MODELS):
        col = 0
        for ds in DATASETS:
            for j, h in enumerate(HORIZONS):
                ours = OURS[model][ds][j]
                paper = PAPER[model][ds][j]
                delta[i, col] = 100 * (ours - paper) / paper
                col += 1
    return delta


# ─── Heatmap ────────────────────────────────────────────────────────────
def plot_heatmap(out_dir: Path):
    delta = compute_delta_matrix()

    fig, ax = plt.subplots(figsize=(16, 5))

    # Diverging colormap centered at 0; clip range to ±20% for readability
    vmax = 20
    im = ax.imshow(delta, cmap='RdYlGn_r', vmin=-vmax, vmax=vmax,
                   aspect='auto')

    # Cell text annotations
    for i in range(delta.shape[0]):
        for j in range(delta.shape[1]):
            v = delta[i, j]
            color = 'white' if abs(v) > 12 else 'black'
            ax.text(j, i, f'{v:+.1f}%',
                    ha='center', va='center', fontsize=10,
                    color=color, fontweight='bold')

    # Y axis: model names
    ax.set_yticks(range(len(MODELS)))
    ax.set_yticklabels(MODELS, fontsize=11)

    # X axis: 12 (dataset, horizon) labels
    x_labels = []
    for ds in DATASETS:
        for h in HORIZONS:
            x_labels.append(f'h={h}')
    ax.set_xticks(range(12))
    ax.set_xticklabels(x_labels, fontsize=10)

    # Vertical separators between datasets
    for x in [3.5, 7.5]:
        ax.axvline(x, color='black', linewidth=2)

    # Dataset labels above
    for i, ds in enumerate(DATASETS):
        center_x = i * 4 + 1.5
        ax.text(center_x, -0.95, ds, ha='center', va='center',
                fontsize=13, fontweight='bold',
                transform=ax.transData)

    ax.set_xlim(-0.5, 11.5)

    # Colorbar
    cbar = fig.colorbar(im, ax=ax, fraction=0.025, pad=0.02)
    cbar.set_label('Δ% (ours vs paper)', fontsize=11)
    cbar.ax.tick_params(labelsize=10)

    ax.set_title(
        'Reproduction Quality — Relative deviation from published numbers\n'
        '(green = within paper range; red = above paper; '
        '|Δ|>20% clamped to red)',
        fontsize=13, fontweight='bold', pad=30,
    )

    fig.tight_layout()
    fig.savefig(out_dir / 'reproduction_heatmap.png', dpi=300,
                bbox_inches='tight')
    fig.savefig(out_dir / 'reproduction_heatmap.pdf',
                bbox_inches='tight')
    plt.close(fig)
    print(f"Saved → {out_dir / 'reproduction_heatmap.png'}")


# ─── Aggregate bar chart ────────────────────────────────────────────────
def plot_summary(out_dir: Path):
    """Mean |Δ|% per (model, dataset)."""
    fig, ax = plt.subplots(figsize=(11, 5.5))

    model_colors = {
        'PatchTST/64':  '#1B3A5C',
        'PatchTST/42':  '#2E86AB',
        'iTransformer': '#E63946',
        'DLinear-I':    '#386641',
        'DLinear':      '#A7C957',
    }

    n_datasets = len(DATASETS)
    n_models = len(MODELS)
    width = 0.16
    x = np.arange(n_datasets)

    for i, model in enumerate(MODELS):
        means = []
        for ds in DATASETS:
            ours = np.array(OURS[model][ds])
            paper = np.array(PAPER[model][ds])
            mean_abs_delta = 100 * np.mean(np.abs((ours - paper) / paper))
            means.append(mean_abs_delta)
        offset = (i - 2) * width
        bars = ax.bar(x + offset, means, width, label=model,
                      color=model_colors[model], edgecolor='white',
                      linewidth=0.5)
        for b, v in zip(bars, means):
            ax.text(b.get_x() + b.get_width() / 2, v + 0.4,
                    f'{v:.1f}', ha='center', fontsize=9, color='black')

    ax.set_xticks(x)
    ax.set_xticklabels(DATASETS, fontsize=12, fontweight='bold')
    ax.set_ylabel('Mean |Δ|%  (lower = closer to paper)', fontsize=12)
    ax.set_title('Reproduction Quality: Mean Absolute Deviation per Dataset',
                 fontsize=13, fontweight='bold')
    ax.legend(fontsize=10, frameon=False, ncol=5,
              loc='upper left', bbox_to_anchor=(0.0, 1.0))
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    ax.set_axisbelow(True)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    # Reference line at 5% (typical "good reproduction")
    ax.axhline(5, color='#FF6B35', linestyle='--', alpha=0.6, linewidth=1)
    ax.text(2.55, 5.3, '5% (good reproduction threshold)',
            fontsize=9, color='#FF6B35', style='italic')

    fig.tight_layout()
    fig.savefig(out_dir / 'reproduction_summary.png', dpi=300,
                bbox_inches='tight')
    fig.savefig(out_dir / 'reproduction_summary.pdf',
                bbox_inches='tight')
    plt.close(fig)
    print(f"Saved → {out_dir / 'reproduction_summary.png'}")


# ─── Compact text table (for poster) ────────────────────────────────────
def plot_text_table(out_dir: Path):
    """A compact table figure listing per-model mean |Δ| per dataset."""
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.axis('off')

    # Build cell text
    rows = []
    for model in MODELS:
        row_cells = [model]
        for ds in DATASETS:
            ours = np.array(OURS[model][ds])
            paper = np.array(PAPER[model][ds])
            mean_abs_delta = 100 * np.mean(np.abs((ours - paper) / paper))
            row_cells.append(f'{mean_abs_delta:.1f}%')
        # Overall mean
        all_d = []
        for ds in DATASETS:
            ours = np.array(OURS[model][ds])
            paper = np.array(PAPER[model][ds])
            all_d.extend(np.abs((ours - paper) / paper).tolist())
        row_cells.append(f'{100 * np.mean(all_d):.1f}%')
        rows.append(row_cells)

    columns = ['Model', 'Weather', 'Electricity', 'Traffic', 'Overall']
    table = ax.table(cellText=rows, colLabels=columns,
                     cellLoc='center', loc='center')

    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1, 1.8)

    # Highlight cells: green if mean |Δ| < 5%, yellow 5-10%, red >10%
    for i, row in enumerate(rows, start=1):
        for j, cell in enumerate(row[1:], start=1):
            v = float(cell.rstrip('%'))
            if v < 5:
                color = '#A8D5A2'   # green
            elif v < 10:
                color = '#FFE5A0'   # yellow
            else:
                color = '#F4A8A8'   # red
            table[(i, j)].set_facecolor(color)

    # Bold header
    for j in range(len(columns)):
        table[(0, j)].set_text_props(weight='bold')
        table[(0, j)].set_facecolor('#D9D9D9')
    # Bold model names
    for i in range(1, len(rows) + 1):
        table[(i, 0)].set_text_props(weight='bold')

    fig.suptitle(
        'Reproduction Faithfulness — Mean |Δ%| vs Published Numbers',
        fontsize=13, fontweight='bold', y=0.95,
    )
    fig.text(0.5, 0.04,
             'Green: < 5% (good).  Yellow: 5–10% (acceptable).  Red: > 10% (undertrained).',
             ha='center', fontsize=10, style='italic', color='#444')

    fig.savefig(out_dir / 'reproduction_table.png', dpi=300,
                bbox_inches='tight')
    fig.savefig(out_dir / 'reproduction_table.pdf',
                bbox_inches='tight')
    plt.close(fig)
    print(f"Saved → {out_dir / 'reproduction_table.png'}")


def main():
    out_dir = Path('result/poster_figures')
    out_dir.mkdir(parents=True, exist_ok=True)

    plot_heatmap(out_dir)
    plot_summary(out_dir)
    plot_text_table(out_dir)

    # Print summary to stdout
    print('\n' + '=' * 65)
    print('Reproduction Quality Summary (mean |Δ|% across all horizons)')
    print('=' * 65)
    print(f'{"Model":<15} {"Weather":>10} {"Electricity":>14} {"Traffic":>10} {"Overall":>10}')
    print('-' * 65)
    for model in MODELS:
        per_ds = []
        all_d = []
        for ds in DATASETS:
            ours = np.array(OURS[model][ds])
            paper = np.array(PAPER[model][ds])
            d = 100 * np.mean(np.abs((ours - paper) / paper))
            per_ds.append(d)
            all_d.extend(np.abs((ours - paper) / paper).tolist())
        overall = 100 * np.mean(all_d)
        print(f'{model:<15} {per_ds[0]:>9.1f}% {per_ds[1]:>13.1f}% '
              f'{per_ds[2]:>9.1f}% {overall:>9.1f}%')


if __name__ == '__main__':
    main()
