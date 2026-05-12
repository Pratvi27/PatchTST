"""Cross-channel correlation heatmaps for Weather / Electricity / Traffic.

Generates one heatmap per dataset plus a 3-panel poster figure that visually
motivates the CI-vs-CD comparison: low / medium / high cross-variable
correlation aligns with PatchTST's vs iTransformer's regimes of dominance.

Usage (defaults match the PatchTST notebook's Linux paths):
    python make_correlation_heatmap.py

Override individual paths if needed:
    python make_correlation_heatmap.py \\
        --weather     /path/to/weather.csv \\
        --electricity /path/to/electricity.csv \\
        --traffic     /path/to/traffic.csv \\
        --out_dir     ./result/heatmaps
"""

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402


DEFAULT_PATHS = {
    'weather':     '/model/patchTST/all_six_datasets/weather/weather.csv',
    'electricity': '/model/patchTST/all_six_datasets/electricity/electricity.csv',
    'traffic':     '/model/patchTST/all_six_datasets/traffic/traffic.csv',
}


def load_variables(csv_path: Path) -> pd.DataFrame:
    """Load CSV and drop the first column (assumed timestamp), keep numeric only."""
    df = pd.read_csv(csv_path)
    df = df.iloc[:, 1:]
    df = df.select_dtypes(include=[np.number])
    return df


def compute_correlation(df: pd.DataFrame) -> np.ndarray:
    """M x M Pearson correlation."""
    return df.corr().to_numpy().astype(np.float32)


def correlation_stats(corr: np.ndarray) -> dict:
    """Summary stats over off-diagonal entries."""
    M = corr.shape[0]
    mask = ~np.eye(M, dtype=bool)
    off = corr[mask]
    abs_off = np.abs(off)
    return {
        'M': int(M),
        'mean_abs_off': float(abs_off.mean()),
        'median_abs_off': float(np.median(abs_off)),
        'max_off': float(off.max()),
        'min_off': float(off.min()),
        'frac_abs_above_0.5': float((abs_off > 0.5).mean()),
        'frac_abs_above_0.8': float((abs_off > 0.8).mean()),
    }


def plot_single(corr: np.ndarray, name: str, stats: dict, save_path: Path):
    M = corr.shape[0]
    fig, ax = plt.subplots(figsize=(8, 7))
    im = ax.imshow(corr, cmap='RdBu_r', vmin=-1, vmax=1, aspect='equal')

    ax.set_title(
        f'{name.capitalize()}  (M={M})\n'
        f'mean |off-diag corr| = {stats["mean_abs_off"]:.3f},  '
        f'frac |corr|>0.5 = {stats["frac_abs_above_0.5"]:.1%}',
        fontsize=13, fontweight='bold',
    )
    ax.set_xlabel('Variable index')
    ax.set_ylabel('Variable index')

    # Show tick labels only for the small case; otherwise the labels are noise.
    if M <= 30:
        ax.set_xticks(range(M))
        ax.set_yticks(range(M))
        ax.tick_params(axis='both', labelsize=8)
    else:
        ax.set_xticks([])
        ax.set_yticks([])

    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label('Pearson correlation', fontsize=11)

    fig.tight_layout()
    fig.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"  → {save_path}")


def plot_three_panels(corrs: dict, stats: dict, save_path: Path):
    """3-panel poster figure with a shared colorbar."""
    n = len(corrs)
    fig, axes = plt.subplots(1, n, figsize=(6.5 * n, 6))
    if n == 1:
        axes = [axes]

    im = None
    for ax, (name, corr) in zip(axes, corrs.items()):
        M = corr.shape[0]
        im = ax.imshow(corr, cmap='RdBu_r', vmin=-1, vmax=1, aspect='equal')
        s = stats[name]
        ax.set_title(
            f'{name.capitalize()}  (M={M})\n'
            f'mean |corr| = {s["mean_abs_off"]:.3f}   '
            f'|corr|>0.5: {s["frac_abs_above_0.5"]:.1%}',
            fontsize=14, fontweight='bold',
        )
        ax.set_xticks([])
        ax.set_yticks([])

    # One shared colorbar on the right.
    fig.subplots_adjust(right=0.92, wspace=0.08)
    cbar_ax = fig.add_axes([0.935, 0.22, 0.012, 0.55])
    cbar = fig.colorbar(im, cax=cbar_ax)
    cbar.set_label('Pearson correlation', fontsize=12)

    fig.suptitle(
        'Cross-variable correlation structure across datasets',
        fontsize=16, fontweight='bold', y=1.02,
    )
    fig.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"  → {save_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--weather',     type=str, default=DEFAULT_PATHS['weather'])
    parser.add_argument('--electricity', type=str, default=DEFAULT_PATHS['electricity'])
    parser.add_argument('--traffic',     type=str, default=DEFAULT_PATHS['traffic'])
    parser.add_argument('--out_dir',     type=str, default='./result/heatmaps')
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    csv_paths = {
        'weather':     Path(args.weather),
        'electricity': Path(args.electricity),
        'traffic':     Path(args.traffic),
    }

    corrs = {}
    stats = {}

    for name, path in csv_paths.items():
        if not path.exists():
            print(f"[WARN] {name}: csv not found at {path} — skipping")
            continue
        print(f"\n[{name}] loading {path}")
        df = load_variables(path)
        print(f"  shape after dropping timestamp: T={df.shape[0]}, M={df.shape[1]}")

        corr = compute_correlation(df)
        s = correlation_stats(corr)
        corrs[name] = corr
        stats[name] = s

        print(f"  stats: M={s['M']}, mean|corr|={s['mean_abs_off']:.3f}, "
              f"frac|corr|>0.5={s['frac_abs_above_0.5']:.1%}, "
              f"frac|corr|>0.8={s['frac_abs_above_0.8']:.1%}")

        plot_single(corr, name, s, save_path=out_dir / f'{name}_correlation.png')

    if len(corrs) >= 2:
        plot_three_panels(corrs, stats, save_path=out_dir / 'correlation_panels.png')
    else:
        print(f"\nOnly {len(corrs)} dataset(s) loaded; skipping panels figure.")

    stats_path = out_dir / 'stats.json'
    with open(stats_path, 'w') as f:
        json.dump(stats, f, indent=2)
    print(f"\nstats: {stats_path}")
    print(json.dumps(stats, indent=2))

    print(f"\nDone. All artifacts in: {out_dir}")


if __name__ == '__main__':
    main()
