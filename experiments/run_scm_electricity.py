"""PatchTST + Stochastic/Latent Channel Mixing (SCM) on Electricity.

Paper-aligned hyperparameters (PatchTST/42 — sl=336, d=128, nhead=16, dim_ff=256,
dropout=0.2, lr=1e-4, batch=16, epochs=20) combined with sliding-window step=4
to make an 11-point alpha sweep tractable within compute budget.

The mixer is a deterministic-interpolation residual:
    z_out = z + alpha * (mixer(z) - z)
This decouples the architectural question (how much mixing helps) from training
dynamics (mixer convergence): the mixer receives gradient signal at every batch
regardless of alpha, instead of only on the alpha-fraction of batches as in the
stochastic-gating variant.

Outputs (all under result_dir/scm_electricity_<TS>/):
    summary.json                  aggregate alpha -> {test_mse, test_mae}
    alpha_sweep_mse.png           MSE vs alpha curve
    alpha_sweep_mae.png           MAE vs alpha curve
    config.json                   exact CFG used
    run.log                       full stdout + stderr tee
    alpha_<X.XX>/results.json     per-alpha test_mse, test_mae, history
    alpha_<X.XX>/training.png     per-alpha train/valid loss curve
    alpha_<X.XX>/best_model.pt    per-alpha best checkpoint

Linux usage:
    python run_scm_electricity.py \
        --csv_path /model/patchTST/all_six_datasets/electricity/electricity.csv \
        --alphas "0.0 0.25 0.5 0.75 1.0" \
        --epochs 20

Override examples:
    python run_scm_electricity.py --alphas "0.0 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1.0"
    python run_scm_electricity.py --forecast_len 192 --batch_size 8
    python run_scm_electricity.py --device cuda:1
"""

import argparse
import json
import logging
import math
import os
import random
import sys
from datetime import datetime
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import torch  # noqa: E402
import torch.nn as nn  # noqa: E402
import torch.nn.functional as F  # noqa: E402
from torch.utils.data import DataLoader, TensorDataset  # noqa: E402


# ─── PatchTST/42 + SCM config (aligned with paper, step=4 for speed) ────────
DEFAULT_CFG = {
    # Time structure (PatchTST/42)
    'seq_len': 336,
    'forecast_len': 96,
    'patch_size': 16,
    'stride': 8,

    # Architecture (paper-aligned)
    'd_model': 128,
    'nhead': 16,
    'num_layers': 3,
    'dim_feedforward': 256,
    'dropout': 0.2,

    # Training (paper-aligned, with one deviation)
    'batch_size': 16,
    'epochs': 20,
    'patience': 3,
    'learning_rate': 1e-4,

    # Data
    'train_ratio': 0.7,
    'valid_ratio': 0.1,
    'step': 4,                 # ← deviation: step=1 in paper, step=4 here for speed

    'seed': 2021,
}


# ─── Argparse ────────────────────────────────────────────────────────────
def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--csv_path', type=str,
                   default='/model/patchTST/all_six_datasets/electricity/electricity.csv',
                   help='Electricity CSV path (timestamp first column will be dropped)')
    p.add_argument('--result_dir', type=str, default='./result',
                   help='Parent dir under which scm_electricity_<TS>/ is created')
    p.add_argument('--alphas', type=str, default='0.0 0.25 0.5 0.75 1.0',
                   help='Space-separated alpha values to sweep')
    p.add_argument('--forecast_len', type=int, default=DEFAULT_CFG['forecast_len'])
    p.add_argument('--seq_len', type=int, default=DEFAULT_CFG['seq_len'])
    p.add_argument('--epochs', type=int, default=DEFAULT_CFG['epochs'])
    p.add_argument('--batch_size', type=int, default=DEFAULT_CFG['batch_size'])
    p.add_argument('--learning_rate', type=float, default=DEFAULT_CFG['learning_rate'])
    p.add_argument('--step', type=int, default=DEFAULT_CFG['step'])
    p.add_argument('--patience', type=int, default=DEFAULT_CFG['patience'])
    p.add_argument('--seed', type=int, default=DEFAULT_CFG['seed'])
    p.add_argument('--device', type=str, default=None,
                   help='Device override: cpu / cuda / cuda:0')
    p.add_argument('--data_fraction', type=float, default=1.0,
                   help='Fraction of total timesteps to use (1.0 = all)')
    return p.parse_args()


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def setup_logging(result_dir: Path):
    log_path = result_dir / 'run.log'
    fmt = '%(asctime)s | %(levelname)s | %(message)s'
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    for h in list(logger.handlers):
        logger.removeHandler(h)
    fh = logging.FileHandler(log_path, mode='w', encoding='utf-8')
    fh.setFormatter(logging.Formatter(fmt))
    logger.addHandler(fh)
    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(logging.Formatter(fmt))
    logger.addHandler(sh)


# ─── Model components ───────────────────────────────────────────────────
class RevIN(nn.Module):
    def __init__(self, eps=1e-5):
        super().__init__()
        self.eps = eps
        self.affine_weight = nn.Parameter(torch.ones(1))
        self.affine_bias = nn.Parameter(torch.zeros(1))

    def normalize(self, x):
        self.mean = x.mean(dim=-1, keepdim=True)
        self.std = x.std(dim=-1, keepdim=True) + self.eps
        x = (x - self.mean) / self.std
        x = x * self.affine_weight + self.affine_bias
        return x

    def denormalize(self, x):
        x = (x - self.affine_bias) / (self.affine_weight + self.eps)
        x = x * self.std + self.mean
        return x


class PatchEmbedding(nn.Module):
    def __init__(self, patch_size, d_model, num_patches):
        super().__init__()
        self.Wp = nn.Linear(patch_size, d_model, bias=False)
        self.Wpos = nn.Parameter(torch.randn(1, num_patches, d_model) * 0.02)

    def forward(self, x_p):
        return self.Wp(x_p) + self.Wpos


class BatchNormTransformerLayer(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward, dropout):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        self.ff1 = nn.Linear(d_model, dim_feedforward)
        self.ff2 = nn.Linear(dim_feedforward, d_model)
        self.dropout = nn.Dropout(dropout)
        self.bn1 = nn.BatchNorm1d(d_model)
        self.bn2 = nn.BatchNorm1d(d_model)

    def forward(self, x):
        attn_out, _ = self.self_attn(x, x, x)
        x = x + self.dropout(attn_out)
        x = self.bn1(x.transpose(1, 2)).transpose(1, 2)
        ff_out = self.ff2(self.dropout(F.gelu(self.ff1(x))))
        x = x + self.dropout(ff_out)
        x = self.bn2(x.transpose(1, 2)).transpose(1, 2)
        return x


class PatchTST_CI(nn.Module):
    def __init__(self, seq_len, forecast_len, patch_size, stride,
                 d_model, nhead, num_layers, dim_feedforward, dropout):
        super().__init__()
        self.seq_len = seq_len
        self.forecast_len = forecast_len
        self.patch_size = patch_size
        self.stride = stride
        self.d_model = d_model
        self.num_patches = math.floor((seq_len - patch_size) / stride) + 2

        self.revin = RevIN()
        self.patch_embed = PatchEmbedding(patch_size, d_model, self.num_patches)
        self.encoder_layers = nn.ModuleList([
            BatchNormTransformerLayer(d_model, nhead, dim_feedforward, dropout)
            for _ in range(num_layers)
        ])
        self.flatten = nn.Flatten(start_dim=1)
        self.linear_head = nn.Linear(self.num_patches * d_model, forecast_len)

    def _make_patches(self, x):
        x = F.pad(x, (0, self.stride), mode='replicate')
        return x.unfold(dimension=-1, size=self.patch_size, step=self.stride)

    def encode(self, x):
        B, L, M = x.shape
        x_uni = x.permute(0, 2, 1).reshape(B * M, L)
        x_uni = self.revin.normalize(x_uni)
        z = self._make_patches(x_uni)
        z = self.patch_embed(z)
        for layer in self.encoder_layers:
            z = layer(z)
        return z, B, M

    def decode(self, z, B, M):
        out = self.linear_head(self.flatten(z))
        out = self.revin.denormalize(out)
        return out.reshape(B, M, self.forecast_len)

    def forward(self, x):
        z, B, M = self.encode(x)
        return self.decode(z, B, M)


class CrossChannelMixingLayer(nn.Module):
    def __init__(self, d_model, nhead, dropout):
        super().__init__()
        self.attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, z, B, M):
        BM, P, D = z.shape
        z_chan = z.reshape(B, M, P, D).permute(0, 2, 1, 3).reshape(B * P, M, D)
        attn_out, _ = self.attn(z_chan, z_chan, z_chan, need_weights=False)
        z_chan = self.norm(z_chan + self.dropout(attn_out))
        z_out = z_chan.reshape(B, P, M, D).permute(0, 2, 1, 3).reshape(B * M, P, D)
        return z_out


class PatchTST_SCM(nn.Module):
    """PatchTST with deterministic-interpolation cross-channel mixing.

    z_out = z + alpha * (mixer(z) - z)

    alpha=0 → pure CI baseline (mixer output not propagated; mixer params receive
              zero gradient, matching pure PatchTST in expectation).
    alpha=1 → mixer fully replaces z (always-on cross-channel attention).
    """
    def __init__(self, seq_len, forecast_len, patch_size, stride,
                 d_model, nhead, num_layers, dim_feedforward, dropout,
                 mix_alpha=0.0, mix_heads=None):
        super().__init__()
        self.mix_alpha = float(mix_alpha)
        self.backbone = PatchTST_CI(
            seq_len=seq_len, forecast_len=forecast_len,
            patch_size=patch_size, stride=stride,
            d_model=d_model, nhead=nhead, num_layers=num_layers,
            dim_feedforward=dim_feedforward, dropout=dropout,
        )
        self.cross_mixer = CrossChannelMixingLayer(
            d_model=d_model, nhead=mix_heads or nhead, dropout=dropout,
        )

    def forward(self, x):
        z, B, M = self.backbone.encode(x)
        z_mix = self.cross_mixer(z, B, M)
        z_out = z + self.mix_alpha * (z_mix - z)
        return self.backbone.decode(z_out, B, M)


def count_parameters(model):
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable


# ─── Data ────────────────────────────────────────────────────────────────
def load_timeseries_csv(csv_path: Path) -> pd.DataFrame:
    if not csv_path.exists():
        raise FileNotFoundError(f'CSV not found at {csv_path}')
    df = pd.read_csv(csv_path)
    logging.info('Raw shape: %s', df.shape)
    numeric_df = df.select_dtypes(include=[np.number]).copy()
    if numeric_df.shape[1] == 0:
        raise ValueError('No numeric columns found in CSV.')
    numeric_df = numeric_df.replace([np.inf, -np.inf], np.nan).ffill().bfill()
    logging.info('Numeric feature shape: %s', numeric_df.shape)
    return numeric_df.astype(np.float32)


def create_dataloaders(data, seq_len, forecast_len, batch_size, train_ratio,
                       valid_ratio, step):
    T_total, M = data.shape
    train_end = int(train_ratio * T_total)
    valid_end = int((train_ratio + valid_ratio) * T_total)

    train_raw = data[:train_end]
    mean = train_raw.mean(axis=0, keepdims=True)
    std = train_raw.std(axis=0, keepdims=True) + 1e-8

    train_data = (data[:train_end] - mean) / std
    valid_data = (data[train_end:valid_end] - mean) / std
    test_data = (data[valid_end:] - mean) / std

    def sliding_window(d):
        X, y = [], []
        total = seq_len + forecast_len
        for i in range(0, len(d) - total + 1, step):
            X.append(d[i:i + seq_len])
            y.append(d[i + seq_len:i + total])
        if not X:
            raise ValueError('Not enough data for chosen seq_len + forecast_len.')
        return (torch.tensor(np.array(X), dtype=torch.float32),
                torch.tensor(np.array(y), dtype=torch.float32))

    X_train, y_train = sliding_window(train_data)
    X_valid, y_valid = sliding_window(valid_data)
    X_test, y_test = sliding_window(test_data)

    logging.info('Splits: train=%d/%d, valid=%d/%d, test=%d/%d (samples / timesteps)',
                 len(X_train), train_end,
                 len(X_valid), valid_end - train_end,
                 len(X_test), T_total - valid_end)
    logging.info('Window shapes: train=%s, valid=%s, test=%s',
                 tuple(X_train.shape), tuple(X_valid.shape), tuple(X_test.shape))

    return (
        DataLoader(TensorDataset(X_train, y_train), batch_size=batch_size, shuffle=True),
        DataLoader(TensorDataset(X_valid, y_valid), batch_size=batch_size, shuffle=False),
        DataLoader(TensorDataset(X_test, y_test), batch_size=batch_size, shuffle=False),
        M,
    )


# ─── Train / eval ────────────────────────────────────────────────────────
def evaluate(model, loader, device):
    """Return (mse, mae) on the loader."""
    model.eval()
    mse_total, mae_total, n_samples = 0.0, 0.0, 0
    with torch.no_grad():
        for xb, yb in loader:
            xb = xb.to(device)
            yb = yb.permute(0, 2, 1).to(device)
            pred = model(xb)
            mse_total += F.mse_loss(pred, yb, reduction='sum').item()
            mae_total += (pred - yb).abs().sum().item()
            n_samples += yb.numel()
    return mse_total / n_samples, mae_total / n_samples


def train_one_alpha(alpha, train_loader, valid_loader, test_loader,
                    cfg, device, save_dir: Path):
    logging.info('=' * 60)
    logging.info('Training PatchTST_SCM with mix_alpha = %.2f', alpha)
    logging.info('=' * 60)

    model = PatchTST_SCM(
        seq_len=cfg['seq_len'],
        forecast_len=cfg['forecast_len'],
        patch_size=cfg['patch_size'],
        stride=cfg['stride'],
        d_model=cfg['d_model'],
        nhead=cfg['nhead'],
        num_layers=cfg['num_layers'],
        dim_feedforward=cfg['dim_feedforward'],
        dropout=cfg['dropout'],
        mix_alpha=alpha,
    ).to(device)
    total, trainable = count_parameters(model)
    logging.info('Parameters: total=%d, trainable=%d', total, trainable)

    optimizer = torch.optim.Adam(model.parameters(), lr=cfg['learning_rate'])
    criterion = nn.MSELoss()
    history = {'train_loss': [], 'valid_loss': []}

    best_valid = float('inf')
    best_state = None
    bad_epochs = 0

    for epoch in range(1, cfg['epochs'] + 1):
        model.train()
        running, n_batches = 0.0, 0
        for xb, yb in train_loader:
            xb = xb.to(device)
            yb = yb.permute(0, 2, 1).to(device)
            optimizer.zero_grad(set_to_none=True)
            pred = model(xb)
            loss = criterion(pred, yb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            running += loss.item()
            n_batches += 1

        train_loss = running / max(n_batches, 1)
        valid_mse, _ = evaluate(model, valid_loader, device)
        history['train_loss'].append(train_loss)
        history['valid_loss'].append(valid_mse)
        logging.info('  epoch %02d: train=%.6f valid=%.6f', epoch, train_loss, valid_mse)

        if valid_mse < best_valid - 1e-7:
            best_valid = valid_mse
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            bad_epochs = 0
        else:
            bad_epochs += 1
            if bad_epochs >= cfg['patience']:
                logging.info('  early stopping at epoch %d (best valid=%.6f)', epoch, best_valid)
                break

    if best_state is not None:
        model.load_state_dict(best_state)

    test_mse, test_mae = evaluate(model, test_loader, device)
    logging.info('  final test_mse=%.6f, test_mae=%.6f', test_mse, test_mae)

    # Save artifacts.
    save_dir.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), save_dir / 'best_model.pt')
    with open(save_dir / 'results.json', 'w') as f:
        json.dump({
            'mix_alpha': alpha,
            'test_mse': test_mse,
            'test_mae': test_mae,
            'best_valid_mse': best_valid,
            'history': history,
            'parameters_total': total,
            'parameters_trainable': trainable,
        }, f, indent=2)
    plot_training_history(history, save_dir / 'training.png', alpha=alpha)

    return {'alpha': alpha, 'test_mse': test_mse, 'test_mae': test_mae,
            'best_valid_mse': best_valid, 'history': history}


# ─── Plots ───────────────────────────────────────────────────────────────
def plot_training_history(history, save_path: Path, alpha: float):
    fig, ax = plt.subplots(figsize=(8, 5))
    epochs = range(1, len(history['train_loss']) + 1)
    ax.plot(epochs, history['train_loss'], 'o-', label='Train MSE', linewidth=2)
    ax.plot(epochs, history['valid_loss'], 's-', label='Valid MSE', linewidth=2)
    ax.set_xlabel('Epoch', fontsize=11)
    ax.set_ylabel('MSE', fontsize=11)
    ax.set_title(f'Training History  (mix_alpha = {alpha:.2f})',
                 fontsize=12, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(save_path, dpi=200, bbox_inches='tight')
    plt.close(fig)


def plot_alpha_sweep(results, save_dir: Path, cfg):
    alphas = [r['alpha'] for r in results]
    mses = [r['test_mse'] for r in results]
    maes = [r['test_mae'] for r in results]

    # MSE curve.
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(alphas, mses, 'o-', linewidth=2, markersize=8, color='#2E86AB')
    ax.axhline(mses[0], color='gray', linestyle='--', alpha=0.5,
               label=f'pure CI (alpha=0): {mses[0]:.4f}')
    best_idx = int(np.argmin(mses))
    ax.scatter([alphas[best_idx]], [mses[best_idx]], color='red', s=120, zorder=5,
               label=f'best: alpha={alphas[best_idx]:.2f}, MSE={mses[best_idx]:.4f}')
    ax.set_xlabel('mix_alpha (cross-channel mixing strength)', fontsize=11)
    ax.set_ylabel('Test MSE', fontsize=11)
    ax.set_title(
        f'PatchTST_SCM Alpha Sweep — Electricity\n'
        f'sl={cfg["seq_len"]}, d_model={cfg["d_model"]}, '
        f'epochs={cfg["epochs"]}, step={cfg["step"]}',
        fontsize=12, fontweight='bold',
    )
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(-0.05, 1.05)
    fig.tight_layout()
    fig.savefig(save_dir / 'alpha_sweep_mse.png', dpi=200, bbox_inches='tight')
    plt.close(fig)

    # MAE curve.
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(alphas, maes, 's-', linewidth=2, markersize=8, color='#A23B72')
    ax.axhline(maes[0], color='gray', linestyle='--', alpha=0.5,
               label=f'pure CI (alpha=0): {maes[0]:.4f}')
    best_idx_mae = int(np.argmin(maes))
    ax.scatter([alphas[best_idx_mae]], [maes[best_idx_mae]], color='red', s=120, zorder=5,
               label=f'best: alpha={alphas[best_idx_mae]:.2f}, MAE={maes[best_idx_mae]:.4f}')
    ax.set_xlabel('mix_alpha', fontsize=11)
    ax.set_ylabel('Test MAE', fontsize=11)
    ax.set_title('PatchTST_SCM Alpha Sweep — Electricity (MAE)',
                 fontsize=12, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(-0.05, 1.05)
    fig.tight_layout()
    fig.savefig(save_dir / 'alpha_sweep_mae.png', dpi=200, bbox_inches='tight')
    plt.close(fig)


# ─── Main ────────────────────────────────────────────────────────────────
def main():
    args = parse_args()
    set_seed(args.seed)

    cfg = dict(DEFAULT_CFG)
    cfg.update({
        'forecast_len': args.forecast_len,
        'seq_len': args.seq_len,
        'epochs': args.epochs,
        'batch_size': args.batch_size,
        'learning_rate': args.learning_rate,
        'step': args.step,
        'patience': args.patience,
        'seed': args.seed,
    })

    alphas = [float(a) for a in args.alphas.split()]

    # Output dir.
    ts = datetime.now().strftime('%Y%m%d-%H%M%S')
    summary_dir = Path(args.result_dir) / f'scm_electricity_h{cfg["forecast_len"]}_{ts}'
    summary_dir.mkdir(parents=True, exist_ok=True)
    setup_logging(summary_dir)

    if args.device:
        device = torch.device(args.device)
    else:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    logging.info('=' * 60)
    logging.info('PatchTST_SCM — Electricity (paper-aligned + step=%d)', cfg['step'])
    logging.info('=' * 60)
    logging.info('csv: %s', args.csv_path)
    logging.info('result_dir: %s', summary_dir)
    logging.info('device: %s', device)
    logging.info('alphas: %s', alphas)
    logging.info('cfg: %s', json.dumps(cfg, indent=2))

    with open(summary_dir / 'config.json', 'w') as f:
        json.dump({**cfg, 'alphas': alphas, 'csv_path': str(args.csv_path)}, f, indent=2)

    # Load data once; rebuild loaders only if needed (here: once).
    df = load_timeseries_csv(Path(args.csv_path))
    data = df.to_numpy(dtype=np.float32)
    T_use = int(len(data) * args.data_fraction)
    data = data[:T_use]
    logging.info('Using %d timesteps (data_fraction=%.2f)', len(data), args.data_fraction)

    train_loader, valid_loader, test_loader, M = create_dataloaders(
        data,
        seq_len=cfg['seq_len'],
        forecast_len=cfg['forecast_len'],
        batch_size=cfg['batch_size'],
        train_ratio=cfg['train_ratio'],
        valid_ratio=cfg['valid_ratio'],
        step=cfg['step'],
    )
    logging.info('Variables (channels): %d', M)

    # Sweep.
    all_results = []
    for alpha in alphas:
        # Re-seed before each alpha so that models start from the same init pattern.
        set_seed(args.seed)
        alpha_dir = summary_dir / f'alpha_{alpha:.2f}'
        result = train_one_alpha(alpha, train_loader, valid_loader, test_loader,
                                 cfg, device, alpha_dir)
        all_results.append(result)

        # Update summary after every alpha so partial results survive a crash.
        partial = {f'alpha_{r["alpha"]:.2f}':
                   {'test_mse': r['test_mse'], 'test_mae': r['test_mae'],
                    'best_valid_mse': r['best_valid_mse']}
                   for r in all_results}
        with open(summary_dir / 'summary.json', 'w') as f:
            json.dump(partial, f, indent=2)
        plot_alpha_sweep(all_results, summary_dir, cfg)

    # Final logging.
    logging.info('=' * 60)
    logging.info('SCM alpha sweep completed.')
    logging.info('=' * 60)
    for r in all_results:
        logging.info('  alpha=%.2f  test_mse=%.6f  test_mae=%.6f',
                     r['alpha'], r['test_mse'], r['test_mae'])
    best = min(all_results, key=lambda r: r['test_mse'])
    logging.info('Best by MSE: alpha=%.2f -> mse=%.6f, mae=%.6f',
                 best['alpha'], best['test_mse'], best['test_mae'])
    logging.info('All artifacts saved to: %s', summary_dir)


if __name__ == '__main__':
    main()
