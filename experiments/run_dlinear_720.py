"""DLinear-I (individual=True) at h=720 on Electricity AND Traffic.

Converted from DLinear_720.ipynb. Runs both datasets sequentially with the
exact hyperparameters from the notebook:

  Electricity:  SEQ_LEN=336, BATCH=8, ACCUM=2, LR=0.001, EPOCHS=20
  Traffic:      SEQ_LEN=336, BATCH=8, ACCUM=2, LR=0.05  (peak via OneCycleLR), EPOCHS=20

Both use INDIVIDUAL=True (DLinear-I, per-channel Linear weights), kernel_size=25,
StandardScaler fit on training set only, OneCycleLR scheduler.

Outputs (under result_dir/dlinear720_<TS>/):
    run.log
    config.json                      exact configs used
    summary.json                     {electricity: {test_mse, test_mae}, traffic: {...}}
    {electricity,traffic}/
        results.json                 {test_mse, test_mae, history}
        training.png                 train/valid loss curve
        best_model.pt                checkpoint of best valid epoch

Linux usage:
    python run_dlinear_720.py
    python run_dlinear_720.py --dataset electricity   # only one
    python run_dlinear_720.py --dataset traffic
    python run_dlinear_720.py \
        --electricity_csv /model/patchTST/all_six_datasets/electricity/electricity.csv \
        --traffic_csv     /model/patchTST/all_six_datasets/traffic/traffic.csv

Override examples:
    python run_dlinear_720.py --epochs 30 --patience 8
    python run_dlinear_720.py --device cuda:1
"""

import argparse
import json
import logging
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
from sklearn.preprocessing import StandardScaler  # noqa: E402
from tqdm import tqdm  # noqa: E402


# ─── Default per-dataset configs (matching DLinear_720.ipynb) ──────────────
ELECTRICITY_CFG = {
    'name': 'electricity',
    'csv_path': '/model/patchTST/all_six_datasets/electricity/electricity.csv',
    'seq_len': 336,
    'pred_len': 720,
    'individual': True,
    'batch_size': 8,
    'accum_steps': 2,
    'lr': 0.001,
    'epochs': 20,
    'patience': 5,
    'kernel_size': 25,
    'expected': (0.203, 0.301),  # paper number for DLinear ECL h=720
}

TRAFFIC_CFG = {
    'name': 'traffic',
    'csv_path': '/model/patchTST/all_six_datasets/traffic/traffic.csv',
    'seq_len': 336,
    'pred_len': 720,
    'individual': True,
    'batch_size': 8,
    'accum_steps': 2,
    'lr': 0.05,                # ⚠ 50× larger than Electricity (notebook setting)
    'epochs': 20,
    'patience': 5,
    'kernel_size': 25,
    'expected': (0.466, 0.315),  # paper number for DLinear Traffic h=720
}


# ─── CLI ─────────────────────────────────────────────────────────────────
def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--dataset', choices=['electricity', 'traffic', 'both'],
                   default='both', help='Which dataset(s) to run')
    p.add_argument('--electricity_csv', type=str,
                   default=ELECTRICITY_CFG['csv_path'])
    p.add_argument('--traffic_csv', type=str,
                   default=TRAFFIC_CFG['csv_path'])
    p.add_argument('--result_dir', type=str, default='./result',
                   help='Parent dir under which dlinear720_<TS>/ is created')
    p.add_argument('--epochs', type=int, default=None,
                   help='Override epochs for both datasets')
    p.add_argument('--patience', type=int, default=None)
    p.add_argument('--batch_size', type=int, default=None)
    p.add_argument('--seed', type=int, default=2021)
    p.add_argument('--device', type=str, default=None)
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


# ─── RevIN (kept for API compatibility, not used in DLinearAdapted) ─────
class RevIN(nn.Module):
    def __init__(self, num_features, eps=1e-5, affine=True, subtract_last=False):
        super().__init__()
        self.num_features = num_features
        self.eps = eps
        self.affine = affine
        self.subtract_last = subtract_last
        if self.affine:
            self.affine_weight = nn.Parameter(torch.ones(num_features))
            self.affine_bias = nn.Parameter(torch.zeros(num_features))

    def forward(self, x, mode):
        if mode == 'norm':
            self._get_statistics(x)
            return self._normalize(x)
        elif mode == 'denorm':
            return self._denormalize(x)
        raise ValueError(f"Unknown mode {mode}")

    def _get_statistics(self, x):
        dim2reduce = tuple(range(1, x.ndim - 1))
        if self.subtract_last:
            self.last = x[:, -1, :].unsqueeze(1)
        else:
            self.mean = torch.mean(x, dim=dim2reduce, keepdim=True).detach()
        self.stdev = torch.sqrt(torch.var(x, dim=dim2reduce, keepdim=True,
                                          unbiased=False) + self.eps).detach()

    def _normalize(self, x):
        if self.subtract_last:
            x = x - self.last
        else:
            x = x - self.mean
        x = x / self.stdev
        if self.affine:
            x = x * self.affine_weight + self.affine_bias
        return x

    def _denormalize(self, x):
        if self.affine:
            x = (x - self.affine_bias) / (self.affine_weight + self.eps * self.eps)
        x = x * self.stdev
        if self.subtract_last:
            x = x + self.last
        else:
            x = x + self.mean
        return x


# ─── DLinear components ──────────────────────────────────────────────────
class moving_avg(nn.Module):
    """Moving average for trend extraction."""
    def __init__(self, kernel_size, stride):
        super().__init__()
        self.kernel_size = kernel_size
        self.avg = nn.AvgPool1d(kernel_size=kernel_size, stride=stride, padding=0)

    def forward(self, x):
        # x: (B, L, C)
        front = x[:, 0:1, :].repeat(1, (self.kernel_size - 1) // 2, 1)
        end = x[:, -1:, :].repeat(1, (self.kernel_size - 1) // 2, 1)
        x = torch.cat([front, x, end], dim=1)
        x = self.avg(x.permute(0, 2, 1)).permute(0, 2, 1)
        return x


class series_decomp(nn.Module):
    """Series decomposition: x = trend + seasonal."""
    def __init__(self, kernel_size):
        super().__init__()
        self.moving_avg = moving_avg(kernel_size, stride=1)

    def forward(self, x):
        moving_mean = self.moving_avg(x)
        residual = x - moving_mean
        return residual, moving_mean


class DLinearAdapted(nn.Module):
    """DLinear with output (B, C, pred_len) to match Trainer convention."""
    def __init__(self, seq_len, forecast_len, enc_in, individual=False, kernel_size=25):
        super().__init__()
        self.seq_len = seq_len
        self.pred_len = forecast_len
        self.channels = enc_in
        self.individual = individual
        self.decompsition = series_decomp(kernel_size)

        if self.individual:
            self.Linear_Seasonal = nn.ModuleList(
                [nn.Linear(seq_len, forecast_len) for _ in range(enc_in)])
            self.Linear_Trend = nn.ModuleList(
                [nn.Linear(seq_len, forecast_len) for _ in range(enc_in)])
        else:
            self.Linear_Seasonal = nn.Linear(seq_len, forecast_len)
            self.Linear_Trend = nn.Linear(seq_len, forecast_len)

    def forward(self, x):
        # x: (B, seq_len, C)
        seasonal, trend = self.decompsition(x)
        seasonal = seasonal.permute(0, 2, 1)   # (B, C, seq_len)
        trend = trend.permute(0, 2, 1)

        if self.individual:
            s_out = torch.zeros(seasonal.size(0), self.channels, self.pred_len,
                                dtype=seasonal.dtype, device=seasonal.device)
            t_out = torch.zeros_like(s_out)
            for i in range(self.channels):
                s_out[:, i, :] = self.Linear_Seasonal[i](seasonal[:, i, :])
                t_out[:, i, :] = self.Linear_Trend[i](trend[:, i, :])
        else:
            s_out = self.Linear_Seasonal(seasonal)   # (B, C, pred_len)
            t_out = self.Linear_Trend(trend)

        return s_out + t_out                          # (B, C, pred_len)


# ─── Data ────────────────────────────────────────────────────────────────
class TimeSeriesDataset:
    def __init__(self, csv_path, exclude_columns=None):
        self.csv_path = csv_path
        self.data = pd.read_csv(csv_path)
        # Drop the first column (timestamp) — same convention as DLinear/PatchTST
        self.data = self.data[self.data.columns[1:]]
        if exclude_columns is not None:
            self.data = self.data.drop(columns=exclude_columns, errors='ignore')
        self.num_variables = self.data.shape[1]
        self.total_timesteps = self.data.shape[0]
        logging.info("Loaded dataset: %d timesteps, %d variables",
                     self.total_timesteps, self.num_variables)

    def get_numpy(self):
        return self.data.to_numpy().astype(np.float32)


def create_dataloaders(data, seq_len, forecast_len, batch_size,
                       train_ratio=0.7, valid_ratio=0.1, step=1):
    T_total = data.shape[0]
    train_end = int(train_ratio * T_total)
    valid_end = int((train_ratio + valid_ratio) * T_total)

    logging.info("Data Split:")
    logging.info("  Total timesteps: %d", T_total)
    logging.info("  Train: 0..%d (%.0f%%)", train_end, train_ratio * 100)
    logging.info("  Valid: %d..%d (%.0f%%)", train_end, valid_end, valid_ratio * 100)
    logging.info("  Test:  %d..%d (%.0f%%)", valid_end, T_total,
                 (1 - train_ratio - valid_ratio) * 100)

    train_data = data[:train_end]
    valid_data = data[train_end:valid_end]
    test_data = data[valid_end:]

    def sliding_window(d):
        X, y = [], []
        total = seq_len + forecast_len
        for i in range(0, len(d) - total + 1, step):
            X.append(d[i:i + seq_len])
            y.append(d[i + seq_len:i + total])
        return (torch.from_numpy(np.array(X)).float(),
                torch.from_numpy(np.array(y)).float())

    X_train, y_train = sliding_window(train_data)
    X_valid, y_valid = sliding_window(valid_data)
    X_test, y_test = sliding_window(test_data)

    logging.info("Dataloader shapes: train X=%s, valid X=%s, test X=%s",
                 tuple(X_train.shape), tuple(X_valid.shape), tuple(X_test.shape))

    return (
        DataLoader(TensorDataset(X_train, y_train), batch_size=batch_size, shuffle=True),
        DataLoader(TensorDataset(X_valid, y_valid), batch_size=batch_size, shuffle=False),
        DataLoader(TensorDataset(X_test, y_test), batch_size=batch_size, shuffle=False),
    )


# ─── Trainer ─────────────────────────────────────────────────────────────
class Trainer:
    def __init__(self, model, device='cpu', best_model_path='./best_model.pt'):
        self.model = model.to(device)
        self.device = device
        self.criterion = nn.MSELoss()
        self.history = {'train_loss': [], 'valid_loss': [], 'test_loss': None}
        self.best_model_path = str(best_model_path)

    def train_epoch(self, train_loader, optimizer, scheduler, accumulation_steps=1):
        self.model.train()
        epoch_loss = 0.0
        optimizer.zero_grad()
        pbar = tqdm(enumerate(train_loader), total=len(train_loader), desc='Training')
        i = -1
        for i, (x_batch, y_batch) in pbar:
            x_batch = x_batch.to(self.device)
            y_batch = y_batch.permute(0, 2, 1).to(self.device)
            out = self.model(x_batch)
            loss = self.criterion(out, y_batch) / accumulation_steps
            loss.backward()
            if (i + 1) % accumulation_steps == 0:
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
            epoch_loss += loss.item() * accumulation_steps
            pbar.set_postfix({'loss': f'{loss.item() * accumulation_steps:.4f}'})
        if i >= 0 and (i + 1) % accumulation_steps != 0:
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
        return epoch_loss / max(len(train_loader), 1)

    def validate(self, valid_loader):
        self.model.eval()
        loss = 0.0
        with torch.no_grad():
            for x_batch, y_batch in valid_loader:
                x_batch = x_batch.to(self.device)
                y_batch = y_batch.permute(0, 2, 1).to(self.device)
                out = self.model(x_batch)
                loss += self.criterion(out, y_batch).item()
        return loss / max(len(valid_loader), 1)

    def train(self, train_loader, valid_loader, epochs=20, lr=1e-3,
              accumulation_steps=1, patience=5):
        optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        # OneCycleLR: warm up to max_lr at 30%, then decay
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer=optimizer,
            steps_per_epoch=len(train_loader),
            pct_start=0.3,
            epochs=epochs,
            max_lr=lr,
        )
        best_valid = float('inf')
        bad_epochs = 0

        logging.info("Training: epochs=%d, max_lr=%g, accum=%d, patience=%d, device=%s",
                     epochs, lr, accumulation_steps, patience, self.device)

        for epoch in range(epochs):
            logging.info("=" * 60)
            logging.info("Epoch %d/%d", epoch + 1, epochs)
            train_loss = self.train_epoch(train_loader, optimizer, scheduler,
                                          accumulation_steps)
            valid_loss = self.validate(valid_loader)
            self.history['train_loss'].append(train_loss)
            self.history['valid_loss'].append(valid_loss)
            logging.info("  train=%.4f valid=%.4f lr=%.6f",
                         train_loss, valid_loss, scheduler.get_last_lr()[0])

            if valid_loss < best_valid:
                best_valid = valid_loss
                torch.save(self.model.state_dict(), self.best_model_path)
                logging.info("  ✓ new best (valid=%.4f) saved to %s",
                             best_valid, self.best_model_path)
                bad_epochs = 0
            else:
                bad_epochs += 1
                logging.info("  no improvement (%d/%d)", bad_epochs, patience)
                if bad_epochs >= patience:
                    logging.info("Early stopping at epoch %d", epoch + 1)
                    break

        self.model.load_state_dict(torch.load(self.best_model_path))
        logging.info("Training done. Best valid=%.4f", best_valid)
        return self.history

    def evaluate(self, test_loader):
        self.model.eval()
        mse_total, mae_total, n_samples = 0.0, 0.0, 0
        logging.info("Evaluating on test set ...")
        with torch.no_grad():
            for x_batch, y_batch in tqdm(test_loader):
                x_batch = x_batch.to(self.device)
                y_batch = y_batch.permute(0, 2, 1).to(self.device)
                out = self.model(x_batch)
                mse_total += F.mse_loss(out, y_batch, reduction='sum').item()
                mae_total += (out - y_batch).abs().sum().item()
                n_samples += y_batch.numel()
        test_mse = mse_total / n_samples
        test_mae = mae_total / n_samples
        self.history['test_loss'] = test_mse
        logging.info("Test MSE=%.4f  RMSE=%.4f  MAE=%.4f",
                     test_mse, np.sqrt(test_mse), test_mae)
        return test_mse, test_mae


# ─── Per-dataset run ─────────────────────────────────────────────────────
def plot_training(history, save_path: Path, title: str):
    fig, ax = plt.subplots(figsize=(8, 5))
    epochs = range(1, len(history['train_loss']) + 1)
    ax.plot(epochs, history['train_loss'], 'o-', label='Train MSE', linewidth=2)
    ax.plot(epochs, history['valid_loss'], 's-', label='Valid MSE', linewidth=2)
    ax.set_xlabel('Epoch', fontsize=11)
    ax.set_ylabel('MSE', fontsize=11)
    ax.set_title(title, fontsize=12, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(save_path, dpi=200, bbox_inches='tight')
    plt.close(fig)


def run_one(cfg, csv_path, save_dir: Path, device, seed):
    """Run DLinear-I on one dataset at h=720."""
    set_seed(seed)
    save_dir.mkdir(parents=True, exist_ok=True)

    logging.info("#" * 60)
    logging.info("DLinear-I  |  %s  |  L=%d, T=%d  |  individual=%s, lr=%g",
                 cfg['name'], cfg['seq_len'], cfg['pred_len'],
                 cfg['individual'], cfg['lr'])
    logging.info("#" * 60)
    logging.info("csv: %s", csv_path)
    logging.info("save_dir: %s", save_dir)

    # Load data + standard scaling on train only.
    ts = TimeSeriesDataset(csv_path)
    data = ts.get_numpy()
    enc_in = ts.num_variables
    logging.info("Channels=%d  total_timesteps=%d", enc_in, data.shape[0])

    scaler = StandardScaler()
    train_end = int(0.7 * len(data))
    scaler.fit(data[:train_end])
    data_scaled = scaler.transform(data).astype(np.float32)

    train_loader, valid_loader, test_loader = create_dataloaders(
        data=data_scaled,
        seq_len=cfg['seq_len'],
        forecast_len=cfg['pred_len'],
        batch_size=cfg['batch_size'],
        train_ratio=0.7, valid_ratio=0.1, step=1,
    )

    model = DLinearAdapted(
        seq_len=cfg['seq_len'],
        forecast_len=cfg['pred_len'],
        enc_in=enc_in,
        individual=cfg['individual'],
        kernel_size=cfg['kernel_size'],
    )
    n_params = sum(p.numel() for p in model.parameters())
    logging.info("Model parameters: %d", n_params)

    trainer = Trainer(model, device=device, best_model_path=save_dir / 'best_model.pt')
    history = trainer.train(
        train_loader=train_loader,
        valid_loader=valid_loader,
        epochs=cfg['epochs'],
        lr=cfg['lr'],
        accumulation_steps=cfg['accum_steps'],
        patience=cfg['patience'],
    )
    test_mse, test_mae = trainer.evaluate(test_loader)

    e_mse, e_mae = cfg['expected']
    logging.info("=" * 60)
    logging.info("[%s h=%d]  MSE=%.4f (paper %.3f)  MAE=%.4f (paper %.3f)",
                 cfg['name'], cfg['pred_len'], test_mse, e_mse, test_mae, e_mae)
    logging.info("=" * 60)

    # Save artifacts.
    with open(save_dir / 'results.json', 'w') as f:
        json.dump({
            'dataset': cfg['name'],
            'pred_len': cfg['pred_len'],
            'seq_len': cfg['seq_len'],
            'individual': cfg['individual'],
            'kernel_size': cfg['kernel_size'],
            'batch_size': cfg['batch_size'],
            'accum_steps': cfg['accum_steps'],
            'lr': cfg['lr'],
            'epochs': cfg['epochs'],
            'patience': cfg['patience'],
            'enc_in': enc_in,
            'parameters': n_params,
            'test_mse': test_mse,
            'test_mae': test_mae,
            'paper_expected_mse': e_mse,
            'paper_expected_mae': e_mae,
            'history': history,
        }, f, indent=2)

    plot_training(history, save_dir / 'training.png',
                  title=f"DLinear-I {cfg['name']} h={cfg['pred_len']}  (lr={cfg['lr']})")

    return {'test_mse': test_mse, 'test_mae': test_mae,
            'paper_mse': e_mse, 'paper_mae': e_mae}


# ─── Main ────────────────────────────────────────────────────────────────
def main():
    args = parse_args()

    ts = datetime.now().strftime('%Y%m%d-%H%M%S')
    summary_dir = Path(args.result_dir) / f'dlinear720_{ts}'
    summary_dir.mkdir(parents=True, exist_ok=True)
    setup_logging(summary_dir)

    # Apply CLI overrides (if any) to both configs.
    e_cfg = dict(ELECTRICITY_CFG)
    t_cfg = dict(TRAFFIC_CFG)
    e_cfg['csv_path'] = args.electricity_csv
    t_cfg['csv_path'] = args.traffic_csv
    if args.epochs is not None:
        e_cfg['epochs'] = t_cfg['epochs'] = args.epochs
    if args.patience is not None:
        e_cfg['patience'] = t_cfg['patience'] = args.patience
    if args.batch_size is not None:
        e_cfg['batch_size'] = t_cfg['batch_size'] = args.batch_size

    if args.device:
        device = torch.device(args.device)
    else:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    logging.info("=" * 60)
    logging.info("DLinear-I  h=720 sweep  |  device=%s  |  seed=%d", device, args.seed)
    logging.info("=" * 60)

    with open(summary_dir / 'config.json', 'w') as f:
        json.dump({
            'dataset_target': args.dataset,
            'electricity': e_cfg,
            'traffic': t_cfg,
            'seed': args.seed,
        }, f, indent=2)

    summary = {}

    if args.dataset in ('electricity', 'both'):
        summary['electricity'] = run_one(
            e_cfg, args.electricity_csv,
            summary_dir / 'electricity', device, args.seed,
        )
        with open(summary_dir / 'summary.json', 'w') as f:
            json.dump(summary, f, indent=2)

    if args.dataset in ('traffic', 'both'):
        summary['traffic'] = run_one(
            t_cfg, args.traffic_csv,
            summary_dir / 'traffic', device, args.seed,
        )
        with open(summary_dir / 'summary.json', 'w') as f:
            json.dump(summary, f, indent=2)

    logging.info("=" * 60)
    logging.info("All runs done. Artifacts in: %s", summary_dir)
    logging.info("Summary:")
    for k, v in summary.items():
        logging.info("  %s h=720:  MSE=%.4f (paper %.3f)  MAE=%.4f (paper %.3f)",
                     k, v['test_mse'], v['paper_mse'], v['test_mae'], v['paper_mae'])


if __name__ == '__main__':
    main()
