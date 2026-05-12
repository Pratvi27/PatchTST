"""CLI entry point for PatchTST training/evaluation.

Usage:
    python main.py --csv_path ./data/traffic/traffic.csv \
                   --config patchtst64 \
                   --forecast_len 96 \
                   --result_dir ./result/patchtst64_h96_20260502-143000 \
                   --seed 2021
"""

import argparse
import json
import logging
import os
import random
import sys
from pathlib import Path

import numpy as np
import torch

from src.config import get_config
from src.data_utils import TimeSeriesDataset, create_dataloaders
from src.model import PatchTST
from src.trainer import Trainer
from src.visualize import (
    plot_error_distribution,
    plot_forecast_comparison,
    plot_predictions,
    plot_training_history,
    plot_variable_performance,
)


def parse_args():
    p = argparse.ArgumentParser(description='Train PatchTST on time series data')
    p.add_argument('--csv_path', type=str, default='./data/traffic/traffic.csv',
                   help='Path to input CSV (first column treated as timestamp index)')
    p.add_argument('--config', type=str, default='patchtst64',
                   choices=['default', 'fast_test', 'high_performance',
                            'low_memory', 'patchtst64', 'patchtst42'],
                   help='Configuration preset')
    p.add_argument('--result_dir', type=str, default='./result/run',
                   help='Directory to save logs, plots, model, and metrics')
    p.add_argument('--seed', type=int, default=2021,
                   help='Random seed')

    # CLI overrides — each defaults to None so we only override when explicitly set.
    p.add_argument('--forecast_len', type=int, default=None,
                   help='Forecast horizon (overrides preset)')
    p.add_argument('--seq_len', type=int, default=None,
                   help='Lookback window length (overrides preset)')
    p.add_argument('--epochs', type=int, default=20,
                   help='Number of training epochs (notebook default: 20)')
    p.add_argument('--patience', type=int, default=10,
                   help='Early stopping patience (notebook default: 10)')
    p.add_argument('--batch_size', type=int, default=8,
                   help='Training batch size (notebook default: 8)')
    p.add_argument('--accumulation_steps', type=int, default=3,
                   help='Gradient accumulation steps (notebook default: 3)')
    p.add_argument('--step', type=int, default=1,
                   help='Sliding window step (notebook default: 1)')
    p.add_argument('--learning_rate', type=float, default=None,
                   help='Optimizer learning rate (overrides preset)')
    p.add_argument('--device', type=str, default=None,
                   help='Device: cuda / cpu / cuda:0. Auto-detect if omitted.')
    return p.parse_args()


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def setup_logging(result_dir: Path):
    """Configure logging to stdout + result_dir/run.log, and tee print() output."""
    log_path = result_dir / 'run.log'
    log_path.parent.mkdir(parents=True, exist_ok=True)

    root = logging.getLogger()
    root.setLevel(logging.INFO)
    for h in list(root.handlers):
        root.removeHandler(h)

    fmt = logging.Formatter('%(asctime)s [%(levelname)s] %(message)s', datefmt='%Y-%m-%d %H:%M:%S')

    fh = logging.FileHandler(log_path, mode='w', encoding='utf-8')
    fh.setFormatter(fmt)
    root.addHandler(fh)

    sh = logging.StreamHandler(sys.__stdout__)
    sh.setFormatter(fmt)
    root.addHandler(sh)

    # Tee any plain print() output (which the original notebook uses heavily)
    # to the run.log as well, while preserving terminal output.
    class _Tee:
        def __init__(self, *streams):
            self._streams = streams

        def write(self, data):
            for s in self._streams:
                try:
                    s.write(data)
                    s.flush()
                except Exception:
                    pass

        def flush(self):
            for s in self._streams:
                try:
                    s.flush()
                except Exception:
                    pass

    log_file_stream = open(log_path, 'a', encoding='utf-8', buffering=1)
    sys.stdout = _Tee(sys.__stdout__, log_file_stream)
    sys.stderr = _Tee(sys.__stderr__, log_file_stream)

    return log_path


def main():
    args = parse_args()

    result_dir = Path(args.result_dir).resolve()
    plots_dir = result_dir / 'plots'
    plots_dir.mkdir(parents=True, exist_ok=True)

    log_path = setup_logging(result_dir)

    print("=" * 60)
    print("PatchTST Pipeline")
    print("=" * 60)
    print(f"  csv_path:    {args.csv_path}")
    print(f"  config:      {args.config}")
    print(f"  result_dir:  {result_dir}")
    print(f"  seed:        {args.seed}")
    print(f"  log file:    {log_path}")

    set_seed(args.seed)

    device = args.device or ('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"  device:      {device}")

    # 1. Load data
    print("\n[1/6] Loading data...")
    csv_path = Path(args.csv_path).resolve()
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV not found: {csv_path}")
    dataset = TimeSeriesDataset(csv_path=str(csv_path))
    data = dataset.get_numpy()

    # 2. Configuration
    print("\n[2/6] Loading configuration...")
    config = get_config(args.config)

    overrides = {
        'epochs': args.epochs,
        'patience': args.patience,
        'batch_size': args.batch_size,
        'accumulation_steps': args.accumulation_steps,
        'step': args.step,
    }
    if args.forecast_len is not None:
        overrides['forecast_len'] = args.forecast_len
    if args.seq_len is not None:
        overrides['seq_len'] = args.seq_len
    if args.learning_rate is not None:
        overrides['learning_rate'] = args.learning_rate
    config.update(**overrides)
    config.print_config()

    # Persist the resolved config so each run is reproducible.
    config_path = result_dir / 'config.json'
    with open(config_path, 'w', encoding='utf-8') as f:
        json.dump(config.to_dict(), f, indent=2)
    print(f"Config saved to: {config_path}")

    # 3. Dataloaders
    print("\n[3/6] Creating dataloaders...")
    train_loader, valid_loader, test_loader = create_dataloaders(
        data=data,
        seq_len=config.seq_len,
        forecast_len=config.forecast_len,
        batch_size=config.batch_size,
        train_ratio=config.train_ratio,
        valid_ratio=config.valid_ratio,
        step=config.step,
    )

    # 4. Model
    print("\n[4/6] Building model...")
    model = PatchTST(
        seq_len=config.seq_len,
        forecast_len=config.forecast_len,
        patch_size=config.patch_size,
        stride=config.stride,
        d_model=config.d_model,
        nhead=config.nhead,
        num_layers=config.num_layers,
        dim_feedforward=config.dim_feedforward,
        dropout=config.dropout,
        use_checkpoint=config.use_checkpoint,
    )

    # 5. Train
    print("\n[5/6] Training...")
    best_ckpt = result_dir / 'best_model.pt'
    trainer = Trainer(model=model, device=device, best_model_path=str(best_ckpt))
    history = trainer.train(
        train_loader=train_loader,
        valid_loader=valid_loader,
        epochs=config.epochs,
        lr=config.learning_rate,
        accumulation_steps=config.accumulation_steps,
        patience=config.patience,
    )

    # 6. Evaluate
    print("\n[6/6] Evaluating...")
    test_loss, predictions, targets = trainer.evaluate(test_loader, return_predictions=True)

    # Plots
    print("\nGenerating plots...")
    plot_training_history(
        history=history,
        save_path=str(plots_dir / 'training_history.png'),
    )
    plot_predictions(
        predictions=predictions, targets=targets,
        num_samples=3, num_variables=5,
        save_path=str(plots_dir / 'predictions.png'),
    )
    plot_error_distribution(
        predictions=predictions, targets=targets,
        save_path=str(plots_dir / 'error_distribution.png'),
    )
    plot_variable_performance(
        predictions=predictions, targets=targets,
        save_path=str(plots_dir / 'variable_performance.png'),
    )
    plot_forecast_comparison(
        predictions=predictions, targets=targets, sample_idx=0,
        save_path=str(plots_dir / 'forecast_comparison.png'),
    )

    # Save final model (snapshot of best-loaded weights) and metrics summary.
    final_model_path = result_dir / 'final_model.pt'
    torch.save(model.state_dict(), final_model_path)
    print(f"Model saved to: {final_model_path}")

    results = {
        'train_loss': float(history['train_loss'][-1]),
        'valid_loss': float(history['valid_loss'][-1]),
        'test_loss': float(test_loss),
        'test_rmse': float(np.sqrt(test_loss)),
        'num_variables': int(data.shape[1]),
        'total_timesteps': int(data.shape[0]),
        'config': config.to_dict(),
        'seed': args.seed,
    }
    results_path = result_dir / 'results.json'
    with open(results_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2)
    print(f"Results saved to: {results_path}")

    print("\n" + "=" * 60)
    print("PIPELINE COMPLETED SUCCESSFULLY")
    print("=" * 60)
    print(f"  Train Loss: {results['train_loss']:.4f}")
    print(f"  Valid Loss: {results['valid_loss']:.4f}")
    print(f"  Test Loss : {results['test_loss']:.4f}")
    print(f"  Test RMSE : {results['test_rmse']:.4f}")
    print(f"  Outputs   : {result_dir}")


if __name__ == '__main__':
    main()
