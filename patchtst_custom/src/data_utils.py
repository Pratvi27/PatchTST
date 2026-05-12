"""Data preprocessing: CSV loading, sliding-window dataloaders, z-score normalization."""

import numpy as np
import pandas as pd
import torch
from torch.utils.data import TensorDataset, DataLoader


class TimeSeriesDataset:
    """Manages time series data loading and preprocessing."""

    def __init__(self, csv_path, target_columns=None, exclude_columns=None):
        self.csv_path = csv_path
        self.data = pd.read_csv(csv_path)

        if target_columns is not None:
            self.data = self.data[target_columns]
        else:
            # Drop the first column (assumed timestamp/index).
            self.data = self.data[self.data.columns[1:]]

        if exclude_columns is not None:
            self.data = self.data.drop(columns=exclude_columns, errors='ignore')

        self.num_variables = self.data.shape[1]
        self.total_timesteps = self.data.shape[0]

        print(f"Loaded dataset: {self.total_timesteps} timesteps, {self.num_variables} variables")

    def get_data(self):
        return self.data

    def get_numpy(self):
        return self.data.to_numpy().astype(np.float32)


def create_dataloaders(data, seq_len, forecast_len, batch_size=64,
                       train_ratio=0.7, valid_ratio=0.1, step=1):
    """Build train / valid / test DataLoaders with z-score stats fit on training split only."""
    T_total = data.shape[0]
    train_end = int(train_ratio * T_total)
    valid_end = int((train_ratio + valid_ratio) * T_total)

    print(f"\nData Split:")
    print(f"  Total timesteps: {T_total}")
    print(f"  Train: 0 to {train_end} ({train_ratio*100:.0f}%)")
    print(f"  Valid: {train_end} to {valid_end} ({valid_ratio*100:.0f}%)")
    print(f"  Test: {valid_end} to {T_total} ({(1-train_ratio-valid_ratio)*100:.0f}%)")

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
        return (torch.from_numpy(np.array(X)).float(),
                torch.from_numpy(np.array(y)).float())

    X_train, y_train = sliding_window(train_data)
    X_valid, y_valid = sliding_window(valid_data)
    X_test, y_test = sliding_window(test_data)

    print(f"\nDataloader Shapes:")
    print(f"  Train: X={X_train.shape}, y={y_train.shape}")
    print(f"  Valid: X={X_valid.shape}, y={y_valid.shape}")
    print(f"  Test:  X={X_test.shape}, y={y_test.shape}")

    train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=batch_size, shuffle=True)
    valid_loader = DataLoader(TensorDataset(X_valid, y_valid), batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(TensorDataset(X_test, y_test), batch_size=batch_size, shuffle=False)

    return train_loader, valid_loader, test_loader


def normalize_data(data):
    if isinstance(data, pd.DataFrame):
        data = data.to_numpy().astype(np.float32)
    mean = data.mean(axis=0, keepdims=True)
    std = data.std(axis=0, keepdims=True)
    return (data - mean) / (std + 1e-8)


def denormalize_data(data_norm, mean, std):
    if isinstance(data_norm, pd.DataFrame):
        data_norm = data_norm.to_numpy().astype(np.float32)
    return data_norm * (std + 1e-8) + mean
