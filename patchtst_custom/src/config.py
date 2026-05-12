"""PatchTSTConfig + named presets (default / fast_test / high_performance / low_memory / patchtst64 / patchtst42)."""

import math


class PatchTSTConfig:
    """Configuration for PatchTST model and training."""

    def __init__(self):
        # Data
        self.seq_len = 336
        self.forecast_len = 96
        self.train_ratio = 0.7
        self.valid_ratio = 0.1

        # Patching
        self.patch_size = 16
        self.stride = 8

        # Architecture
        self.d_model = 128
        self.nhead = 16
        self.num_layers = 3
        self.dim_feedforward = 256
        self.dropout = 0.2
        self.use_checkpoint = True

        # Training
        self.batch_size = 4
        self.accumulation_steps = 16
        self.epochs = 10
        self.learning_rate = 1e-4
        self.patience = 5

        # Sliding window step
        self.step = 8

        self._recompute_derived()

    def _recompute_derived(self):
        self.num_patches = math.floor((self.seq_len - self.patch_size) / self.stride) + 2
        self.effective_batch_size = self.batch_size * self.accumulation_steps

    def update(self, **kwargs):
        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)
            else:
                raise ValueError(f"Unknown configuration parameter: {key}")
        self._recompute_derived()

    def print_config(self):
        print("\n" + "=" * 60)
        print("PatchTST Configuration")
        print("=" * 60)

        print("\nData Parameters:")
        print(f"  Sequence Length (L): {self.seq_len}")
        print(f"  Forecast Length (T): {self.forecast_len}")
        print(f"  Train/Valid/Test Split: {self.train_ratio:.1%}/{self.valid_ratio:.1%}/"
              f"{1-self.train_ratio-self.valid_ratio:.1%}")
        print(f"  Sliding Window Step: {self.step}")

        print("\nPatching Parameters:")
        print(f"  Patch Size (P): {self.patch_size}")
        print(f"  Stride (S): {self.stride}")
        print(f"  Number of Patches (N): {self.num_patches}")

        print("\nModel Architecture:")
        print(f"  Model Dimension (D): {self.d_model}")
        print(f"  Attention Heads (H): {self.nhead}")
        print(f"  Transformer Layers: {self.num_layers}")
        print(f"  Feedforward Dimension: {self.dim_feedforward}")
        print(f"  Dropout: {self.dropout}")
        print(f"  Gradient Checkpointing: {self.use_checkpoint}")

        print("\nTraining Parameters:")
        print(f"  Batch Size: {self.batch_size}")
        print(f"  Accumulation Steps: {self.accumulation_steps}")
        print(f"  Effective Batch Size: {self.effective_batch_size}")
        print(f"  Epochs: {self.epochs}")
        print(f"  Learning Rate: {self.learning_rate}")
        print(f"  Early Stopping Patience: {self.patience}")
        print("=" * 60 + "\n")

    def to_dict(self):
        return {
            'seq_len': self.seq_len,
            'forecast_len': self.forecast_len,
            'patch_size': self.patch_size,
            'stride': self.stride,
            'd_model': self.d_model,
            'nhead': self.nhead,
            'num_layers': self.num_layers,
            'dim_feedforward': self.dim_feedforward,
            'dropout': self.dropout,
            'use_checkpoint': self.use_checkpoint,
            'batch_size': self.batch_size,
            'accumulation_steps': self.accumulation_steps,
            'epochs': self.epochs,
            'learning_rate': self.learning_rate,
            'train_ratio': self.train_ratio,
            'valid_ratio': self.valid_ratio,
            'step': self.step,
            'patience': self.patience,
            'num_patches': self.num_patches,
            'effective_batch_size': self.effective_batch_size,
        }


def _build_configs():
    """Build a fresh dict of named presets. Always rebuild — don't share instances."""
    configs = {
        'default': PatchTSTConfig(),
        'fast_test': PatchTSTConfig(),
        'high_performance': PatchTSTConfig(),
        'low_memory': PatchTSTConfig(),
        'patchtst64': PatchTSTConfig(),
        'patchtst42': PatchTSTConfig(),
    }

    configs['fast_test'].update(
        epochs=5,
        batch_size=8,
        d_model=64,
        num_layers=2,
        nhead=8,
        dim_feedforward=128,
    )

    configs['high_performance'].update(
        epochs=20,
        d_model=256,
        num_layers=4,
        nhead=16,
        dim_feedforward=512,
        dropout=0.2,
        patience=10,
    )

    configs['low_memory'].update(
        batch_size=2,
        accumulation_steps=32,
        d_model=64,
        num_layers=2,
        nhead=8,
        dim_feedforward=128,
        use_checkpoint=True,
    )

    configs['patchtst64'].update(
        seq_len=512,
        forecast_len=96,
    )

    configs['patchtst42'].update(
        seq_len=336,
        forecast_len=96,
    )

    return configs


def get_config(name='default'):
    """Return a fresh PatchTSTConfig for the named preset (mutating it does not affect other presets)."""
    configs = _build_configs()
    if name not in configs:
        raise ValueError(f"Unknown configuration: {name}. Available: {list(configs.keys())}")
    return configs[name]
