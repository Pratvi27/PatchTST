"""PatchTST model: RevIN + patch embedding + BatchNorm transformer encoder + linear head."""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint


class RevIN(nn.Module):
    """Reversible Instance Normalization over the time dimension."""

    def __init__(self, eps=1e-5):
        super().__init__()
        self.eps = eps
        self.affine_weight = nn.Parameter(torch.ones(1))
        self.affine_bias = nn.Parameter(torch.zeros(1))

    def normalize(self, x: torch.Tensor) -> torch.Tensor:
        self.mean = x.mean(dim=-1, keepdim=True)
        self.std = x.std(dim=-1, keepdim=True) + self.eps
        x = (x - self.mean) / self.std
        x = x * self.affine_weight + self.affine_bias
        return x

    def denormalize(self, x: torch.Tensor) -> torch.Tensor:
        x = (x - self.affine_bias) / (self.affine_weight + self.eps)
        x = x * self.std + self.mean
        return x


class PatchEmbedding(nn.Module):
    def __init__(self, patch_size: int, d_model: int, num_patches: int):
        super().__init__()
        self.patch_size = patch_size
        self.d_model = d_model
        self.num_patches = num_patches
        self.Wp = nn.Linear(patch_size, d_model, bias=False)
        self.Wpos = nn.Parameter(torch.randn(d_model, num_patches))

    def forward(self, x_p: torch.Tensor) -> torch.Tensor:
        x_d = self.Wp(x_p)
        x_d = x_d.transpose(-1, -2)
        x_d = x_d + self.Wpos
        return x_d


class BatchNormTransformerLayer(nn.Module):
    """Transformer encoder layer using BatchNorm1d in place of LayerNorm."""

    def __init__(self, d_model: int, nhead: int, dim_feedforward: int, dropout: float):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        self.ff1 = nn.Linear(d_model, dim_feedforward)
        self.ff2 = nn.Linear(dim_feedforward, d_model)
        self.dropout = nn.Dropout(dropout)
        self.bn1 = nn.BatchNorm1d(d_model)
        self.bn2 = nn.BatchNorm1d(d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        attn_out, _ = self.self_attn(x, x, x)
        x = x + self.dropout(attn_out)
        x = x.transpose(1, 2)
        x = self.bn1(x)
        x = x.transpose(1, 2)

        ff_out = self.ff2(self.dropout(F.gelu(self.ff1(x))))
        x = x + self.dropout(ff_out)
        x = x.transpose(1, 2)
        x = self.bn2(x)
        x = x.transpose(1, 2)

        return x


class PatchTST(nn.Module):
    """PatchTST: time series transformer with patching and channel independence."""

    def __init__(self, seq_len, forecast_len, patch_size, stride,
                 d_model, nhead=8, num_layers=3, dim_feedforward=256,
                 dropout=0.1, use_checkpoint=True):
        super().__init__()
        self.seq_len = seq_len
        self.patch_size = patch_size
        self.stride = stride
        self.forecast_len = forecast_len
        self.num_patches = math.floor((seq_len - patch_size) / stride) + 2
        self.use_checkpoint = use_checkpoint

        self.revin = RevIN()
        self.patch_embed = PatchEmbedding(patch_size, d_model, self.num_patches)

        self.encoder_layers = nn.ModuleList([
            BatchNormTransformerLayer(d_model, nhead, dim_feedforward, dropout)
            for _ in range(num_layers)
        ])

        self.flatten = nn.Flatten(start_dim=-2)
        self.linear_head = nn.Linear(d_model * self.num_patches, forecast_len)

    def _patch(self, x):
        S, P = self.stride, self.patch_size
        last_val = x[:, -1:]
        x_padded = torch.cat([x, last_val.expand(-1, S)], dim=-1)
        return x_padded.unfold(dimension=1, size=P, step=S)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, L, M = x.shape
        x = x.permute(0, 2, 1).reshape(B * M, L)
        x = self.revin.normalize(x)

        x_p = self._patch(x)
        x_d = self.patch_embed(x_p)
        x_d = x_d.transpose(-1, -2)

        for layer in self.encoder_layers:
            if self.use_checkpoint:
                x_d = checkpoint(layer, x_d, use_reentrant=False)
            else:
                x_d = layer(x_d)

        out = self.flatten(x_d)
        out = self.linear_head(out)
        out = self.revin.denormalize(out)
        return out.reshape(B, M, self.forecast_len)
