from __future__ import annotations

import torch
from torch import nn


def pool_2x2_spatial(feats: torch.Tensor) -> torch.Tensor:
    """Average each 2x2 block of a row-major 14x14 grid.

    Accepts ``[196, D]`` or ``[B, 196, D]`` and returns 49 tokens.
    The feature cache always stores the uncompressed 196 tokens; compression
    happens here, immediately before the connector.
    """
    squeeze = feats.dim() == 2
    if squeeze:
        feats = feats.unsqueeze(0)
    if feats.dim() != 3:
        raise ValueError(f"Expected [B, 196, D] features, got {tuple(feats.shape)}")
    batch, tokens, dim = feats.shape
    if tokens != 196:
        raise ValueError(f"2x2 pooling requires 196 patch tokens, got {tokens}")
    grid = feats.reshape(batch, 14, 14, dim)
    pooled = grid.reshape(batch, 7, 2, 7, 2, dim).mean(dim=(2, 4))
    pooled = pooled.reshape(batch, 49, dim)
    return pooled.squeeze(0) if squeeze else pooled


class Connector(nn.Module):
    """Linear 768→1024, GELU, Linear 1024→1024, then a calibrated output scale.

    ``output_scale`` is a buffer, not a parameter. It is set so the mean L2
    norm of connector outputs matches the mean L2 norm of text token embeddings.
    """

    def __init__(
        self,
        in_dim: int = 768,
        hidden_dim: int = 1024,
        out_dim: int = 1024,
        num_visual_tokens: int = 196,
    ):
        super().__init__()
        if num_visual_tokens not in {49, 196}:
            raise ValueError("num_visual_tokens must be 196 or 49")
        self.in_dim = int(in_dim)
        self.hidden_dim = int(hidden_dim)
        self.out_dim = int(out_dim)
        self.num_visual_tokens = int(num_visual_tokens)
        self.fc1 = nn.Linear(self.in_dim, self.hidden_dim)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(self.hidden_dim, self.out_dim)
        self.register_buffer("output_scale", torch.tensor(1.0, dtype=torch.float32), persistent=True)

    def compress(self, feats: torch.Tensor) -> torch.Tensor:
        if self.num_visual_tokens == 196:
            return feats
        return pool_2x2_spatial(feats)

    def forward(self, feats: torch.Tensor) -> torch.Tensor:
        hidden = self.compress(feats)
        hidden = self.fc2(self.act(self.fc1(hidden)))
        scale = self.output_scale.to(device=hidden.device, dtype=hidden.dtype)
        return hidden * scale
