"""Published deep learning baseline architectures and canonical protocol mappings:
  1. BuresLarrosaClassifier (Nature 2023 spirit): multi-scale dilated 1D-CNN + residual
     temporal pooling over irregular concentration curves + FiLM design conditioning ->
     closed-set Softmax classifier (NO parameter normalizing flow, NO factorized head).
  2. Canonical BL-20 mappable mechanism subset (20 mechanisms spanning C1-C4 & M1-M8),
     also serving as the fixed 20-mechanism multicore canonical sublibrary for SMC-ABC.
"""

from __future__ import annotations

from typing import List, Tuple
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from quomech.grammar import ReactionGrammar


BL20_CANONICAL_NAMES: Tuple[str, ...] = (
    # 4 Clean catalytic cores (no side-reaction modules)
    "C1",
    "C2",
    "C3",
    "C4",
    # Activation (M1)
    "C1+M1",
    "C4+M1",
    # Catalyst deactivation (M2: 1st-order, 2nd-order, complex)
    "C1+M2:free1",
    "C1+M2:free2",
    "C1+M2:complex",
    # Excess substrate inhibition (M3: A-inhib on C1, B-inhib on C2)
    "C1+M3",
    "C2+M3",
    # Product inhibition (M4)
    "C1+M4",
    "C4+M4",
    # Off-cycle dimeric resting state (M5)
    "C1+M5",
    # Background non-catalytic reaction (M6)
    "C1+M6",
    # Reversible product release (M7)
    "C1+M7",
    # Bimetallic / 2nd-order in catalyst (M8)
    "C1+M8",
    # Key 2-module BL-20 / literature combinations
    "C1+M1+M2:free1",
    "C1+M2:free1+M4",
    "C1+M1+M5",
)


def get_bl20_mappable_mech_ids(grammar: ReactionGrammar) -> np.ndarray:
    """Return sorted 20-element int64 array of mechanism IDs for the BL-20 mappable protocol."""
    ids = [grammar.get_mech_id_by_name(name) for name in BL20_CANONICAL_NAMES]
    assert len(ids) == 20 and len(set(ids)) == 20
    return np.array(ids, dtype=np.int64)


class ResConv1DBlock(nn.Module):
    """Multi-scale dilated residual 1D-CNN block with design FiLM modulation."""

    def __init__(self, channels: int, cond_dim: int, kernel_size: int = 5, dilation: int = 1) -> None:
        super().__init__()
        pad = (kernel_size - 1) * dilation // 2
        self.conv1 = nn.Conv1d(channels, channels, kernel_size, padding=pad, dilation=dilation)
        self.norm1 = nn.GroupNorm(8, channels)
        self.conv2 = nn.Conv1d(channels, channels, kernel_size, padding=pad, dilation=dilation)
        self.norm2 = nn.GroupNorm(8, channels)
        self.film = nn.Linear(cond_dim, 2 * channels)

    def forward(self, x: torch.Tensor, cond: torch.Tensor, valid_mask: torch.Tensor) -> torch.Tensor:
        # x: (N, C, L), cond: (N, cond_dim), valid_mask: (N, 1, L) float
        h = F.gelu(self.norm1(self.conv1(x))) * valid_mask
        gamma, beta = self.film(cond).chunk(2, dim=-1)
        h = h * (1.0 + gamma.unsqueeze(-1)) + beta.unsqueeze(-1)
        h = F.gelu(self.norm2(self.conv2(h))) * valid_mask
        return (x + h) * valid_mask


class BuresLarrosaClassifier(nn.Module):
    """Closed-set 1D-CNN + multi-experiment pooling mechanism classifier (Burés & Larrosa 2023 baseline).

    Accepts the exact same batch tensors as QuoMechNet:
      (design_feats, times, obs_vals, obs_masks, time_pad_mask, exp_pad_mask)
    and outputs closed-set logits (B, n_classes) without parameter normalizing flow.
    """

    def __init__(
        self,
        design_dim: int,
        n_obs_channels: int,
        n_classes: int,
        hidden: int = 256,
    ) -> None:
        super().__init__()
        self.n_classes = n_classes
        self.hidden = hidden
        in_ch = 2 + 2 * n_obs_channels  # [t_norm, log1p(t), obs_vals*mask, obs_masks]

        self.des_mlp = nn.Sequential(
            nn.Linear(design_dim, hidden),
            nn.GELU(),
            nn.Linear(hidden, hidden),
        )
        self.in_conv = nn.Conv1d(in_ch, hidden, kernel_size=3, padding=1)
        self.blocks = nn.ModuleList([
            ResConv1DBlock(hidden, hidden, kernel_size=3, dilation=1),
            ResConv1DBlock(hidden, hidden, kernel_size=5, dilation=1),
            ResConv1DBlock(hidden, hidden, kernel_size=5, dilation=2),
            ResConv1DBlock(hidden, hidden, kernel_size=5, dilation=4),
            ResConv1DBlock(hidden, hidden, kernel_size=3, dilation=1),
        ])
        self.exp_head = nn.Sequential(
            nn.Linear(hidden * 3, hidden),
            nn.LayerNorm(hidden),
            nn.GELU(),
            nn.Linear(hidden, hidden),
        )
        self.classifier = nn.Sequential(
            nn.Linear(hidden * 2, hidden * 2),
            nn.LayerNorm(hidden * 2),
            nn.GELU(),
            nn.Linear(hidden * 2, hidden),
            nn.GELU(),
            nn.Linear(hidden, n_classes),
        )

    def forward_logits(
        self,
        design_feats: torch.Tensor,   # (B, E, design_dim)
        times: torch.Tensor,          # (B, E, L)
        obs_vals: torch.Tensor,       # (B, E, L, C)
        obs_masks: torch.Tensor,      # (B, E, L, C)
        time_pad_mask: torch.Tensor,  # (B, E, L) bool (True=padded)
        exp_pad_mask: torch.Tensor,   # (B, E) bool (True=padded)
    ) -> torch.Tensor:
        B, E, L, C = obs_vals.shape
        N = B * E

        df = design_feats.reshape(N, -1)
        tm = times.reshape(N, L)
        ov = obs_vals.reshape(N, L, C)
        om = obs_masks.reshape(N, L, C)
        valid_t = (~time_pad_mask.reshape(N, L)).to(ov.dtype)  # (N, L)

        t_norm = (tm / 4.0).unsqueeze(-1)
        t_log = torch.log1p(tm.clamp(min=0.0)).unsqueeze(-1)
        seq = torch.cat([t_norm, t_log, ov * om, om], dim=-1)  # (N, L, 2+2C)
        x = seq.transpose(1, 2)  # (N, in_ch, L)
        vm = valid_t.unsqueeze(1)  # (N, 1, L)

        des_emb = self.des_mlp(df)  # (N, H)
        h = F.gelu(self.in_conv(x)) * vm
        for blk in self.blocks:
            h = blk(h, des_emb, vm)

        # Masked mean + max pooling over time
        h_mean = h.sum(dim=-1) / valid_t.sum(dim=-1, keepdim=True).clamp(min=1.0)
        h_max = h.masked_fill(vm < 0.5, -1e4).max(dim=-1).values
        exp_emb = self.exp_head(torch.cat([h_mean, h_max, des_emb], dim=-1)).view(B, E, self.hidden)

        # Permutation-invariant set pooling (mean + max) over valid experiments
        valid_e = (~exp_pad_mask).to(exp_emb.dtype).unsqueeze(-1)  # (B, E, 1)
        set_mean = (exp_emb * valid_e).sum(dim=1) / valid_e.sum(dim=1).clamp(min=1.0)
        set_max = exp_emb.masked_fill(exp_pad_mask.unsqueeze(-1), -1e4).max(dim=1).values
        return self.classifier(torch.cat([set_mean, set_max], dim=-1))

    def forward(
        self,
        design_feats: torch.Tensor,
        times: torch.Tensor,
        obs_vals: torch.Tensor,
        obs_masks: torch.Tensor,
        time_pad_mask: torch.Tensor,
        exp_pad_mask: torch.Tensor,
        targets: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        logits = self.forward_logits(design_feats, times, obs_vals, obs_masks, time_pad_mask, exp_pad_mask)
        loss = F.cross_entropy(logits.float(), targets)
        return loss, logits
