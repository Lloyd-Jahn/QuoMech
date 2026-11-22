"""Exact Rational-Quadratic Neural Spline Flow (Durkan et al. 2019) in PyTorch.

Clean, self-contained, numerically stable conditional normalizing flow for continuous
rate-constant posterior inference q(theta | m, D) with exact forward log_prob and exact
analytical inverse sampling. All spline computations run in float32 for precision.
"""

from __future__ import annotations

import math
from typing import Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F


DEFAULT_MIN_BIN_WIDTH = 1e-3
DEFAULT_MIN_BIN_HEIGHT = 1e-3
DEFAULT_MIN_DERIVATIVE = 1e-3


def unconstrained_rational_quadratic_spline(
    inputs: torch.Tensor,
    unnormalized_widths: torch.Tensor,
    unnormalized_heights: torch.Tensor,
    unnormalized_derivatives: torch.Tensor,
    inverse: bool = False,
    tail_bound: float = 5.0,
    min_bin_width: float = DEFAULT_MIN_BIN_WIDTH,
    min_bin_height: float = DEFAULT_MIN_BIN_HEIGHT,
    min_derivative: float = DEFAULT_MIN_DERIVATIVE,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Branchless (zero CPU-GPU sync) piecewise rational-quadratic spline on [-tail_bound, tail_bound]."""
    inside = (inputs >= -tail_bound) & (inputs <= tail_bound)
    x = inputs.clamp(-tail_bound + 1e-6, tail_bound - 1e-6)
    uw = unnormalized_widths
    uh = unnormalized_heights
    ud = unnormalized_derivatives

    num_bins = uw.shape[-1]
    widths = F.softmax(uw, dim=-1)
    widths = min_bin_width + (1.0 - min_bin_width * num_bins) * widths
    cumwidths = torch.cumsum(widths, dim=-1)
    cumwidths = F.pad(cumwidths, (1, 0), value=0.0)
    cumwidths = (2.0 * tail_bound) * cumwidths - tail_bound
    cumwidths[..., 0] = -tail_bound
    cumwidths[..., -1] = tail_bound
    widths = cumwidths[..., 1:] - cumwidths[..., :-1]

    heights = F.softmax(uh, dim=-1)
    heights = min_bin_height + (1.0 - min_bin_height * num_bins) * heights
    cumheights = torch.cumsum(heights, dim=-1)
    cumheights = F.pad(cumheights, (1, 0), value=0.0)
    cumheights = (2.0 * tail_bound) * cumheights - tail_bound
    cumheights[..., 0] = -tail_bound
    cumheights[..., -1] = tail_bound
    heights = cumheights[..., 1:] - cumheights[..., :-1]

    inner_derivs = min_derivative + F.softplus(ud)
    ones = torch.ones_like(inner_derivs[..., :1])
    derivatives = torch.cat([ones, inner_derivs, ones], dim=-1)

    if inverse:
        bin_idx = torch.searchsorted(cumheights.contiguous(), x[..., None].contiguous(), right=True) - 1
    else:
        bin_idx = torch.searchsorted(cumwidths.contiguous(), x[..., None].contiguous(), right=True) - 1
    bin_idx = bin_idx.clamp(0, num_bins - 1)

    input_cumwidths = cumwidths.gather(-1, bin_idx)[..., 0]
    input_bin_widths = widths.gather(-1, bin_idx)[..., 0]
    input_cumheights = cumheights.gather(-1, bin_idx)[..., 0]
    input_bin_heights = heights.gather(-1, bin_idx)[..., 0]
    input_delta = input_bin_heights / input_bin_widths
    input_deriv = derivatives.gather(-1, bin_idx)[..., 0]
    input_deriv_plus_one = derivatives.gather(-1, bin_idx + 1)[..., 0]

    if inverse:
        a = (x - input_cumheights) * (input_deriv + input_deriv_plus_one - 2.0 * input_delta) + input_bin_heights * (input_delta - input_deriv)
        b = input_bin_heights * input_deriv - (x - input_cumheights) * (input_deriv + input_deriv_plus_one - 2.0 * input_delta)
        c = -input_delta * (x - input_cumheights)
        disc = (b * b - 4.0 * a * c).clamp(min=0.0)
        root = (2.0 * c) / (-b - torch.sqrt(disc) - 1e-12)
        root = root.clamp(0.0, 1.0)
        y = root * input_bin_widths + input_cumwidths

        theta_one_minus_theta = root * (1.0 - root)
        denom = input_delta + (input_deriv + input_deriv_plus_one - 2.0 * input_delta) * theta_one_minus_theta
        deriv_num = input_delta * input_delta * (
            input_deriv_plus_one * root * root
            + 2.0 * input_delta * theta_one_minus_theta
            + input_deriv * (1.0 - root) * (1.0 - root)
        )
        lad = -(torch.log(deriv_num.clamp(min=1e-12)) - 2.0 * torch.log(denom.clamp(min=1e-12)))
    else:
        theta = ((x - input_cumwidths) / input_bin_widths).clamp(0.0, 1.0)
        theta_one_minus_theta = theta * (1.0 - theta)
        num = input_bin_heights * (input_delta * theta * theta + input_deriv * theta_one_minus_theta)
        denom = input_delta + (input_deriv + input_deriv_plus_one - 2.0 * input_delta) * theta_one_minus_theta
        y = input_cumheights + num / denom

        deriv_num = input_delta * input_delta * (
            input_deriv_plus_one * theta * theta
            + 2.0 * input_delta * theta_one_minus_theta
            + input_deriv * (1.0 - theta) * (1.0 - theta)
        )
        lad = torch.log(deriv_num.clamp(min=1e-12)) - 2.0 * torch.log(denom.clamp(min=1e-12))

    outputs = torch.where(inside, y, inputs)
    logabsdet = torch.where(inside, lad, torch.zeros_like(lad))
    return outputs, logabsdet


class RQSCouplingLayer(nn.Module):
    """Conditional Rational-Quadratic Spline coupling layer."""

    def __init__(
        self,
        dim: int,
        cond_dim: int,
        mask: torch.Tensor,
        hidden_dim: int = 192,
        num_bins: int = 8,
        tail_bound: float = 5.0,
    ) -> None:
        super().__init__()
        self.dim = dim
        self.num_bins = num_bins
        self.tail_bound = tail_bound
        self.register_buffer("mask", mask.float())
        self.n_trans = int((mask == 0).sum().item())
        self.n_id = int((mask == 1).sum().item())
        self.register_buffer("id_idx", torch.where(mask == 1)[0])
        self.register_buffer("trans_idx", torch.where(mask == 0)[0])

        out_per_dim = 3 * num_bins - 1  # widths (B), heights (B), interior derivs (B-1)
        self.net = nn.Sequential(
            nn.Linear(self.n_id + cond_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, self.n_trans * out_per_dim),
        )
        # Initialize last layer near zero -> near-identity initial flow
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def _get_params(self, x_id: torch.Tensor, cond: torch.Tensor):
        h = self.net(torch.cat([x_id, cond], dim=-1))
        h = h.view(*x_id.shape[:-1], self.n_trans, 3 * self.num_bins - 1)
        uw = h[..., : self.num_bins]
        uh = h[..., self.num_bins : 2 * self.num_bins]
        ud = h[..., 2 * self.num_bins :]
        return uw, uh, ud

    def forward(self, x: torch.Tensor, cond: torch.Tensor, inverse: bool = False) -> Tuple[torch.Tensor, torch.Tensor]:
        x_f = x.float()
        c_f = cond.float()
        x_id = x_f.index_select(-1, self.id_idx)
        x_tr = x_f.index_select(-1, self.trans_idx)
        uw, uh, ud = self._get_params(x_id, c_f)
        y_tr, lad = unconstrained_rational_quadratic_spline(
            x_tr, uw, uh, ud, inverse=inverse, tail_bound=self.tail_bound
        )
        y = x_f.clone()
        y.index_copy_(-1, self.trans_idx, y_tr)
        return y, lad.sum(dim=-1)


class ConditionalNSF(nn.Module):
    """Conditional Neural Spline Flow q(theta | cond) over continuous dim-D vector."""

    def __init__(
        self,
        dim: int,
        cond_dim: int,
        num_layers: int = 6,
        hidden_dim: int = 192,
        num_bins: int = 8,
        tail_bound: float = 5.0,
    ) -> None:
        super().__init__()
        self.dim = dim
        self.cond_dim = cond_dim
        layers = []
        for l in range(num_layers):
            # Alternating / shifted binary masks so every dimension interacts with every other
            mask = torch.zeros(dim, dtype=torch.long)
            if l % 2 == 0:
                mask[::2] = 1
            else:
                mask[1::2] = 1
            if l >= 2 and l % 4 == 2:
                mask = torch.zeros(dim, dtype=torch.long)
                mask[: dim // 2] = 1
            elif l >= 2 and l % 4 == 3:
                mask = torch.zeros(dim, dtype=torch.long)
                mask[dim // 2 :] = 1
            layers.append(
                RQSCouplingLayer(
                    dim=dim,
                    cond_dim=cond_dim,
                    mask=mask,
                    hidden_dim=hidden_dim,
                    num_bins=num_bins,
                    tail_bound=tail_bound,
                )
            )
        self.layers = nn.ModuleList(layers)

    def log_prob(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        """Compute exact log q(x | cond) in float32."""
        z = x.float()
        c = cond.float()
        total_lad = torch.zeros(z.shape[:-1], device=z.device, dtype=torch.float32)
        for layer in self.layers:
            z, lad = layer(z, c, inverse=False)
            total_lad = total_lad + lad
        base_lp = -0.5 * (z * z + math.log(2.0 * math.pi)).sum(dim=-1)
        return base_lp + total_lad

    @torch.no_grad()
    def sample(self, cond: torch.Tensor, num_samples: int = 1) -> torch.Tensor:
        """Sample from q(x | cond). If num_samples==1, returns (*batch, dim); else (num_samples, *batch, dim)."""
        c = cond.float()
        if num_samples == 1:
            z = torch.randn(*c.shape[:-1], self.dim, device=c.device, dtype=torch.float32)
            for layer in reversed(self.layers):
                z, _ = layer(z, c, inverse=True)
            return z
        else:
            c_exp = c.unsqueeze(0).expand(num_samples, *c.shape).reshape(-1, c.shape[-1])
            z = torch.randn(c_exp.shape[0], self.dim, device=c.device, dtype=torch.float32)
            for layer in reversed(self.layers):
                z, _ = layer(z, c_exp, inverse=True)
            return z.view(num_samples, *c.shape[:-1], self.dim)
