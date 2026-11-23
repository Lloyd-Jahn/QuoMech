"""Amortized Bayesian Posterior Networks: QuoMech (Factorized Mechanism Head) & Flat NPE Baseline.

Architecture (Section 7.2):
  1. ExperimentEncoder: 4-layer Transformer (d=192) over prepended continuous design token
     and irregular time-series observation tokens (supporting arbitrary missingness & variable L).
  2. SetEncoder: 2-layer Set Transformer (ISAB + PMA) -> permutation-invariant summary f(D) in R^256.
  3. MechanismHead:
     - QuoMech (factorized): per-module option embeddings -> 2-layer small Transformer -> g(m),
       bilinear/MLP scorer s(D, m) = MLP([f; g(m); f * g(m)]) over all M mechanisms.
     - Flat NPE (baseline): MLP(f(D)) -> M logits directly + learned mechanism table for flow cond.
  4. RateConstantHead: 15D Conditional Rational-Quadratic Neural Spline Flow conditioned on [f; g(m)],
     operating in prior-standardized coordinates so active and N(0,1)-padded inactive slots are O(1).
"""

from __future__ import annotations

import math
from typing import Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F

from quomech.flows import ConditionalNSF


class TimeEncoding(nn.Module):
    """Multi-scale linear + log1p + Fourier encoding for variable-window timestamps t in [0, 4.0]."""

    def __init__(self, out_dim: int, n_freqs: int = 16, t_max: float = 4.0) -> None:
        super().__init__()
        self.t_max = t_max
        freqs = torch.exp(torch.linspace(0.0, math.log(64.0), n_freqs))
        self.register_buffer("freqs", freqs)
        self.proj = nn.Linear(2 + 2 * n_freqs, out_dim)

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        # t: (..., 1) in [0, t_max]
        t_norm = t / self.t_max
        t_log = torch.log1p(t.clamp(min=0.0))
        ang = t_norm * self.freqs * math.pi
        feats = torch.cat([t_norm, t_log, torch.sin(ang), torch.cos(ang)], dim=-1)
        return self.proj(feats)


class FastTransformerBlock(nn.Module):
    """Fused SDPA Pre-LN Transformer block with QK-Norm for long-horizon BF16 stability."""

    def __init__(self, d_model: int = 192, n_heads: int = 6, dropout: float = 0.0) -> None:
        super().__init__()
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        self.ln1 = nn.LayerNorm(d_model)
        self.qkv = nn.Linear(d_model, 3 * d_model)
        self.q_norm = nn.LayerNorm(self.d_head)
        self.k_norm = nn.LayerNorm(self.d_head)
        self.proj = nn.Linear(d_model, d_model)
        self.ln2 = nn.LayerNorm(d_model)
        self.ff = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.GELU(),
            nn.Linear(d_model * 4, d_model),
        )

    def forward(self, x: torch.Tensor, attn_bias: torch.Tensor) -> torch.Tensor:
        N, S, D = x.shape
        qkv = self.qkv(self.ln1(x)).view(N, S, 3, self.n_heads, self.d_head).permute(2, 0, 3, 1, 4)
        q = self.q_norm(qkv[0]).to(qkv.dtype)
        k = self.k_norm(qkv[1]).to(qkv.dtype)
        v = qkv[2]
        bias = attn_bias.to(qkv.dtype) if attn_bias.dtype != qkv.dtype else attn_bias
        out = F.scaled_dot_product_attention(q, k, v, attn_mask=bias)
        x = x + self.proj(out.transpose(1, 2).reshape(N, S, D))
        x = x + self.ff(self.ln2(x))
        return x


class ExperimentEncoder(nn.Module):
    """4-layer Pre-LN Transformer encoding a single experiment (design token + L irregular obs tokens)."""

    def __init__(
        self,
        design_dim: int,
        n_obs_channels: int,
        d_model: int = 192,
        n_heads: int = 6,
        n_layers: int = 4,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.d_model = d_model
        self.design_proj = nn.Sequential(
            nn.Linear(design_dim, d_model),
            nn.SiLU(),
            nn.Linear(d_model, d_model),
        )
        self.time_enc = TimeEncoding(d_model // 2)
        self.obs_proj = nn.Sequential(
            nn.Linear(2 * n_obs_channels, d_model // 2),
            nn.SiLU(),
            nn.Linear(d_model // 2, d_model // 2),
        )
        self.tok_fuse = nn.Linear(d_model, d_model)
        self.type_emb = nn.Embedding(2, d_model)  # 0 = design token, 1 = obs token
        self.layers = nn.ModuleList([FastTransformerBlock(d_model, n_heads, dropout) for _ in range(n_layers)])
        self.out_norm = nn.LayerNorm(d_model)

    def forward(
        self,
        design_feat: torch.Tensor,  # (N, design_dim)
        times: torch.Tensor,        # (N, L)
        obs_vals: torch.Tensor,     # (N, L, C) normalized values (0 where masked)
        obs_masks: torch.Tensor,    # (N, L, C) binary channel mask
        time_pad_mask: torch.Tensor,# (N, L) bool, True where time slot is PADDED (invalid)
    ) -> torch.Tensor:
        N, L, _ = obs_vals.shape
        device = design_feat.device

        des_tok = self.design_proj(design_feat).unsqueeze(1)  # (N, 1, d)
        des_tok = des_tok + self.type_emb(torch.zeros((1, 1), dtype=torch.long, device=device))

        t_emb = self.time_enc(times.unsqueeze(-1))  # (N, L, d/2)
        y_in = torch.cat([obs_vals * obs_masks, obs_masks], dim=-1)
        y_emb = self.obs_proj(y_in)  # (N, L, d/2)
        obs_tok = self.tok_fuse(torch.cat([t_emb, y_emb], dim=-1))
        obs_tok = obs_tok + self.type_emb(torch.ones((1, 1), dtype=torch.long, device=device))

        seq = torch.cat([des_tok, obs_tok], dim=1)  # (N, 1+L, d)
        pad_full = torch.cat(
            [torch.zeros((N, 1), dtype=torch.bool, device=device), time_pad_mask], dim=1
        )
        attn_bias = torch.zeros((N, 1, 1, 1 + L), device=device, dtype=seq.dtype).masked_fill(
            pad_full.view(N, 1, 1, 1 + L), -1e4
        )
        h = seq
        for layer in self.layers:
            h = layer(h, attn_bias)
        h = self.out_norm(h)
        des_out = h[:, 0]
        valid = (~time_pad_mask).to(h.dtype).unsqueeze(-1)
        obs_mean = (h[:, 1:] * valid).sum(dim=1) / valid.sum(dim=1).clamp(min=1.0)
        return 0.5 * (des_out + obs_mean)


class MAB(nn.Module):
    """Fused SDPA Pre-LN Cross/Self Multihead Attention Block with KV-LayerNorm & QK-Norm."""

    def __init__(self, d_model: int, n_heads: int = 8) -> None:
        super().__init__()
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        self.q_proj = nn.Linear(d_model, d_model)
        self.kv_proj = nn.Linear(d_model, 2 * d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        self.ln1 = nn.LayerNorm(d_model)
        self.ln_kv = nn.LayerNorm(d_model)
        self.q_norm = nn.LayerNorm(self.d_head)
        self.k_norm = nn.LayerNorm(self.d_head)
        self.ln2 = nn.LayerNorm(d_model)
        self.ff = nn.Sequential(
            nn.Linear(d_model, d_model * 2),
            nn.GELU(),
            nn.Linear(d_model * 2, d_model),
        )

    def forward(
        self,
        X: torch.Tensor,
        Y: torch.Tensor,
        key_padding_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        B, Sq, D = X.shape
        Sk = Y.shape[1]
        q_in = self.ln1(X)
        kv_in = self.ln1(Y) if X is Y else self.ln_kv(Y)
        q = self.q_proj(q_in).view(B, Sq, self.n_heads, self.d_head).transpose(1, 2)
        kv = self.kv_proj(kv_in).view(B, Sk, 2, self.n_heads, self.d_head).permute(2, 0, 3, 1, 4)
        q = self.q_norm(q).to(kv.dtype)
        k = self.k_norm(kv[0]).to(kv.dtype)
        v = kv[1]
        attn_bias = None
        if key_padding_mask is not None:
            attn_bias = torch.zeros((B, 1, 1, Sk), device=X.device, dtype=q.dtype).masked_fill(
                key_padding_mask.view(B, 1, 1, Sk), -1e4
            )
        out = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_bias)
        H = X + self.out_proj(out.transpose(1, 2).reshape(B, Sq, D))
        H = H + self.ff(self.ln2(H))
        return H


class ISAB(nn.Module):
    """Induced Set Attention Block (strictly permutation-equivariant)."""

    def __init__(self, d_model: int, n_heads: int = 8, num_inducing: int = 16) -> None:
        super().__init__()
        self.I = nn.Parameter(torch.randn(1, num_inducing, d_model) * (d_model ** -0.5))
        self.mab1 = MAB(d_model, n_heads)
        self.mab2 = MAB(d_model, n_heads)

    def forward(self, X: torch.Tensor, exp_pad_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        B = X.shape[0]
        I = self.I.expand(B, -1, -1)
        H = self.mab1(I, X, key_padding_mask=exp_pad_mask)
        return self.mab2(X, H, key_padding_mask=None)


class PMA(nn.Module):
    """Pooling by Multihead Attention (strictly permutation-invariant)."""

    def __init__(self, d_model: int, n_heads: int = 8, num_seeds: int = 4) -> None:
        super().__init__()
        self.S = nn.Parameter(torch.randn(1, num_seeds, d_model) * (d_model ** -0.5))
        self.mab = MAB(d_model, n_heads)

    def forward(self, X: torch.Tensor, exp_pad_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        B = X.shape[0]
        S = self.S.expand(B, -1, -1)
        return self.mab(S, X, key_padding_mask=exp_pad_mask)


class SetEncoder(nn.Module):
    """2-layer Set Transformer (2x ISAB + PMA) outputting permutation-invariant f(D) in R^d_out."""

    def __init__(self, d_in: int = 192, d_set: int = 256, n_heads: int = 8, num_inducing: int = 16) -> None:
        super().__init__()
        self.in_proj = nn.Linear(d_in, d_set)
        self.isab1 = ISAB(d_set, n_heads=n_heads, num_inducing=num_inducing)
        self.isab2 = ISAB(d_set, n_heads=n_heads, num_inducing=num_inducing)
        self.pma = PMA(d_set, n_heads=n_heads, num_seeds=4)
        self.out_proj = nn.Sequential(
            nn.LayerNorm(d_set * 4),
            nn.Linear(d_set * 4, d_set),
            nn.GELU(),
            nn.Linear(d_set, d_set),
            nn.LayerNorm(d_set),
        )

    def forward(self, exp_embs: torch.Tensor, exp_pad_mask: torch.Tensor) -> torch.Tensor:
        # exp_embs: (B, E, d_in), exp_pad_mask: (B, E) True where padded
        H = self.in_proj(exp_embs)
        H = self.isab1(H, exp_pad_mask=exp_pad_mask)
        H = self.isab2(H, exp_pad_mask=exp_pad_mask)
        P = self.pma(H, exp_pad_mask=exp_pad_mask)  # (B, 4, d_set)
        return self.out_proj(P.reshape(P.shape[0], -1))


class FactorizedMechanismHead(nn.Module):
    """Factorized compositional mechanism encoder & scorer (QuoMech).

    Encodes each mechanism m from its discrete module option vector via per-module
    embeddings + 2-layer Transformer -> g(m) in R^d_g, then scores s(D, m) = MLP([f; g; f * g]).
    """

    def __init__(
        self,
        mech_options: torch.Tensor,  # (M, n_modules) long
        module_num_options: Tuple[int, ...],
        d_f: int = 256,
        d_g: int = 128,
    ) -> None:
        super().__init__()
        self.register_buffer("mech_options", mech_options.long())
        self.n_mechs, self.n_modules = mech_options.shape
        self.d_f = d_f
        self.d_g = d_g

        self.mod_embs = nn.ModuleList([nn.Embedding(n_opt, d_g) for n_opt in module_num_options])
        self.mod_pos = nn.Parameter(torch.randn(1, self.n_modules, d_g) * (d_g ** -0.5))
        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_g, nhead=4, dim_feedforward=d_g * 2,
            dropout=0.0, activation="gelu", batch_first=True, norm_first=True
        )
        self.mod_tf = nn.TransformerEncoder(enc_layer, num_layers=2, enable_nested_tensor=False)
        self.g_norm = nn.LayerNorm(d_g)

        self.f_to_g = nn.Linear(d_f, d_g)
        # Exact factorization of first layer Linear([f; g; f_g * g], 256) to avoid 151x redundant f-matmul
        self.w_f = nn.Linear(d_f, 256)
        self.w_g = nn.Linear(d_g, 256, bias=False)
        self.w_p = nn.Linear(d_g, 256, bias=False)
        self.scorer_tail = nn.Sequential(
            nn.SiLU(),
            nn.Linear(256, 128),
            nn.SiLU(),
            nn.Linear(128, 1),
        )

    def encode_all_mechanisms(self) -> torch.Tensor:
        # Returns G: (M, d_g)
        toks = torch.stack(
            [self.mod_embs[j](self.mech_options[:, j]) for j in range(self.n_modules)],
            dim=1,
        )  # (M, n_modules, d_g)
        toks = toks + self.mod_pos
        h = self.mod_tf(toks)
        return self.g_norm(h.mean(dim=1))

    def forward(self, f: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """f: (B, d_f) -> logits: (B, M), G: (M, d_g)."""
        G = self.encode_all_mechanisms()  # (M, d_g)
        fg = self.f_to_g(f)               # (B, d_g)
        prod = fg.unsqueeze(1) * G.unsqueeze(0)  # (B, M, d_g)
        h = self.w_f(f).unsqueeze(1) + self.w_g(G).unsqueeze(0) + self.w_p(prod)
        logits = self.scorer_tail(h).squeeze(-1)  # (B, M)
        return logits, G


class FlatMechanismHead(nn.Module):
    """Flat softmax mechanism classification head (Esquivel et al.-style baseline)."""

    def __init__(self, n_mechs: int, d_f: int = 256, d_g: int = 128) -> None:
        super().__init__()
        self.n_mechs = n_mechs
        self.d_g = d_g
        self.classifier = nn.Sequential(
            nn.Linear(d_f, 256),
            nn.SiLU(),
            nn.Linear(256, n_mechs),
        )
        self.mech_emb = nn.Embedding(n_mechs, d_g)

    def encode_all_mechanisms(self) -> torch.Tensor:
        return self.mech_emb.weight

    def forward(self, f: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        logits = self.classifier(f)
        return logits, self.mech_emb.weight


class QuoMechNet(nn.Module):
    """Complete Amortized Posterior Network q(m, theta | D) = q(m | D) q(theta | m, D)."""

    def __init__(
        self,
        design_dim: int,
        n_obs_channels: int,
        n_params: int,
        mech_options: torch.Tensor,
        module_num_options: Tuple[int, ...],
        mech_prior_mean: torch.Tensor,  # (M, n_params)
        mech_prior_std: torch.Tensor,   # (M, n_params)
        mech_param_mask: torch.Tensor,  # (M, n_params)
        head_type: str = "factorized",  # "factorized" (QuoMech) or "flat" (baseline)
        d_exp: int = 192,
        d_set: int = 256,
        d_g: int = 128,
        nsf_layers: int = 6,
        nsf_hidden: int = 192,
        nsf_bins: int = 8,
    ) -> None:
        super().__init__()
        self.head_type = head_type
        self.n_params = n_params
        self.n_mechs = mech_options.shape[0]
        self.register_buffer("prior_mean", mech_prior_mean.float())
        self.register_buffer("prior_std", mech_prior_std.float())
        self.register_buffer("param_mask", mech_param_mask.float())

        self.exp_encoder = ExperimentEncoder(
            design_dim=design_dim,
            n_obs_channels=n_obs_channels,
            d_model=d_exp,
            n_heads=6,
            n_layers=4,
        )
        self.set_encoder = SetEncoder(d_in=d_exp, d_set=d_set, n_heads=8, num_inducing=16)
        if head_type == "factorized":
            self.mech_head = FactorizedMechanismHead(
                mech_options=mech_options,
                module_num_options=module_num_options,
                d_f=d_set,
                d_g=d_g,
            )
        elif head_type == "flat":
            self.mech_head = FlatMechanismHead(n_mechs=self.n_mechs, d_f=d_set, d_g=d_g)
        else:
            raise ValueError(f"Unknown head_type: {head_type}")

        self.flow = ConditionalNSF(
            dim=n_params,
            cond_dim=d_set + d_g,
            num_layers=nsf_layers,
            hidden_dim=nsf_hidden,
            num_bins=nsf_bins,
            tail_bound=5.0,
        )

    def encode_dataset(
        self,
        design_feats: torch.Tensor,   # (B, E, design_dim)
        times: torch.Tensor,          # (B, E, L)
        obs_vals: torch.Tensor,       # (B, E, L, C)
        obs_masks: torch.Tensor,      # (B, E, L, C)
        time_pad_mask: torch.Tensor,  # (B, E, L) bool (True=padded)
        exp_pad_mask: torch.Tensor,   # (B, E) bool (True=padded)
    ) -> torch.Tensor:
        B, E, L, C = obs_vals.shape
        # Branchless (zero CPU-GPU sync): design CLS token at pos 0 is always unpadded so attention is finite
        enc = self.exp_encoder(
            design_feats.reshape(B * E, -1),
            times.reshape(B * E, L),
            obs_vals.reshape(B * E, L, C),
            obs_masks.reshape(B * E, L, C),
            time_pad_mask.reshape(B * E, L),
        ).view(B, E, -1)
        exp_embs = enc * (~exp_pad_mask).to(enc.dtype).unsqueeze(-1)
        return self.set_encoder(exp_embs, exp_pad_mask=exp_pad_mask)

    def standardize_theta(self, theta: torch.Tensor, mech_ids: torch.Tensor) -> torch.Tensor:
        """Map active theta slots to (theta - mu)/std; leave inactive slots (already N(0,1)) unchanged."""
        mu = self.prior_mean[mech_ids]
        sd = self.prior_std[mech_ids]
        mask = self.param_mask[mech_ids]
        return mask * ((theta - mu) / sd) + (1.0 - mask) * theta

    def unstandardize_theta(self, z_theta: torch.Tensor, mech_ids: torch.Tensor) -> torch.Tensor:
        mu = self.prior_mean[mech_ids]
        sd = self.prior_std[mech_ids]
        mask = self.param_mask[mech_ids]
        return mask * (mu + sd * z_theta) + (1.0 - mask) * z_theta

    def forward(
        self,
        design_feats: torch.Tensor,
        times: torch.Tensor,
        obs_vals: torch.Tensor,
        obs_masks: torch.Tensor,
        time_pad_mask: torch.Tensor,
        exp_pad_mask: torch.Tensor,
        mech_ids: torch.Tensor,
        thetas: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """DDP-compatible forward alias calling forward_loss."""
        return self.forward_loss(
            design_feats, times, obs_vals, obs_masks, time_pad_mask, exp_pad_mask, mech_ids, thetas
        )

    def forward_loss(
        self,
        design_feats: torch.Tensor,
        times: torch.Tensor,
        obs_vals: torch.Tensor,
        obs_masks: torch.Tensor,
        time_pad_mask: torch.Tensor,
        exp_pad_mask: torch.Tensor,
        mech_ids: torch.Tensor,
        thetas: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Joint NLL loss = -log q(m | D) - log q(z_theta | m, D)."""
        f = self.encode_dataset(design_feats, times, obs_vals, obs_masks, time_pad_mask, exp_pad_mask)
        logits, G = self.mech_head(f)
        loss_m = F.cross_entropy(logits.float(), mech_ids)

        g_m = G[mech_ids]  # (B, d_g)
        cond = torch.cat([f.float(), g_m.float()], dim=-1)
        z_th = self.standardize_theta(thetas.float(), mech_ids)
        # Replace inactive slots with fresh N(0,1) noise so flow never memorizes dummy constants
        mask = self.param_mask[mech_ids]
        z_th = mask * z_th + (1.0 - mask) * torch.randn_like(z_th)
        lp_th = self.flow.log_prob(z_th, cond)
        loss_th = -lp_th.mean()
        return loss_m + loss_th, loss_m, loss_th, logits

    @torch.no_grad()
    def predict_posterior(
        self,
        design_feats: torch.Tensor,
        times: torch.Tensor,
        obs_vals: torch.Tensor,
        obs_masks: torch.Tensor,
        time_pad_mask: torch.Tensor,
        exp_pad_mask: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return (probs (B, M), f (B, d_set), G (M, d_g))."""
        f = self.encode_dataset(design_feats, times, obs_vals, obs_masks, time_pad_mask, exp_pad_mask)
        logits, G = self.mech_head(f)
        probs = F.softmax(logits.float(), dim=-1)
        return probs, f.float(), G.float()

    @torch.no_grad()
    def encode_experiments_only(
        self,
        design_feats: torch.Tensor,   # (B, E, design_dim)
        times: torch.Tensor,          # (B, E, L)
        obs_vals: torch.Tensor,       # (B, E, L, C)
        obs_masks: torch.Tensor,      # (B, E, L, C)
        time_pad_mask: torch.Tensor,  # (B, E, L)
        exp_pad_mask: torch.Tensor,   # (B, E)
    ) -> torch.Tensor:
        """Encode valid experiments without running SetEncoder yet -> (B, E, d_exp)."""
        B, E, L, C = obs_vals.shape
        valid_exp = ~exp_pad_mask
        flat_valid = valid_exp.reshape(-1)
        d_out = self.exp_encoder.d_model
        exp_embs_flat = torch.zeros(B * E, d_out, device=obs_vals.device, dtype=design_feats.dtype)
        if torch.any(flat_valid):
            enc_v = self.exp_encoder(
                design_feats.reshape(B * E, -1)[flat_valid],
                times.reshape(B * E, L)[flat_valid],
                obs_vals.reshape(B * E, L, C)[flat_valid],
                obs_masks.reshape(B * E, L, C)[flat_valid],
                time_pad_mask.reshape(B * E, L)[flat_valid],
            )
            exp_embs_flat[flat_valid] = enc_v.to(exp_embs_flat.dtype)
        return exp_embs_flat.view(B, E, d_out)

    @torch.no_grad()
    def probs_from_exp_embs(
        self,
        exp_embs: torch.Tensor,       # (B, E, d_exp)
        exp_pad_mask: torch.Tensor,   # (B, E)
    ) -> torch.Tensor:
        """Run SetEncoder + MechHead from pre-encoded experiment embeddings -> probs (B, M)."""
        f = self.set_encoder(exp_embs, exp_pad_mask=exp_pad_mask)
        logits, _ = self.mech_head(f)
        return F.softmax(logits.float(), dim=-1)

    @torch.no_grad()
    def sample_theta(
        self,
        f: torch.Tensor,      # (B, d_set)
        G: torch.Tensor,      # (M, d_g)
        mech_ids: torch.Tensor,  # (B,) or (N,)
        num_samples: int = 1,
    ) -> torch.Tensor:
        """Sample unstandardized log10_k ~ q(theta | mech_ids, f)."""
        g_m = G[mech_ids]
        cond = torch.cat([f, g_m], dim=-1)
        z_samp = self.flow.sample(cond, num_samples=num_samples)
        if num_samples == 1:
            return self.unstandardize_theta(z_samp, mech_ids)
        else:
            # z_samp: (S, B, P)
            mu = self.prior_mean[mech_ids].unsqueeze(0)
            sd = self.prior_std[mech_ids].unsqueeze(0)
            mask = self.param_mask[mech_ids].unsqueeze(0)
            return mask * (mu + sd * z_samp) + (1.0 - mask) * z_samp
