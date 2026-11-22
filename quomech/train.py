"""PyTorch BF16/FP32 single-GPU and multi-GPU DDP training loop for QuoMech & Flat NPE."""

from __future__ import annotations

import json
import math
import os
import time
from pathlib import Path
from typing import Dict, Optional
import numpy as np
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

from quomech.data_gen import OnlineKineticsDataset, get_observable_channel_indices
from quomech.grammar import ReactionGrammar
from quomech.models import QuoMechNet


def worker_init_fn(worker_id: int) -> None:
    seed = (torch.initial_seed() + worker_id * 10007) % (2 ** 32 - 1)
    np.random.seed(seed)


def compute_ece(probs: np.ndarray, targets: np.ndarray, n_bins: int = 15) -> float:
    """Top-label Expected Calibration Error (ECE)."""
    conf = np.max(probs, axis=1)
    preds = np.argmax(probs, axis=1)
    acc = (preds == targets).astype(np.float64)
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    n = len(targets)
    for b in range(n_bins):
        lo, hi = edges[b], edges[b + 1]
        mask = (conf >= lo) & (conf <= hi if b == n_bins - 1 else conf < hi)
        if np.any(mask):
            ece += (np.sum(mask) / n) * abs(float(np.mean(acc[mask])) - float(np.mean(conf[mask])))
    return float(ece)


@torch.no_grad()
def evaluate_loader(
    model: QuoMechNet,
    loader: DataLoader,
    device: torch.device,
    use_bf16: bool = True,
    max_batches: Optional[int] = None,
) -> Dict[str, float]:
    model.eval()
    tot_loss = 0.0
    tot_lm = 0.0
    tot_lth = 0.0
    n_seen = 0
    all_probs = []
    all_mids = []

    for b_idx, batch in enumerate(loader):
        if max_batches is not None and b_idx >= max_batches:
            break
        df, tm, ov, om, tp, ep, mids, ths = [x.to(device, non_blocking=True) for x in batch]
        with torch.cuda.amp.autocast(enabled=use_bf16, dtype=torch.bfloat16):
            loss, lm, lth, logits = model.forward_loss(df, tm, ov, om, tp, ep, mids, ths)
        bs = mids.shape[0]
        tot_loss += float(loss.item()) * bs
        tot_lm += float(lm.item()) * bs
        tot_lth += float(lth.item()) * bs
        n_seen += bs
        all_probs.append(torch.softmax(logits.float(), dim=-1).cpu().numpy())
        all_mids.append(mids.cpu().numpy())

    probs = np.concatenate(all_probs, axis=0)
    mids = np.concatenate(all_mids, axis=0)
    top1 = float(np.mean(np.argmax(probs, axis=1) == mids))
    top3_idx = np.argpartition(probs, -3, axis=1)[:, -3:]
    top3 = float(np.mean(np.any(top3_idx == mids[:, None], axis=1)))
    ece = compute_ece(probs, mids)

    return {
        "loss": tot_loss / n_seen,
        "loss_m": tot_lm / n_seen,
        "loss_th": tot_lth / n_seen,
        "mech_top1": top1,
        "mech_top3": top3,
        "mech_ece": ece,
        "n_eval": n_seen,
    }


def build_model(grammar: ReactionGrammar, head_type: str) -> QuoMechNet:
    obs_ch = get_observable_channel_indices(grammar)
    return QuoMechNet(
        design_dim=6 + grammar.n_species,  # [A0, B0, P0, log10(cat0), a_pre, log10(t_end), obs_mask...]
        n_obs_channels=len(obs_ch),
        n_params=grammar.n_params,
        mech_options=torch.tensor(grammar.mech_options, dtype=torch.long),
        module_num_options=tuple(len(m.options) for m in grammar.modules),
        mech_prior_mean=torch.tensor(grammar.mech_prior_mean, dtype=torch.float32),
        mech_prior_std=torch.tensor(grammar.mech_prior_std, dtype=torch.float32),
        mech_param_mask=torch.tensor(grammar.mech_param_mask, dtype=torch.float32),
        head_type=head_type,
        d_exp=192,
        d_set=256,
        d_g=128,
        nsf_layers=6,
        nsf_hidden=192,
        nsf_bins=8,
    )


def train_model(
    grammar: ReactionGrammar,
    data_dir: Path,
    out_dir: Path,
    head_type: str = "factorized",
    gpu_id: int = 0,
    total_steps: int = 35000,
    batch_size: int = 128,
    grad_accum: int = 1,
    lr: float = 6e-4,
    weight_decay: float = 1e-4,
    warmup_steps: int = 1500,
    eval_every: int = 2500,
    num_workers: int = 6,
    seed: int = 42,
) -> Dict:
    # Auto-detect torchrun DDP environment
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", str(gpu_id)))
    is_ddp = world_size > 1
    is_main = (rank == 0)

    if is_ddp:
        dist.init_process_group(backend="nccl")
        torch.cuda.set_device(local_rank)
        device = torch.device(f"cuda:{local_rank}")
    else:
        device = torch.device(f"cuda:{gpu_id}" if torch.cuda.is_available() else "cpu")

    if is_main:
        out_dir.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(seed + rank)
    np.random.seed(seed + rank)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    use_bf16 = torch.cuda.is_available() and torch.cuda.is_bf16_supported()

    if is_main:
        print(
            f"[Train:{head_type}] DDP={is_ddp} world={world_size} dev={device} bf16={use_bf16} "
            f"eff_bs={batch_size * grad_accum * world_size}",
            flush=True,
        )

    # Use mmap in DDP so 4 ranks share OS page cache cleanly (zero 4x RAM duplication!)
    train_ds = OnlineKineticsDataset(grammar, data_dir / "train", max_exps=8, max_len=30, in_memory=(not is_ddp))
    val_ds = OnlineKineticsDataset(grammar, data_dir / "val", max_exps=8, max_len=30, in_memory=True)

    train_sampler = DistributedSampler(train_ds, num_replicas=world_size, rank=rank, shuffle=True, seed=seed) if is_ddp else None
    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=(train_sampler is None),
        sampler=train_sampler,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
        persistent_workers=(num_workers > 0),
        worker_init_fn=worker_init_fn,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=256,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
        persistent_workers=True,
        worker_init_fn=worker_init_fn,
    ) if is_main else None

    raw_model = build_model(grammar, head_type=head_type).to(device)
    n_params_model = sum(p.numel() for p in raw_model.parameters() if p.requires_grad)
    if is_main:
        print(f"[Train:{head_type}] Model parameters: {n_params_model:,}", flush=True)

    model = DDP(raw_model, device_ids=[local_rank], output_device=local_rank, find_unused_parameters=False) if is_ddp else raw_model
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    def lr_lambda(step: int) -> float:
        if step < warmup_steps:
            return float(step + 1) / float(max(1, warmup_steps))
        prog = float(step - warmup_steps) / float(max(1, total_steps - warmup_steps))
        return 0.05 + 0.95 * 0.5 * (1.0 + math.cos(math.pi * min(1.0, prog)))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    step = 0
    epoch = 0
    micro = 0
    best_val_loss = float("inf")
    history = []
    t0 = time.perf_counter()
    t_log = t0
    run_lm = 0.0
    run_lth = 0.0
    run_cnt = 0

    model.train()
    optimizer.zero_grad(set_to_none=True)
    while step < total_steps:
        if train_sampler is not None:
            train_sampler.set_epoch(epoch)
        epoch += 1
        for batch in train_loader:
            if step >= total_steps:
                break
            df, tm, ov, om, tp, ep, mids, ths = [x.to(device, non_blocking=True) for x in batch]
            with torch.cuda.amp.autocast(enabled=use_bf16, dtype=torch.bfloat16):
                # Call forward_loss via module or DDP forward wrapper
                if is_ddp:
                    loss, lm, lth, _ = model(df, tm, ov, om, tp, ep, mids, ths)
                else:
                    loss, lm, lth, _ = model.forward_loss(df, tm, ov, om, tp, ep, mids, ths)

            if not torch.isfinite(loss):
                if is_main:
                    print(f"[Train:{head_type}] WARNING non-finite loss at step {step}", flush=True)
                optimizer.zero_grad(set_to_none=True)
                micro = 0
                continue

            (loss / grad_accum).backward()
            run_lm += float(lm.item())
            run_lth += float(lth.item())
            run_cnt += 1
            micro += 1

            if micro < grad_accum:
                continue

            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=2.0)
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            scheduler.step()
            micro = 0
            step += 1

            if is_main and (step % 500 == 0 or step == 1):
                dt = time.perf_counter() - t_log
                sps = (500 / dt) if step >= 500 else (1 / max(dt, 1e-6))
                print(
                    f"[Train:{head_type}] step {step:6d}/{total_steps} | "
                    f"loss_m={run_lm/run_cnt:.4f} loss_th={run_lth/run_cnt:.4f} | "
                    f"lr={scheduler.get_last_lr()[0]:.2e} | {sps:.1f} step/s | elapsed={time.perf_counter()-t0:.0f}s",
                    flush=True,
                )
                run_lm, run_lth, run_cnt = 0.0, 0.0, 0
                t_log = time.perf_counter()

            if step % eval_every == 0 or step == total_steps:
                if is_main:
                    val_metrics = evaluate_loader(raw_model, val_loader, device, use_bf16=use_bf16, max_batches=30)
                    val_metrics["step"] = step
                    val_metrics["elapsed_s"] = time.perf_counter() - t0
                    history.append(val_metrics)
                    print(
                        f"  [Val:{head_type} @ {step}] loss={val_metrics['loss']:.4f} "
                        f"(m={val_metrics['loss_m']:.4f}, th={val_metrics['loss_th']:.4f}) | "
                        f"top1={val_metrics['mech_top1']:.3f} top3={val_metrics['mech_top3']:.3f} "
                        f"ece={val_metrics['mech_ece']:.4f}",
                        flush=True,
                    )
                    if val_metrics["loss"] < best_val_loss:
                        best_val_loss = val_metrics["loss"]
                        torch.save(
                            {
                                "step": step,
                                "head_type": head_type,
                                "model_state": raw_model.state_dict(),
                                "val_metrics": val_metrics,
                            },
                            out_dir / "best.pt",
                        )
                    model.train()
                if is_ddp:
                    dist.barrier()

    summary = {}
    if is_main:
        torch.save(
            {"step": step, "head_type": head_type, "model_state": raw_model.state_dict()},
            out_dir / "last.pt",
        )
        summary = {
            "head_type": head_type,
            "seed": seed,
            "total_steps": total_steps,
            "batch_size_per_gpu": batch_size,
            "grad_accum": grad_accum,
            "world_size": world_size,
            "effective_batch_size": batch_size * grad_accum * world_size,
            "n_params": n_params_model,
            "best_val_loss": best_val_loss,
            "elapsed_s": time.perf_counter() - t0,
            "history": history,
        }
        with open(out_dir / "train_summary.json", "w") as f:
            json.dump(summary, f, indent=2)

    if is_ddp:
        dist.destroy_process_group()
    return summary
