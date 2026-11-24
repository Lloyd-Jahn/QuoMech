#!/usr/bin/env python3
"""Phase 2 Stage 2b: Train Burés & Larrosa (Nature 2023) closed-set 1D-CNN classifier
in both protocols across 3 random seeds [42, 43, 44]:
  1) --protocol full527 : 527-class closed-set classifier on data/pool_multicore_527
  2) --protocol bl20    : 20-class Nature 2023 mappable BL-20 subset classifier
"""

from __future__ import annotations

import argparse
import json
import math
import os
import time
from pathlib import Path
import numpy as np
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

from quomech.baselines import BuresLarrosaClassifier, get_bl20_mappable_mech_ids
from quomech.data_gen import get_observable_channel_indices
from quomech.grammar import build_multicore_grammar
from quomech.train import GPUKineticBatchSampler, compute_ece


@torch.no_grad()
def eval_bl(model, sampler, id_remap_t, use_bf16=True, n_batches=30, batch_size=256, eval_seed=54321):
    model.eval()
    dev = sampler.device
    tot_loss = 0.0
    n_seen = 0
    all_probs, all_tgts = [], []
    ctx = torch.random.fork_rng(devices=[dev] if dev.type == "cuda" else [])
    with ctx:
        torch.manual_seed(eval_seed)
        if dev.type == "cuda":
            torch.cuda.manual_seed(eval_seed)
        for _ in range(n_batches):
            df, tm, ov, om, tp, ep, mids, _ = sampler.sample_batch(batch_size)
            tgts = id_remap_t[mids] if id_remap_t is not None else mids
            with torch.cuda.amp.autocast(enabled=use_bf16, dtype=torch.bfloat16):
                loss, logits = model(df, tm, ov, om, tp, ep, tgts)
            bs = tgts.shape[0]
            tot_loss += float(loss.item()) * bs
            n_seen += bs
            all_probs.append(torch.softmax(logits.float(), dim=-1).cpu().numpy())
            all_tgts.append(tgts.cpu().numpy())
    probs = np.concatenate(all_probs, axis=0)
    tgts_np = np.concatenate(all_tgts, axis=0)
    top1 = float(np.mean(np.argmax(probs, axis=1) == tgts_np))
    k = min(3, probs.shape[1])
    top3 = float(np.mean(np.any(np.argpartition(probs, -k, axis=1)[:, -k:] == tgts_np[:, None], axis=1)))
    ece = compute_ece(probs, tgts_np)
    return {"loss_m": tot_loss / n_seen, "top1": top1, "top3": top3, "ece": ece}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--protocol", type=str, required=True, choices=["full527", "bl20"])
    ap.add_argument("--data-dir", type=str, default="data/pool_multicore_527")
    ap.add_argument("--out-dir", type=str, required=True)
    ap.add_argument("--steps", type=int, default=40000)
    ap.add_argument("--bs", type=int, default=512)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--warmup", type=int, default=1500)
    ap.add_argument("--eval-every", type=int, default=2500)
    ap.add_argument("--hidden", type=int, default=256)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--resume", type=str, default="auto")
    args = ap.parse_args()

    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    is_ddp = world_size > 1
    is_main = (rank == 0)

    if is_ddp:
        dist.init_process_group(backend="nccl")
        torch.cuda.set_device(local_rank)
        device = torch.device(f"cuda:{local_rank}")
    else:
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    out_dir = Path(args.out_dir)
    if is_main:
        out_dir.mkdir(parents=True, exist_ok=True)

    torch.manual_seed(args.seed + rank * 1009)
    np.random.seed(args.seed + rank * 1009)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    use_bf16 = torch.cuda.is_available() and torch.cuda.is_bf16_supported()

    g = build_multicore_grammar()
    obs_ch = get_observable_channel_indices(g)

    allowed_ids = None
    id_remap_t = None
    if args.protocol == "bl20":
        allowed_ids = get_bl20_mappable_mech_ids(g)
        n_classes = len(allowed_ids)
        remap = np.full(g.n_mechs, -1, dtype=np.int64)
        for cls_i, m_id in enumerate(allowed_ids):
            remap[m_id] = cls_i
        id_remap_t = torch.from_numpy(remap).to(device)
    else:
        n_classes = g.n_mechs

    data_dir = Path(args.data_dir)
    train_sampler = GPUKineticBatchSampler(
        g, data_dir / "train", device=device, max_exps=8, max_len=30,
        shard_rank=rank, shard_world=world_size, allowed_mech_ids=allowed_ids,
    )
    val_sampler = GPUKineticBatchSampler(
        g, data_dir / "val", device=device, max_exps=8, max_len=30,
        shard_rank=0, shard_world=1, allowed_mech_ids=allowed_ids,
    ) if is_main else None

    raw_model = BuresLarrosaClassifier(
        design_dim=6 + g.n_species,
        n_obs_channels=len(obs_ch),
        n_classes=n_classes,
        hidden=args.hidden,
    ).to(device)
    n_params = sum(p.numel() for p in raw_model.parameters() if p.requires_grad)
    if is_main:
        print(
            f"[BL:{args.protocol}] Params={n_params:,} n_classes={n_classes} "
            f"world={world_size} bs/gpu={args.bs} eff_bs={args.bs*world_size} steps={args.steps} seed={args.seed}",
            flush=True,
        )

    model = DDP(raw_model, device_ids=[local_rank], output_device=local_rank) if is_ddp else raw_model
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)

    def lr_lambda(step: int) -> float:
        if step < args.warmup:
            return float(step + 1) / float(max(1, args.warmup))
        prog = float(step - args.warmup) / float(max(1, args.steps - args.warmup))
        return 0.05 + 0.95 * 0.5 * (1.0 + math.cos(math.pi * min(1.0, prog)))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    start_step = 0
    best_val_loss = float("inf")
    history = []
    elapsed_prev = 0.0
    if args.resume == "auto" and (out_dir / "latest_ckpt.pt").exists():
        ckpt = torch.load(out_dir / "latest_ckpt.pt", map_location=device)
        raw_model.load_state_dict(ckpt["model_state"])
        optimizer.load_state_dict(ckpt["optimizer_state"])
        scheduler.load_state_dict(ckpt["scheduler_state"])
        start_step = int(ckpt["step"])
        best_val_loss = float(ckpt["best_val_loss"])
        history = list(ckpt.get("history", []))
        elapsed_prev = float(ckpt.get("elapsed_s", 0.0))
        if is_ddp:
            dist.barrier()

    def _atomic_save(payload, path: Path):
        tmp = path.with_suffix(path.suffix + ".tmp")
        torch.save(payload, tmp)
        os.replace(tmp, path)

    t0 = time.perf_counter()
    t_log = t0
    run_lm, run_cnt = 0.0, 0
    model.train()

    for step in range(start_step + 1, args.steps + 1):
        optimizer.zero_grad(set_to_none=True)
        df, tm, ov, om, tp, ep, mids, _ = train_sampler.sample_batch(args.bs)
        tgts = id_remap_t[mids] if id_remap_t is not None else mids
        with torch.cuda.amp.autocast(enabled=use_bf16, dtype=torch.bfloat16):
            loss, _ = model(df, tm, ov, om, tp, ep, tgts)
        loss.backward()
        gnorm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        if torch.isfinite(gnorm):
            optimizer.step()
        scheduler.step()
        if step % 100 == 0 or step == 1:
            run_lm += float(loss.item())
            run_cnt += 1

        if is_main and (step % 2000 == 0 or step == start_step + 1):
            dt = time.perf_counter() - t_log
            sps = (2000 / dt) if (step % 2000 == 0 and step > start_step + 1) else (1 / max(dt, 1e-6))
            print(
                f"[BL:{args.protocol}] step {step:5d}/{args.steps} | loss_m={run_lm/max(run_cnt,1):.4f} | "
                f"lr={scheduler.get_last_lr()[0]:.2e} | {sps:.1f} step/s | elapsed={elapsed_prev + time.perf_counter()-t0:.0f}s",
                flush=True,
            )
            run_lm, run_cnt = 0.0, 0
            t_log = time.perf_counter()

        if step % args.eval_every == 0 or step == args.steps:
            if is_main:
                tr = eval_bl(raw_model, train_sampler, id_remap_t, use_bf16=use_bf16, eval_seed=12345)
                va = eval_bl(raw_model, val_sampler, id_remap_t, use_bf16=use_bf16, eval_seed=54321)
                rec = {
                    "step": step,
                    "train_loss_m": tr["loss_m"], "train_top1": tr["top1"],
                    "val_loss_m": va["loss_m"], "val_top1": va["top1"], "val_top3": va["top3"], "val_ece": va["ece"],
                    "elapsed_s": elapsed_prev + time.perf_counter() - t0,
                }
                history.append(rec)
                print(
                    f"  [Val BL:{args.protocol} @ {step}] tr_lm={tr['loss_m']:.4f} (top1={tr['top1']:.3f}) | "
                    f"va_lm={va['loss_m']:.4f} (top1={va['top1']:.3f}, top3={va['top3']:.3f}, ece={va['ece']:.4f})",
                    flush=True,
                )
                if va["loss_m"] < best_val_loss:
                    best_val_loss = va["loss_m"]
                    _atomic_save(
                        {
                            "step": step, "protocol": args.protocol, "hidden": args.hidden,
                            "n_classes": n_classes,
                            "allowed_mech_ids": allowed_ids.tolist() if allowed_ids is not None else None,
                            "model_state": raw_model.state_dict(), "val_metrics": rec,
                        },
                        out_dir / "best.pt",
                    )
                _atomic_save(
                    {
                        "step": step, "protocol": args.protocol, "hidden": args.hidden,
                        "model_state": raw_model.state_dict(),
                        "optimizer_state": optimizer.state_dict(),
                        "scheduler_state": scheduler.state_dict(),
                        "best_val_loss": best_val_loss, "history": history,
                        "elapsed_s": elapsed_prev + time.perf_counter() - t0,
                    },
                    out_dir / "latest_ckpt.pt",
                )
                model.train()
            if is_ddp:
                dist.barrier()

    if is_main:
        summary = {
            "protocol": args.protocol, "seed": args.seed, "n_classes": n_classes,
            "n_params": n_params, "steps": args.steps, "best_val_loss_m": best_val_loss,
            "elapsed_s": elapsed_prev + time.perf_counter() - t0, "history": history,
        }
        with open(out_dir / "train_summary.json", "w") as f:
            json.dump(summary, f, indent=2)
    if is_ddp:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
