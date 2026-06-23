"""
Phase 1.5 SFT Training (FSDP + Memory Optimized)

Core technique: Two-Pass Embedding Hook
  Pass 1 (no grad): model(input_ids) → hidden_states[-1]
  Hook: register_forward_hook on embedding, replace <SKIP> with Adapter(h_{t-1})
  Pass 2 (with grad): model(input_ids) through hooked embedding → logits → loss

Bug fixes vs previous version:
  * scheduler NOT prepared by accelerator (was causing 4x speedup of cosine cycle)
  * AdapterNorm logging changed to per-dim std (sqrt(d) norm is expected for LayerNorm)

Usage (single GPU):
    python train_phase15.py --checkpoint ./checkpoints/phase1/best \
        --sft_data ./phase15_sft_data/sft_training_data.pt

Usage (4-GPU FSDP, via run_phase15.sh):
    sbatch run_phase15.sh
"""

import os
import sys
import json
import time
import random
import argparse
import logging
from typing import Dict, Tuple
from collections import defaultdict
from latent_adapter import SkipAdapter

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, get_cosine_schedule_with_warmup
from accelerate import Accelerator

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("phase15_sft")


# ============================================================
# Dataset
# ============================================================

class SkipSFTDataset(Dataset):
    def __init__(self, data_path: str, max_seq_len: int = 2048):
        self.data = torch.load(data_path, map_location="cpu", weights_only=False)
        orig = len(self.data)
        self.data = [d for d in self.data if d["sft_ids"].shape[0] <= max_seq_len]
        if len(self.data) < orig:
            logger.info(f"Filtered {orig - len(self.data)} sequences > {max_seq_len}")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return {
            "input_ids": self.data[idx]["sft_ids"],
            "prompt_length": self.data[idx]["prompt_length"],
            "target_types": self.data[idx].get("target_types", torch.zeros_like(self.data[idx]["sft_ids"]))
        }


# ============================================================
# Core: Two-Pass Embedding Hook Forward
# ============================================================

def two_pass_forward(
    model: nn.Module,
    skip_adapter: nn.Module,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    skip_token_id: int,
    accelerator: Accelerator,
) -> Tuple[torch.Tensor, Dict]:
    """
    Two-Pass Embedding Hook Forward.

    Latent feedback rule:
      input(t) = Adapter(h_{t-1})  if token(t) == <SKIP>
      input(t) = E(token(t))       otherwise
    """
    device = input_ids.device
    B, L = input_ids.shape
    skip_mask = (input_ids == skip_token_id)

    if not skip_mask.any():
        out = model(input_ids=input_ids, attention_mask=attention_mask, use_cache=False)
        return out.logits, {"adapter_norm_mean": 0.0, "num_skip_positions": 0}

    # ── Pass 1: hidden states (no grad) ──
    with torch.no_grad():
        out1 = model(input_ids=input_ids, attention_mask=attention_mask,
                     output_hidden_states=True, use_cache=False)
        hidden_last = out1.hidden_states[-1].detach()
        del out1
        torch.cuda.empty_cache()

    # ── Adapter outputs (WITH grad) ──
    d = hidden_last.shape[-1]
    adapter_norms = []
    adapter_vals_list = []

    for b in range(B):
        skip_pos = skip_mask[b].nonzero(as_tuple=True)[0]
        if len(skip_pos) == 0:
            adapter_vals_list.append(
                torch.zeros(L, d, device=device, dtype=hidden_last.dtype)
            )
            continue

        prev_pos = (skip_pos - 1).clamp(min=0)
        h_prev = hidden_last[b, prev_pos]
        z_skips = skip_adapter(h_prev)

        adapter_norms.extend(z_skips.detach().norm(dim=-1).tolist())

        n_skips = len(skip_pos)
        one_hot = torch.zeros(L, n_skips, device=device, dtype=z_skips.dtype)
        one_hot[skip_pos, torch.arange(n_skips, device=device)] = 1.0
        adapter_vals_list.append(one_hot @ z_skips)

    adapter_vals = torch.stack(adapter_vals_list, dim=0)
    del hidden_last

    # ── Pass 2: Hooked forward (with grad) ──
    skip_mask_3d = skip_mask.unsqueeze(-1)

    def embedding_hook(module, input, output):
        return torch.where(skip_mask_3d, adapter_vals, output)

    unwrapped = accelerator.unwrap_model(model)
    embed_layer = unwrapped.get_input_embeddings()
    handle = embed_layer.register_forward_hook(embedding_hook)

    try:
        out2 = model(input_ids=input_ids, attention_mask=attention_mask, use_cache=False)
    finally:
        handle.remove()

    stats = {
        "adapter_norm_mean": np.mean(adapter_norms) if adapter_norms else 0.0,
        "num_skip_positions": len(adapter_norms),
    }
    return out2.logits, stats


# ============================================================
# Loss
# ============================================================

def compute_sft_loss(
    logits: torch.Tensor,
    input_ids: torch.Tensor,
    target_types: torch.Tensor, # [added] pass in the mask that was just assembled
    attention_mask: torch.Tensor,
    prompt_lengths: list,
    adapter_loss_weight: float = 2.0,
) -> Tuple[torch.Tensor, Dict]:
    B, seq_len = input_ids.shape
    device = input_ids.device

    labels = torch.full((B, seq_len), -100, dtype=torch.long, device=device)
    for b in range(B):
        pl = prompt_lengths[b]
        actual_len = min(int(attention_mask[b].sum().item()), seq_len)
        if actual_len > pl + 1:
            labels[b, pl:actual_len - 1] = input_ids[b, pl + 1:actual_len]

    # Shift the sequence: logits[t] predicts labels[t] (its type is target_types[t+1])
    shift_logits = logits[:, :-1, :].contiguous()
    shift_labels = labels[:, :-1].contiguous()
    shift_types = target_types[:, 1:].contiguous() # shift the label types left in sync

    flat_logits = shift_logits.view(-1, shift_logits.shape[-1])
    flat_labels = shift_labels.view(-1)
    flat_types = shift_types.view(-1)

    weights = torch.ones_like(flat_labels, dtype=torch.float, device=device)
    
    # * Core loss routing logic *
    # Type 0: normal explicit, weight = 1.0
    # Type 1: Entry (predicting the first SKIP), weight = 1.0 (teaches when to enter)
    # Type 2: Mid (predicting intermediate SKIPs), weight = 0.0 (no ground truth, free degrees of freedom)
    weights[flat_types == 2] = 0.0
    # Type 3: Exit (predicting the real token from SKIP), weight = adapter_loss_weight (the core driving signal)
    weights[flat_types == 3] = adapter_loss_weight

    per_token_loss = F.cross_entropy(flat_logits, flat_labels, reduction="none")
    valid = (flat_labels != -100) & (weights > 0) # intermediate SKIPs are also excluded from the valid denominator
    
    if valid.sum() == 0:
        return torch.tensor(0.0, device=device, requires_grad=True), {}

    weighted_loss = (per_token_loss * weights * valid.float()).sum() / valid.float().sum()

    with torch.no_grad():
        v_exit = valid & (flat_types == 3)
        v_norm = valid & (flat_types != 3)
        l_e = (per_token_loss * v_exit.float()).sum() / max(v_exit.sum(), 1)
        l_n = (per_token_loss * v_norm.float()).sum() / max(v_norm.sum(), 1)

    return weighted_loss, {
        "loss_total": weighted_loss.item(),
        "loss_normal": l_n.item(),
        "loss_adapter_exit": l_e.item(), # dedicated monitoring of Exit loss
    }

# ============================================================
# P(SKIP | context) monitor
# ============================================================

@torch.no_grad()
def check_skip_probability(model, tokenizer, skip_token_id, sft_data, n=100):
    model.eval()
    ps = []
    n = min(n, len(sft_data))
    for idx in range(n):
        ids = sft_data[idx]["sft_ids"]
        pl = sft_data[idx]["prompt_length"]
        skip_pos = (ids == skip_token_id).nonzero(as_tuple=True)[0]
        for pos in skip_pos[:3]:
            if pos <= pl:
                continue
            ctx = ids[:pos].unsqueeze(0).to(next(model.parameters()).device)
            out = model(input_ids=ctx, use_cache=False)
            p = F.softmax(out.logits[0, -1, :].float(), dim=-1)[skip_token_id].item()
            ps.append(p)
    model.train()
    return {
        "mean": np.mean(ps) if ps else 0,
        "gt3pct": np.mean([p > 0.03 for p in ps]) if ps else 0,
        "n": len(ps),
    }


# ============================================================
# FSDP-safe checkpoint
# ============================================================

def save_ckpt(accelerator, model, skip_adapter, tokenizer, save_dir, step, args):
    os.makedirs(save_dir, exist_ok=True)
    unwrapped_model = accelerator.unwrap_model(model)

    # 1. Gather the weights of the main model
    state_dict = accelerator.get_state_dict(model, unwrap=False)

    # 2. * Gather the Adapter weights (essential: makes FSDP gather all the shards together)
    adapter_state_dict = accelerator.get_state_dict(skip_adapter, unwrap=False)

    if accelerator.is_main_process and state_dict is not None:
        unwrapped_model.save_pretrained(
            save_dir, state_dict=state_dict, safe_serialization=True
        )
        tokenizer.save_pretrained(save_dir)

        # 3. * Save the full gathered Adapter state dict
        torch.save(
            adapter_state_dict, 
            os.path.join(save_dir, "skip_adapter.pt"),
        )
        with open(os.path.join(save_dir, "training_config.json"), "w") as f:
            json.dump({
                "phase": "1.5", "step": step,
                "lr_base": args.lr_base, "lr_adapter": args.lr_adapter,
                "adapter_loss_weight": args.adapter_loss_weight,
            }, f, indent=2)
        logger.info(f"  Saved: {save_dir}")

    accelerator.wait_for_everyone()


# ============================================================
# Main
# ============================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--sft_data", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default="./checkpoints/phase15")
    parser.add_argument("--lr_adapter", type=float, default=5e-6)
    parser.add_argument("--lr_base", type=float, default=1e-6)
    parser.add_argument("--adapter_loss_weight", type=float, default=2.0)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--grad_accum", type=int, default=4)
    parser.add_argument("--num_steps", type=int, default=400)
    parser.add_argument("--warmup_steps", type=int, default=30)
    parser.add_argument("--max_seq_len", type=int, default=2048)
    parser.add_argument("--eval_every", type=int, default=100)
    parser.add_argument("--save_every", type=int, default=100)
    parser.add_argument("--max_grad_norm", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    accelerator = Accelerator(mixed_precision="bf16")
    is_main = accelerator.is_main_process
    device = accelerator.device

    os.makedirs(args.output_dir, exist_ok=True)
    torch.manual_seed(args.seed + accelerator.process_index)
    random.seed(args.seed)
    np.random.seed(args.seed)

    if is_main:
        logger.info("=" * 70)
        logger.info("Phase 1.5 SFT (Two-Pass Hook + FSDP)")
        logger.info(f"  GPUs: {accelerator.num_processes}, "
                    f"grad_accum: {args.grad_accum}, "
                    f"batch_size: {args.batch_size}")
        logger.info("=" * 70)

    # ── 1. Model ──
    if is_main:
        logger.info("Loading model...")
    model = AutoModelForCausalLM.from_pretrained(
        args.checkpoint, torch_dtype=torch.bfloat16,
        attn_implementation="sdpa", trust_remote_code=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(args.checkpoint, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    d_model = model.config.hidden_size
    weights_tied = (model.get_input_embeddings().weight is model.get_output_embeddings().weight)

    # ── 2. <SKIP> token ──
    if "<SKIP>" in tokenizer.get_vocab():
        skip_token_id = tokenizer.convert_tokens_to_ids("<SKIP>")
    else:
        tokenizer.add_special_tokens({"additional_special_tokens": ["<SKIP>"]})
        skip_token_id = tokenizer.convert_tokens_to_ids("<SKIP>")
        model.resize_token_embeddings(len(tokenizer))

    with torch.no_grad():
        el = model.get_input_embeddings()
        seeds = [tokenizer.encode(t, add_special_tokens=False)[0]
                 for t in ["skip", "pass", "omit", "therefore"]]
        init_emb = el.weight[seeds].mean(0) + torch.randn(d_model) * 0.01
        el.weight[skip_token_id] = init_emb
        if not weights_tied:
            model.lm_head.weight[skip_token_id] = init_emb
    if is_main:
        logger.info(f"  skip_token_id={skip_token_id}, tied={weights_tied}")

    # ── 3. Adapter ──
    # * Fix 1: removed bottleneck_ratio, enabling a full-dimension residual network
    skip_adapter = SkipAdapter(
        hidden_size=d_model
    ).to(dtype=torch.bfloat16, device=device)

    ap = os.path.join(args.checkpoint, "skip_adapter.pt")
    if os.path.exists(ap):
        try:
            # * Fix 2: catch the size mismatch with old 384-dim weights for a safe restart
            state_dict = torch.load(ap, map_location="cpu", weights_only=True)
            skip_adapter.load_state_dict(state_dict, strict=True)
            if is_main:
                logger.info(f"  Adapter loaded from {ap}")
        except RuntimeError as e:
            if is_main:
                logger.warning("="*60)
                logger.warning("  ⚠️ Architecture Mismatch Detected in Skip Adapter!")
                logger.warning("  Discarding old bottleneck weights. Using fresh residual initialization.")
                logger.warning("="*60)
    if is_main:
        logger.info(f"  Adapter: {sum(p.numel() for p in skip_adapter.parameters())/1e6:.2f}M")

    # ── 4. Gradient checkpointing ──
    model.gradient_checkpointing_enable()
    model.config.use_cache = False
    if is_main:
        logger.info("  Gradient checkpointing: ON")

    # ── 5. Data ──
    dataset = SkipSFTDataset(args.sft_data, args.max_seq_len)
    if is_main:
        logger.info(f"  Data: {len(dataset)} trajectories")

    pad_token_id = tokenizer.pad_token_id or tokenizer.eos_token_id

    def inf_loader(ds, bsz, pad_id, offset=0):
        ep = 0
        while True:
            idx = list(range(len(ds)))
            random.Random(args.seed + offset + ep).shuffle(idx)
            batch = []
            for i in idx:
                batch.append(ds[i])
                if len(batch) == bsz:
                    max_len = max(len(item["input_ids"]) for item in batch)
                    b_ids, b_mask, b_pl, b_tt = [], [], [], []
                    for item in batch:
                        seq = item["input_ids"]
                        pad_len = max_len - len(seq)
                        # Padding input_ids
                        b_ids.append(F.pad(seq, (0, pad_len), value=pad_id))
                        # Padding attention_mask
                        b_mask.append(torch.cat([
                            torch.ones(len(seq), dtype=seq.dtype),
                            torch.zeros(pad_len, dtype=seq.dtype),
                        ]))
                        b_pl.append(item["prompt_length"])

                        # * Added: pad target_types
                        # fill the blank region with 0 (Normal type)
                        tt = item["target_types"]
                        b_tt.append(F.pad(tt, (0, pad_len), value=0))

                    yield {
                        "input_ids": torch.stack(b_ids),
                        "attention_mask": torch.stack(b_mask),
                        "prompt_lengths": b_pl,
                        "target_types": torch.stack(b_tt), # * added to the returned dict
                    }
                    batch = []
            ep += 1

    data_iter = iter(inf_loader(dataset, args.batch_size, pad_token_id,
                                accelerator.process_index))

    # ── 6. Optimizer ──
    no_decay_kw = ["bias", "layer_norm", "layernorm", "rmsnorm"]
    optimizer = torch.optim.AdamW([
        {"params": [p for n, p in model.named_parameters()
                    if p.requires_grad and not any(k in n.lower() for k in no_decay_kw)],
         "lr": args.lr_base, "weight_decay": 0.01},
        {"params": [p for n, p in model.named_parameters()
                    if p.requires_grad and any(k in n.lower() for k in no_decay_kw)],
         "lr": args.lr_base, "weight_decay": 0.0},
        {"params": skip_adapter.parameters(), "lr": args.lr_adapter, "weight_decay": 0.01},
    ])

    # * Key fix: the scheduler does NOT go through accelerator.prepare()
    # Reason: accelerator.prepare(scheduler) wraps it as an AcceleratedScheduler,
    # which on a 4-GPU setup only advances one real step every 4 .step() calls,
    # causing the cosine schedule to finish its entire cycle in 400/4=100 steps.
    scheduler = get_cosine_schedule_with_warmup(
        optimizer, args.warmup_steps, args.num_steps
    )

    # ── 7. FSDP prepare (does NOT include the scheduler!) ──
    model, skip_adapter, optimizer = accelerator.prepare(
        model, skip_adapter, optimizer
    )
    if is_main:
        use_fsdp = getattr(accelerator.state, "fsdp_plugin", None) is not None
        logger.info(f"  FSDP active: {use_fsdp}")
        logger.info(f"  * Scheduler: manual (NOT wrapped by accelerator)")

    # ── 8. Train ──
    if is_main:
        logger.info("Training...")
        logger.info("=" * 70)

    model.train()
    skip_adapter.train()

    LOG_EVERY = 10
    run = defaultdict(float)
    run_count = 0
    start_time = time.time()

    for step in range(1, args.num_steps + 1):
        optimizer.zero_grad()

        for _ in range(args.grad_accum):
            batch = next(data_iter)
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            target_types = batch["target_types"].to(device) # * fetch target_types
            prompt_lengths = batch["prompt_lengths"]

            try:
                logits, fwd_stats = two_pass_forward(
                    model, skip_adapter, input_ids, attention_mask,
                    skip_token_id, accelerator,
                )
                # * Fix: use keyword arguments to ensure target_types is passed correctly, and drop the redundant skip_token_id
                loss, loss_stats = compute_sft_loss(
                    logits=logits, 
                    input_ids=input_ids, 
                    target_types=target_types,
                    attention_mask=attention_mask, 
                    prompt_lengths=prompt_lengths,
                    adapter_loss_weight=args.adapter_loss_weight,
                )
                accelerator.backward(loss / args.grad_accum)
            except RuntimeError as e:
                if "out of memory" in str(e):
                    torch.cuda.empty_cache()
                    if is_main:
                        logger.warning(f"  OOM at step {step}, skipping")
                    continue
                raise

            run["loss"] += loss_stats.get("loss_total", 0)
            run["loss_n"] += loss_stats.get("loss_normal", 0)
            run["loss_a"] += loss_stats.get("loss_adapter_exit", 0) # * unified dict key name
            run["anorm"] += fwd_stats.get("adapter_norm_mean", 0)
            run_count += 1

        gn = accelerator.clip_grad_norm_(
            list(model.parameters()) + list(skip_adapter.parameters()),
            args.max_grad_norm,
        )
        optimizer.step()
        scheduler.step()  # * call the raw scheduler directly, advancing one step at a time

        # Logging
        if is_main and step % LOG_EVERY == 0:
            n = max(run_count, 1)
            t = time.time() - start_time
            lr = scheduler.get_last_lr()[0]
            gn_val = gn.item() if isinstance(gn, torch.Tensor) else gn
            logger.info(
                f"Step {step}/{args.num_steps} | "
                f"Loss:{run['loss']/n:.4f} (N:{run['loss_n']/n:.4f} A:{run['loss_a']/n:.4f}) | "
                f"ANorm:{run['anorm']/n:.2f} | GN:{gn_val:.3f} | LR:{lr:.2e} | {t:.0f}s"
            )
            run.clear()
            run_count = 0

        # Eval (all ranks participate in the forward pass, only the main rank prints)
        if step % args.eval_every == 0 or step == args.num_steps:
            if is_main:
                logger.info(f"\n--- Eval step {step} ---")
            sp = check_skip_probability(model, tokenizer, skip_token_id, dataset.data)
            if is_main:
                logger.info(f"  P(SKIP) mean:{sp['mean']:.4f}, >3%:{sp['gt3pct']:.1%} (n={sp['n']})")
                logger.info("---\n")

        # Save
        if step % args.save_every == 0:
            save_ckpt(accelerator, model, skip_adapter, tokenizer,
                      os.path.join(args.output_dir, f"step-{step}"), step, args)

    # Final
    save_ckpt(accelerator, model, skip_adapter, tokenizer,
              os.path.join(args.output_dir, "best"), args.num_steps, args)
    save_ckpt(accelerator, model, skip_adapter, tokenizer,
              os.path.join(args.output_dir, "final"), args.num_steps, args)

    if is_main:
        logger.info("=" * 70)
        logger.info("Phase 1.5 complete! Next: Phase 2 GRPO")
        logger.info("=" * 70)


if __name__ == "__main__":
    main()