"""
Anchored Step 3: Chunked Sequential Forward SFT

Core technique: for <THINK> segments, run a sequential forward pass with the
KV-Cache (keeping the causal chain intact); for explicit segments, use the
standard parallel forward. Train a 3-level curriculum that progressively
deepens the latent reasoning.

Difference from the two-pass hook:
  two-pass: the Adapter input comes from the <THINK> token embedding context (incorrect)
  chunked:  the Adapter input comes from the previous Adapter output after passing
            through the Transformer (correct)

Usage:
    accelerate launch --num_processes=2 --mixed_precision=bf16 \
        train_phase15_emerge.py \
        --checkpoint ./checkpoints/phase1/best \
        --sft_dir ./sft_emerge_data \
        --output_dir ./checkpoints/emerge_sft
"""

import os, sys, json, time, random, argparse, logging, math
from typing import Dict, List, Tuple, Optional
from collections import defaultdict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, get_cosine_schedule_with_warmup
from accelerate import Accelerator
from latent_adapter import SkipAdapter

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s",
                    handlers=[logging.StreamHandler(sys.stdout)])
logger = logging.getLogger("emerge_sft")


# ============================================================
# Dataset
# ============================================================

class EmergeSFTDataset(Dataset):
    def __init__(self, data_path: str, think_token_id: int, max_seq_len: int = 2048):
        raw = torch.load(data_path, map_location="cpu", weights_only=False)
        self.data = []
        for d in raw:
            ids = d["sft_ids"].clone()
            # replace placeholder -1 with the real think_token_id
            ids[ids == -1] = think_token_id
            if len(ids) <= max_seq_len:
                d["sft_ids"] = ids
                self.data.append(d)
        logger.info(f"  Loaded {len(self.data)} sequences from {data_path}")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        d = self.data[idx]
        return {
            "input_ids": d["sft_ids"],
            "target_types": d["target_types"],
            "prompt_length": d["prompt_length"],
            "think_segments": d["think_segments"],
            "intermediate_targets": d["intermediate_targets"],
        }


# ============================================================
# Chunked Sequential Forward (core)
# ============================================================

def chunked_forward(
    model: nn.Module,
    adapter: nn.Module,
    input_ids: torch.Tensor,       # (seq_len,) single sequence
    target_types: torch.Tensor,    # (seq_len,)
    think_token_id: int,
    intermediate_targets: List[List[int]],
    exit_weight: float = 3.0,
    intermediate_weight: float = 0.3,
    device: torch.device = None,
) -> Tuple[torch.Tensor, Dict]:
    """
    Run a Chunked Sequential Forward on a single sequence.

    Explicit segment: parallel forward, using the KV-Cache
    <THINK> segment: step-by-step sequential forward, each step Adapter(h) → Transformer(z, kv_cache)

    Returns:
        loss: scalar loss
        stats: statistics
    """
    if device is None:
        device = input_ids.device
    
    seq_len = len(input_ids)
    think_mask = (input_ids == think_token_id)
    
    # find the boundaries of all chunks
    # chunk type: 'text' (consecutive non-<THINK>) or 'think' (consecutive <THINK>)
    chunks = []
    i = 0
    while i < seq_len:
        if think_mask[i]:
            start = i
            while i < seq_len and think_mask[i]:
                i += 1
            chunks.append(("think", start, i))  # [start, i) is <THINK>
        else:
            start = i
            while i < seq_len and not think_mask[i]:
                i += 1
            chunks.append(("text", start, i))

    # forward chunk by chunk
    kv_cache = None
    h_last = None
    all_logits = []  # (position, logit_vector) used for computing the loss
    think_seg_idx = 0  # track which <THINK> segment we are on
    
    total_loss = torch.tensor(0.0, device=device)
    n_loss_tokens = 0
    stats = {"n_text_chunks": 0, "n_think_chunks": 0,
             "n_exit_tokens": 0, "n_intermediate": 0}
    
    for chunk_type, c_start, c_end in chunks:
        c_len = c_end - c_start
        
        if chunk_type == "text":
            stats["n_text_chunks"] += 1
            
            # parallel forward
            chunk_ids = input_ids[c_start:c_end].unsqueeze(0).to(device)

            if kv_cache is None:
                # first chunk: full forward
                out = model(input_ids=chunk_ids, use_cache=True,
                           output_hidden_states=True)
            else:
                # subsequent chunk: with KV-Cache
                # if the previous chunk was <THINK>, h_last was already updated by the Adapter
                # the first token uses the normal embedding
                out = model(input_ids=chunk_ids,
                           past_key_values=kv_cache,
                           use_cache=True,
                           output_hidden_states=True)

            kv_cache = out.past_key_values
            h_last = out.hidden_states[-1][0, -1, :]  # hidden at the last position
            chunk_logits = out.logits[0]  # (c_len, V)

            # compute the loss for explicit tokens
            for pos_in_chunk in range(c_len):
                abs_pos = c_start + pos_in_chunk
                tt = target_types[abs_pos].item()

                # determine the target token
                if abs_pos + 1 < seq_len:
                    target = input_ids[abs_pos + 1].item()
                else:
                    continue

                if tt == 0:  # prompt: no loss
                    continue
                elif tt == 1:  # normal explicit: weight 1.0
                    weight = 1.0
                elif tt == 3:  # exit: weight exit_weight
                    weight = exit_weight
                    stats["n_exit_tokens"] += 1
                else:
                    continue
                
                logit = chunk_logits[pos_in_chunk]
                token_loss = F.cross_entropy(logit.unsqueeze(0),
                                              torch.tensor([target], device=device))
                total_loss = total_loss + weight * token_loss
                n_loss_tokens += 1
            
            del out
        
        elif chunk_type == "think":
            stats["n_think_chunks"] += 1
            
            if h_last is None:
                # should not happen: the first chunk should not be <THINK>
                # fall back to a zero vector
                h_last = torch.zeros(model.config.hidden_size,
                                     device=device, dtype=torch.bfloat16)

            # get the intermediate supervision targets
            inter_targets = []
            if think_seg_idx < len(intermediate_targets):
                inter_targets = intermediate_targets[think_seg_idx]
            think_seg_idx += 1

            # sequential forward: each step Adapter → Transformer
            for k in range(c_len):
                z = adapter(h_last.unsqueeze(0))  # (1, d) - has gradient!

                out = model(inputs_embeds=z.unsqueeze(0),  # (1, 1, d)
                           past_key_values=kv_cache,
                           use_cache=True,
                           output_hidden_states=True)
                kv_cache = out.past_key_values
                h_last = out.hidden_states[-1][0, -1, :]

                # intermediate supervision
                if intermediate_weight > 0 and k < len(inter_targets):
                    target_tok = inter_targets[k]
                    logit_k = out.logits[0, -1, :]
                    inter_loss = F.cross_entropy(
                        logit_k.unsqueeze(0),
                        torch.tensor([target_tok], device=device))
                    total_loss = total_loss + intermediate_weight * inter_loss
                    stats["n_intermediate"] += 1
                
                # last step: if the next token is explicit, compute the exit loss
                # (the exit loss is handled in the text chunk, since target_types[next_text_start]=3)

                del out
    
    if n_loss_tokens > 0:
        total_loss = total_loss / n_loss_tokens
    
    stats["n_loss_tokens"] = n_loss_tokens
    return total_loss, stats


# ============================================================
# Batched Chunked Forward (handles a mini-batch)
# ============================================================

def batched_chunked_forward(
    model, adapter, batch, think_token_id,
    exit_weight=3.0, intermediate_weight=0.3, device=None,
):
    """
    Run a chunked forward on each sequence in a batch one at a time,
    accumulate the loss, and return the mean.

    Note: because each sequence has a different chunk structure, true
    parallelism is not possible. However, the text segments within a single
    sequence are processed in parallel.
    """
    total_loss = torch.tensor(0.0, device=device, requires_grad=True)
    total_stats = defaultdict(float)
    B = len(batch["input_ids"])
    
    for b in range(B):
        ids = batch["input_ids"][b].to(device)
        tt = batch["target_types"][b].to(device)
        inter_targets = batch["intermediate_targets"][b]

        # strip padding
        actual_len = (ids != 0).sum().item()  # assumes pad=0
        if actual_len < 10:
            continue
        ids = ids[:actual_len]
        tt = tt[:actual_len]
        
        loss_b, stats_b = chunked_forward(
            model, adapter, ids, tt, think_token_id,
            inter_targets, exit_weight, intermediate_weight, device)
        
        total_loss = total_loss + loss_b
        for k, v in stats_b.items():
            total_stats[k] += v
    
    if B > 0:
        total_loss = total_loss / B
    
    return total_loss, dict(total_stats)


# ============================================================
# P(THINK | context) monitor
# ============================================================

@torch.no_grad()
def check_think_probability(model, tokenizer, think_token_id, dataset, n=50):
    model.eval()
    ps = []
    n = min(n, len(dataset))
    for idx in range(n):
        item = dataset[idx]
        ids = item["input_ids"]
        think_pos = (ids == think_token_id).nonzero(as_tuple=True)[0]
        for pos in think_pos[:3]:
            if pos <= item["prompt_length"]:
                continue
            ctx = ids[:pos].unsqueeze(0).to(next(model.parameters()).device)
            out = model(input_ids=ctx, use_cache=False)
            p = F.softmax(out.logits[0, -1, :].float(), dim=-1)[think_token_id].item()
            ps.append(p)
    model.train()
    return {
        "mean": float(np.mean(ps)) if ps else 0,
        "gt3pct": float(np.mean([p > 0.03 for p in ps])) if ps else 0,
        "n": len(ps),
    }


# ============================================================
# Collate function
# ============================================================

def collate_fn(batch, pad_id=0):
    max_len = max(len(item["input_ids"]) for item in batch)
    b_ids, b_types, b_pl, b_inter = [], [], [], []
    for item in batch:
        ids = item["input_ids"]
        tt = item["target_types"]
        pad_len = max_len - len(ids)
        b_ids.append(F.pad(ids, (0, pad_len), value=pad_id))
        b_types.append(F.pad(tt, (0, pad_len), value=0))
        b_pl.append(item["prompt_length"])
        b_inter.append(item["intermediate_targets"])
    return {
        "input_ids": torch.stack(b_ids),
        "target_types": torch.stack(b_types),
        "prompt_lengths": b_pl,
        "intermediate_targets": b_inter,
    }


# ============================================================
# Checkpoint
# ============================================================

def save_ckpt(accelerator, model, adapter, tokenizer, save_dir, step, args):
    os.makedirs(save_dir, exist_ok=True)
    unwrapped = accelerator.unwrap_model(model)
    state = accelerator.get_state_dict(model, unwrap=False)
    adapter_state = accelerator.get_state_dict(adapter, unwrap=False)
    if accelerator.is_main_process and state is not None:
        unwrapped.save_pretrained(save_dir, state_dict=state, safe_serialization=True)
        tokenizer.save_pretrained(save_dir)
        torch.save(adapter_state, os.path.join(save_dir, "skip_adapter.pt"))
        with open(os.path.join(save_dir, "training_config.json"), "w") as f:
            json.dump({"phase": "1.5-emerge", "step": step,
                       "lr_base": args.lr_base, "lr_adapter": args.lr_adapter},
                      f, indent=2)
        logger.info(f"  Saved: {save_dir}")
    accelerator.wait_for_everyone()


# ============================================================
# Main
# ============================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--sft_dir", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default="./checkpoints/emerge_sft")
    parser.add_argument("--lr_adapter", type=float, default=1e-4)
    parser.add_argument("--lr_base", type=float, default=1e-6)
    parser.add_argument("--exit_weight", type=float, default=3.0)
    parser.add_argument("--intermediate_weight", type=float, default=0.3)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--grad_accum", type=int, default=8)
    parser.add_argument("--steps_per_level", type=int, default=300)
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
    
    total_steps = args.steps_per_level * 3
    os.makedirs(args.output_dir, exist_ok=True)
    torch.manual_seed(args.seed + accelerator.process_index)
    random.seed(args.seed)
    np.random.seed(args.seed)
    
    if is_main:
        logger.info("=" * 70)
        logger.info("Anchored Phase 1.5: Chunked Sequential Forward SFT")
        logger.info(f"  3 Levels × {args.steps_per_level} steps = {total_steps} total")
        logger.info("=" * 70)

    # model
    if is_main: logger.info("Loading model...")
    model = AutoModelForCausalLM.from_pretrained(
        args.checkpoint, torch_dtype=torch.bfloat16,
        attn_implementation="sdpa", trust_remote_code=True)
    tokenizer = AutoTokenizer.from_pretrained(args.checkpoint, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    d_model = model.config.hidden_size
    
    # <THINK> token
    if "<THINK>" not in tokenizer.get_vocab():
        tokenizer.add_special_tokens({"additional_special_tokens": ["<THINK>"]})
        model.resize_token_embeddings(len(tokenizer))
    think_token_id = tokenizer.convert_tokens_to_ids("<THINK>")
    
    with torch.no_grad():
        el = model.get_input_embeddings()
        seeds = [tokenizer.encode(t, add_special_tokens=False)[0]
                 for t in ["think", "pause", "wait", "hmm"]]
        init_emb = el.weight[seeds].mean(0) + torch.randn(d_model) * 0.01
        el.weight[think_token_id] = init_emb
    
    if is_main:
        logger.info(f"  think_token_id={think_token_id}, d_model={d_model}")
    
    # Adapter
    adapter = SkipAdapter(hidden_size=d_model).to(dtype=torch.bfloat16, device=device)
    ap = os.path.join(args.checkpoint, "skip_adapter.pt")
    if os.path.exists(ap):
        try:
            adapter.load_state_dict(torch.load(ap, map_location="cpu", weights_only=True))
            if is_main: logger.info(f"  Adapter loaded from {ap}")
        except RuntimeError:
            if is_main: logger.warning("  Adapter shape mismatch, using fresh init")
    if is_main:
        logger.info(f"  Adapter: {sum(p.numel() for p in adapter.parameters())/1e6:.2f}M")
    
    model.gradient_checkpointing_enable()
    model.config.use_cache = False  # chunked forward manages the cache internally

    # data
    datasets = {}
    for level in [1, 2, 3]:
        path = os.path.join(args.sft_dir, f"sft_level{level}.pt")
        if os.path.exists(path):
            datasets[level] = EmergeSFTDataset(path, think_token_id, args.max_seq_len)
    
    # Optimizer
    no_decay_kw = ["bias", "layer_norm", "layernorm", "rmsnorm"]
    optimizer = torch.optim.AdamW([
        {"params": [p for n, p in model.named_parameters()
                    if p.requires_grad and not any(k in n.lower() for k in no_decay_kw)],
         "lr": args.lr_base, "weight_decay": 0.01},
        {"params": [p for n, p in model.named_parameters()
                    if p.requires_grad and any(k in n.lower() for k in no_decay_kw)],
         "lr": args.lr_base, "weight_decay": 0.0},
        {"params": adapter.parameters(), "lr": args.lr_adapter, "weight_decay": 0.01},
    ])
    
    scheduler = get_cosine_schedule_with_warmup(optimizer, args.warmup_steps, total_steps)
    
    model, adapter, optimizer = accelerator.prepare(model, adapter, optimizer)

    # data iterator
    def make_loader(dataset, bsz, seed_offset=0):
        ep = 0
        while True:
            idx = list(range(len(dataset)))
            random.Random(args.seed + seed_offset + ep).shuffle(idx)
            batch = []
            for i in idx:
                batch.append(dataset[i])
                if len(batch) == bsz:
                    yield collate_fn(batch, pad_id=tokenizer.pad_token_id or 0)
                    batch = []
            ep += 1

    # training
    if is_main:
        logger.info("Training...")
        logger.info("=" * 70)
    
    model.train()
    adapter.train()
    
    global_step = 0
    start_time = time.time()
    run = defaultdict(float)
    run_count = 0
    LOG_EVERY = 10
    
    for level in [1, 2, 3]:
        if level not in datasets:
            if is_main: logger.warning(f"  Level {level} data not found, skipping")
            continue
        
        if is_main:
            logger.info(f"\n{'='*70}")
            logger.info(f"  Level {level} starting at step {global_step}")
            logger.info(f"{'='*70}")
        
        data_iter = iter(make_loader(datasets[level], args.batch_size,
                                      accelerator.process_index + level * 100))
        
        for step_in_level in range(args.steps_per_level):
            optimizer.zero_grad()
            
            for _ in range(args.grad_accum):
                batch = next(data_iter)
                
                try:
                    # Chunked forward must process sequences one at a time
                    # but we accumulate over grad_accum
                    loss, fwd_stats = batched_chunked_forward(
                        accelerator.unwrap_model(model),
                        accelerator.unwrap_model(adapter),
                        batch, think_token_id,
                        args.exit_weight, args.intermediate_weight, device)
                    
                    accelerator.backward(loss / args.grad_accum)
                    
                except RuntimeError as e:
                    if "out of memory" in str(e).lower():
                        torch.cuda.empty_cache()
                        if is_main: logger.warning(f"  OOM at step {global_step}")
                        continue
                    raise
                
                run["loss"] += loss.item()
                run["exit"] += fwd_stats.get("n_exit_tokens", 0)
                run["inter"] += fwd_stats.get("n_intermediate", 0)
                run_count += 1
            
            gn = accelerator.clip_grad_norm_(
                list(model.parameters()) + list(adapter.parameters()),
                args.max_grad_norm)
            optimizer.step()
            scheduler.step()
            global_step += 1
            
            # Logging
            if is_main and global_step % LOG_EVERY == 0:
                n = max(run_count, 1)
                t = time.time() - start_time
                lr = scheduler.get_last_lr()[0]
                gn_val = gn.item() if isinstance(gn, torch.Tensor) else gn
                logger.info(
                    f"Step {global_step}/{total_steps} [L{level}] | "
                    f"Loss:{run['loss']/n:.4f} | "
                    f"Exit:{run['exit']/n:.1f} Inter:{run['inter']/n:.1f} | "
                    f"GN:{gn_val:.3f} LR:{lr:.2e} | {t:.0f}s")
                run.clear()
                run_count = 0
            
            # Eval
            if global_step % args.eval_every == 0:
                if is_main:
                    logger.info(f"\n--- Eval step {global_step} ---")
                    sp = check_think_probability(model, tokenizer, think_token_id,
                                                  datasets[level].data)
                    logger.info(f"  P(THINK) mean:{sp['mean']:.4f}, "
                                f">3%:{sp['gt3pct']:.1%} (n={sp['n']})")
                    logger.info("---\n")
            
            # Save
            if global_step % args.save_every == 0:
                save_ckpt(accelerator, model, adapter, tokenizer,
                          os.path.join(args.output_dir, f"step-{global_step}"),
                          global_step, args)
        
        # save after each level finishes
        save_ckpt(accelerator, model, adapter, tokenizer,
                  os.path.join(args.output_dir, f"level{level}-done"),
                  global_step, args)

    # final save
    save_ckpt(accelerator, model, adapter, tokenizer,
              os.path.join(args.output_dir, "final"),
              global_step, args)

    if is_main:
        logger.info("=" * 70)
        logger.info("Anchored Phase 1.5 complete! Next: Phase 2 RL")
        logger.info("=" * 70)


if __name__ == "__main__":
    main()
