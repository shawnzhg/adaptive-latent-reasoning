"""
SAL (Semantic Anchoring Loss) Verification — Training Script

Standalone, no dependencies on the older pipeline. Fresh adapter from zero.

Fixes vs previous version:
  - No torch.compile (catastrophic with dynamic chunk shapes)
  - No hooks (unreliable) — use output_hidden_states=True
  - float32 loss computation (prevents bf16 NaN overflow)
  - Prompt under no_grad (40% speedup on backward)

Usage:
    python train_sal_test.py \
        --checkpoint ./checkpoints/phase1/best \
        --data_dir ./sal_test_data \
        --output_dir ./sal_test_models/group_a \
        --use_sal false

    python train_sal_test.py \
        --checkpoint ./checkpoints/phase1/best \
        --data_dir ./sal_test_data \
        --output_dir ./sal_test_models/group_b_1.0 \
        --use_sal true --beta 1.0
"""

import os, sys, json, time, random, argparse, logging, gc
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from transformers import AutoModelForCausalLM, AutoTokenizer, get_cosine_schedule_with_warmup

from config import get_config, DataConfig
from data_utils import (GSM8KEvalDataset, build_prompt, extract_model_answer, check_answer)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s",
                    handlers=[logging.StreamHandler(sys.stdout)])
logger = logging.getLogger("sal")


# ============================================================
# Adapter (inline, fresh)
# ============================================================

class LatentBridgeAdapter(nn.Module):
    """
    e = h + W_down(SiLU(W_up(RMSNorm(h))))
    W_down zero-init → identity at step 0.
    """
    def __init__(self, d):
        super().__init__()
        self.norm = nn.RMSNorm(d, elementwise_affine=True)
        self.up = nn.Linear(d, d, bias=False)
        self.down = nn.Linear(d, d, bias=False)
        nn.init.zeros_(self.down.weight)

    def forward(self, h):
        # h: (..., d)
        return h + self.down(F.silu(self.up(self.norm(h))))


# ============================================================
# Chunked Forward with SAL
# ============================================================

def chunked_forward_sal(
    model, adapter,
    prompt_ids: torch.Tensor,       # (P,) on device
    response_ids: torch.Tensor,     # (R,) on device
    think_mask: torch.Tensor,       # (R,)
    anchor_targets: torch.Tensor,   # (R,)
    think_token_id: int,
    use_sal: bool = False,
    beta: float = 1.0,
    exit_weight: float = 3.0,
) -> Tuple[torch.Tensor, Dict]:
    """
    Causally correct chunked forward for ONE sample.

    Text chunks → parallel forward (prompt under no_grad).
    Think chunks → sequential adapter→transformer with KV-cache.

    All loss computed in float32 to prevent bf16 NaN.
    """
    device = prompt_ids.device
    full_ids = torch.cat([prompt_ids, response_ids])
    P = prompt_ids.shape[0]
    R = response_ids.shape[0]
    vocab_size = model.config.vocab_size

    # ---- Parse chunks ----
    chunks = []
    if P > 0:
        chunks.append((0, P, False))
    i = 0
    while i < R:
        start = P + i
        is_think = (think_mask[i].item() == 1)
        j = i
        while j < R and (think_mask[j].item() == 1) == is_think:
            j += 1
        chunks.append((start, P + j, is_think))
        i = j

    kv = None
    h_last = None  # (1, d) — last position's hidden state
    ce_losses = []
    exit_losses = []
    anchor_losses = []

    for ci, (cs, ce, is_think) in enumerate(chunks):
        clen = ce - cs

        if is_think:
            # ======== SEQUENTIAL THINK STEPS ========
            resp_off = cs - P
            for step_i in range(clen):
                t = resp_off + step_i  # index in response

                # Adapter: h_last (1, d) → z (1, d)
                z = adapter(h_last)

                # Single-token forward: (1, 1, d) → model
                out = model(
                    inputs_embeds=z.unsqueeze(1),  # (1, 1, d)
                    past_key_values=kv,
                    use_cache=True,
                    output_hidden_states=True,
                )
                kv = out.past_key_values
                h_last = out.hidden_states[-1][:, -1, :]  # (1, d)

                # ★ SAL: force latent state to predict original token
                if use_sal and anchor_targets[t].item() != -100:
                    # float32 for numerical stability
                    logits_f32 = model.lm_head(h_last).float()  # (1, vocab)
                    target = anchor_targets[t:t+1].to(device)
                    anchor_losses.append(F.cross_entropy(logits_f32, target))

            # ★ EXIT LOSS: last think h must predict next explicit token
            if ci + 1 < len(chunks):
                next_start = chunks[ci + 1][0]
                if next_start < len(full_ids):
                    exit_logits = model.lm_head(h_last).float()  # (1, vocab)
                    exit_target = full_ids[next_start:next_start+1]
                    exit_losses.append(F.cross_entropy(exit_logits, exit_target))

        else:
            # ======== TEXT CHUNK (parallel) ========
            chunk_ids = full_ids[cs:ce].unsqueeze(0)  # (1, clen)
            is_prompt = (ce <= P)

            if is_prompt:
                # Prompt: no gradients needed, just build KV-cache
                with torch.no_grad():
                    out = model(input_ids=chunk_ids, past_key_values=kv,
                                use_cache=True, output_hidden_states=True)
                    kv = out.past_key_values
                    h_last = out.hidden_states[-1][:, -1, :].clone()
                # Re-enable grad for h_last (adapter needs it)
                h_last.requires_grad_(True)
            else:
                # Response text: needs gradients
                out = model(input_ids=chunk_ids, past_key_values=kv,
                            use_cache=True, output_hidden_states=True)
                kv = out.past_key_values
                h_last = out.hidden_states[-1][:, -1, :]  # (1, d)

                # CE loss: each position predicts next explicit token
                logits_f32 = out.logits[0].float()  # (clen, vocab)

                for tc in range(clen):
                    src_abs = cs + tc
                    tgt_abs = src_abs + 1

                    # Skip prompt-only positions
                    if src_abs < P:
                        continue
                    # Target must exist and be in response
                    if tgt_abs >= len(full_ids):
                        continue
                    # Skip if target is a <THINK> token
                    tgt_r = tgt_abs - P
                    if 0 <= tgt_r < R and think_mask[tgt_r].item() == 1:
                        continue

                    ce_losses.append(
                        F.cross_entropy(logits_f32[tc:tc+1],
                                        full_ids[tgt_abs:tgt_abs+1]))

    # ---- Aggregate (float32) ----
    device_f32 = torch.float32
    zero = torch.tensor(0.0, device=device, dtype=device_f32)

    if ce_losses:
        avg_ce = torch.stack(ce_losses).mean()
    else:
        avg_ce = zero

    if exit_losses:
        avg_exit = torch.stack(exit_losses).mean()
    else:
        avg_exit = zero

    if anchor_losses:
        avg_anchor = torch.stack(anchor_losses).mean()
    else:
        avg_anchor = zero

    total = avg_ce + exit_weight * avg_exit
    if use_sal and anchor_losses:
        total = total + beta * avg_anchor

    # Safety: if still NaN (shouldn't happen with f32), return zero loss
    if torch.isnan(total):
        total = zero
        logger.warning("  NaN loss detected, returning zero")

    return total, {
        "ce": avg_ce.item() if not torch.isnan(avg_ce) else 0.0,
        "exit": avg_exit.item() if not torch.isnan(avg_exit) else 0.0,
        "anchor": avg_anchor.item() if not torch.isnan(avg_anchor) else 0.0,
        "total": total.item() if not torch.isnan(total) else 0.0,
        "n_ce": len(ce_losses), "n_exit": len(exit_losses),
        "n_anch": len(anchor_losses),
    }


# ============================================================
# Batched eval (no latent)
# ============================================================

@torch.inference_mode()
def batched_eval(model, tokenizer, problems, data_config,
                 max_eval=200, batch_size=32, max_new_tokens=420):
    model.eval()
    device = next(model.parameters()).device
    pad_id = tokenizer.pad_token_id or 0
    probs = problems[:max_eval]
    correct = 0

    for bi in range(0, len(probs), batch_size):
        batch = probs[bi:bi+batch_size]
        texts = [build_prompt(p["question"], tokenizer, data_config) for p in batch]
        encs = [tokenizer(t, return_tensors="pt", add_special_tokens=False)["input_ids"].squeeze(0)
                for t in texts]
        ml = max(len(e) for e in encs)
        B = len(encs)
        ids = torch.full((B, ml), pad_id, dtype=torch.long, device=device)
        attn = torch.zeros((B, ml), dtype=torch.long, device=device)
        for i, e in enumerate(encs):
            ids[i, ml-len(e):] = e.to(device)
            attn[i, ml-len(e):] = 1
        out = model.generate(ids, attention_mask=attn, max_new_tokens=max_new_tokens,
                             do_sample=False, pad_token_id=pad_id)
        for j, p in enumerate(batch):
            resp = tokenizer.decode(out[j][ml:], skip_special_tokens=True)
            if check_answer(extract_model_answer(resp), p["answer"]):
                correct += 1
        torch.cuda.empty_cache()

    return correct / max(len(probs), 1), correct, len(probs)


# ============================================================
# Main
# ============================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--data_dir", type=str, default="./sal_test_data")
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--use_sal", type=str, default="false")
    parser.add_argument("--beta", type=float, default=1.0)
    parser.add_argument("--total_steps", type=int, default=800)
    parser.add_argument("--grad_accum", type=int, default=8)
    parser.add_argument("--model_lr", type=float, default=1e-6)
    parser.add_argument("--adapter_lr", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    args.use_sal = args.use_sal.lower() in ("true", "1", "yes")

    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    device = torch.device("cuda:0")
    config = get_config(); data_config = DataConfig()
    os.makedirs(args.output_dir, exist_ok=True)

    # ---- Model ----
    logger.info(f"Loading from {args.checkpoint}")
    model = AutoModelForCausalLM.from_pretrained(
        args.checkpoint, torch_dtype=torch.bfloat16,
        attn_implementation="sdpa", trust_remote_code=True).to(device)
    tokenizer = AutoTokenizer.from_pretrained(args.checkpoint, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    think_id = tokenizer.convert_tokens_to_ids("<SKIP>")

    # ---- Fresh adapter ----
    adapter = LatentBridgeAdapter(model.config.hidden_size).to(device=device, dtype=torch.bfloat16)
    n_params = sum(p.numel() for p in adapter.parameters())
    logger.info(f"  Fresh adapter: {n_params/1e6:.1f}M params, zero-init")

    # ---- Data (pre-tensorize on GPU) ----
    all_data = torch.load(os.path.join(args.data_dir, "sal_data.pt"), weights_only=False)
    for K in [1, 2, 3, 4]:
        for s in all_data[K]:
            s["_p"] = torch.tensor(s["prompt_ids"], dtype=torch.long, device=device)
            s["_r"] = torch.tensor(s["response_ids"], dtype=torch.long, device=device)
            s["_m"] = torch.tensor(s["think_mask"], dtype=torch.long, device=device)
            s["_a"] = torch.tensor(s["anchor_targets"], dtype=torch.long, device=device)
    total_samples = sum(len(all_data[K]) for K in [1,2,3,4])
    logger.info(f"  {total_samples} samples pre-loaded on GPU")

    # ---- Eval data ----
    eval_probs = []
    try:
        ed = GSM8KEvalDataset(config.data, tokenizer)
        eval_probs = ed.problems if hasattr(ed, "problems") else ed
    except: pass

    # ---- Optimizer ----
    nd = ["bias", "layer_norm", "layernorm", "rmsnorm"]
    optimizer = AdamW([
        {"params": [p for n, p in model.named_parameters()
                    if p.requires_grad and not any(k in n.lower() for k in nd)],
         "weight_decay": 0.01, "lr": args.model_lr},
        {"params": [p for n, p in model.named_parameters()
                    if p.requires_grad and any(k in n.lower() for k in nd)],
         "weight_decay": 0.0, "lr": args.model_lr},
        {"params": adapter.parameters(), "lr": args.adapter_lr, "weight_decay": 0.01},
    ], lr=args.model_lr, betas=(0.9, 0.95), eps=1e-8)
    scheduler = get_cosine_schedule_with_warmup(
        optimizer, int(args.total_steps * 0.10), args.total_steps)

    # ---- Curriculum ----
    ks = []
    ks.extend([1] * int(args.total_steps * 0.50))
    ks.extend([2] * int(args.total_steps * 0.20))
    ks.extend([3] * int(args.total_steps * 0.15))
    ks.extend([4] * int(args.total_steps * 0.15))
    while len(ks) < args.total_steps: ks.append(4)

    logger.info("=" * 70)
    sal_str = f"WITH SAL β={args.beta}" if args.use_sal else "NO SAL"
    logger.info(f"SAL Experiment — {sal_str}")
    logger.info(f"  Fresh adapter, chunked sequential forward, float32 loss")
    logger.info(f"  Steps: {args.total_steps}, grad_accum: {args.grad_accum}")
    logger.info(f"  K schedule: 1[{int(args.total_steps*0.50)}] → 2[{int(args.total_steps*0.20)}] "
                f"→ 3[{int(args.total_steps*0.15)}] → 4[{int(args.total_steps*0.15)}]")
    logger.info("=" * 70)

    if eval_probs:
        acc, c, t = batched_eval(model, tokenizer, eval_probs, data_config, batch_size=32)
        logger.info(f"  Initial eval: {acc:.4f} ({c}/{t})")

    log_history = []
    best_exit = float("inf")

    for step in range(args.total_steps):
        t0 = time.time()
        model.train(); adapter.train()
        model.config.use_cache = True

        K = ks[step]
        pool = all_data[K]
        optimizer.zero_grad(set_to_none=True)

        agg = {"ce": 0.0, "exit": 0.0, "anchor": 0.0, "total": 0.0}
        ok = 0

        for gi in range(args.grad_accum):
            s = random.choice(pool)
            try:
                loss, met = chunked_forward_sal(
                    model, adapter,
                    s["_p"], s["_r"], s["_m"], s["_a"],
                    think_id, use_sal=args.use_sal, beta=args.beta)

                if torch.isfinite(loss) and loss.item() > 0:
                    (loss / args.grad_accum).backward()
                    for k in agg:
                        agg[k] += met.get(k, 0) / args.grad_accum
                    ok += 1
                else:
                    # Zero or non-finite loss — skip but don't crash
                    pass
            except torch.cuda.OutOfMemoryError:
                torch.cuda.empty_cache()
            except Exception as e:
                if step < 5:
                    logger.warning(f"  step={step} gi={gi}: {type(e).__name__}: {e}")

        if ok > 0:
            gn = torch.nn.utils.clip_grad_norm_(
                list(model.parameters()) + list(adapter.parameters()), 1.0)
            gn = gn.item() if isinstance(gn, torch.Tensor) else gn
            optimizer.step()
        else:
            gn = 0.0
        scheduler.step()

        elapsed = time.time() - t0

        if step % 10 == 0:
            gpu_mb = torch.cuda.max_memory_allocated() // (1024*1024)
            logger.info(
                f"Step {step}/{args.total_steps} [K={K}] | "
                f"CE:{agg['ce']:.3f} Exit:{agg['exit']:.3f} "
                f"Anch:{agg['anchor']:.3f} Tot:{agg['total']:.3f} | "
                f"GN:{gn:.3f} OK:{ok}/{args.grad_accum} "
                f"T:{elapsed:.1f}s GPU:{gpu_mb}MB")
            log_history.append({
                "step": step, "K": K, **agg,
                "gn": gn, "time": elapsed, "ok": ok,
            })
            torch.cuda.reset_peak_memory_stats()

        if (step > 0 and step % 100 == 0) or step == args.total_steps - 1:
            if eval_probs:
                model.eval()
                acc, c, t = batched_eval(model, tokenizer, eval_probs, data_config, batch_size=32)
                logger.info(f"  Eval: {acc:.4f} ({c}/{t})")
                model.train()

            if 0 < agg["exit"] < best_exit:
                best_exit = agg["exit"]
                _save(model, adapter, tokenizer, step, log_history, args.output_dir, "best")
            _save(model, adapter, tokenizer, step, log_history, args.output_dir, f"step-{step}")

    _save(model, adapter, tokenizer, args.total_steps, log_history, args.output_dir, "final")
    with open(os.path.join(args.output_dir, "training_log.json"), "w") as f:
        json.dump(log_history, f, indent=2)
    logger.info("Done.")


def _save(model, adapter, tokenizer, step, log_history, output_dir, tag):
    sd = os.path.join(output_dir, tag); os.makedirs(sd, exist_ok=True)
    m = model._orig_mod if hasattr(model, "_orig_mod") else model
    m.save_pretrained(sd, safe_serialization=True)
    tokenizer.save_pretrained(sd)
    torch.save(adapter.state_dict(), os.path.join(sd, "adapter.pt"))
    with open(os.path.join(sd, "meta.json"), "w") as f:
        json.dump({"step": step, "log": log_history[-20:]}, f, indent=2, default=str)
    logger.info(f"  Saved: {sd}")


if __name__ == "__main__":
    main()