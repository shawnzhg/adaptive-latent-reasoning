"""
Fast GSM8K evaluation script.

Optimized for a single-GPU 1.5B model, maximizing parallelism:
  - Large-batch prefill (fill 64-128 prompts at once)
  - Two modes: skip-disabled (plain model) + skip-enabled (Golden Rule)
  - GSM8K test set has 1319 problems, expected to finish in 5-10 minutes

Usage:
    python eval_gsm8k_fast.py --checkpoint ./checkpoints/phase15/best
    python eval_gsm8k_fast.py --checkpoint ./checkpoints/phase15/best --mode both
    python eval_gsm8k_fast.py --checkpoint Qwen/Qwen2.5-1.5B-Instruct --mode no_skip  # original baseline
"""

import os, sys, json, time, argparse, logging
from typing import List, Dict, Tuple

import torch
import torch.nn.functional as F
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer

from config import get_config, DataConfig
from data_utils import GSM8KEvalDataset, build_prompt, extract_model_answer, check_answer
from latent_adapter import SkipAdapter
from information_gain import MCIGComputer, MCIGState

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("eval")


# ============================================================
# Batched generation (no skip, plain model)
# ============================================================

@torch.inference_mode()
def generate_batch_no_skip(
    model, tokenizer, prompts: List[str],
    max_new_tokens: int = 512, temperature: float = 0.0,
) -> List[str]:
    """
    Large-batch parallel generation (no skip token).
    temperature=0 -> greedy, deterministic evaluation.
    """
    device = next(model.parameters()).device
    pad_id = tokenizer.pad_token_id or 0
    eos_id = tokenizer.eos_token_id
    B = len(prompts)

    # Tokenize + left-pad
    all_ids = []
    all_lengths = []
    for p in prompts:
        ids = tokenizer(p, return_tensors="pt", add_special_tokens=False)["input_ids"].squeeze(0)
        all_ids.append(ids)
        all_lengths.append(len(ids))

    max_prompt_len = max(all_lengths)
    input_ids = torch.full((B, max_prompt_len), pad_id, dtype=torch.long, device=device)
    attn_mask = torch.zeros((B, max_prompt_len), dtype=torch.long, device=device)

    for i, ids in enumerate(all_ids):
        pl = len(ids)
        input_ids[i, max_prompt_len - pl:] = ids.to(device)
        attn_mask[i, max_prompt_len - pl:] = 1

    # Prefill
    out = model(input_ids=input_ids, attention_mask=attn_mask,
                use_cache=True, output_hidden_states=False)
    past_kv = out.past_key_values
    is_eos = torch.zeros(B, dtype=torch.bool, device=device)

    generated = [[] for _ in range(B)]

    for step in range(max_new_tokens):
        if is_eos.all():
            break

        logits = out.logits[:, -1, :]

        if temperature <= 0:
            next_tokens = logits.argmax(dim=-1)
        else:
            probs = F.softmax(logits / temperature, dim=-1)
            next_tokens = torch.multinomial(probs, 1).squeeze(-1)

        hit_eos = (next_tokens == eos_id)

        for b in range(B):
            if not is_eos[b] and not hit_eos[b]:
                generated[b].append(next_tokens[b].item())

        is_eos = is_eos | hit_eos
        if is_eos.all():
            break

        new_attn = (~is_eos).long().unsqueeze(1)
        attn_mask = torch.cat([attn_mask, new_attn], dim=1)

        out = model(input_ids=next_tokens.unsqueeze(1),
                    attention_mask=attn_mask,
                    past_key_values=past_kv, use_cache=True)
        past_kv = out.past_key_values

    return [tokenizer.decode(g, skip_special_tokens=True) for g in generated]


# ============================================================
# Batched generation (Golden Rule skip)
# ============================================================

@torch.inference_mode()
def generate_batch_with_skip(
    model, adapter, tokenizer, prompts: List[str],
    skip_token_id: int, max_new_tokens: int = 512,
    temperature: float = 0.0, k_max: int = 8,
) -> Tuple[List[str], List[int], List[int]]:
    """
    Large-batch Golden Rule generation.
    Returns: (responses, skip_counts, response_lengths)
    """
    device = next(model.parameters()).device
    pad_id = tokenizer.pad_token_id or 0
    eos_id = tokenizer.eos_token_id
    B = len(prompts)

    # Tokenize + left-pad
    all_ids = []
    all_lengths = []
    for p in prompts:
        ids = tokenizer(p, return_tensors="pt", add_special_tokens=False)["input_ids"].squeeze(0)
        all_ids.append(ids)
        all_lengths.append(len(ids))

    max_prompt_len = max(all_lengths)
    input_ids = torch.full((B, max_prompt_len), pad_id, dtype=torch.long, device=device)
    attn_mask = torch.zeros((B, max_prompt_len), dtype=torch.long, device=device)

    for i, ids in enumerate(all_ids):
        pl = len(ids)
        input_ids[i, max_prompt_len - pl:] = ids.to(device)
        attn_mask[i, max_prompt_len - pl:] = 1

    # Prefill
    out = model(input_ids=input_ids, attention_mask=attn_mask,
                use_cache=True, output_hidden_states=True)
    past_kv = out.past_key_values
    last_h = out.hidden_states[-1][:, -1, :]

    is_eos = torch.zeros(B, dtype=torch.bool, device=device)
    consecutive_skips = torch.zeros(B, dtype=torch.long, device=device)
    generated = [[] for _ in range(B)]
    skip_counts = [0] * B

    for step in range(max_new_tokens):
        if is_eos.all():
            break

        logits = out.logits[:, -1, :]

        if temperature <= 0:
            next_tokens = logits.argmax(dim=-1)
        else:
            probs = F.softmax(logits / temperature, dim=-1)
            next_tokens = torch.multinomial(probs, 1).squeeze(-1)

        # K_max enforcement
        is_skip = (next_tokens == skip_token_id)
        consecutive_skips = torch.where(is_skip, consecutive_skips + 1, 0)
        violators = consecutive_skips > k_max
        if violators.any():
            logits_masked = logits.clone()
            logits_masked[:, skip_token_id] = -float('inf')
            alt_tokens = logits_masked.argmax(dim=-1)
            next_tokens = torch.where(violators, alt_tokens, next_tokens)
            consecutive_skips = torch.where(violators, 0, consecutive_skips)
            is_skip = (next_tokens == skip_token_id)

        hit_eos = (next_tokens == eos_id)

        for b in range(B):
            if not is_eos[b] and not hit_eos[b]:
                generated[b].append(next_tokens[b].item())
                if is_skip[b].item():
                    skip_counts[b] += 1

        is_eos = is_eos | hit_eos
        if is_eos.all():
            break

        # Golden Rule routing
        z_skip = adapter(last_h)  # (B, dim)
        emb = model.get_input_embeddings()(next_tokens.unsqueeze(1))  # (B, 1, dim)
        next_inputs = torch.where(is_skip.view(-1, 1, 1), z_skip.unsqueeze(1), emb)

        new_attn = (~is_eos).long().unsqueeze(1)
        attn_mask = torch.cat([attn_mask, new_attn], dim=1)

        out = model(inputs_embeds=next_inputs,
                    attention_mask=attn_mask,
                    past_key_values=past_kv, use_cache=True,
                    output_hidden_states=True)
        past_kv = out.past_key_values
        last_h = out.hidden_states[-1][:, -1, :]

    # Decode (skip tokens are excluded from text)
    responses = []
    resp_lengths = []
    for b in range(B):
        visible = [t for t in generated[b] if t != skip_token_id]
        responses.append(tokenizer.decode(visible, skip_special_tokens=True))
        resp_lengths.append(len(generated[b]))

    return responses, skip_counts, resp_lengths


# ============================================================
# Main evaluation logic
# ============================================================

def evaluate_gsm8k(
    model, tokenizer, problems: List[Dict], data_config,
    adapter=None, skip_token_id=None,
    batch_size: int = 64, max_new_tokens: int = 512,
    temperature: float = 0.0, mode: str = "no_skip",
) -> Dict:
    """
    Batched GSM8K evaluation.

    mode:
      "no_skip"   - plain model, no Adapter
      "with_skip" - Golden Rule, uses Adapter
    """
    model.eval()
    if adapter is not None:
        adapter.eval()

    total = len(problems)
    correct = 0
    total_skips = 0
    total_tokens = 0
    results = []

    for start in range(0, total, batch_size):
        end = min(start + batch_size, total)
        batch = problems[start:end]

        # Build prompts
        prompt_texts = [build_prompt(p["question"], tokenizer, data_config) for p in batch]
        gt_answers = [p["answer"] for p in batch]

        t0 = time.time()

        if mode == "no_skip":
            responses = generate_batch_no_skip(
                model, tokenizer, prompt_texts, max_new_tokens, temperature)
            batch_skips = [0] * len(batch)
            batch_lengths = [0] * len(batch)  # exact length not needed
        else:
            responses, batch_skips, batch_lengths = generate_batch_with_skip(
                model, adapter, tokenizer, prompt_texts,
                skip_token_id, max_new_tokens, temperature)

        elapsed = time.time() - t0

        # Check answers
        batch_correct = 0
        for i, (resp, gt) in enumerate(zip(responses, gt_answers)):
            pred = extract_model_answer(resp)
            is_correct = check_answer(pred, gt)
            if is_correct:
                correct += 1
                batch_correct += 1
            total_skips += batch_skips[i]
            if mode == "with_skip":
                total_tokens += batch_lengths[i]

            results.append({
                "question": batch[i]["question"],
                "ground_truth": gt,
                "prediction": pred,
                "correct": is_correct,
                "skips": batch_skips[i],
                "response_preview": resp[:200],
            })

        # Progress
        done = min(end, total)
        acc_so_far = correct / done
        speed = len(batch) / elapsed
        skip_info = f" Skips/q:{np.mean(batch_skips):.1f}" if mode == "with_skip" else ""
        logger.info(
            f"  [{done}/{total}] Acc: {acc_so_far:.4f} ({correct}/{done})"
            f" | Batch: {elapsed:.1f}s ({speed:.1f} q/s){skip_info}")

    acc = correct / total
    summary = {
        "accuracy": acc,
        "correct": correct,
        "total": total,
        "mode": mode,
    }
    if mode == "with_skip" and total_tokens > 0:
        summary["mean_skips"] = total_skips / total
        summary["skip_ratio"] = total_skips / total_tokens
        summary["mean_response_len"] = total_tokens / total

    return summary, results


# ============================================================
# Main
# ============================================================

def main():
    parser = argparse.ArgumentParser(description="Fast GSM8K evaluation")
    parser.add_argument("--checkpoint", type=str, required=True,
                        help="Model checkpoint path (or HF model name)")
    parser.add_argument("--mode", type=str, default="both",
                        choices=["no_skip", "with_skip", "both"],
                        help="Evaluation mode")
    parser.add_argument("--batch_size", type=int, default=64,
                        help="Batch size for generation (tune to GPU memory)")
    parser.add_argument("--max_new_tokens", type=int, default=512)
    parser.add_argument("--temperature", type=float, default=0.0,
                        help="0.0 = greedy (deterministic)")
    parser.add_argument("--max_problems", type=int, default=0,
                        help="0 = all problems, else subsample")
    parser.add_argument("--save_results", action="store_true",
                        help="Save per-problem results to JSON")
    parser.add_argument("--output_dir", type=str, default="./eval_results")
    args = parser.parse_args()

    device = torch.device("cuda:0")

    # -- Load model --
    logger.info(f"Loading model from {args.checkpoint}")
    model = AutoModelForCausalLM.from_pretrained(
        args.checkpoint, torch_dtype=torch.bfloat16,
        attn_implementation="sdpa", trust_remote_code=True).to(device)
    tokenizer = AutoTokenizer.from_pretrained(args.checkpoint, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model.eval()

    # -- Load adapter (if exists) --
    adapter = None
    skip_token_id = None
    ap = os.path.join(args.checkpoint, "skip_adapter.pt")
    if os.path.exists(ap):
        d_model = model.config.hidden_size
        adapter = SkipAdapter(hidden_size=d_model, bottleneck_ratio=4)
        adapter.load_state_dict(torch.load(ap, map_location="cpu", weights_only=False))
        adapter = adapter.to(device=device, dtype=model.dtype)
        adapter.eval()
        skip_token_id = tokenizer.convert_tokens_to_ids("<SKIP>")
        logger.info(f"  Adapter loaded, skip_token_id={skip_token_id}")
    elif args.mode in ("with_skip", "both"):
        logger.warning("  No adapter found! Falling back to no_skip mode.")
        if args.mode == "with_skip":
            args.mode = "no_skip"
        elif args.mode == "both":
            args.mode = "no_skip"

    # -- Load GSM8K --
    config = get_config()
    data_config = config.data
    try:
        eval_ds = GSM8KEvalDataset(data_config, tokenizer)
        problems = eval_ds.problems
    except Exception:
        # Fallback: try loading directly
        from datasets import load_dataset
        ds = load_dataset("openai/gsm8k", "main", split="test")
        problems = [{"question": ex["question"], "answer": ex["answer"]} for ex in ds]

    if args.max_problems > 0:
        problems = problems[:args.max_problems]

    logger.info(f"GSM8K test set: {len(problems)} problems")
    logger.info(f"Batch size: {args.batch_size}, Temperature: {args.temperature}")

    # -- GPU info --
    gpu_mem = torch.cuda.get_device_properties(0).total_memory / 1024**3
    logger.info(f"GPU: {torch.cuda.get_device_name(0)}, {gpu_mem:.0f} GB")

    os.makedirs(args.output_dir, exist_ok=True)
    all_summaries = {}

    # -- Run evaluation --
    modes = []
    if args.mode == "both":
        modes = ["no_skip", "with_skip"]
    else:
        modes = [args.mode]

    for mode in modes:
        logger.info("=" * 60)
        logger.info(f"Mode: {mode}")
        logger.info("=" * 60)

        t_start = time.time()

        summary, results = evaluate_gsm8k(
            model, tokenizer, problems, data_config,
            adapter=adapter if mode == "with_skip" else None,
            skip_token_id=skip_token_id if mode == "with_skip" else None,
            batch_size=args.batch_size,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            mode=mode,
        )

        total_time = time.time() - t_start
        summary["total_time_seconds"] = total_time
        summary["problems_per_second"] = len(problems) / total_time

        logger.info("-" * 60)
        logger.info(f"  Accuracy:   {summary['accuracy']:.4f} ({summary['correct']}/{summary['total']})")
        logger.info(f"  Time:       {total_time:.1f}s ({summary['problems_per_second']:.1f} q/s)")
        if mode == "with_skip":
            logger.info(f"  Mean skips: {summary.get('mean_skips', 0):.2f}")
            logger.info(f"  Skip ratio: {summary.get('skip_ratio', 0):.4f}")
            logger.info(f"  Mean len:   {summary.get('mean_response_len', 0):.0f}")
        logger.info("-" * 60)

        all_summaries[mode] = summary

        if args.save_results:
            out_path = os.path.join(args.output_dir, f"gsm8k_{mode}.json")
            with open(out_path, "w") as f:
                json.dump({"summary": summary, "results": results}, f, indent=2, ensure_ascii=False)
            logger.info(f"  Results saved to {out_path}")

    # -- Comparison (if both modes) --
    if len(all_summaries) == 2:
        no_skip = all_summaries["no_skip"]
        with_skip = all_summaries["with_skip"]
        gap = no_skip["accuracy"] - with_skip["accuracy"]

        logger.info("=" * 60)
        logger.info("COMPARISON")
        logger.info("=" * 60)
        logger.info(f"  No-skip accuracy:   {no_skip['accuracy']:.4f}")
        logger.info(f"  With-skip accuracy: {with_skip['accuracy']:.4f}")
        logger.info(f"  Gap:                {gap:+.4f} ({'OK' if gap < 0.05 else 'WARNING: large gap'})")
        if "mean_skips" in with_skip:
            logger.info(f"  Mean skips/query:   {with_skip['mean_skips']:.2f}")
            logger.info(f"  Skip ratio:         {with_skip['skip_ratio']:.4f}")

    # Save summary
    summary_path = os.path.join(args.output_dir, "eval_summary.json")
    with open(summary_path, "w") as f:
        json.dump(all_summaries, f, indent=2)
    logger.info(f"\nSummary saved to {summary_path}")


if __name__ == "__main__":
    main()