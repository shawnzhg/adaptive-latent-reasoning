"""
Anchored Step 1: Counterfactual Anchor Detection

For each token in a correct trajectory, use the attention-masking method to test:
  after deleting this token, does the model's prediction over the answer region drop significantly?
  if yes  → this token is an anchor (must be output explicitly)
  if no   → this token can be folded into the latent space

Additional syntax protection: digits, math symbols, and \\boxed are forced to be anchors.

Output: an anchor_mask per trajectory (1=anchor, 0=foldable)

Usage:
    python detect_anchors.py \
        --checkpoint ./checkpoints/phase1/best \
        --calibration_data ./calibration/calibration_raw_data.pt \
        --output_dir ./anchor_data \
        --epsilon 0.15 \
        --n_answer_tokens 20 \
        --max_trajectories 3000
"""

import os, sys, argparse, logging, json, glob, math, re
import numpy as np
import torch
import torch.nn.functional as F
from typing import Dict, List, Tuple, Optional

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s",
                    handlers=[logging.StreamHandler(sys.stdout)])
logger = logging.getLogger("detect_anchors")


# ============================================================
# syntax protection: forced anchor detection
# ============================================================

def detect_syntax_anchors(token_ids: torch.Tensor, tokenizer) -> np.ndarray:
    """
    Detect tokens that must be output explicitly for syntactic reasons.
    Returns a bool array, True = forced anchor.
    """
    T = len(token_ids)
    forced = np.zeros(T, dtype=bool)

    # characters/patterns to protect
    protect_chars = set('0123456789+-×÷=^{}\\$')
    protect_keywords = ['boxed', '\\frac', '\\sqrt', '\\times', '\\div',
                        '\\left', '\\right', '\\text', '\\cdot']

    for t in range(T):
        tok_str = tokenizer.decode([token_ids[t].item()])
        # contains a digit
        if any(c.isdigit() for c in tok_str):
            forced[t] = True
        # contains a math symbol
        elif any(c in tok_str for c in protect_chars):
            forced[t] = True
        # contains a LaTeX keyword
        elif any(kw in tok_str for kw in protect_keywords):
            forced[t] = True

    return forced


# ============================================================
# counterfactual ablation: per-token detection
# ============================================================

@torch.no_grad()
def detect_anchors_for_trajectory(
    model,
    full_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    prompt_length: int,
    tokenizer,
    epsilon: float = 0.15,
    n_answer_tokens: int = 20,
    batch_ablation: int = 8,  # how many tokens to ablate in parallel per pass
) -> Tuple[np.ndarray, Dict]:
    """
    Run counterfactual anchor detection on a single trajectory.

    Method: for each token in the response, mask it out via attention masking
    and check whether the likelihood over the answer region (the last
    n_answer_tokens tokens) drops significantly.

    Returns:
        anchor_mask: (resp_len,) bool array, True = anchor
        stats: statistics
    """
    device = next(model.parameters()).device
    seq_len = full_ids.shape[0]
    resp_len = seq_len - prompt_length

    if resp_len < n_answer_tokens + 5:
        # mark all tokens of a too-short trajectory as anchors
        return np.ones(resp_len, dtype=bool), {"too_short": True}

    # answer region
    answer_start = max(seq_len - n_answer_tokens, prompt_length + 1)
    answer_end = seq_len

    # ── Step 1: compute the answer-region log-likelihood of the full sequence ──
    inp = full_ids.unsqueeze(0).to(device)
    mask_full = attention_mask.unsqueeze(0).to(device)
    
    out_full = model(input_ids=inp, attention_mask=mask_full, use_cache=False)
    logits_full = out_full.logits[0]
    
    ll_full = 0.0
    n_ans = 0
    for t in range(answer_start, answer_end):
        target = full_ids[t].item()
        lp = F.log_softmax(logits_full[t - 1].float(), dim=-1)
        ll_full += lp[target].item()
        n_ans += 1
    ll_full_avg = ll_full / max(n_ans, 1)
    
    del out_full

    # ── Step 2: syntax protection ──
    resp_ids = full_ids[prompt_length:]
    syntax_anchors = detect_syntax_anchors(resp_ids, tokenizer)

    # ── Step 3: per-token counterfactual ablation ──
    # only run counterfactual detection on non-syntax-anchor tokens (syntax anchors are already determined)
    causal_anchors = np.zeros(resp_len, dtype=bool)
    delta_lls = np.full(resp_len, 0.0)

    tokens_to_check = [t for t in range(resp_len) if not syntax_anchors[t]]

    # do not check the answer region itself (the answer region defaults to anchors)
    ans_start_resp = answer_start - prompt_length
    tokens_to_check = [t for t in tokens_to_check if t < ans_start_resp]

    # batched ablation
    for batch_start in range(0, len(tokens_to_check), batch_ablation):
        batch_tokens = tokens_to_check[batch_start:batch_start + batch_ablation]
        B = len(batch_tokens)

        # build B ablated versions of the attention mask
        inp_batch = inp.expand(B, -1)  # (B, seq_len)
        masks = mask_full.expand(B, -1).clone()  # (B, seq_len)

        for i, t_resp in enumerate(batch_tokens):
            t_abs = prompt_length + t_resp
            masks[i, t_abs] = 0  # mask out this token

        out_abl = model(input_ids=inp_batch, attention_mask=masks, use_cache=False)
        logits_abl = out_abl.logits  # (B, seq_len, V)
        
        for i, t_resp in enumerate(batch_tokens):
            ll_abl = 0.0
            for t in range(answer_start, answer_end):
                target = full_ids[t].item()
                lp = F.log_softmax(logits_abl[i, t - 1].float(), dim=-1)
                ll_abl += lp[target].item()
            ll_abl_avg = ll_abl / max(n_ans, 1)
            
            delta = ll_abl_avg - ll_full_avg
            delta_lls[t_resp] = delta
            
            if delta < -epsilon:
                causal_anchors[t_resp] = True
        
        del out_abl
        torch.cuda.empty_cache()
    
    # ── Step 4: merge anchors ──
    # mark the entire answer region as anchors
    answer_anchors = np.zeros(resp_len, dtype=bool)
    answer_anchors[ans_start_resp:] = True
    
    anchor_mask = syntax_anchors | causal_anchors | answer_anchors
    
    stats = {
        "resp_len": resp_len,
        "n_syntax_anchors": int(syntax_anchors.sum()),
        "n_causal_anchors": int(causal_anchors.sum()),
        "n_answer_anchors": int(answer_anchors.sum()),
        "n_total_anchors": int(anchor_mask.sum()),
        "anchor_ratio": float(anchor_mask.sum()) / max(resp_len, 1),
        "mean_delta_ll": float(np.mean(delta_lls[tokens_to_check])) if tokens_to_check else 0.0,
    }
    
    return anchor_mask, stats


# ============================================================
# data loading
# ============================================================

def load_correct_trajectories(calibration_data_path=None, raw_chunks_dir=None,
                               max_trajectories=None):
    trajectories = []
    
    if calibration_data_path and os.path.exists(calibration_data_path):
        logger.info(f"Loading from {calibration_data_path}")
        data = torch.load(calibration_data_path, map_location="cpu", weights_only=False)
        if isinstance(data, list):
            trajectories = data
        elif isinstance(data, dict) and "trajectories" in data:
            trajectories = data["trajectories"]
    elif raw_chunks_dir and os.path.exists(raw_chunks_dir):
        logger.info(f"Loading from chunks in {raw_chunks_dir}")
        for cf in sorted(glob.glob(os.path.join(raw_chunks_dir, "chunk_*.pt"))):
            try:
                chunk = torch.load(cf, map_location="cpu", weights_only=False)
                raw = chunk.get("trajectories", [])
                if raw and isinstance(raw[0], list):
                    raw = [t for g in raw for t in g]
                trajectories.extend(raw)
            except Exception as e:
                logger.warning(f"Failed: {cf}: {e}")
    
    correct = [t for t in trajectories if t.get("is_correct", False)]
    logger.info(f"Total: {len(trajectories)}, correct: {len(correct)}")
    
    if max_trajectories and len(correct) > max_trajectories:
        import random; random.seed(42)
        correct = random.sample(correct, max_trajectories)
    
    return correct


# ============================================================
# Main
# ============================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--calibration_data", type=str, default=None)
    parser.add_argument("--raw_chunks_dir", type=str, default=None)
    parser.add_argument("--output_dir", type=str, default="./anchor_data")
    parser.add_argument("--epsilon", type=float, default=0.15)
    parser.add_argument("--n_answer_tokens", type=int, default=20)
    parser.add_argument("--batch_ablation", type=int, default=8)
    parser.add_argument("--max_trajectories", type=int, default=3000)
    args = parser.parse_args()
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    logger.info("=" * 70)
    logger.info("Counterfactual Anchor Detection")
    logger.info("=" * 70)

    # load the model
    logger.info("Loading model...")
    from transformers import AutoModelForCausalLM, AutoTokenizer
    model = AutoModelForCausalLM.from_pretrained(
        args.checkpoint, torch_dtype=torch.bfloat16,
        attn_implementation="sdpa", trust_remote_code=True)
    tokenizer = AutoTokenizer.from_pretrained(args.checkpoint, trust_remote_code=True)
    model.eval()
    device = torch.device("cuda:0")
    model = model.to(device)
    logger.info(f"  Model loaded on {device}")

    # load data
    logger.info("Loading correct trajectories...")
    trajectories = load_correct_trajectories(
        args.calibration_data, args.raw_chunks_dir, args.max_trajectories)
    logger.info(f"  {len(trajectories)} trajectories")
    
    # detect anchors
    logger.info("Detecting anchors...")
    results = []
    all_stats = []
    
    for i, traj in enumerate(trajectories):
        full_ids = traj["full_ids"]
        attn_mask = traj.get("full_attention_mask", torch.ones_like(full_ids))
        pl = traj["prompt_length"]
        
        anchor_mask, stats = detect_anchors_for_trajectory(
            model, full_ids, attn_mask, pl, tokenizer,
            epsilon=args.epsilon,
            n_answer_tokens=args.n_answer_tokens,
            batch_ablation=args.batch_ablation)
        
        results.append({
            "full_ids": full_ids,
            "prompt_length": pl,
            "anchor_mask": torch.tensor(anchor_mask, dtype=torch.bool),
            "ground_truth": traj.get("ground_truth", traj.get("answer", "")),
        })
        all_stats.append(stats)
        
        if (i + 1) % 100 == 0:
            avg_ratio = np.mean([s["anchor_ratio"] for s in all_stats])
            logger.info(f"  {i+1}/{len(trajectories)} | avg anchor ratio: {avg_ratio:.1%}")
            torch.cuda.empty_cache()
    
    # save
    save_path = os.path.join(args.output_dir, "anchor_results.pt")
    torch.save(results, save_path)
    logger.info(f"  Saved: {save_path}")

    # statistics report
    logger.info("\n" + "=" * 70)
    logger.info("ANCHOR DETECTION REPORT")
    logger.info("=" * 70)
    ratios = [s["anchor_ratio"] for s in all_stats]
    syntax_counts = [s["n_syntax_anchors"] for s in all_stats]
    causal_counts = [s["n_causal_anchors"] for s in all_stats]
    logger.info(f"  Trajectories: {len(results)}")
    logger.info(f"  Anchor ratio: {np.mean(ratios):.1%} (mean), {np.median(ratios):.1%} (median)")
    logger.info(f"  Syntax anchors/traj: {np.mean(syntax_counts):.1f}")
    logger.info(f"  Causal anchors/traj: {np.mean(causal_counts):.1f}")
    logger.info(f"  → ~{100-np.mean(ratios)*100:.0f}% of tokens can be folded into latent space")
    
    stats_path = os.path.join(args.output_dir, "anchor_stats.json")
    with open(stats_path, "w") as f:
        json.dump({"mean_anchor_ratio": float(np.mean(ratios)),
                    "median_anchor_ratio": float(np.median(ratios)),
                    "n_trajectories": len(results),
                    "config": vars(args)}, f, indent=2)
    logger.info(f"  Stats: {stats_path}")
    logger.info("=" * 70)


if __name__ == "__main__":
    main()
