"""
Anchored Step 2: curriculum SFT data construction

Build three-level compressed SFT data from the anchor-detection results:
  Level 1: 30% latent, K_max=3  (shallow latent, short bridging)
  Level 2: 60% latent, K_max=8  (medium latent, mid-range reasoning)
  Level 3: 85% latent, K_max=15 (deep latent, keep only the key anchors)

Non-anchor tokens are replaced with <THINK>, count = max(1, floor(N/R))
R = compression ratio (Level 1: 3, Level 2: 4, Level 3: 5)

Usage:
    python build_sft_emerge.py \
        --anchor_data ./anchor_data/anchor_results.pt \
        --output_dir ./sft_emerge_data \
        --think_token "<THINK>"
"""

import os, sys, argparse, logging, json, math
import numpy as np
import torch
from typing import Dict, List, Tuple

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s",
                    handlers=[logging.StreamHandler(sys.stdout)])
logger = logging.getLogger("build_sft_emerge")


# ============================================================
# Level configs
# ============================================================

LEVEL_CONFIGS = {
    1: {"fold_ratio": 0.30, "compress_R": 3, "max_consec_think": 3},
    2: {"fold_ratio": 0.60, "compress_R": 4, "max_consec_think": 8},
    3: {"fold_ratio": 0.85, "compress_R": 5, "max_consec_think": 15},
}


# ============================================================
# select foldable tokens
# ============================================================

def select_foldable_tokens(anchor_mask: np.ndarray, fold_ratio: float,
                           delta_lls: np.ndarray = None) -> np.ndarray:
    """
    Select a fold_ratio fraction of the non-anchor tokens to fold.

    If delta_ll is available (the likelihood drop under ablation), prefer folding
    the lowest-impact tokens. Otherwise select randomly.

    Returns:
        fold_mask: bool array, True = this token is folded into <THINK>
    """
    T = len(anchor_mask)
    non_anchor_idx = np.where(~anchor_mask)[0]
    n_non_anchor = len(non_anchor_idx)
    
    if n_non_anchor == 0:
        return np.zeros(T, dtype=bool)
    
    n_to_fold = min(int(T * fold_ratio), n_non_anchor)
    
    if delta_lls is not None and len(delta_lls) == T:
        # sort by delta_ll ascending (fold the lowest-impact tokens first)
        # delta_ll near 0 = removal has no effect = safest fold candidate
        scores = np.abs(delta_lls[non_anchor_idx])
        sorted_idx = non_anchor_idx[np.argsort(scores)]
        selected = sorted_idx[:n_to_fold]
    else:
        np.random.seed(42)
        selected = np.random.choice(non_anchor_idx, n_to_fold, replace=False)
    
    fold_mask = np.zeros(T, dtype=bool)
    fold_mask[selected] = True
    return fold_mask


# ============================================================
# assemble a single SFT sequence
# ============================================================

def assemble_emerge_trajectory(
    full_ids: torch.Tensor,
    prompt_length: int,
    fold_mask: np.ndarray,
    think_token_id: int,
    compress_R: int,
    max_consec_think: int,
) -> Tuple[torch.Tensor, torch.Tensor, List[Tuple[int, int]]]:
    """
    Replace the folded regions with <THINK> tokens.

    For each contiguous folded segment (N tokens), replace it with K = max(1, floor(N/R)) <THINK> tokens.

    Returns:
        sft_ids: the assembled token sequence
        target_types: the type of each position
            0 = prompt (no loss)
            1 = normal explicit token (loss weight 1.0)
            2 = inside <THINK> (intermediate supervision weight 0.3)
            3 = <THINK> → explicit (exit, loss weight 3.0)
        think_segments: [(start_in_sft, end_in_sft, n_original_tokens), ...]
    """
    resp_start = prompt_length
    resp_ids = full_ids[resp_start:]
    T = len(resp_ids)

    # find contiguous folded segments
    segments = []  # [(start, end), ...] in response space, inclusive
    in_seg = False
    seg_start = 0
    for t in range(T):
        if fold_mask[t]:
            if not in_seg:
                seg_start = t
                in_seg = True
        else:
            if in_seg:
                segments.append((seg_start, t - 1))
                in_seg = False
    if in_seg:
        segments.append((seg_start, T - 1))

    # assemble
    parts_ids = []
    parts_types = []
    think_segs_in_sft = []

    # Prompt
    parts_ids.append(full_ids[:prompt_length])
    parts_types.append(torch.zeros(prompt_length, dtype=torch.long))

    prev_end = 0  # position within the response

    for seg_start, seg_end in segments:
        seg_n = seg_end - seg_start + 1
        n_think = max(1, min(max_consec_think, seg_n // compress_R))

        # explicit tokens before the segment
        if seg_start > prev_end:
            abs_s = resp_start + prev_end
            abs_e = resp_start + seg_start
            n_explicit = abs_e - abs_s
            parts_ids.append(full_ids[abs_s:abs_e])

            types = torch.ones(n_explicit, dtype=torch.long)  # all type=1 (normal explicit)
            # if the previous segment was <THINK>, the first token is exit
            if prev_end > 0 and len(parts_types) > 0:
                # check whether the previous segment was think
                last_part = parts_types[-1]
                if len(last_part) > 0 and last_part[-1].item() in (2, ):
                    types[0] = 3  # exit token
            parts_types.append(types)

        # <THINK> tokens
        sft_pos_start = sum(len(p) for p in parts_ids)
        think_ids = torch.full((n_think,), think_token_id, dtype=full_ids.dtype)
        think_types = torch.full((n_think,), 2, dtype=torch.long)  # intermediate <THINK>

        parts_ids.append(think_ids)
        parts_types.append(think_types)

        think_segs_in_sft.append((sft_pos_start, sft_pos_start + n_think - 1, seg_n))

        prev_end = seg_end + 1

    # final explicit tokens
    if prev_end < T:
        abs_s = resp_start + prev_end
        n_remaining = len(full_ids) - abs_s
        parts_ids.append(full_ids[abs_s:])

        types = torch.ones(n_remaining, dtype=torch.long)
        # if the preceding segment was <THINK>, mark exit
        if len(parts_types) > 0:
            last_part = parts_types[-1]
            if len(last_part) > 0 and last_part[-1].item() == 2:
                types[0] = 3
        parts_types.append(types)
    
    sft_ids = torch.cat(parts_ids)
    target_types = torch.cat(parts_types)
    
    assert len(sft_ids) == len(target_types)
    
    return sft_ids, target_types, think_segs_in_sft


# ============================================================
# generate intermediate supervision targets
# ============================================================

def generate_intermediate_targets(
    full_ids: torch.Tensor,
    prompt_length: int,
    fold_mask: np.ndarray,
    segments: List[Tuple[int, int]],  # folded segments in the original response
    n_think_per_seg: List[int],       # number of <THINK> tokens per segment
) -> List[List[int]]:
    """
    Generate intermediate supervision targets for each <THINK> segment.

    The target of the k-th <THINK> step = the token at position s + floor(k*N/K) in the original sequence.
    """
    resp_start = prompt_length
    all_targets = []
    
    for (seg_s, seg_e), n_think in zip(segments, n_think_per_seg):
        seg_n = seg_e - seg_s + 1
        targets = []
        for k in range(n_think):
            orig_pos = seg_s + min(int(k * seg_n / n_think), seg_n - 1)
            target_token = full_ids[resp_start + orig_pos].item()
            targets.append(target_token)
        all_targets.append(targets)
    
    return all_targets


# ============================================================
# Main
# ============================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--anchor_data", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default="./sft_emerge_data")
    parser.add_argument("--think_token", type=str, default="<THINK>")
    parser.add_argument("--max_seq_len", type=int, default=2048)
    args = parser.parse_args()
    
    os.makedirs(args.output_dir, exist_ok=True)
    np.random.seed(42)
    
    logger.info("=" * 70)
    logger.info("Curriculum SFT Data Construction")
    logger.info("=" * 70)

    # load anchor data
    logger.info("Loading anchor data...")
    anchor_results = torch.load(args.anchor_data, map_location="cpu", weights_only=False)
    logger.info(f"  {len(anchor_results)} trajectories")

    # build data for each level
    for level in [1, 2, 3]:
        cfg = LEVEL_CONFIGS[level]
        logger.info(f"\n--- Level {level}: fold={cfg['fold_ratio']:.0%}, R={cfg['compress_R']}, K_max={cfg['max_consec_think']} ---")
        
        level_data = []
        level_stats = {"n_traj": 0, "total_original": 0, "total_sft": 0,
                       "total_think": 0, "total_explicit": 0}
        
        for i, item in enumerate(anchor_results):
            full_ids = item["full_ids"]
            pl = item["prompt_length"]
            anchor_mask = item["anchor_mask"].numpy()

            # select tokens to fold
            fold_mask = select_foldable_tokens(anchor_mask, cfg["fold_ratio"])

            # find contiguous folded segments (used for intermediate supervision)
            segments_orig = []
            in_seg = False
            for t in range(len(fold_mask)):
                if fold_mask[t]:
                    if not in_seg: seg_s = t; in_seg = True
                else:
                    if in_seg: segments_orig.append((seg_s, t-1)); in_seg = False
            if in_seg: segments_orig.append((seg_s, len(fold_mask)-1))

            # assemble
            sft_ids, target_types, think_segs = assemble_emerge_trajectory(
                full_ids, pl, fold_mask,
                think_token_id=-1,  # placeholder, replaced at training time
                compress_R=cfg["compress_R"],
                max_consec_think=cfg["max_consec_think"])

            if len(sft_ids) > args.max_seq_len:
                continue

            # intermediate supervision targets
            n_think_list = [s[1] - s[0] + 1 for s in think_segs]  # temporarily use the segment length in the sft
            # recompute based on the original segments
            n_think_list_orig = []
            for seg_s, seg_e in segments_orig:
                seg_n = seg_e - seg_s + 1
                n_think_list_orig.append(
                    max(1, min(cfg["max_consec_think"], seg_n // cfg["compress_R"])))
            
            inter_targets = generate_intermediate_targets(
                full_ids, pl, fold_mask, segments_orig, n_think_list_orig)
            
            entry = {
                "sft_ids": sft_ids,
                "target_types": target_types,
                "prompt_length": pl,
                "original_length": len(full_ids),
                "think_segments": think_segs,  # (start, end, n_orig) in sft space
                "intermediate_targets": inter_targets,
                "ground_truth": item.get("ground_truth", ""),
                "anchor_ratio": float(anchor_mask.sum()) / max(len(anchor_mask), 1),
                "fold_ratio": float(fold_mask.sum()) / max(len(fold_mask), 1),
            }
            level_data.append(entry)

            n_think = (sft_ids == -1).sum().item()  # -1 is the placeholder
            n_explicit = len(sft_ids) - pl - n_think
            level_stats["n_traj"] += 1
            level_stats["total_original"] += len(full_ids) - pl
            level_stats["total_sft"] += len(sft_ids) - pl
            level_stats["total_think"] += n_think
            level_stats["total_explicit"] += n_explicit
        
        # save
        save_path = os.path.join(args.output_dir, f"sft_level{level}.pt")
        torch.save(level_data, save_path)
        
        s = level_stats
        logger.info(f"  Trajectories: {s['n_traj']}")
        logger.info(f"  Original resp tokens: {s['total_original']}")
        logger.info(f"  SFT resp tokens: {s['total_sft']}")
        logger.info(f"  <THINK> tokens: {s['total_think']} ({s['total_think']/max(s['total_sft'],1):.1%})")
        logger.info(f"  Explicit tokens: {s['total_explicit']} ({s['total_explicit']/max(s['total_sft'],1):.1%})")
        logger.info(f"  Saved: {save_path}")
    
    logger.info("\n" + "=" * 70)
    logger.info("All levels constructed.")
    logger.info(f"Next: python train_phase15_emerge.py --sft_dir {args.output_dir}")
    logger.info("=" * 70)


if __name__ == "__main__":
    main()
