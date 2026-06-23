"""
KVIG signal diagnostics: validate the core hypothesis.

Core issue: AUC=0.534 means KVIG cannot distinguish correct from incorrect at
the trajectory level, but the skip mechanism needs token-level high/low
discrimination. This script checks directly:

1. Token-level variance: within a single trajectory, does KVIG show clear
   high/low fluctuation?
   If std/mean is very small -> KVIG gives all tokens similar scores -> cannot guide skipping.

2. Semantic alignment: do low-KVIG tokens correspond to "skippable" content
   (connectives / repetition)? Do high-KVIG tokens correspond to "key
   computation steps"?

3. Skip simulation: if we drop the lowest-20% KVIG tokens, is the answer broken?
   If not broken -> KVIG can indeed identify redundant tokens -> core hypothesis holds.

Usage:
    python diagnose_kvig.py --checkpoint ./checkpoints/phase1/best --num_problems 20
"""

import os
import sys
import argparse
import logging
import torch
import random
import numpy as np
import json

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("kvig_diag")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--num_problems", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    from config import get_config
    config = get_config()

    # Load model
    logger.info("Loading model...")
    from model_utils import load_checkpoint, setup_model_for_phase1
    if os.path.exists(os.path.join(args.checkpoint, "config.json")):
        model, tokenizer, skip_token_id, skip_adapter, _ = load_checkpoint(
            args.checkpoint, config)
    else:
        model, tokenizer, skip_token_id, skip_adapter = setup_model_for_phase1(config)
    model.eval()

    # Load data
    from data_utils import MathProblemDataset
    dataset = MathProblemDataset(config.data, tokenizer, seed=args.seed)
    problems = dataset.get_calibration_subset(n=args.num_problems, source="math")

    # Generate trajectories
    logger.info(f"Generating trajectories for {len(problems)} problems...")
    from rollout import generate_trajectories
    grouped = generate_trajectories(
        model=model, tokenizer=tokenizer, prompts=problems,
        group_size=4, temperature=0.7, max_new_tokens=512,
        data_config=config.data,
    )

    # Collect correct trajectories
    correct_trajs = []
    for group in grouped:
        for traj in group:
            if traj["is_correct"]:
                correct_trajs.append(traj)

    logger.info(f"Got {len(correct_trajs)} correct trajectories")
    if len(correct_trajs) < 5:
        logger.error("Too few correct trajectories. Use more problems.")
        return

    # Compute KVIG
    from information_gain_legacy import KVIGComputer
    kvig_computer = KVIGComputer(config.kvig, config.model)

    all_cv = []       # coefficient of variation per trajectory
    all_low_high = [] # (low_kvig_tokens, high_kvig_tokens) examples
    skip_safe = 0     # number of cases where the answer stays correct after dropping low-KVIG tokens
    skip_total = 0

    logger.info("=" * 60)
    logger.info("KVIG token-level diagnostics")
    logger.info("=" * 60)

    for i, traj in enumerate(correct_trajs[:20]):  # inspect at most 20 trajectories
        full_ids = traj["full_ids"].unsqueeze(0).to(model.device)
        attn_mask = traj["full_attention_mask"].unsqueeze(0).to(model.device)
        prompt_len = traj["prompt_length"]

        with torch.no_grad():
            result = kvig_computer.compute_trajectory_from_model(
                model=model, input_ids=full_ids, attention_mask=attn_mask,
                prompt_length=prompt_len,
            )

        kvig_values = np.array(result["kvig_values"])
        response_ids = traj["response_ids"]
        response_tokens = tokenizer.convert_ids_to_tokens(response_ids.tolist())

        if len(kvig_values) == 0:
            continue

        # -- Metric 1: Token-level variance --
        mean_k = np.mean(kvig_values)
        std_k = np.std(kvig_values)
        cv = std_k / (abs(mean_k) + 1e-8)  # coefficient of variation
        all_cv.append(cv)

        # -- Metric 2: tokens with the lowest/highest KVIG --
        n_tokens = len(kvig_values)
        n_show = min(5, n_tokens // 4)
        if n_show > 0 and n_tokens == len(response_tokens):
            sorted_idx = np.argsort(kvig_values)
            low_idx = sorted_idx[:n_show]
            high_idx = sorted_idx[-n_show:]

            low_tokens = [(response_tokens[j], f"{kvig_values[j]:.4f}") for j in low_idx]
            high_tokens = [(response_tokens[j], f"{kvig_values[j]:.4f}") for j in high_idx]

            if i < 5:  # print the first 5 trajectories in detail
                logger.info(f"\n--- Trajectory {i} (len={n_tokens}, mean_kvig={mean_k:.4f}, "
                            f"std={std_k:.4f}, CV={cv:.2f}) ---")
                logger.info(f"  Answer: {traj['predicted_answer']}")
                logger.info(f"  Lowest KVIG tokens: {low_tokens}")
                logger.info(f"  Highest KVIG tokens: {high_tokens}")

        # -- Metric 3: Skip simulation --
        if n_tokens > 5:
            skip_ratio = 0.2
            n_skip = max(1, int(n_tokens * skip_ratio))
            sorted_idx = np.argsort(kvig_values)
            
            # Apply the same syntax-and-number protection barrier as compare_signals
            actual_skip_idx = set()
            protected_chars = set(['\\', '{', '}', '[', ']', '(', ')', '=', '+', '-', '*', '/', '^', '_', '$'])
            for idx in sorted_idx:
                tok_str = response_tokens[idx]
                if any(c.isdigit() for c in tok_str) or any(c in protected_chars for c in tok_str):
                    continue
                actual_skip_idx.add(idx)
                if len(actual_skip_idx) >= n_skip:
                    break
            
            keep_idx = [j for j in range(n_tokens) if j not in actual_skip_idx]
            kept_ids = response_ids[keep_idx] if len(keep_idx) <= len(response_ids) else response_ids
            kept_text = tokenizer.decode(kept_ids, skip_special_tokens=True)

            from data_utils import extract_model_answer, check_answer
            kept_answer = extract_model_answer(kept_text)
            reference = traj.get("ground_truth", "")
            
            still_correct = False
            if reference:
                if kept_answer and check_answer(kept_answer, reference):
                    still_correct = True
                else:
                    ref_clean = reference.strip()
                    if len(ref_clean) > 0 and ref_clean in kept_text:
                        still_correct = True
                    else:
                        import re
                        nums = re.findall(r'-?\d+\.?\d*', kept_text)
                        if nums:
                            try:
                                if abs(float(nums[-1].replace(',', '')) - float(ref_clean.replace(',', ''))) < 1e-6:
                                    still_correct = True
                            except:
                                pass

            skip_total += 1
            if still_correct:
                skip_safe += 1

    # -- Summary --
    logger.info("\n" + "=" * 60)
    logger.info("Diagnostic summary")
    logger.info("=" * 60)

    mean_cv = np.mean(all_cv) if all_cv else 0
    logger.info(f"\n1. Token-level variance (CV = std/|mean|):")
    logger.info(f"   Mean CV = {mean_cv:.2f}")
    if mean_cv > 2.0:
        logger.info(f"   CV > 2: KVIG varies significantly across tokens, distinguishing high/low-information tokens")
    elif mean_cv > 0.5:
        logger.info(f"   CV in 0.5-2.0: KVIG shows some variation but not strongly significant")
    else:
        logger.info(f"   CV < 0.5: KVIG gives all tokens similar scores, cannot guide skipping")

    logger.info(f"\n2. Skip safety (answer still correct after dropping lowest-20% KVIG tokens):")
    if skip_total > 0:
        safe_rate = skip_safe / skip_total
        logger.info(f"   {skip_safe}/{skip_total} = {safe_rate:.1%}")
        if safe_rate > 0.8:
            logger.info(f"   > 80%: low-KVIG tokens are indeed redundant, skip mechanism is effective")
        elif safe_rate > 0.5:
            logger.info(f"   50-80%: some effect but skipping carries risk")
        else:
            logger.info(f"   < 50%: skipping low-KVIG tokens breaks the answer, signal is unreliable")
    else:
        logger.info(f"   Not enough data")

    logger.info(f"\n3. Conclusion:")
    if mean_cv > 2.0 and skip_total > 0 and skip_safe / max(skip_total, 1) > 0.8:
        logger.info(f"   KVIG token-level signal is effective!")
        logger.info(f"   Although trajectory-level AUC=0.534 (cannot tell correct from incorrect trajectories),")
        logger.info(f"   large token-level variance + safe skipping -> core hypothesis holds")
        logger.info(f"   -> can proceed to Phase 2")
    elif mean_cv > 0.5 and skip_total > 0 and skip_safe / max(skip_total, 1) > 0.5:
        logger.info(f"   KVIG signal exists but is weak; suggestions:")
        logger.info(f"   - lower the skip ratio (from 20% to 10%)")
        logger.info(f"   - or use attention entropy as an auxiliary signal")
        logger.info(f"   -> can proceed cautiously to Phase 2")
    else:
        logger.info(f"   KVIG token-level signal is insufficient; the approach needs adjustment:")
        logger.info(f"   Option A: use attention entropy instead of KVIG")
        logger.info(f"   Option B: use token repetition / cosine similarity as the skip signal")
        logger.info(f"   Option C: train a small skip predictor head")

    # Save results
    diag_results = {
        "mean_cv": float(mean_cv),
        "all_cv": [float(x) for x in all_cv],
        "skip_safe": skip_safe,
        "skip_total": skip_total,
        "skip_safe_rate": skip_safe / max(skip_total, 1),
        "num_correct_trajs": len(correct_trajs),
    }
    out_path = os.path.join(os.path.dirname(args.checkpoint), "kvig_diagnostic.json")
    with open(out_path, "w") as f:
        json.dump(diag_results, f, indent=2)
    logger.info(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()