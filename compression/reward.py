"""
Reward function

Phase 1: R_outcome = +1 (correct) / -1 (incorrect)
Advantage: Dr. GRPO (Group-Relative, no std normalization)

Dr. GRPO paper (Liu et al., 2025):
  Standard GRPO: A_i = (R_i - mean(R)) / std(R)  <- has a difficulty bias
  Dr. GRPO:      A_i = R_i - mean(R)             <- no bias
"""

import math
from typing import List, Dict, Optional
from dataclasses import dataclass

from config import RewardConfig, KVIGConfig


# ============================================================
# Phase 1: Outcome Reward (+1 / -1)
# ============================================================

def compute_outcome_reward(is_correct: bool) -> float:
    return 1.0 if is_correct else -1.0


def compute_total_reward(
    trajectory: Dict,
    phase: int = 1,
    config: RewardConfig = None,
    **kwargs,
) -> float:
    """Phase 1: use only the outcome reward"""
    if config is None:
        config = RewardConfig()
    is_correct = trajectory["is_correct"]
    if phase == 1:
        return compute_outcome_reward(is_correct)
    # Phase 2: extension (not yet implemented)
    return compute_outcome_reward(is_correct)


# ============================================================
# Advantage computation
# ============================================================

def compute_group_advantages(
    rewards: List[float],
    use_std_norm: bool = False,
    eps: float = 1e-8,
) -> List[float]:
    """
    Group-Relative Advantage (the core of GRPO)

    Dr. GRPO (default, use_std_norm=False):
        A_i = R_i - mean(R)
        Does not divide by std -> removes difficulty bias
        All correct / all wrong -> mean=+-1 -> A_i=0 -> no gradient signal (correct behavior)

    Standard GRPO (use_std_norm=True):
        A_i = (R_i - mean(R)) / (std(R) + eps)
        All correct / all wrong -> std=0 -> returns 0
    """
    n = len(rewards)
    if n == 0:
        return []

    mean_r = sum(rewards) / n

    if not use_std_norm:
        # Dr. GRPO: do not divide by std
        return [r - mean_r for r in rewards]

    # Standard GRPO: divide by std
    var_r = sum((r - mean_r) ** 2 for r in rewards) / n
    std_r = var_r ** 0.5
    if std_r < eps:
        return [0.0 for _ in rewards]
    return [(r - mean_r) / (std_r + eps) for r in rewards]