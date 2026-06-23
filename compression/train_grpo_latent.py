"""
Phase 2 — v5 (MCIG Token-Level Reward)

Core changes vs v4:
  1. R_skip redesign: each skip's reward/penalty is determined by its MCIG value
     - MCIG < tau_low  → +0.15 (good skip, redundant region)
     - MCIG > tau_high → -0.30 (bad skip, critical region)
     - middle region → 0 (no reward, no penalty)
  2. Remove the uniform bias: keep only the MCIG-guided bias, eliminating blind exploration
  3. Late Phase A exploration check: turn on the bias but set l_skip=0 to verify accuracy stability
  4. Runtime MCIG threshold calibration: set automatically from the MCIG distribution collected in Phase A

Core problem solved:
  In v4 accuracy collapsed from 0.45 to 0.15 because the uniform bias pushed up skip
  everywhere, including critical reasoning nodes. v5 only encourages skip at positions
  that MCIG confirms are redundant.
"""

import contextlib
import math
import os, sys, json, time, random, argparse, logging, gc, copy
from typing import Dict, List, Tuple, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from transformers import AutoModelForCausalLM, AutoTokenizer, get_cosine_schedule_with_warmup

from config import get_config, SPARKConfig, DataConfig
from data_utils import (MathProblemDataset, GSM8KEvalDataset, load_math500,
                        build_prompt, extract_model_answer, check_answer)
from information_gain import MCIGComputer, MCIGState
from latent_adapter import SkipAdapter
from reward import compute_group_advantages

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s",
                    handlers=[logging.StreamHandler(sys.stdout)])
logger = logging.getLogger("phase2")


# ============================================================
# MCIG Calibrator — collect the distribution from Phase A, set thresholds automatically
# ============================================================

class MCIGCalibrator:
    """
    Collect all MCIG values during Phase A, then compute quantile thresholds at the start of Phase B.
    """
    def __init__(self):
        self.all_values = []
        self.tau_low = 0.05    # default value, overwritten at the start of Phase B
        self.tau_high = 0.15
        self.calibrated = False

    def collect(self, mcig_values_list: List[List[float]]):
        """Collect MCIG values at every step of Phase A"""
        for vals in mcig_values_list:
            self.all_values.extend(vals)

    def calibrate(self):
        """Called at the end of Phase A to compute the quantile thresholds"""
        if len(self.all_values) < 100:
            logger.warning(f"  MCIG calibration: only {len(self.all_values)} values, using defaults")
            return
        arr = np.array(self.all_values)
        arr = arr[np.isfinite(arr)]
        if len(arr) < 100:
            return

        self.tau_low = float(np.percentile(arr, 25))
        self.tau_high = float(np.percentile(arr, 75))
        self.calibrated = True

        logger.info(f"  MCIG Calibrated from {len(arr)} values:")
        logger.info(f"    tau_low  (p25) = {self.tau_low:.4f}")
        logger.info(f"    tau_high (p75) = {self.tau_high:.4f}")
        logger.info(f"    mean = {arr.mean():.4f}, std = {arr.std():.4f}")
        logger.info(f"    min = {arr.min():.4f}, max = {arr.max():.4f}")

        self.all_values = []  # free memory


# ============================================================
# Exploration — pure MCIG-guided, no uniform bias
# ============================================================

def get_exploration_params(phase: str, calibrator: MCIGCalibrator) -> Dict:
    """
    Phase A (0-10%):  no exploration
    Phase A-late (8-10%): turn on the MCIG bias but l_skip=0 (pure verification)
    Phase B (10-40%): MCIG-guided bias (push skip only where MCIG is low)
    Phase C (40-70%): MCIG-guided bias gradually decays
    Phase D (70-100%): turn off the bias, rely entirely on the policy
    """
    if phase == "A":
        return {"use_mcig_bias": False, "bias_boost": 0.0, "bias_brake": 0.0}
    elif phase == "B":
        return {
            "use_mcig_bias": True,
            "bias_boost": 2.0,    # low MCIG: +2.0 (exp(2/0.9)≈9x probability)
            "bias_brake": -1.5,   # high MCIG: -1.5 (exp(-1.5/0.9)≈0.19x)
        }
    elif phase == "C":
        return {
            "use_mcig_bias": True,
            "bias_boost": 1.0,    # gradually weakening
            "bias_brake": -1.0,
        }
    else:  # D
        return {"use_mcig_bias": False, "bias_boost": 0.0, "bias_brake": 0.0}


# ============================================================
# Curriculum
# ============================================================

class AdaptiveCurriculum:
    PHASE_LAMBDAS = {
        # Phase A: pure outcome, collect the MCIG baseline
        "A": {"l_skip": (0.0, 0.0),  "lp": (0.0, 0.0),  "l2": (0.0, 0.0),  "lmod": (0.0, 0.0)},
        # Phase B: skip reward turned on (MCIG quality-based, no need for an extremely high lambda)
        "B": {"l_skip": (0.5, 1.0),  "lp": (0.0, 0.0),  "l2": (0.0, 0.0),  "lmod": (0.0, 0.0)},
        # Phase C: introduce the efficiency reward
        "C": {"l_skip": (1.0, 0.8),  "lp": (0.0, 0.0),  "l2": (0.0, 0.05), "lmod": (0.0, 0.1)},
        # Phase D: stabilize
        "D": {"l_skip": (0.8, 0.8),  "lp": (0.0, 0.0),  "l2": (0.05, 0.05),"lmod": (0.1, 0.1)},
    }

    def __init__(self, total_steps=600):
        self.total_steps = total_steps
        self.phase_thresholds = {
            "A": int(total_steps * 0.10),
            "B": int(total_steps * 0.40),
            "C": int(total_steps * 0.70),
        }
        self.current_phase = "A"
        self.global_step = 0

    def step(self, metrics):
        self.global_step += 1
        if self.global_step < self.phase_thresholds["A"]:
            self.current_phase = "A"
        elif self.global_step < self.phase_thresholds["B"]:
            self.current_phase = "B"
        elif self.global_step < self.phase_thresholds["C"]:
            self.current_phase = "C"
        else:
            self.current_phase = "D"

    def get_lambdas(self):
        targets = self.PHASE_LAMBDAS[self.current_phase]
        ss = 0
        if self.current_phase == "B": ss = self.phase_thresholds["A"]
        elif self.current_phase == "C": ss = self.phase_thresholds["B"]
        elif self.current_phase == "D": ss = self.phase_thresholds["C"]
        es = self.phase_thresholds.get(self.current_phase, self.total_steps)
        t = min(max((self.global_step - ss) / max(es - ss, 1), 0), 1)
        return {k: s + (e - s) * t for k, (s, e) in targets.items()}

    def get_state(self):
        return {"phase": self.current_phase, "global_step": self.global_step}


# ============================================================
# Reward v5 — MCIG token-level quality reward
# ============================================================

def compute_phase2_reward(traj, mcig_values, skip_positions, lambdas,
                          calibrator: MCIGCalibrator,
                          t_ref=350.0, beta_eff=3.0,
                          l_soft=800, l_hard=1024, kappa=0.5,
                          k_max=32):
    """
    Core change: R_skip is no longer count-based but quality-based.
    The MCIG value at each skip position determines that skip's reward/penalty.

    Good skip (low MCIG, redundant region): +0.15
    Bad skip (high MCIG, critical region): -0.30 (penalty is 2x the reward, biased toward safety)
    Middle skip: 0
    """
    is_correct = traj["is_correct"]
    resp_len = traj["response_length"]
    n_skips = len(skip_positions) if skip_positions else 0
    skip_ratio = n_skips / max(resp_len, 1)

    r_outcome = 1.0 if is_correct else -1.0

    # * MCIG quality-based skip reward
    tau_low = calibrator.tau_low
    tau_high = calibrator.tau_high
    good_skips = 0
    bad_skips = 0
    neutral_skips = 0

    if skip_positions and mcig_values:
        for t in skip_positions:
            if t >= len(mcig_values):
                continue
            m = mcig_values[t]
            if m < tau_low:
                good_skips += 1
            elif m > tau_high:
                bad_skips += 1
            else:
                neutral_skips += 1

    # reward good skips, penalize bad skips (applies whether the answer is right or wrong)
    r_skip_quality = good_skips * 0.15 - bad_skips * 0.30

    # extra efficiency reward when correct: encourage more good skips
    if is_correct and good_skips >= 3:
        r_skip_quality += 0.5  # correct + at least 3 good skips = extra 0.5

    # efficiency reward (only when correct)
    if is_correct and t_ref > 0:
        explicit_len = resp_len - n_skips
        r_efficiency = beta_eff * max(0.0, 1.0 - explicit_len / t_ref)
    else:
        r_efficiency = 0.0

    # consecutive-skip penalty
    max_consec = 0
    if skip_positions:
        c = 1
        for i in range(1, len(skip_positions)):
            if skip_positions[i] == skip_positions[i-1] + 1:
                c += 1
            else:
                max_consec = max(max_consec, c)
                c = 1
        max_consec = max(max_consec, c)
    r_overthink = -0.05 * max(0.0, max_consec - k_max)

    # length penalty
    r_overlong = 0.0
    if resp_len > l_soft:
        r_overlong = -kappa * min((resp_len - l_soft) / max(l_hard - l_soft, 1), 1.0)

    l_skip = lambdas["l_skip"]
    l2 = lambdas["l2"]
    final_skip = l_skip * r_skip_quality
    final_eff = l2 * r_efficiency

    r_total = r_outcome + final_skip + final_eff + r_overlong + r_overthink

    return r_total, {
        "r_outcome": r_outcome,
        "r_skip": final_skip,
        "r_skip_quality_raw": r_skip_quality,
        "r_efficiency": final_eff,
        "r_overlong": r_overlong,
        "r_overthink": r_overthink,
        "r_total": r_total,
        "n_skips": n_skips,
        "good_skips": good_skips,
        "bad_skips": bad_skips,
        "neutral_skips": neutral_skips,
        "skip_ratio": skip_ratio,
    }


# ============================================================
# NGRPO
# ============================================================

def compute_advantages_ngrpo(rewards):
    if not rewards:
        return []
    std = np.std(rewards)
    if std > 1e-6:
        return [r - np.mean(rewards) for r in rewards]
    return [0.0] * len(rewards)


# ============================================================
# Generation — pure MCIG-guided bias
# ============================================================

@torch.inference_mode()
def generate_all_trajectories(
    model, adapter, tokenizer, mcig_computer, prompts, data_config,
    skip_token_id, group_size=8, max_new_tokens=512, temperature=0.7,
    top_p=0.95, k_max=32, explore_params=None, calibrator=None,
):
    device = next(model.parameters()).device
    num_prompts = len(prompts)
    prompt_texts, prompt_lengths, gt_answers = [], [], []
    for pd in prompts:
        text = build_prompt(pd["question"], tokenizer, data_config)
        ids = tokenizer(text, return_tensors="pt", add_special_tokens=False)["input_ids"].squeeze(0)
        prompt_texts.append(text)
        prompt_lengths.append(len(ids))
        gt_answers.append(pd["answer"])

    max_pl = max(prompt_lengths)
    all_ids, all_masks, pl_seq, gt_seq, pi_seq = [], [], [], [], []
    pad_id = tokenizer.pad_token_id or 0
    for pi in range(num_prompts):
        ids = tokenizer(prompt_texts[pi], return_tensors="pt", add_special_tokens=False)["input_ids"].squeeze(0)
        pl = len(ids)
        padded = torch.cat([torch.full((max_pl - pl,), pad_id, dtype=torch.long), ids])
        attn = torch.cat([torch.zeros(max_pl - pl, dtype=torch.long), torch.ones(pl, dtype=torch.long)])
        for _ in range(group_size):
            all_ids.append(padded)
            all_masks.append(attn)
            pl_seq.append(pl)
            gt_seq.append(gt_answers[pi])
            pi_seq.append(pi)

    input_ids = torch.stack(all_ids).to(device)
    attn_mask = torch.stack(all_masks).to(device)
    B = input_ids.shape[0]

    out = model(input_ids=input_ids, attention_mask=attn_mask, use_cache=True, output_hidden_states=True)
    past_kv = out.past_key_values
    last_h = out.hidden_states[-1][:, -1, :]

    mcig_states = [MCIGState() for _ in range(B)]
    for b in range(B):
        _, mcig_states[b] = mcig_computer.compute_step(last_h[b], out.logits[b, -1, :], mcig_states[b])

    gen_ids = [[] for _ in range(B)]
    mcig_vals = [[] for _ in range(B)]
    skip_pos = [[] for _ in range(B)]
    consec = torch.zeros(B, dtype=torch.long, device=device)
    is_eos = torch.zeros(B, dtype=torch.bool, device=device)
    cur_attn = attn_mask

    use_mcig = explore_params.get("use_mcig_bias", False) if explore_params else False
    bias_boost = explore_params.get("bias_boost", 0.0) if explore_params else 0.0
    bias_brake = explore_params.get("bias_brake", 0.0) if explore_params else 0.0

    tau_low = calibrator.tau_low if calibrator else 0.05
    tau_high = calibrator.tau_high if calibrator else 0.15

    for step in range(max_new_tokens):
        if is_eos.all():
            break

        logits = out.logits[:, -1, :]
        slogits = logits.clone()

        # * Pure MCIG-guided bias: push only where MCIG is low, suppress where MCIG is high
        if use_mcig and (bias_boost > 0 or bias_brake < 0):
            for b in range(B):
                if is_eos[b]:
                    continue
                if len(mcig_vals[b]) < 3:
                    # cold start: the first few tokens lack enough MCIG data, add no bias
                    continue

                window = mcig_vals[b][-15:]
                current = window[-1]

                # use the calibrated absolute thresholds, not relative quantiles
                if current < tau_low:
                    slogits[b, skip_token_id] += bias_boost
                elif current > tau_high:
                    slogits[b, skip_token_id] += bias_brake  # negative value
                # middle region: no intervention

        if temperature <= 0:
            next_tokens = slogits.argmax(dim=-1)
        else:
            probs = F.softmax(slogits / temperature, dim=-1)
            if top_p < 1.0:
                sp, si = torch.sort(probs, descending=True)
                cum = sp.cumsum(dim=-1)
                sp[cum - sp > top_p] = 0.0
                sp /= sp.sum(dim=-1, keepdim=True)
                next_tokens = torch.gather(si, -1, torch.multinomial(sp, 1)).squeeze(-1)
            else:
                next_tokens = torch.multinomial(probs, 1).squeeze(-1)

        is_skip = (next_tokens == skip_token_id)
        consec = torch.where(is_skip, consec + 1, 0)

        # K_max safety valve
        viol = consec > k_max
        if viol.any():
            ml = logits.clone()
            ml[:, skip_token_id] = -float('inf')
            next_tokens = torch.where(viol, ml.argmax(dim=-1), next_tokens)
            consec = torch.where(viol, 0, consec)
            is_skip = (next_tokens == skip_token_id)

        hit_eos = (next_tokens == tokenizer.eos_token_id)
        for b in range(B):
            if not is_eos[b] and not hit_eos[b]:
                gen_ids[b].append(next_tokens[b].item())
                if is_skip[b].item():
                    skip_pos[b].append(len(gen_ids[b]) - 1)
        is_eos = is_eos | hit_eos
        if is_eos.all():
            break

        z = adapter(last_h)
        emb = model.get_input_embeddings()(next_tokens.unsqueeze(1))
        ni = torch.where(is_skip.view(-1, 1, 1), z.unsqueeze(1), emb)
        cur_attn = torch.cat([cur_attn, (~is_eos).long().unsqueeze(1)], dim=1)
        out = model(inputs_embeds=ni, attention_mask=cur_attn, past_key_values=past_kv,
                    use_cache=True, output_hidden_states=True)
        past_kv = out.past_key_values
        last_h = out.hidden_states[-1][:, -1, :]
        for b in range(B):
            if not is_eos[b]:
                score, mcig_states[b] = mcig_computer.compute_step(
                    last_h[b], out.logits[b, -1, :], mcig_states[b])
                mcig_vals[b].append(float(score))

    del past_kv, out, cur_attn, last_h, input_ids, attn_mask, consec, is_eos
    torch.cuda.empty_cache()

    grouped = [[] for _ in range(num_prompts)]
    for b in range(B):
        pi = pi_seq[b]
        resp_ids = torch.tensor(gen_ids[b], dtype=torch.long)
        orig = tokenizer(prompt_texts[pi], return_tensors="pt", add_special_tokens=False)["input_ids"].squeeze(0)
        full = torch.cat([orig, resp_ids])
        txt = tokenizer.decode([t for t in gen_ids[b] if t != skip_token_id], skip_special_tokens=True)
        grouped[pi].append({
            "prompt_ids": orig, "prompt_length": pl_seq[b],
            "response_ids": resp_ids, "response_text": txt,
            "full_ids": full, "full_attention_mask": torch.ones_like(full),
            "predicted_answer": extract_model_answer(txt), "ground_truth": gt_seq[b],
            "is_correct": check_answer(extract_model_answer(txt), gt_seq[b]),
            "response_length": len(gen_ids[b]),
            "mcig_values": mcig_vals[b], "skip_positions": skip_pos[b],
        })
    return grouped


# ============================================================
# Two-Pass Log Probs
# ============================================================

def compute_log_probs_batch(model, skip_adapter, trajs, skip_token_id, device,
                            no_grad=False, cached_hidden=None):
    if not trajs:
        return [], None
    pad_id = 0
    ml = max(t["full_ids"].shape[-1] for t in trajs)
    bids = [F.pad(t["full_ids"].squeeze(), (0, ml - t["full_ids"].shape[-1]), value=pad_id) for t in trajs]
    bmsk = [torch.cat([torch.ones(t["full_ids"].shape[-1]),
                        torch.zeros(ml - t["full_ids"].shape[-1])]).long() for t in trajs]
    iids = torch.stack(bids).to(device)
    amsk = torch.stack(bmsk).to(device)
    B, L = iids.shape
    d = model.config.hidden_size
    sm = (iids == skip_token_id)
    outer = torch.no_grad() if no_grad else contextlib.nullcontext()
    with outer:
        if cached_hidden is not None:
            hidden = cached_hidden
            rh = None
        else:
            with torch.no_grad():
                o1 = model(input_ids=iids, attention_mask=amsk, output_hidden_states=True, use_cache=False)
                hidden = o1.hidden_states[-1].detach()
                del o1
                torch.cuda.empty_cache()
            rh = hidden
        avl = []
        for b in range(B):
            sp = sm[b].nonzero(as_tuple=True)[0]
            if len(sp) == 0:
                dz = skip_adapter(hidden[b, 0:1])
                avl.append(torch.zeros(L, d, device=device, dtype=hidden.dtype) + dz.sum() * 0.0)
                continue
            prev = (sp - 1).clamp(min=0)
            z = skip_adapter(hidden[b, prev])
            n = len(sp)
            oh = torch.zeros(L, n, device=device, dtype=z.dtype)
            oh[sp, torch.arange(n, device=device)] = 1.0
            avl.append(oh @ z)
        av = torch.stack(avl)
        sm3 = sm.unsqueeze(-1)

        def hook(m, i, o):
            return torch.where(sm3, av, o)
        emb = model.get_input_embeddings()
        h = emb.register_forward_hook(hook)
        try:
            o2 = model(input_ids=iids, attention_mask=amsk, use_cache=False)
        finally:
            h.remove()
        lpa = F.log_softmax(o2.logits.float(), dim=-1)
    result = []
    for i, t in enumerate(trajs):
        pl = t["prompt_length"]
        sl = t["full_ids"].shape[-1]
        rid = t["response_ids"].to(device)
        result.append(lpa[i, pl - 1:sl - 1, :].gather(-1, rid.unsqueeze(-1)).squeeze(-1))
    return result, rh


# ============================================================
# Trainer
# ============================================================

class Phase2Trainer:
    def __init__(self, model, ref_model, skip_adapter, tokenizer, skip_token_id,
                 config, dataset, eval_datasets, device, t_ref=350.0,
                 max_steps=600, ref_device=None):
        self.model = model.to(device)
        self.ref_device = ref_device or device
        self.ref_model = ref_model.to(self.ref_device) if ref_model is not None else None
        self.skip_adapter = skip_adapter.to(device=device, dtype=model.dtype)
        self.tokenizer = tokenizer
        self.skip_token_id = skip_token_id
        self.config = config
        self.dataset = dataset
        self.eval_datasets = eval_datasets or {}
        self.device = device
        self.t_ref = t_ref
        self.data_config = DataConfig()
        self.mcig_computer = MCIGComputer(config.kvig, config.model)
        self.max_steps = max_steps
        self.group_size = 8
        self.batch_size = 12
        self.temperature = 0.9
        self.kl_coeff = 0.01
        self.clip_range = 0.1
        self.max_new_tokens = config.model.max_new_tokens
        self.max_grad_norm = 0.5
        self.eval_interval = 50
        self.save_interval = 50
        self.log_interval = 1
        self.output_dir = "./checkpoints/phase2_v5"
        self.loss_token_norm = float(config.model.max_new_tokens)
        self.logprob_mb = 32
        self.ppo_mb = 16

        self.curriculum = AdaptiveCurriculum(total_steps=max_steps)
        self.calibrator = MCIGCalibrator()

        model.gradient_checkpointing_enable()
        model.config.use_cache = False
        if ref_model:
            ref_model.eval()
            for p in ref_model.parameters():
                p.requires_grad = False
        self.ref_adapter = self.skip_adapter
        if ref_model and self.ref_device != device:
            self.ref_adapter = copy.deepcopy(skip_adapter).to(device=self.ref_device, dtype=model.dtype)
            self.ref_adapter.eval()
            for p in self.ref_adapter.parameters():
                p.requires_grad = False

        nd = ["bias", "layer_norm", "layernorm", "rmsnorm", "embed_tokens", "wte", "lm_head"]
        self.optimizer = AdamW([
            {"params": [p for n, p in model.named_parameters()
                        if p.requires_grad and not any(k in n.lower() for k in nd)],
             "weight_decay": 0.01},
            {"params": [p for n, p in model.named_parameters()
                        if p.requires_grad and any(k in n.lower() for k in nd)],
             "weight_decay": 0.0},
            {"params": skip_adapter.parameters(), "lr": 5e-6, "weight_decay": 0.01},
        ], lr=1e-6, betas=(0.9, 0.95), eps=1e-8)
        self.scheduler = get_cosine_schedule_with_warmup(
            self.optimizer, int(max_steps * 0.05), max_steps)
        self.global_step = 0
        self.log_history = []
        self.zero_std_count = 0
        self.total_groups = 0

    def train_step(self):
        self.model.train()
        self.skip_adapter.train()
        t0 = time.time()
        step = self.global_step
        phase = self.curriculum.current_phase
        lambdas = self.curriculum.get_lambdas()

        # exploration parameters
        explore = get_exploration_params(phase, self.calibrator)

        # late Phase A: trigger calibration
        phase_a_end = self.curriculum.phase_thresholds["A"]
        if step == phase_a_end - 1 and not self.calibrator.calibrated:
            self.calibrator.calibrate()

        # Generation
        batch = self.dataset.sample_batch(self.batch_size)
        self.model.eval()
        self.skip_adapter.eval()
        self.model.config.use_cache = True
        try:
            grouped = generate_all_trajectories(
                self.model, self.skip_adapter, self.tokenizer, self.mcig_computer,
                batch, self.data_config, self.skip_token_id, self.group_size,
                self.max_new_tokens, self.temperature,
                explore_params=explore, calibrator=self.calibrator)
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            half = len(batch) // 2
            grouped = []
            for sub in [batch[:half], batch[half:]]:
                grouped.extend(generate_all_trajectories(
                    self.model, self.skip_adapter, self.tokenizer, self.mcig_computer,
                    sub, self.data_config, self.skip_token_id, self.group_size,
                    self.max_new_tokens, self.temperature,
                    explore_params=explore, calibrator=self.calibrator))
                torch.cuda.empty_cache()
        self.model.train()
        self.skip_adapter.train()
        self.model.config.use_cache = False
        torch.cuda.empty_cache()
        gen_time = time.time() - t0

        # Rewards
        all_trajs, all_r, all_bd = [], [], []
        gb = []
        off = 0
        bc, bt = 0, 0
        for group in grouped:
            for traj in group:
                r, bk = compute_phase2_reward(
                    traj, traj.get("mcig_values", []),
                    traj.get("skip_positions", []),
                    lambdas, self.calibrator, self.t_ref)
                all_trajs.append(traj)
                all_r.append(r)
                all_bd.append(bk)
                if traj["is_correct"]:
                    bc += 1
                bt += 1
            gb.append((off, off + len(group)))
            off += len(group)
        N = len(all_trajs)
        if N == 0:
            self.global_step += 1
            return {"step": step, "skipped": True}

        # Phase A: collect MCIG values for calibration
        if phase == "A":
            self.calibrator.collect([t.get("mcig_values", []) for t in all_trajs])

        # Log Probs
        MB = self.logprob_mb
        old_lps, ref_lps, ch = [], [], []
        for i in range(0, N, MB):
            chunk = all_trajs[i:i + MB]
            co, h = compute_log_probs_batch(
                self.model, self.skip_adapter, chunk,
                self.skip_token_id, self.device, no_grad=True)
            old_lps.extend([l.detach() for l in co])
            ch.append(h)
            if self.ref_model and self.kl_coeff > 0:
                cr, _ = compute_log_probs_batch(
                    self.ref_model, self.ref_adapter, chunk,
                    self.skip_token_id, self.ref_device, no_grad=True)
                ref_lps.extend([l.detach().to(self.device) for l in cr])
            else:
                ref_lps.extend([None] * len(chunk))
            torch.cuda.empty_cache()

        adj = list(all_r)
        total_kl = 0.0
        if self.ref_model and self.kl_coeff > 0:
            for idx in range(N):
                if ref_lps[idx] is not None:
                    resp_ids = all_trajs[idx]["response_ids"].to(self.device)
                    emask = (resp_ids != self.skip_token_id).float()
                    kl_arr = (old_lps[idx] - ref_lps[idx]) * emask
                    n_exp = max(emask.sum().item(), 1.0)
                    kl = kl_arr.sum().item() / n_exp
                    total_kl += kl
                    adj[idx] -= self.kl_coeff * kl

        fa = []
        szs = 0
        for s, e in gb:
            gr = adj[s:e]
            fa.extend(compute_advantages_ngrpo(gr))
            self.total_groups += 1
            if np.std(gr) < 1e-6:
                szs += 1
                self.zero_std_count += 1
        torch.cuda.empty_cache()
        lp_time = time.time() - t0 - gen_time

        # PPO
        tl, tc = 0.0, 0.0
        lmod = lambdas["lmod"]
        self.optimizer.zero_grad()
        for ci, start in enumerate(range(0, N, self.ppo_mb)):
            end = min(start + self.ppo_mb, N)
            chunk = all_trajs[start:end]
            chh = ch[ci] if (self.ppo_mb == MB and ci < len(ch)) else None
            clps, _ = compute_log_probs_batch(
                self.model, self.skip_adapter, chunk,
                self.skip_token_id, self.device, no_grad=False, cached_hidden=chh)
            cl = torch.tensor(0.0, device=self.device, requires_grad=True)
            for j, clp in enumerate(clps):
                idx = start + j
                olp = old_lps[idx]
                av = fa[idx]
                if lmod > 0:
                    mv = all_trajs[idx].get("mcig_values", [])
                    rl = clp.shape[0]
                    if mv and len(mv) >= rl:
                        ma = np.array(mv[:rl])
                        mod = np.maximum(0.1, 1.0 + lmod * (ma - ma.mean()))
                        at = torch.full_like(clp, av) * torch.tensor(
                            mod, device=self.device, dtype=clp.dtype)
                    else:
                        at = torch.full_like(clp, av)
                else:
                    at = torch.full_like(clp, av)
                at = at.clamp(-10, 10)
                lr_ = (clp - olp).clamp(-20, 20)
                ratio = torch.exp(lr_)
                s1 = ratio * at
                s2 = torch.clamp(ratio, 1 - self.clip_range, 1 + self.clip_range) * at
                pl_ = -torch.min(s1, s2).sum() / self.loss_token_norm
                cl = cl + pl_ / N
                tl += abs(pl_.item())
                tc += ((ratio < 1 - self.clip_range) | (ratio > 1 + self.clip_range)).sum().item() / max(ratio.numel(), 1)
            cl.backward()
            del clps, cl
            torch.cuda.empty_cache()
        del ch
        torch.cuda.empty_cache()
        gn = torch.nn.utils.clip_grad_norm_(
            list(self.model.parameters()) + list(self.skip_adapter.parameters()),
            self.max_grad_norm)
        gn = gn.item() if isinstance(gn, torch.Tensor) else gn
        self.optimizer.step()
        self.scheduler.step()
        del old_lps, ref_lps
        gc.collect()
        torch.cuda.empty_cache()

        # Metrics
        ms = np.mean([len(t.get("skip_positions", [])) for t in all_trajs])
        msr = np.mean([b.get("skip_ratio", 0) for b in all_bd])
        mg = np.mean([b.get("good_skips", 0) for b in all_bd])
        mb_ = np.mean([b.get("bad_skips", 0) for b in all_bd])
        rba = {k: np.mean([b[k] for b in all_bd])
               for k in ["r_outcome", "r_skip", "r_efficiency", "r_overlong"]}

        metrics = {
            "step": step, "loss": tl / N, "kl": total_kl / N,
            "grad_norm": gn, "clip_frac": tc / N,
            "mean_reward": np.mean(all_r),
            "accuracy": bc / max(bt, 1),
            "mean_skips": ms, "mean_skip_ratio": msr,
            "good_skips": mg, "bad_skips": mb_,
            "mean_response_len": np.mean([t["response_length"] for t in all_trajs]),
            "lr": self.scheduler.get_last_lr()[0],
            "time": time.time() - t0, "time_gen": gen_time, "time_lp": lp_time,
            "gpu_mb": torch.cuda.memory_allocated() // (1024 * 1024),
            "zero_std": szs,
            "tau_low": self.calibrator.tau_low,
            "tau_high": self.calibrator.tau_high,
            **{f"reward/{k}": v for k, v in rba.items()},
            **{f"lambda/{k}": v for k, v in lambdas.items()},
        }

        self.curriculum.step(metrics)
        metrics["phase"] = self.curriculum.current_phase
        self.global_step += 1
        return metrics

    def train(self, resume_step=0):
        logger.info("=" * 70)
        logger.info("Phase 2 GRPO v5 — MCIG Token-Level Reward")
        logger.info(f"  B={self.batch_size}, G={self.group_size}, max_steps={self.max_steps}")
        logger.info("=" * 70)
        os.makedirs(self.output_dir, exist_ok=True)

        best_acc, acc_floor, cz = 0.0, 0.59, 0
        step = resume_step

        while step < self.max_steps:
            if os.path.exists(os.path.join(self.output_dir, "STOP")):
                break

            m = self.train_step()
            if m.get("skipped"):
                step += 1
                continue
            if m.get("kl", 0) > 2.0:
                self._save(step, "emergency_kl")
                break
            if m["accuracy"] == 0 and m["mean_reward"] <= -0.9:
                cz += 1
            else:
                cz = 0
            if cz >= 5:
                self._save(step, "emergency")
                break

            if step % self.log_interval == 0:
                logger.info(
                    f"Step {step}/{self.max_steps} [{m['phase']}] | "
                    f"Loss:{m['loss']:.4f} KL:{m['kl']:.4f} GN:{m['grad_norm']:.3f} | "
                    f"Rew:{m['mean_reward']:.3f} Acc:{m['accuracy']:.3f} "
                    f"Skips:{m['mean_skips']:.1f} SkipR:{m['mean_skip_ratio']:.1%} "
                    f"Good:{m['good_skips']:.1f} Bad:{m['bad_skips']:.1f} "
                    f"Len:{m['mean_response_len']:.0f} | "
                    f"R_out:{m['reward/r_outcome']:.2f} R_sk:{m['reward/r_skip']:.3f} "
                    f"R_eff:{m['reward/r_efficiency']:.3f} | "
                    f"tau:[{m['tau_low']:.3f},{m['tau_high']:.3f}] ZStd:{m['zero_std']} | "
                    f"Tot:{m['time']:.0f}s GPU:{m['gpu_mb']}MB")
                self.log_history.append(m)

            if (step > 0 and step % self.eval_interval == 0) or step == 20:
                er = self._run_eval()
                for n, r in er.items():
                    logger.info(
                        f"  Eval {n}: Acc={r['accuracy']:.4f} "
                        f"Skips={r['avg_skips']:.1f} SkipR={r['skip_ratio']:.1%} "
                        f"ExplTok={r['avg_explicit_len']:.0f}")
                acc = er.get("gsm8k", {}).get("accuracy", 0)
                if acc > best_acc:
                    best_acc = acc
                    self._save(step, "best", {"accuracy": acc})
                if acc < acc_floor:
                    self._save(step, "emergency_acc")
                    break
            if step > 0 and step % self.save_interval == 0:
                self._save(step)
            step += 1

        fe = self._run_eval()
        for n, r in fe.items():
            logger.info(
                f"  Final {n}: Acc={r['accuracy']:.4f} Skips={r['avg_skips']:.1f} "
                f"SkipR={r['skip_ratio']:.1%}")
        self._save(self.global_step, "final")
        logger.info(
            f"  Zero-std: {self.zero_std_count}/{self.total_groups} "
            f"({self.zero_std_count / max(self.total_groups, 1):.1%})")
        with open(os.path.join(self.output_dir, "training_log.json"), "w") as f:
            json.dump(self.log_history, f, indent=2)

    @torch.inference_mode()
    def _run_eval(self, max_eval=200):
        self.model.eval()
        self.skip_adapter.eval()
        self.model.config.use_cache = True
        results = {}
        for name, ed in self.eval_datasets.items():
            problems = (ed.problems if hasattr(ed, "problems") else ed)[:max_eval]
            total = len(problems)
            ts, te, correct = 0, 0, 0
            eb = 16
            for i in range(0, total, eb):
                bp = problems[i:i + eb]
                prompts = [build_prompt(p["question"], self.tokenizer, self.data_config) for p in bp]
                gts = [p["answer"] for p in bp]
                B = len(prompts)
                pad_id = self.tokenizer.pad_token_id or 0
                eos_id = self.tokenizer.eos_token_id
                aids = [self.tokenizer(p, return_tensors="pt", add_special_tokens=False)["input_ids"].squeeze(0) for p in prompts]
                mpl = max(len(x) for x in aids)
                iids = torch.full((B, mpl), pad_id, dtype=torch.long, device=self.device)
                att = torch.zeros((B, mpl), dtype=torch.long, device=self.device)
                for j, ids in enumerate(aids):
                    iids[j, mpl - len(ids):] = ids.to(self.device)
                    att[j, mpl - len(ids):] = 1
                out = self.model(input_ids=iids, attention_mask=att, use_cache=True, output_hidden_states=True)
                pkv = out.past_key_values
                lh = out.hidden_states[-1][:, -1, :]
                eof = torch.zeros(B, dtype=torch.bool, device=self.device)
                gen = [[] for _ in range(B)]
                sc = [0] * B
                cc = [0] * B
                for si in range(self.max_new_tokens):
                    if eof.all():
                        break
                    lo = out.logits[:, -1, :]
                    nt = lo.argmax(dim=-1)
                    isk = (nt == self.skip_token_id)
                    heos = (nt == eos_id)
                    for b in range(B):
                        if isk[b].item() and not eof[b].item():
                            cc[b] += 1
                            if cc[b] > 32:
                                ml = lo[b].clone()
                                ml[self.skip_token_id] = -float('inf')
                                nt[b] = ml.argmax()
                                isk[b] = False
                                cc[b] = 0
                        elif not eof[b].item():
                            cc[b] = 0
                    for b in range(B):
                        if not eof[b] and not heos[b]:
                            gen[b].append(nt[b].item())
                            sc[b] += int(isk[b].item())
                    eof = eof | heos
                    if eof.all():
                        break
                    zs = self.skip_adapter(lh)
                    em = self.model.get_input_embeddings()(nt.unsqueeze(1))
                    ni = torch.where(isk.view(-1, 1, 1), zs.unsqueeze(1), em)
                    att = torch.cat([att, (~eof).long().unsqueeze(1)], dim=1)
                    out = self.model(inputs_embeds=ni, attention_mask=att,
                                    past_key_values=pkv, use_cache=True, output_hidden_states=True)
                    pkv = out.past_key_values
                    lh = out.hidden_states[-1][:, -1, :]
                for j in range(B):
                    exp = [t for t in gen[j] if t != self.skip_token_id]
                    if check_answer(extract_model_answer(
                            self.tokenizer.decode(exp, skip_special_tokens=True)), gts[j]):
                        correct += 1
                    ts += sc[j]
                    te += (len(gen[j]) - sc[j])
                torch.cuda.empty_cache()
            results[name] = {
                "accuracy": correct / max(total, 1),
                "correct": correct, "total": total,
                "avg_skips": ts / max(total, 1),
                "avg_explicit_len": te / max(total, 1),
                "skip_ratio": ts / max(ts + te, 1),
            }
        self.model.train()
        self.skip_adapter.train()
        self.model.config.use_cache = False
        return results

    def _save(self, step, tag=None, extra=None):
        sd = os.path.join(self.output_dir, tag or f"step-{step}")
        os.makedirs(sd, exist_ok=True)
        self.model.save_pretrained(sd, safe_serialization=True)
        self.tokenizer.save_pretrained(sd)
        torch.save(self.skip_adapter.state_dict(), os.path.join(sd, "skip_adapter.pt"))
        torch.save({"optimizer": self.optimizer.state_dict(),
                     "scheduler": self.scheduler.state_dict()},
                   os.path.join(sd, "training_optim.pt"))
        meta = {
            "step": step, "global_step": self.global_step,
            "curriculum": self.curriculum.get_state(),
            "calibrator": {"tau_low": self.calibrator.tau_low,
                           "tau_high": self.calibrator.tau_high,
                           "calibrated": self.calibrator.calibrated},
            "log_history": self.log_history[-20:],
            **(extra or {}),
        }
        with open(os.path.join(sd, "training_state.json"), "w") as f:
            json.dump(meta, f, indent=2, default=str)
        logger.info(f"  Saved: {sd}")

    def resume_from(self, rd):
        ap = os.path.join(rd, "skip_adapter.pt")
        if os.path.exists(ap):
            self.skip_adapter.load_state_dict(
                torch.load(ap, map_location=self.device, weights_only=False))
        op = os.path.join(rd, "training_optim.pt")
        if os.path.exists(op):
            st = torch.load(op, map_location=self.device, weights_only=False)
            self.optimizer.load_state_dict(st["optimizer"])
            self.scheduler.load_state_dict(st["scheduler"])
        mp = os.path.join(rd, "training_state.json")
        if os.path.exists(mp):
            with open(mp) as f:
                meta = json.load(f)
            rs = meta.get("global_step", meta.get("step", 0))
            self.global_step = rs
            if "curriculum" in meta:
                self.curriculum.current_phase = meta["curriculum"].get("phase", "A")
            if "calibrator" in meta:
                cal = meta["calibrator"]
                self.calibrator.tau_low = cal.get("tau_low", 0.05)
                self.calibrator.tau_high = cal.get("tau_high", 0.15)
                self.calibrator.calibrated = cal.get("calibrated", False)
            return rs
        return 0


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default="./checkpoints/phase2_v5")
    parser.add_argument("--max_steps", type=int, default=600)
    parser.add_argument("--batch_size", type=int, default=12)
    parser.add_argument("--group_size", type=int, default=8)
    parser.add_argument("--temperature", type=float, default=0.9)
    parser.add_argument("--max_new_tokens", type=int, default=420)
    parser.add_argument("--no_ref_model", action="store_true")
    parser.add_argument("--logprob_mb", type=int, default=32)
    parser.add_argument("--ppo_mb", type=int, default=16)
    parser.add_argument("--resume", type=str, default=None)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.resume:
        args.checkpoint = args.resume
    device = torch.device("cuda:0")
    config = get_config()
    config.model.max_new_tokens = args.max_new_tokens
    logger.info("=" * 70)
    logger.info("Phase 2 v5 — MCIG Token-Level Reward")
    logger.info(f"  Checkpoint: {args.checkpoint}")
    logger.info("=" * 70)
    model = AutoModelForCausalLM.from_pretrained(
        args.checkpoint, torch_dtype=torch.bfloat16,
        attn_implementation="sdpa", trust_remote_code=True)
    tokenizer = AutoTokenizer.from_pretrained(args.checkpoint, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    skip_token_id = tokenizer.convert_tokens_to_ids("<SKIP>")
    skip_adapter = SkipAdapter(hidden_size=model.config.hidden_size)
    ap = os.path.join(args.checkpoint, "skip_adapter.pt")
    if os.path.exists(ap):
        skip_adapter.load_state_dict(torch.load(ap, map_location="cpu", weights_only=False))
    ref_model = None
    if not args.no_ref_model:
        ref_model = copy.deepcopy(model)
        ref_model.eval()
        for p in ref_model.parameters():
            p.requires_grad = False
    dataset = MathProblemDataset(config.data, tokenizer, seed=args.seed)
    eval_datasets = {}
    try:
        eval_datasets["gsm8k"] = GSM8KEvalDataset(config.data, tokenizer)
    except Exception:
        pass
    ref_device = (torch.device("cuda:1")
                  if torch.cuda.device_count() >= 2 and ref_model else device)
    trainer = Phase2Trainer(
        model=model, ref_model=ref_model, skip_adapter=skip_adapter,
        tokenizer=tokenizer, skip_token_id=skip_token_id, config=config,
        dataset=dataset, eval_datasets=eval_datasets, device=device,
        t_ref=350.0, max_steps=args.max_steps, ref_device=ref_device)
    trainer.output_dir = args.output_dir
    trainer.batch_size = args.batch_size
    trainer.group_size = args.group_size
    trainer.temperature = args.temperature
    trainer.logprob_mb = args.logprob_mb
    trainer.ppo_mb = args.ppo_mb
    resume_step = trainer.resume_from(args.resume) if args.resume else 0
    trainer.train(resume_step=resume_step)
    if ref_model:
        del ref_model
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()