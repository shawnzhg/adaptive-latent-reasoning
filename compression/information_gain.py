#!/usr/bin/env python3
import torch
import torch.nn.functional as F
import numpy as np
from typing import Optional, Dict, Tuple, List
from dataclasses import dataclass

# Assumes config.py renamed KVIGConfig to MCIGConfig; if not renamed, this stays compatible
from config import KVIGConfig as MCIGConfig, ModelConfig

@dataclass
class MCIGState:
    """
    Incremental state for online autoregressive generation (Incremental State for Online RL)
    """
    h_norm_prev: Optional[torch.Tensor] = None       # normalized previous hidden state (\hat{h}_{t-1})
    log_norm_prev: Optional[float] = None            # log of the previous hidden-state norm (\log ||h_{t-1}||_2)
    delta_h_norm_prev: Optional[torch.Tensor] = None # previous tangent vector (\Delta \hat{h}_{t-1})
    probs_prev: Optional[torch.Tensor] = None        # previous probability distribution (P_{t-1})
    mcig_env_prev: float = 0.0                       # previous momentum-envelope value
    step: int = 0                                    # current generation step

    def is_first_step(self) -> bool:
        return self.step == 0

class MCIGComputer:
    """
    Manifold causal dynamics computer (Dense-MCIG Engine)
    Implements true OR logic (Max-Pooling) with an orthogonal three-dimensional feature fusion.
    """
    def __init__(self, mcig_config: MCIGConfig, model_config: ModelConfig):
        self.eps = getattr(mcig_config, "eps", 1e-8)
        self.hidden_size = model_config.hidden_size
        self.decay = 0.8  # envelope decay coefficient gamma (aligned with the offline experiments)
        self.alpha_energy = 2.0  # energy alignment coefficient

    @torch.no_grad()
    def compute_step(self, h_t: torch.Tensor, logits_t: torch.Tensor, state: MCIGState, **kwargs) -> Tuple[float, MCIGState]:
        """
        [Phase 2 online-generation only]
        Incrementally computes the Dense-MCIG score for the current token, used for
        token-level reward modulation.
        """
        h_t = h_t.detach().float().squeeze()
        logits_t = logits_t.detach().float().squeeze()

        # 1. Basic state computation
        probs_t = F.softmax(logits_t, dim=-1).clamp(min=1e-10)
        norm_t = h_t.norm(p=2).item()
        log_norm_t = np.log(norm_t + self.eps)
        h_norm_t = F.normalize(h_t, p=2, dim=-1)

        # ====================================================
        # Key fix: dynamically align the CPU tensors in state to the current GPU
        # ====================================================
        device = h_t.device
        if state.h_norm_prev is not None:
            state.h_norm_prev = state.h_norm_prev.to(device)
        if state.delta_h_norm_prev is not None:
            state.delta_h_norm_prev = state.delta_h_norm_prev.to(device)
        if state.probs_prev is not None:
            state.probs_prev = state.probs_prev.to(device)

        # Handle the starting step: no preceding state available
        if state.is_first_step() or state.h_norm_prev is None or state.probs_prev is None:
            mcig_raw = 0.0
            new_delta_h_norm = None
        else:
            # -- (1) C_t: geodesic curvature (direction change) --
            delta_h_norm_t = h_norm_t - state.h_norm_prev
            if state.delta_h_norm_prev is None:
                # When generating the first response token, fall back to the absolute angular difference
                cos_sim = F.cosine_similarity(h_norm_t.unsqueeze(0), state.h_norm_prev.unsqueeze(0), eps=self.eps).item()
                C_t = max(0.0, 1.0 - cos_sim)
            else:
                # Normal tangent-vector angular difference
                cos_sim = F.cosine_similarity(delta_h_norm_t.unsqueeze(0), state.delta_h_norm_prev.unsqueeze(0), eps=self.eps).item()
                C_t = max(0.0, 1.0 - cos_sim)
            new_delta_h_norm = delta_h_norm_t.cpu()

            # -- (2) J_t: symmetric information shock (JSD) --
            M = 0.5 * (probs_t + state.probs_prev)
            kl_curr = (probs_t * (probs_t.log() - M.log())).sum().item()
            kl_prev = (state.probs_prev * (state.probs_prev.log() - M.log())).sum().item()
            J_t = max(0.0, 0.5 * kl_curr + 0.5 * kl_prev)

            # -- (3) E_t: absolute log-energy shock (payload jump) --
            E_t = self.alpha_energy * abs(log_norm_t - state.log_norm_prev)

            # -- (4) Max-Pooling OR-logic fusion --
            mcig_raw = max(C_t, J_t, E_t)

        # Momentum-envelope smoothing
        mcig_env = max(mcig_raw, state.mcig_env_prev * self.decay)

        # Update state
        new_state = MCIGState(
            h_norm_prev=h_norm_t.cpu(),
            log_norm_prev=log_norm_t,
            delta_h_norm_prev=new_delta_h_norm,
            probs_prev=probs_t.cpu(),
            mcig_env_prev=mcig_env,
            step=state.step + 1
        )

        return mcig_env, new_state

    @torch.no_grad()
    def compute_trajectory_from_model(self, model, input_ids: torch.Tensor, attention_mask: Optional[torch.Tensor] = None, prompt_length: int = 0, **kwargs) -> Dict:
        """
        [Phase 1.5 offline extraction only]
        Batch-computes the Dense-MCIG for a single complete trajectory. Strictly aligned with
        the mathematical formulas in `compare_ada.py`.
        """
        device = next(model.parameters()).device
        if input_ids.dim() == 1: input_ids = input_ids.unsqueeze(0)
        if attention_mask is not None and attention_mask.dim() == 1: attention_mask = attention_mask.unsqueeze(0)

        input_ids = input_ids.to(device)
        attention_mask = attention_mask.to(device) if attention_mask is not None else None
        resp_len = input_ids.shape[1] - prompt_length

        if resp_len <= 1:
            empty = [0.0] * max(0, resp_len)
            return {"mcig_values": empty, "mean_mcig": 0.0, "std_mcig": 0.0}

        outputs = model(input_ids=input_ids, attention_mask=attention_mask, output_hidden_states=True, use_cache=False)
        last_hidden = outputs.hidden_states[-1][0].float()
        logits = outputs.logits[0].float()

        h_resp, logits_resp = last_hidden[prompt_length:], logits[prompt_length:]
        h_prev_first, logits_prev_first = last_hidden[prompt_length - 1], logits[prompt_length - 1]

        h_all = torch.cat([h_prev_first.unsqueeze(0), h_resp], dim=0)
        all_logits = torch.cat([logits_prev_first.unsqueeze(0), logits_resp], dim=0)

        # Compute the probability distribution
        probs = F.softmax(all_logits, dim=-1).clamp(min=1e-10)

        # Initialize the base tensors
        C_vec = np.zeros(resp_len, dtype=np.float32)
        J_vec = np.zeros(resp_len, dtype=np.float32)
        E_vec = np.zeros(resp_len, dtype=np.float32)

        if resp_len >= 2:
            # 1. Compute C (Curvature)
            h_all_norm = F.normalize(h_all, p=2, dim=-1)
            delta_h_norm = h_all_norm[1:] - h_all_norm[:-1]
            cos_t0 = F.cosine_similarity(h_all_norm[1].unsqueeze(0), h_all_norm[0].unsqueeze(0), eps=self.eps).item()
            C_vec[0] = max(0.0, 1.0 - cos_t0)
            C_vec[1:] = (1.0 - F.cosine_similarity(delta_h_norm[1:], delta_h_norm[:-1], dim=-1, eps=self.eps)).cpu().numpy()
            C_vec = np.maximum(0.0, C_vec) # safeguard to remove negatives

            # 2. Compute J (JSD Causal Shock)
            P_prev = probs[:-1]
            P_curr = probs[1:]
            M = 0.5 * (P_prev + P_curr)
            kl_prev_M = torch.sum(P_prev * (torch.log(P_prev) - torch.log(M)), dim=-1)
            kl_curr_M = torch.sum(P_curr * (torch.log(P_curr) - torch.log(M)), dim=-1)
            J_vec = (0.5 * (kl_prev_M + kl_curr_M)).cpu().numpy()
            J_vec = np.maximum(0.0, J_vec)

            # 3. Compute E (Absolute Log Energy Spike)
            h_norms = h_all.norm(dim=-1).cpu().numpy()
            E_vec = np.abs(np.log(h_norms[1:] + self.eps) - np.log(h_norms[:-1] + self.eps)) * self.alpha_energy

        # 4. Max-Pooling (true OR-logic fusion)
        sig_mcig = np.maximum(C_vec, np.maximum(J_vec, E_vec))

        # Momentum-envelope propagation
        mcig_env = np.zeros_like(sig_mcig)
        if len(sig_mcig) > 0:
            mcig_env[0] = sig_mcig[0]
            for t in range(1, len(sig_mcig)):
                mcig_env[t] = max(sig_mcig[t], mcig_env[t-1] * self.decay)

        del outputs, logits, last_hidden, h_resp, logits_resp, h_all, all_logits, probs
        if resp_len >= 2:
            del h_all_norm, delta_h_norm, P_prev, P_curr, M
        torch.cuda.empty_cache()

        mcig_env = np.nan_to_num(mcig_env, nan=0.0, posinf=0.0, neginf=0.0)
        mcig_values = mcig_env.tolist()

        return {
            "mcig_values": mcig_values,
            "mean_mcig": float(np.mean(mcig_values)) if mcig_values else 0.0,
            "std_mcig": float(np.std(mcig_values)) if mcig_values else 0.0,
            # Compatible with data pipelines that need the original naming (avoids touching too many external dependencies)
            "kvig_values": mcig_values,
        }

def compute_mcig_batch(mcig_computer, model, batch_input_ids, batch_attention_masks, prompt_lengths, **kwargs):
    return [mcig_computer.compute_trajectory_from_model(model, input_ids=ids, attention_mask=mask, prompt_length=pl)
            for ids, mask, pl in zip(batch_input_ids, batch_attention_masks, prompt_lengths)]
