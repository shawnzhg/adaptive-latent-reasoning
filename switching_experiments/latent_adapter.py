"""
Latent Bridge Adapter (upgraded version).

Upgraded from the old Bottleneck Adapter to a full-dimension residual Adapter:
  Old: h -> W_up(d->d/4) -> ReLU -> W_down(d/4->d) -> LayerNorm -> z
  New: h -> h + W_down(SiLU(W_up(RMSNorm(h))))

Key improvements:
  1. Residual connection: the Adapter only learns the corrective delta h, rather than reconstructing from scratch
  2. Full dimension d->d->d: preserves 100% of the information bandwidth (vs 25% in the old version)
  3. RMSNorm in front: aligns with the Pre-Norm architecture of Qwen-2.5
  4. SiLU activation: avoids the dead-neuron problem of ReLU
  5. W_down zero-initialized: step-0 output = h_last (identity-mapping cold start)

Parameter count: ~25.7M (d^2 x 2 + d x 2), about 0.37% of the 7B model
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class SkipAdapter(nn.Module):
    """
    Latent Bridge Adapter: maps the Transformer output space back to the input embedding space.

    Thanks to the residual structure, the Adapter is equivalent to an identity
    mapping (h_last -> h_last) early in training and gradually learns the optimal
    space translation as training proceeds.

    Args:
        hidden_size: the model hidden dimension d (e.g., 3584 for Qwen-2.5-7B, 2048 for 1.5B)
    """

    def __init__(self, hidden_size: int, **kwargs):
        # **kwargs absorbs old arguments such as bottleneck_ratio to keep the interface compatible
        super().__init__()
        self.hidden_size = hidden_size

        # RMSNorm in front (aligns with the Qwen Pre-Norm convention)
        self.norm = nn.RMSNorm(hidden_size)

        # Full-dimension projection d -> d -> d
        self.w_up = nn.Linear(hidden_size, hidden_size, bias=False)
        self.w_down = nn.Linear(hidden_size, hidden_size, bias=False)

        self._init_weights()

    def _init_weights(self):
        """
        Initialization strategy:
          W_up: Kaiming normal initialization (suited to the SiLU activation)
          W_down: all-zero initialization -> at step 0, delta h = 0, output = h_last
          RMSNorm: default gamma=1
        """
        nn.init.kaiming_normal_(self.w_up.weight, nonlinearity='linear')
        nn.init.zeros_(self.w_down.weight)

    def forward(self, h_last: torch.Tensor) -> torch.Tensor:
        """
        Forward pass: e_next = h_last + W_down(SiLU(W_up(RMSNorm(h_last))))

        Args:
            h_last: the Transformer last-layer hidden state
                    shape: (..., hidden_size)

        Returns:
            e_next: the mapped pseudo-embedding vector, used directly as the next-step inputs_embeds
                    shape: same as the input
        """
        normed = self.norm(h_last)
        delta = self.w_down(F.silu(self.w_up(normed)))
        return h_last + delta

    def get_param_count(self) -> dict:
        """Return parameter statistics."""
        up_params = self.w_up.weight.numel()
        down_params = self.w_down.weight.numel()
        norm_params = sum(p.numel() for p in self.norm.parameters())
        total = up_params + down_params + norm_params
        return {
            "w_up": up_params,
            "w_down": down_params,
            "norm": norm_params,
            "total": total,
            "total_M": total / 1e6,
        }

    def check_output_norm_ratio(self, h_last: torch.Tensor) -> float:
        """Check the output norm ratio: ||e_next|| / ||h_last||"""
        with torch.no_grad():
            e_next = self.forward(h_last)
            in_norm = h_last.norm(dim=-1).mean()
            out_norm = e_next.norm(dim=-1).mean()
            return (out_norm / (in_norm + 1e-8)).item()