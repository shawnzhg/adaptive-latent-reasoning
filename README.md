# adaptive-latent-reasoning

Xiang Zhang, University of Michigan, Ann Arbor

A study of **hidden-state-guided chain-of-thought (CoT) compression**: can we detect *which* CoT tokens
are redundant from the model's own hidden states, and fold only those into latent (continuous) reasoning
while keeping the load-bearing tokens explicit? We use **MCIG (Manifold Causal Information Gain)**, a
training-free per-token redundancy signal that cleanly localizes filler vs. decisive tokens — but **the
compression built on it does not pan out at 1.5B** (GSM8K regresses from ~65% to ~58% once the model folds
non-trivially). The report analyzes *why* (five causes) and motivates a *selective* latent↔explicit switch
over wholesale latent reasoning.

📄 **Full research report:** [`REPORT.md`](REPORT.md) — the experimental process, the negative result,
the five-cause analysis, and why the project pivoted, with citations in [`REFERENCES.md`](REFERENCES.md).

## Key findings

- **MCIG signal:** a training-free per-token score = max-pool of geodesic curvature, Jensen–Shannon
  information shock, and log-energy shock, momentum-smoothed. A token-level heatmap shows it cleanly
  separates connective filler (low MCIG → skip) from decisive reasoning steps (high MCIG → keep); as a
  *whole-trajectory* correctness predictor it is only near-chance (~AUC 0.53).
- **Compressor (negative):** a latent-feedback adapter + MCIG-guided curriculum only matches the base
  when it folds almost nothing; real folding regresses GSM8K to **~58%** from ~65%.
- **Analysis:** five causes (residual-stream overload, error accumulation, KV-cache poisoning, RoPE
  manifold mismatch, and insufficient scale at 1.5B) — the basis for choosing a selective switch.

## Install

```bash
pip install -r requirements.txt
```

## Quickstart

```bash
# Inspect the MCIG redundancy signal
python compression/diagnose_signal.py               # MCIG signal diagnostics
# Compression
python compression/init_model.py                    # base + latent adapter + special token
bash    scripts/run_p1.sh                            # build MCIG + counterfactual curriculum
bash    scripts/run_p15.sh                           # curriculum SFT
bash    scripts/run_p2.sh                            # MCIG-guided GRPO
python  compression/eval_gsm8k.py                    # no-skip vs with-skip GSM8K
python  compression/measure_latent_horizon.py        # latent-reasoning-horizon probe
# Switching experiments
python switching_experiments/train_latent_curriculum.py
```

Set model/data via flags or place under `./data` and `./checkpoints`; no absolute paths are hard-coded.

## Repository layout

```
compression/           MCIG signal + latent-feedback adapter + curriculum + GRPO — REGRESSES at 1.5B
                       information_gain (MCIG), latent_adapter, detect_anchors, train_sft_latent,
                       train_grpo_latent, eval_gsm8k, measure_latent_horizon, diagnose_signal,
                       + results/ (incl. figures/mcig_semantic_heatmap.png)
switching_experiments/ Coconut-style latent↔explicit curricula + Semantic Anchoring Loss + results/
scripts/               run_p1 (data) · run_p15 (SFT) · run_p2 (GRPO)
REPORT.md  full report   ·   REFERENCES.md  bibliography
```
