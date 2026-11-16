# Experimental Results

All values below are taken from executed training logs (notebooks `archive/colab_runs/01` through `08`) or from [`Experiments/triangulation.txt`](../Experiments/triangulation.txt). JSON mirrors appear in [`results/summaries/`](../results/summaries/). Cross-checks are recorded in [`VERIFICATION.md`](VERIFICATION.md).

Unless noted otherwise, I report epoch-2 perplexity on WikiText-103 with the GPT-2 byte-pair vocabulary (50,257 tokens).

---

## Act I: Factorized embedding geometry (E4)

**Hypothesis:** Output embedding width (`k_out`) dominates input width (`k_in`) for perplexity at fixed depth.

**Protocol:** Eight layers, `d_model=192`, factorized embeddings (E4), two training epochs, AdamW-family optimizers (v13 through v16).

| Run | k_in | k_out | Params | ep2 val PPL | ep2 test PPL | Role |
|-----|-----:|------:|-------:|------------:|-------------:|------|
| v13 | 96 | 64 | 11.6M | 53.63 | **50.20** | Baseline asymmetric shape |
| v14 | 128 | 64 | 13.2M | 52.84 | **49.61** | Extra reserve `k_in` rows |
| v15 | 96 | 48 | 10.8M | 57.08 | **53.41** | Minimum `k_out` (worst of trio) |
| v16 | 96 | 96 | 13.2M | 50.11 | **46.46** | Shape-match boundary test |

**Marginal rates (v13 through v15):**

| Axis | PPL improvement per dimension | Relative efficiency |
|------|------------------------------:|--------------------:|
| k_out | 0.201 PPL / dim | **11.2×** vs k_in |
| k_in | 0.018 PPL / dim | 1× (reference) |

**Pre-registered prediction (v16):** epoch-2 test PPL **45.7 ± 2**. **Observed:** **46.46** (confirmed within band). Naive linear extrapolation (**43.77**) did not hold.

**Note:** A separate stalled v16 run recorded test PPL 65.19; I do not use that run in any primary claim. I describe the optimum as a boundary region rather than a single `(k_in, k_out)` pair.

**Sources:** notebooks `01` through `04`; [`act1_embedding_geometry.json`](../results/summaries/act1_embedding_geometry.json).

---

## Act II: Optimizer stack (Muon + SparseMuon)

**Hypothesis:** The v31 Muon + per-row SparseMuon stack improves perplexity relative to prior AdamW-only configurations.

**Protocol:** Eight layers, `d=224`, `k_in=k_out=48`, single vMF head, 9,958,162 parameters, WikiText-103, two epochs.

| Metric | Epoch 2 | Pre-registered criterion |
|--------|--------:|--------------------------|
| val PPL | **53.60** | n/a |
| test PPL | **48.93** | 48.3 ± 2 (confirmed) |

Earlier private runs falsified v30 SGDR basin-hopping and showed v29 tangential SparseMuon did not exceed v31.

**Reproduction:** `python scripts/v31_sparse_muon_train.py`; notebook `05`; [`act2_optimizer_stack.json`](../results/summaries/act2_optimizer_stack.json).

---

## Act III: vMF head vs softmax (head-only matched comparison)

**Hypothesis:** At fixed parameters, data, and optimizer, replacing only the output head (linear softmax with row-normalized vMF) accelerates convergence.

**Matched stack:** Eight layers, `d=224`, `k=48`, 9,958,162 parameters, Muon + SparseMuon + AdamW, MTP, EMA, shortcut, unigram bias. The sole intentional difference is `HEAD_MODE` (`linear` vs `vmf`).

### Epoch trajectory (v41 vMF vs notebook `06` softmax)

| Epoch | Softmax val | Softmax test | vMF val | vMF test | Δ val (vMF − softmax) |
|------:|------------:|-------------:|--------:|---------:|----------------------:|
| 1 | 67.11 | 62.20 | **61.76** | n/a | **−5.35** |
| 2 | 57.05 | 52.80 | **52.82** | n/a | **−4.23 (−7.4%)** |
| 3 | n/a | n/a | **49.21** | n/a | n/a |
| 4 | n/a (stopped) | n/a | **46.84** | **46.16** | n/a |

### Tail tokens (rank > 30,000), cross-entropy at epoch 1

| Head | Tail CE | vs uniform (10.83 nats) |
|------|--------:|-------------------------|
| Softmax | **11.38** | Above uniform (harmful early in training) |
| vMF | **9.54** | Below uniform (tail learning from epoch 1) |

At epoch 2, vMF tail CE was **8.94** (see [`act3_v32_head_comparison.json`](../results/summaries/act3_v32_head_comparison.json)).

### Gradient allocation (softmax epoch 2; long vMF run epoch 3)

| Head | head/body grad ratio | Interpretation |
|------|---------------------:|----------------|
| Softmax | **3.78×** | Head-dominated gradient flow |
| vMF | **1.31×** | More balanced; body receives signal from epoch 1 |

### Extended vMF run (eight epochs, notebook `07`)

| Epoch | val PPL | test PPL | train CE |
|------:|--------:|---------:|---------:|
| 3 | 48.91 | 44.71 | 4.1095 |
| 8 | **42.72** | **39.20** | 3.9313 |

Epochs 1 and 2 of this run were not logged in the repository (checkpoints only). My matched-epoch claims rely on the v41 A/B pair above.

### Pre-registered Zipf gates at vMF epoch 8

| ID | Claim | Gate | Measured | Verdict |
|----|-------|------|----------|---------|
| H1 | Angle ∝ −log rank | Pearson > 0.5 | **0.23** | Weak |
| H2 | Two learning problems | ratio drift | **0.337** (stable) | Null |
| H4 | Radial \|A_out\| vs rank | Pearson > 0.5 | **0.96** | Confirmed |

Pearson(\|A_out\|, −log rank) at epoch 4 in the v41 run: **0.95**.

**Reproduction:** `python scripts/v32_standard_softmax_comparison.py`; `python scripts/v41_vmf_concentrated.py --sota`; [`act3_v32_head_comparison.json`](../results/summaries/act3_v32_head_comparison.json).

---

## Act IV: Scaling on OpenWebText

### IV-A: Chinchilla-E triangulation (confirmed)

**Protocol:** Three matched-architecture models, GPT-2 tokenizer, OpenWebText, 500M tokens per epoch for six epochs, softmax language models.

| Model | Params | E_app (nats) | β_rep | OWT test PPL |
|-------|-------:|-------------:|------:|-------------:|
| A (10M) | 10,165,681 | 4.1379 | 1.8254 | **43** |
| C (25M) | 25,147,153 | 3.6984 | 1.4915 | **39** |
| B (51M) | 51,016,529 | 3.4406 | 1.5933 | **31** |

**Triangulated entropy floor E_true:**

| Method | α | E_true (nats) | Within pre-registered band |
|--------|---|--------------:|:--------------------------:|
| Two-point (A+B) | 0.34 | **2.4863** | Yes |
| Three-point OLS | 0.3531 (free) | **2.5322** | Yes |
| Three-point | 0.34 (fixed) | **2.4852** | Yes |

Pre-registered band: 2.485 < E < 2.855. Fit residuals: ±0.0000 nats on all three anchor points. Tail-to-top CE ratio: approximately **3.7×** across scales.

**Sources:** notebook `08`; [`Experiments/triangulation.txt`](../Experiments/triangulation.txt); [`act4_scaling_laws.json`](../results/summaries/act4_scaling_laws.json).

### IV-B: β_rep ∝ √N (failed)

**Prediction:** β_rep(N) = 0.000161 × N^0.5.

| Model | β_pred | β_actual | Δ | Gate (\|Δ\| < 0.05) |
|-------|-------:|---------:|--:|:--------------------:|
| 25M | 2.8711 | 1.4915 | −1.38 | Fail |
| 51M | 4.0893 | 1.5933 | −2.50 | Fail |

Fitted scaling: β_rep ∝ N^**−0.084** (approximately flat).

### IV-C: A_floor ∝ V²/T (not supported; original protocol invalid)

| Scan | Slope | R² | Verdict |
|------|------:|---:|---------|
| Original | +0.183 | 0.91 | Invalid test (axes mis-specified) |
| Corrected | −0.2525 | 0.49 | Credible negative; short-T variants under-fit |

**Corrected scan, fitted floor A:**

| Variant | A (nats) |
|---------|--------:|
| V50K, T1.0 | 500.0 |
| V25K | 444.2 |
| V12K | 432.3 |
| T0.5 | 285.7 |
| T0.25 | 114.1 |

A is nearly flat in V while V²/T varies by roughly 16×.

### IV-D: C* phase-transition ansatz (falsified as stated)

Fixed (H, c) extrapolation of C* = H + c·(V²/T): deviation from prediction reached **+0.85 nats by epoch 8** (gate: ±0.05). C* here is a training cross-entropy fit intercept, not corpus entropy.

**Scaling scan reproduction:** set `MODE='scaling_scan'` in `scripts/v32_zipf_diagnostics.py`.

---

## Act IV-E: Log-only Chinchilla-E triangulation (zero training)

**Goal:** Estimate corpus-specific irreducible loss E_true from **published training logs** on existing model ladders - no GPU, no new runs. Validates the same triangulation ansatz as Act IV-A on independent public data.

**Full write-up:** [`docs/LOG_ONLY_TRIANGULATION_RESULTS.md`](LOG_ONLY_TRIANGULATION_RESULTS.md).

### IV-E-1: Pythia / The Pile (primary)

**Source:** EleutherAI public W&B TSV exports (70M, 160M, 410M). Script: `scripts/pythia_chinchilla_e_from_logs.py`.

| Model | Params | E_app | Final train loss |
|-------|-------:|------:|-----------------:|
| Pythia-70M | 70.4M | 2.7776 | 2.8004 |
| Pythia-160M | 157.1M | 2.4607 | 2.4990 |
| Pythia-410M | 405.7M | 2.1033 | 2.1780 |

**E_true (fixed α = 0.34): 1.2919 nats.** Holdout 410M ΔE_app = 0.079; LOO std = 0.144 - both pass pre-registered gates.

**Artifacts:** [`results/pythia_chinchilla_from_logs/`](../results/pythia_chinchilla_from_logs/).

### IV-E-2: Meta FAIR Step-2 (second corpus)

**Source:** Public `FS-step2v2_*_ti134698.csv` scaling logs (~256B tokens, matched ladder). Script: `scripts/meta_step2_chinchilla_e_from_logs.py`. E_app = late C* (runs past the training knee).

| Model | Params | E_app (late C*) | Final loss |
|-------|-------:|----------------:|-----------:|
| Step2-h832-L12 | 227.6M | 2.0090 | 2.3865 |
| Step2-h1024-L16 | 393.6M | 1.9596 | 2.2456 |
| Step2-h1280-L20 | 699.6M | 1.8929 | 2.1192 |

**E_true (fixed α = 0.34): 1.6493 nats.** Holdout h1280 Δ = 0.024; LOO std = 0.067 - pass.

**Artifacts:** [`results/meta_step2_chinchilla_from_logs/`](../results/meta_step2_chinchilla_from_logs/).

### IV-E-3: Synthetic pipeline check (Colab)

Known Zipf H = 5.640 nats; recovered E_true = 5.638 nats (|Delta| = 0.002). Run on Colab; `results/synthetic_chinchilla/triangulation.json` not yet copied locally.

### IV-E-4: Robustness battery

Script: `scripts/chinchilla_e_robustness.py`. **Overall PASS** on Pythia + Meta Step-2 holdout/LOO/sanity gates. OLMo/Dolma fails (truncated 13B log). Chinchilla original 400-run grid not re-runnable (MassiveText proprietary).

**Report:** [`results/robustness_chinchilla_e/robustness_report.txt`](../results/robustness_chinchilla_e/robustness_report.txt).

---

## Out of scope for this document

Incomplete runs, supplementary analyses under `archive/internal/`, and approximately forty private ablation versions are excluded. See [`LINEAGE.md`](LINEAGE.md).
