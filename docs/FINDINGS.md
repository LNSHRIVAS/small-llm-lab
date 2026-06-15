# Findings

This document summarizes my conclusions from the executed runs. Measured values appear in [`RESULTS.md`](RESULTS.md). The audit record is in [`VERIFICATION.md`](VERIFICATION.md).

Unless noted otherwise, I use epoch-2 test perplexity on WikiText-103 (GPT-2 BPE, 50,257 vocabulary).

---

## 1. Output embedding width dominates input width (Act I)

I ran four factorized-embedding configurations (eight layers, `d_model=192`, E4) to establish marginal geometry before changing optimizers or output heads:

| Run | k_in | k_out | Params | ep2 test PPL |
|-----|-----:|------:|-------:|-------------:|
| v13 | 96 | 64 | 11.6M | 50.20 |
| v14 | 128 | 64 | 13.2M | 49.61 |
| v15 | 96 | 48 | 10.8M | 53.41 |
| v16 | 96 | 96 | 13.2M | **46.46** |

From marginal rates across v13 through v15, one `k_out` dimension yields **11.2×** more perplexity improvement per dimension than one `k_in` dimension (0.201 vs 0.018 PPL per dimension).

For v16 (`k_in = k_out = 96`), I pre-registered test PPL **45.7 ± 2**. The observed **46.46** falls within that band and refutes naive linear extrapolation (**43.77**). These four runs do not share a fixed parameter budget; the comparison is geometric rather than iso-parameter.

A stalled sister v16 run recorded test PPL 65.19. I rely only on the executed `best_update_v16` run. I interpret the optimum as lying in a boundary region rather than at strict `k_in = k_out`.

---

## 2. Muon + SparseMuon improves the pre-v32 baseline (Act II)

Configuration v31 (eight layers, `d=224`, `k=48`, vMF head, 9,958,162 parameters) with Muon and per-row SparseMuon on `A_in` and `A_out`:

| Metric | Epoch 2 | Pre-registered gate |
|--------|--------:|---------------------|
| test PPL | **48.93** | 48.3 ± 2 (confirmed) |
| val PPL | 53.60 | n/a |

This was my strongest stack before the full v32 Zipf and vMF diagnostic suite. In private runs I falsified v30 SGDR basin-hopping and found v29 tangential SparseMuon did not exceed v31.

---

## 3. vMF head accelerates convergence in a head-only matched comparison (Act III)

I held architecture, optimizer, data, and regularization fixed at 9.96M parameters and changed only the output head (`scripts/v32_standard_softmax_comparison.py` vs `scripts/v41_vmf_concentrated.py --sota`).

### Convergence

| Epoch | Softmax val | Softmax test | vMF val | vMF test |
|------:|------------:|-------------:|--------:|---------:|
| 1 | 67.11 | 62.20 | **61.76** | n/a |
| 2 | 57.05 | 52.80 | **52.82** | n/a |
| 4 | n/a | n/a | **46.84** | **46.16** |
| 8¹ | n/a | n/a | **42.72** | **39.20** |

¹ Epoch 8 from the extended v32 vMF run (notebook `07`); epochs 1 and 2 of that run were not logged in the repository. Matched-epoch claims rest on the v41 A/B pair.

Validation perplexity improved by **7.4%** at epoch 2 (57.05 to 52.82). The vMF arm led from epoch 1 (−5.35 val PPL).

### Mechanism

| Signal | Softmax | vMF |
|--------|--------:|----:|
| head/body grad ratio | **3.78×** | **1.31×** |
| tail rank >30K CE @ ep1 | **11.38** (> uniform 10.83) | **9.54** (< uniform) |

At epoch 1 the softmax head produced tail cross-entropy above the uniform baseline; the vMF head did not. I interpret vMF as a preconditioner: it alters the trajectory toward the loss floor without necessarily changing the asymptotic floor, which appears body-determined.

### Zipf hypothesis gates @ vMF epoch 8

| Gate | Result | Verdict |
|------|--------|---------|
| H1: angle ∝ −log rank | Pearson **0.23** (gate > 0.5) | Weak |
| H2: two learning problems | ratio **0.337**, stable | Null |
| H4: radial \|A_out\| vs rank | Pearson **0.96** | Confirmed |

The convergence result and H4 radial structure stand; the H1 Zipf-angle hypothesis did not pass its pre-registered gate.

I previously mislabeled an internal summary as “vMF ≈ 42 at epoch 2 / 26% faster.” The value 42.72 is epoch-8 validation perplexity. The correct epoch-2 gap is 4.2 val PPL (7.4%), not 26%.

---

## 4. Chinchilla-E floor on OpenWebText (Act IV-A)

I trained three matched softmax models on OpenWebText (500M tokens per epoch, six epochs, GPT-2 tokenizer):

| Model | Params | E_app | OWT test PPL |
|-------|-------:|------:|-------------:|
| A (10M) | 10.2M | 4.14 nats | **43** |
| C (25M) | 25.1M | 3.70 nats | **39** |
| B (51M) | 51.0M | 3.44 nats | **31** |

Triangulated E_true: **2.4863** nats (α=0.34, two-point) and **2.5322** nats (free α=0.353, three-point). Both lie in my pre-registered band **2.485 to 2.855**. Fit residuals were ±0.0000 nats on all three anchors. The tail-to-top CE ratio remained approximately **3.7×** across scales.

Log: [`Experiments/triangulation.txt`](../Experiments/triangulation.txt).

---

## 5. Scaling hypotheses tested and rejected (Act IV-B, IV-C)

### β_rep ∝ √N

I pre-registered β_rep(N) = 0.000161 × √N. At 25M and 51M parameters, |Δ| exceeded 1.3 nats (gate: |Δ| < 0.05). The fitted exponent was **N^−0.084**, essentially flat.

### A_floor ∝ V²/T

1. **Original scan (slope +0.183, R²=0.91):** invalid protocol. Optimization steps were fixed across T variants (~32.8M tokens each); V_eff masked targets without reducing the 50K-way partition; the regression used corpus length rather than tokens trained. The law was not evaluated.

2. **Corrected scan (slope −0.2525, R²=0.49):** true V_eff-way head; T_trained scaled (33.4M / 16.7M / 8.3M). Per-variant A: 500.0, 444.2, 432.3, 285.7, 114.1 nats while V²/T varied sixteenfold. Credible negative evidence, with under-fit short-T variants (R² as low as 0.005).

3. **C* = H + c·(V²/T):** the fixed (H, c) ansatz failed; deviation reached **+0.85 nats by epoch 8** (gate ±0.05). C* is a fit intercept on training CE, not corpus entropy.

I include this audit, including retraction of the invalid first falsification, as part of the scientific record for this repository.

---

## 6. Structural context

Approximately **3,369 of 50,257** GPT-2 vocabulary tokens can receive zero input gradient at small scale in my setup. Standard cross-entropy pretraining optimizes Zipf frequency rather than intervention relevance; that observation motivated my separate work on difference-native pretraining.

---

## 7. Confirmed: cheap corpus-floor recovery from public logs (Act IV-E)

This is the **strongest closed result** in the scaling arc  -  separate from the open bounded-law shape question (§8).

### Claim (confirmed)

The irreducible training loss floor is **corpus-specific** (not one universal number), and you can **estimate it from existing public training ladders at zero training cost** via fixed-α Chinchilla-E triangulation  -  with holdout validation.

| Corpus family | E_true (α=0.34) | n | Holdout Δ | LOO std | Verdict |
|---------------|----------------:|--:|----------:|--------:|---------|
| The Pile (Pythia, 6-size) | **1.48 ± 0.06 nats** | 6 | 0.003 | 0.06 | pass |
| Meta Step-2 (ti145166) | **1.56 ± 0.00 nats** | 7 | 0.006 | 0.00 | pass |
| Meta Step-2 (ti134698, default) | **1.65 ± 0.07 nats** | 3 | 0.024 | 0.07 | pass |
| Kempner OLMo / fineweb-edu (iso-flop) | **2.50 ± 0.01 nats** | 13 | 0.060 | 0.01 | pass |
| Kempner OLMo / smollm-corpus (iso-flop) | **2.66 ± 0.03 nats** | 13 | 0.101 | 0.03 | pass |
| OpenWebText (our runs) | **2.49 nats** | 3 |  -  |  -  | reference |
| Chinchilla paper (MassiveText) | ~1.69 nats |  -  |  -  |  -  | not re-runnable |

**Batch sweep (vendored logs): 7 / 13 corpora pass** holdout + LOO + sanity (`python scripts/public_ladder_sweep.py`). Failures are **diagnostic** (truncated OLMo 13B log; iso-flop ladders with non-monotonic \(E_{\text{app}}(N)\); OPT PILE at 10B tokens shows a mid-ladder bump).

Full protocol and tables: [`LOG_ONLY_TRIANGULATION_RESULTS.md`](LOG_ONLY_TRIANGULATION_RESULTS.md) · catalog: [`PUBLIC_LADDER_CATALOG.md`](PUBLIC_LADDER_CATALOG.md).

### Finding vs “law”  -  how to read the multi-corpus sweep

**Yes, it is a finding  -  a strong one:**

1. **Method finding:** When public logs satisfy a matched ladder (same corpus/stack, deep curves, monotonic \(E_{\text{app}}\downarrow\) with \(N\)), fixed-α Chinchilla triangulation **predicts the largest model’s loss out of sample** with Δ often \< 0.10 nats and LOO std \< 0.07. That held on **three independent corpus families** (Pile/Pythia, Meta Step-2 English mix, Kempner OLMo on fineweb/smollm) without retraining anything.

2. **Ansatz re-validation:** The separable form \(E_{\text{app}}(N) \approx E_{\text{true}} + A N^{-\alpha}\) with \(\alpha \approx 0.34\) is not new (Hoffmann et al.); what the sweep adds is **out-of-sample confirmation on messy public logs**, not discovery of a new functional form.

**No, it is not a single universal “entropy law”:**

- **\(E_{\text{true}}\) is corpus-specific:** Pile ≈ 1.48, Step-2 ≈ 1.55-1.65, fineweb-edu ≈ 2.50, OWT ≈ 2.5  -  the spread is the point, not noise to average away.
- **Failures are real:** When logs are truncated (OLMo 13B) or the ladder protocol breaks monotonicity (iso-flop Kempner on code corpora; OPT at fixed 10B tokens), gates **correctly reject** the fit. A universal law would not need quality gates.
- **Not Shannon entropy:** All numbers are irreducible **training CE** under a given tokenizer and stack.

**If you want one sentence for a paper:** *We confirm a portable, zero-GPU procedure to estimate corpus-specific irreducible CE floors from public scaling ladders, and re-validate the Chinchilla power-law ansatz out-of-sample on three corpus families  -  while showing that floor values themselves are not universal.*

### What this is not

- Not Shannon entropy of raw text (model-corpus irreducible CE).
- Not proof of a **universal within-run decay shape** across sizes  -  that is §8.

---

## 8. Open: universal within-run bounded law (needs 6-size OWT sweep)

**Separate question.** At n=3 OWT models, shared shape exponent \(p \approx 0.93\) fits almost as well as free per-size \(p\) (ΔR² ≈ 0.003), **but** recovering the triangulated floor \(E_{\text{true}} \approx 2.485\) under that shared \(p\) drifts to **2.744** (+0.26 nats). At three sizes, **shape and floor trade off**  -  you cannot test both simultaneously.

**Pre-registered test:** train **6 sizes** (10M-500M, same Act IV protocol), fit

\[
\mathrm{CE} = C_\infty(N) + (H - C_\infty(N))(1 + t/\tau)^{-p}
\]

with **H fixed**, **p shared**, **α=0.34** floor law. **Confirm** if ΔR² < 0.01 *and* recovered \(E_{\text{true}} \in [2.39, 2.58]\). **Reject** if shared \(p\) forces floor off triangulation by >0.10 nat, or ΔR² ≥ 0.01.

Gates and protocol: [`PREREGISTER_owt_6size_bounded_law.md`](PREREGISTER_owt_6size_bounded_law.md). Prior n=3 analysis: `archive/internal/LOCKED_OWT_TWO_ANCHOR_INVESTIGATION.md`.

**Do both:** write up §7 now (done); run §8 sweep to close the shape question either way.

---

## Scope and limitations

I do not claim that every H1 through H8 hypothesis passed (see [`act3_v32_head_comparison.json`](../results/summaries/act3_v32_head_comparison.json)). I do not claim that vMF improves the asymptotic floor, that V²/T or β_rep scaling holds, or that this repository contains all private ablations ([`LINEAGE.md`](LINEAGE.md)).
