# Hypothesis: log-only Chinchilla-E triangulation

**Status:** first live demo on Pythia/The Pile (2026-06). Synthetic gate passed (0.002 nats).

## The bet

Chinchilla spent **400+ training runs** to fit

```text

L(N,D) = E + A/N^alpha + B/D^beta

```

We only need a **1D slice** at fixed data protocol:

```text

E_app(N) ≈ E_true + A * N^(-alpha)

```

If a public **model ladder** already exists (same data, same stack, published losses), we can estimate E_true **without training anything**.

Cost: download logs + CPU fit. Time: seconds.

## Evidence so far

| Experiment | Training cost | E_true (fixed α=0.34) | Ground truth | Gates |
|------------|---------------|------------------------|--------------|-------|
| Synthetic Zipf | ~5 min A100 (Colab) | 5.638 nats | H = 5.640 (known) | PASS (|Δ| = 0.002) |
| OWT (Act IV) | 3 custom runs | ~2.49 nats | unknown | trained here |
| **Pythia / Pile** | **zero** | **1.48 ± 0.06 nats** | unknown | **PASS** (6-size holdout + LOO) |
| **Meta Step-2** | **zero** | **1.65 nats** | unknown | **PASS** (holdout + LOO) |
| OLMo / Dolma | zero | ~2.19 nats | unknown | FAIL (short 13B log) |

Full write-up: [`LOG_ONLY_TRIANGULATION_RESULTS.md`](LOG_ONLY_TRIANGULATION_RESULTS.md).

Scripts: `scripts/pythia_chinchilla_e_from_logs.py`, `scripts/meta_step2_chinchilla_e_from_logs.py`, `scripts/chinchilla_e_robustness.py`

## What would make this a field-level tool

1. **FloorDB:** map `(corpus, architecture family)` → triangulated E_true from public ladders (Pythia/Pile, OLMo/Dolma, LLaMA mixes, …)
2. **6+ point ladders** beat 3-point exact-fit (default: 14m-6.9b on Pile; GitHub TSV + vendored CSV)
3. **Holdout validation:** fit on 70m+160m, predict 410m final loss
4. **Compare to Chinchilla grid** on same corpus where both exist

## Honest limits

- E_true is **model-corpus irreducible loss**, not Shannon entropy of text
- Free-α fits can hit bounds (Pythia run: use fixed α=0.34)
- Public logs are sparse for large models (W&B export in progress on EleutherAI side)
- Same 3-parameter exact-fit caveat as OWT when only three anchors are used

## The punchline

DeepMind paid in **compute**. We pay in **API calls to GitHub**.

If the ansatz holds, the floor of a dataset becomes a **table lookup** for any corpus that already has a public scaling ladder - not a million-dollar science project.
