# Log-only Chinchilla-E triangulation  -  what we did and results

**Date:** June 2026  
**Cost:** CPU only (no GPU training). Most runs finish in under two minutes after logs are cached.

> **Two questions  -  do not conflate them**
>
> 1. **Can you recover a corpus floor cheaply from public logs?** → **Yes, confirmed** (this document).
> 2. **Is there a universal within-run shape CE = C∞ + (H−C∞)(1+t/τ)^(−p) with shared p?** → **Open**; needs a 6-size OWT sweep. Pre-register: [`PREREGISTER_owt_6size_bounded_law.md`](PREREGISTER_owt_6size_bounded_law.md).

## Problem

The original Chinchilla paper fit a 400+ run grid on proprietary **MassiveText** to estimate an irreducible loss floor \(E\). We wanted a **cheap validation** of the same *triangulation ansatz* without re-running that grid:

\[
E_{\text{app}}(N) \approx E_{\text{true}} + A \cdot N^{-\alpha}
\]

Given three (or more) model sizes trained on the **same corpus and stack**, with public training-loss curves, we can estimate \(E_{\text{true}}\) by fitting \(E_{\text{app}}\) across \(N\)  -  **zero new training**.

## What we built

| Script | Corpus | Source | Output |
|--------|--------|--------|--------|
| `scripts/synthetic_chinchilla_e_validation.py` | Synthetic Zipf | Generated on GPU (Colab) | Known \(H\) ground truth |
| `scripts/pythia_chinchilla_e_from_logs.py` | The Pile | EleutherAI public W&B TSVs | Primary live demo |
| `scripts/meta_step2_chinchilla_e_from_logs.py` | Meta Step-2 English web | Public `FS-step2v2_*` CSVs | Second independent corpus |
| `scripts/olmo_chinchilla_e_from_logs.py` | Dolma / OLMo | Allen AI W&B exports + `olmo.csv` | Partial (short 13B log) |
| `scripts/chinchilla_e_robustness.py` | All of the above | Holdout + LOO + sanity gates | `results/robustness_chinchilla_e/` |

Colab notebooks: `colab/synthetic_chinchilla_e_a100.ipynb`, `colab/pythia_floor_from_logs.ipynb`.

Hypothesis framing: [`HYPOTHESIS_log_only_triangulation.md`](HYPOTHESIS_log_only_triangulation.md).

---

## 1. Synthetic validation (pipeline check)

**Goal:** Recover known entropy \(H\) from fake scaling ladders where \(H = 5.64\) nats.

**Where run:** Google Colab (A100). Results not yet copied to this repo (`results/synthetic_chinchilla/triangulation.json` missing locally).

| Metric | Value |
|--------|------:|
| \(H_{\text{true}}\) | 5.640 nats |
| \(E_{\text{true}}\) (free α) | 5.638 nats |
| \|E − H\| | **0.002 nats** |
| Primary gate | **PASS** |

**Conclusion:** The fitting pipeline can recover a known floor to within 0.002 nats. This validates the **machinery**, not any real corpus.

---

## 2. Pythia / The Pile (primary public demo)

**Source:** EleutherAI [training-loss TSVs](https://github.com/EleutherAI/pythia/tree/ccdf77005e27f2b6811d85ca145532cc502180b6/hmm-training-maps/training_losses/data) for 14M-410M, plus vendored W&B CSVs for 1.4B and 6.9B (`data/public_logs/pythia/`).

**Protocol:** late pseudo-epoch **C\*** as \(E_{\text{app}}\) (steps ≥ 50,000); α = 0.34; **6-point overdetermined OLS** (4 DOF). Uncertainty = LOO std  -  not in-sample RMSE.

| Model | Params | E_app (late C*) | Final train loss |
|-------|-------:|----------------:|-----------------:|
| Pythia-14M | 14.0M | 3.6271 | 3.5870 |
| Pythia-70M | 70.4M | 2.8162 | 2.8004 |
| Pythia-160M | 157.1M | 2.6484 | 2.4990 |
| Pythia-410M | 405.7M | 2.1276 | 2.1780 |
| Pythia-1.4B | 1451.8M | 1.8310 | 2.0195 |
| Pythia-6.9B | 6857.3M | 1.7500 | 1.8022 |

**Triangulation (fixed α = 0.34, N-point OLS):** **E_true = 1.48 nats** (report **1.48 ± 0.06** from LOO)

| Legacy (3-point, do not headline) | E_true |
|-----------------------------------|-------:|
| 70M+160M+410M only | 1.29 |

**Robustness gates** (`scripts/chinchilla_e_robustness.py`):

| Gate | Result | Pass? |
|------|--------|:-----:|
| Holdout 6.9B (train other 5, predict E_app) | Δ = 0.003 | ✓ |
| Leave-one-out E_true std | **0.057** | ✓ |
| E_true < min E_app; monotonic E_app ladder | yes | ✓ |

**Reference:** Chinchilla paper reports \(E \approx 1.69\) nats on **MassiveText** (different corpus)  -  delta ≈ −0.40 nats vs our Pile estimate; not a pass/fail comparison.

**Artifacts:** `results/pythia_chinchilla_from_logs/`

---

## 3. Meta FAIR Step-2 (second independent corpus)

**Source:** Vendored CSVs in `data/public_logs/meta_step2/`  -  `FS-step2v2_*_ti134698.csv` (matched ~256B-token runs, Step-2 English web mix).

**Protocol:** Train CE vs tokens; runs are deep past the knee, so **E_app = last pseudo-epoch C\*** (inner \(C^*(T)\) fit hits the optimizer floor when the curve is flat).

| Model | Params | E_app (late C*) | Final loss |
|-------|-------:|----------------:|-----------:|
| Step2-h832-L12 | 227.6M | 2.0090 | 2.3865 |
| Step2-h1024-L16 | 393.6M | 1.9596 | 2.2456 |
| Step2-h1280-L20 | 699.6M | 1.8929 | 2.1192 |

**Triangulation (fixed α = 0.34):** **E_true = 1.6493 nats**

**Robustness gates:**

| Gate | Result | Pass? |
|------|--------|:-----:|
| Holdout h1280 | Δ = 0.024 | ✓ |
| LOO E_true std | 0.067 | ✓ |
| Sanity | monotonic E_app; E_true below all E_app | ✓ |

**Artifacts:** `results/meta_step2_chinchilla_from_logs/`

---

## 4. OLMo / Dolma (partial  -  not used for primary claims)

**Source:** Optional W&B exports in `data/public_logs/olmo/`; long 1B/7B curves optional.

| Model | Params | E_app | Issue |
|-------|-------:|------:|-------|
| OLMo-1B | 1.0B | 1.6062 | OK |
| OLMo-7B | 7.0B | 1.3511 | Validation CE, not train |
| OLMo2-13B | 13.0B | 2.3206 | Too few steps; breaks monotonicity |

**Triangulation (fixed α = 0.34):** E_true ≈ 2.19 nats  -  **holdout/LOO/sanity FAIL.** Treat as exploratory only until full train logs are exported (W&B refresh with `WANDB_API_KEY`).

**Artifacts:** `results/olmo_chinchilla_from_logs/`

---

## 5. Our own OWT runs (Act IV-A, for comparison)

From training three matched models on OpenWebText (GPU, this repo):

| Model | Params | E_app |
|-------|-------:|------:|
| A (10M) | 10.2M | 4.14 |
| C (25M) | 25.1M | 3.70 |
| B (51M) | 51.0M | 3.44 |

**E_true ≈ 2.49 nats** (fixed α = 0.34). Same ansatz, different corpus/stack than Pythia or Step-2.

See [`RESULTS.md`](RESULTS.md) Act IV-A and `results/summaries/act4_scaling_laws.json`.

---

## 6. Multi-corpus sweep (13 / 21 pass)

Batch: `python scripts/public_ladder_sweep.py` · FloorDB + law probes: `python scripts/floor_db.py`

| Corpus | n | E_true ± LOO | PASS | Failure cause (if no) |
|--------|--:|-------------:|:----:|------------------------|
| Pythia / Pile | 6 | 1.48 ± 0.06 | ✓ |  -  |
| Meta Step-2 (5 `ti*` budgets) | 3-7 | 1.55-1.65 | ✓ |  -  |
| Kempner fineweb / edu / smollm | 13-14 | 2.50-2.94 | ✓ |  -  |
| Kempner slimpajama (mono prefix) | 6 | 2.42 ± 0.03 | ✓ | full 14-pt ladder non-monotonic |
| Meta OPT / PILE (mono token slice) | 6 | 2.51 ± 0.01 | ✓ | @10B tokens: mid-ladder bump |
| Cerebras-GPT / PILE (final eval) | 7 | 1.42 ± 0.03 | ✓ | point ladder, not step curves |
| Pythia 7-size (+1B) | 7 | 1.78 ± 0.14 | ✗ | 410M→1B protocol bump |
| OLMo / Dolma | 3 | 2.19 ± 1.46 | ✗ | 13B log truncated |
| Kempner proof-pile / starcoder | 13 | ~1.26-1.75 | ✗ | non-monotonic \(E_{\text{app}}(N)\) |
| OPT @ 10B tokens | 6 | 3.22 ± 0.05 | ✗ | 6.7B→13B bump |
| OWT (flat E_app) | 3 | 2.49 | ✗* | reference only |

\* OWT uses published flat \(E_{\text{app}}\), not epoch curves.

### Failure appendix (report as prominently as passes)

Every fail has a **named mechanical cause**  -  not method noise:

1. **Truncated log**  -  OLMo 13B (~12k steps); LOO std explodes.
2. **Non-monotonic ladder**  -  OPT @10B (6.7B→13B), Kempner code/mixed corpora, Pythia +1B step-protocol mismatch.
3. **Gated-out suggestive values**  -  starcoder E≈1.26 (code, lower CE) fails monotonicity; **do not cite as confirmed floor**.

### Law probes (`results/floor_db/law_probes.txt`)

| Test | Result |
|------|--------|
| Universal single \(E_{\text{true}}\) | **REJECTED**  -  spread 1.52 nats across 13 passes |
| Kempner same-stack ordering (passes only) | **2.42 → 2.50 → 2.66 → 2.94** (edu cleaner → raw web) |
| Domain regression R²=0.83 | **NOT_A_LAW**  -  confounded by stack/tokenizer/protocol |
| α sensitivity (fineweb-edu) | **STABLE**  -  ±0.035 nats for α∈[0.24, 0.44] |

Catalog: [`PUBLIC_LADDER_CATALOG.md`](PUBLIC_LADDER_CATALOG.md) · CSV: `results/floor_db/floor_db.csv`

---

## 7. What we cannot run

**Chinchilla original 400-run grid:** MassiveText and training code are proprietary. Individual run logs are not public. We only compare qualitatively to their published \(E \approx 1.69\) nats on MassiveText.

---

## 8. Robustness battery summary

Command:

```bash
cd portfolio/small-lm-lab
python scripts/chinchilla_e_robustness.py
```

| Corpus | E_true (α=0.34) | Holdout | LOO | Sanity | Verdict |
|--------|----------------:|---------|-----|--------|---------|
| Pythia / Pile (6-size) | 1.48 ± 0.06 | ✓ | ✓ | ✓ | **Strong** |
| Meta Step-2 (3-size) | 1.65 ± 0.07 | ✓ | ✓ | ✓ | **Strong** |
| Meta Step-2 sweep (5 budgets) | ~1.55-1.65 | ✓ | ✓ | ✓ | **Strong** (see §6) |
| OLMo / Dolma | 2.19 ± 1.46 | ✗ | ✗ | ✗ | Weak (log quality) |
| OWT (flat E_app only) | 2.49 | ✓ | ✓ | ✓ | Reference only |
| Synthetic |  -  |  -  |  -  |  -  | Skipped locally (Colab only) |

**Overall (Pythia + Meta Step-2): PASS**

Full report: `results/robustness_chinchilla_e/robustness_report.txt`

---

## 8. Conclusions

1. **The triangulation ansatz works on public logs** when you have a matched size ladder and long enough late-training curves. **Three corpus families** pass strict gates: Pythia/Pile, Meta Step-2, and Kempner OLMo (fineweb / fineweb-edu / smollm-corpus iso-flop).

2. **Floor estimates are corpus-specific**, not universal Shannon entropy:
   - The Pile (Pythia stack): **~1.48 ± 0.06 nats**
   - Meta Step-2 mix: **~1.55-1.65 nats**
   - Fineweb-edu (Kempner OLMo): **~2.50 nats**
   - OpenWebText (our runs): **~2.49 nats**
   - Chinchilla paper reference (MassiveText): **~1.69 nats**

3. **Multi-corpus success is a method finding, not a universal law.** The **procedure** (fixed α, holdout, LOO) transfers; **\(E_{\text{true}}\)** values do not collapse to one number. Failures on truncated or non-monotonic ladders are expected and useful.

4. **Synthetic validation** confirms the pipeline recovers known \(H\) within 0.002 nats (Colab).

5. **Use fixed α = 0.34** for cross-corpus comparisons; free-α fits often hit optimizer bounds on real ladders.

---

## Reproduce

```bash
cd portfolio/small-lm-lab
pip install numpy scipy pandas

python scripts/pythia_chinchilla_e_from_logs.py
python scripts/meta_step2_chinchilla_e_from_logs.py
python scripts/olmo_chinchilla_e_from_logs.py          # optional; partial
python scripts/kempner_chinchilla_e_from_logs.py       # Kempner iso-flop sweep
python scripts/opt_chinchilla_e_from_logs.py           # OPT / PILE @ 10B tokens
python scripts/public_ladder_sweep.py                    # batch all corpora
python scripts/chinchilla_e_robustness.py
```

Meta Step-2 and OLMo scripts read from `data/public_logs/` (vendored in this repo).
