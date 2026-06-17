# Verification Record

I traced every primary metric to an executed notebook cell, script log, or entry in [`Experiments/triangulation.txt`](../Experiments/triangulation.txt), and cross-checked against JSON in `results/summaries/`.

**Procedure:** I read archived notebooks `archive/colab_runs/01` through `08`, extracted epoch summaries, hypothesis verdicts, or triangulation blocks, and compared them to the JSON summaries.

---

## Primary metrics

| Result | Value | Source | Status |
|--------|-------|--------|--------|
| v13 ep2 | test 50.20, val 53.63 | notebook `01` | Match |
| v14 ep2 | test 49.61, val 52.84 | notebook `02` | Match |
| v15 ep2 | test 53.41, val 57.08 | notebook `03` | Match |
| v16 ep2 | test 46.46, val 50.11 | notebook `04` | Match |
| v31 ep2 | test 48.93, val 53.60 | notebook `05` | Match |
| v32 softmax ep2 | test 52.80, val 57.05 | notebook `06` | Match |
| v41 vMF ep1 | val 61.76 | v41 log / act3 JSON | Match |
| v41 vMF ep2 | val 52.82 | v41 log / act3 JSON | Match |
| v41 vMF ep4 | val 46.84, test 46.16 | v41 log / act3 JSON | Match |
| v32 vMF ep3 | test 44.71, val 48.91 | notebook `07` | Match |
| v32 vMF ep8 | test 39.20, val 42.72 | notebook `07` | Match |
| v32 vMF H1 | Pearson 0.2308 | verdict block `07` | Match |
| v32 vMF H4 | Pearson 0.9591 | verdict block `07` | Match |
| Chinchilla-E | E_true 2.4863 (α=0.34) | notebook `08`, triangulation.txt | Match |
| Chinchilla-E | E_true 2.5322 (α=0.3531) | notebook `08` | Match |
| OWT PPL | 10M=43, 25M=39, 51M=31 | notebook `08`, act4 JSON | Match |
| β_rep scaling | N^−0.084, gate fail | triangulation.txt | Match |
| IV-E Pythia E_true | **1.48 ± 0.06 nats** (6-size, α=0.34) | `robustness_chinchilla_e/` | PASS |
| IV-E Pythia holdout | holdout 6.9B Δ=**0.003** | `robustness_chinchilla_e/` | PASS |
| IV-E Meta Step-2 E_true | **1.65 nats** | `meta_step2_chinchilla_from_logs/triangulation.json` | Match |
| IV-E Meta holdout | Δ=**0.024**, LOO std=0.067 | `robustness_chinchilla_e/` | PASS |
| IV-E OLMo E_true | ~2.19 nats | `olmo_chinchilla_from_logs/` | FAIL (truncated log) |

---

## Model configurations

| Run | k_in | k_out | d | L | Head | Params |
|-----|-----:|------:|--:|--:|------|-------:|
| v13 | 96 | 64 | 192 | 8 | E4 softmax | 11,614,816 |
| v14 | 128 | 64 | 192 | 8 | E4 softmax | 13,229,184 |
| v15 | 96 | 48 | 192 | 8 | E4 softmax | 10,807,632 |
| v16 | 96 | 96 | 192 | 8 | E4 softmax | 13,229,184 |
| v31 | 48 | 48 | 224 | 8 | vMF | 9,958,162 |
| v32 softmax | 48 | 48 | 224 | 8 | linear | 9,958,162 |
| v32 / v41 vMF | 48 | 48 | 224 | 8 | vMF | 9,958,162 |

---

## Notebook index

| Notebook | Experiment | Metrics verified |
|----------|------------|------------------|
| `01_v13_k96_k64.ipynb` | Act I baseline | ep2 test 50.20 |
| `02_v14_k128_k64.ipynb` | Act I reserve dimension | ep2 test 49.61 |
| `03_v15_k96_k48.ipynb` | Act I minimum k_out | ep2 test 53.41 |
| `04_v16_k96_k96_shape_match.ipynb` | Act I boundary | ep2 test 46.46 |
| `05_v31_sparse_muon.ipynb` | Act II optimizer | ep2 test 48.93 |
| `06_v32_softmax_baseline.ipynb` | Act III softmax arm | ep2 val 57.05, test 52.80 |
| `07_v32_vmf_ep3-8.ipynb` | Act III extended vMF | ep8 test 39.20; H1/H4 |
| `08_chinchilla_E_owt.ipynb` | Act IV triangulation | E_true, OWT PPL, β_rep |
| `pythia_chinchilla_e_from_logs.py` | Act IV-E log-only | Pythia E_true, holdout |
| `meta_step2_chinchilla_e_from_logs.py` | Act IV-E log-only | Meta Step-2 E_true, holdout |
| `chinchilla_e_robustness.py` | Act IV-E gates | holdout / LOO battery |

JSON summaries: [`results/summaries/`](../results/summaries/). Tabulated results: [`RESULTS.md`](RESULTS.md).

---

## Corrections applied during verification

1. **Act I parameter counts:** I had listed 13.2M for all runs. Correct values: v13 **11.6M**, v15 **10.8M**; v14 and v16 remain 13.2M. The runs are not iso-parameter.
2. **Act III head description:** I had labeled the head “vMF K=96 / tiered.” The executed configuration is a single vMF head with k=48, d=224, matched to the softmax baseline.
3. **Act III convergence claim:** I retracted “26% faster.” The value 42.72 is epoch-8 validation perplexity. The matched-epoch gap at epoch 2 is **7.4%**.
4. **Act III optimizer label:** The softmax baseline uses Muon + SparseMuon + AdamW, not AdamW alone.
5. **Act IV V²/T:** The original +0.183 slope arose from an invalid protocol (see methodology review below).

---

## Hypotheses not supported

| Hypothesis | Gate | Outcome | Source |
|------------|------|---------|--------|
| A_floor ∝ V²/T | log-log slope ∈ [0.5, 1.5] | Not supported; original test invalid; corrected slope −0.2525 | scaling scan |
| C* = H + c·V²/T | delta ±0.05 nats ep3-8 | Falsified as fixed ansatz | notebook `07` |
| β_rep ∝ √N | \|Δ\| < 0.05 | Failed; N^−0.084 | triangulation.txt |
| H1 Zipf-angle | Pearson > 0.5 | Weak; 0.23 @ ep8 | notebook `07` |
| H2 two-learning-problems | ratio drift | Null; 0.337 stable | notebook `07` |

---

## Methodology review: vMF convergence

I confirmed the convergence claim in a head-only, matched-epoch comparison: `v32_standard_softmax_comparison.py` (`HEAD_MODE='linear'`) vs `v41_vmf_concentrated.py --sota` (`HEAD_MODE='vmf'`), sharing 9,958,162 parameters, eight layers (d=224, k=48), Muon + SparseMuon + AdamW, MTP, EMA, and data.

| Epoch | softmax val | vMF val | gap |
|------:|------------:|--------:|----:|
| 1 | 67.11 | 61.76 | −5.35 |
| 2 | 57.05 | 52.82 | −4.23 (−7.4%) |
| 4 | n/a | 46.84 (test 46.16) | n/a |

Mechanism: head/body gradient ratio **3.78×** (softmax) vs **1.31×** (vMF); softmax epoch-1 tail CE **11.38** vs uniform **10.83**; vMF epoch-1 tail CE **9.54**.

The extended v32 vMF run lacks epoch 1-2 logs in the repository (Drive checkpoints only). The first logged point is epoch 3 (val 48.91, test 44.71).

---

## Methodology review: V²/T scaling scan

| Scan | Slope | R² | Verdict |
|------|------:|---:|---------|
| Original | +0.183 | 0.91 | Invalid (T and V axes mis-specified; see Findings §5) |
| Corrected | −0.2525 | 0.49 | Partially valid negative; short-T under-fit |

Per-variant A (corrected scan): 500.0, 444.2, 432.3, 285.7, 114.1 nats.

---

## Known caveats

| Quantity | Issue | Resolution |
|----------|-------|------------|
| v16 ep2 | 46.46 vs stalled 65.19 | Primary claim uses executed v16 run only |
| Shape-match | k_in=k_out vs k_in≥k_out | Claim refers to boundary region |
| Chinchilla-E | 2.4863 / 2.4852 / 2.5322 / 2.6279 | Values differ by α prior and method; all logged |
| Zero-grad tokens | 3369/50257 vs 3369/3373 | Same count; different denominator |

---

## Limitations

I do not claim that V²/T or β_rep scaling laws hold, that vMF heads are novel (prior art includes Kumar & Tsvetkov and spherical classifiers), or that every Zipf gate H1 through H8 passed (H1 weak, H4 confirmed). Incomplete or supplementary experiments are excluded from this public repository.
