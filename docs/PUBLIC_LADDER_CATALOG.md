# Public ladder catalog  -  where the method holds (and where it breaks)

**Purpose:** Map publicly available model-size ladders suitable for log-only Chinchilla-E triangulation, record what we tested, and list high-value sources still to pull.

Run the batch sweep:

```bash
python scripts/public_ladder_sweep.py
# Full Step-2 grid (local monorepo fetched dir):
META_STEP2_LOG_DIR=../data/public_logs/meta_step2 python scripts/public_ladder_sweep.py
```

Output: `results/public_ladder_sweep/sweep_report.txt`

---

## What counts as a valid ladder

Requirements for a **fair** floor-estimation test:

1. **Same corpus + stack** across all sizes (matched data order, tokenizer, training recipe).
2. **Step-level train CE** (or equivalent) logged deep enough to estimate late **C\*** / \(E_{\text{app}}\).
3. **≥3 sizes** (prefer ≥5) on a monotonic \(N\) ladder.
4. **Holdout gate:** fit on all but largest size; predict largest \(E_{\text{app}}\) with Δ < 0.15 nats.
5. **LOO gate:** leave-one-out \(E_{\text{true}}\) std < 0.15 nats.

This tests the **method**, not Shannon entropy of text.

---

## Tested corpora (June 2026 sweep)

| Corpus | Ladder | n | E_true ± LOO (α=0.34) | Holdout Δ | PASS |
|--------|--------|--:|----------------------:|----------:|:----:|
| **Pythia / The Pile** | 14M-6.9B | 6 | **1.48 ± 0.06** | 0.003 | ✓ |
| **Meta Step-2** (ti134698) | h832-h1280 | 3 | 1.65 ± 0.07 | 0.024 | ✓ |
| **Meta Step-2** (ti134698) | h832-h1152 | 5 | 1.65 ± 0.02 | 0.014 | ✓ |
| **Meta Step-2** (ti139508) | h1472-h2048 | 4 | 1.56 ± 0.01 | 0.005 | ✓ |
| **Meta Step-2** (ti145166) | h960-h2176 | 7 | **1.56 ± 0.00** | 0.006 | ✓ |
| **Meta Step-2** (ti153451) | h1280-h2048 | 5 | 1.55 ± 0.01 | 0.005 | ✓ |
| **Meta Step-2** (ti172881) | h1216-h1536 | 3 | 1.56 ± 0.01 | 0.002 | ✓ |
| OLMo / Dolma | 1B-13B | 3 | 2.19 ± 1.46 | 0.544 | ✗ |
| **Kempner OLMo** / fineweb-100b (iso-flop) | 73M-614M | 14 | 2.94 ± 0.01 | 0.041 | ✓ |
| **Kempner OLMo** / fineweb-edu-100b (iso-flop) | 161M-1.2B | 13 | **2.50 ± 0.01** | 0.060 | ✓ |
| **Kempner OLMo** / smollm-corpus (iso-flop) | 46M-415M | 13 | 2.66 ± 0.03 | 0.101 | ✓ |
| Kempner OLMo / proof-pile-2 (iso-flop) | 111M-778M | 13 | 1.75 ± 0.02 | 0.074 | ✗ |
| Kempner OLMo / slimpajama-chunk1 (iso-flop) | 73M-614M | 14 | 2.81 ± 0.02 | 0.084 | ✗ |
| Kempner OLMo / starcoder (iso-flop) | 111M-778M | 13 | 1.26 ± 0.01 | 0.045 | ✗ |
| Meta OPT / PILE @ 10B tokens | 125M-175B | 6 | 3.22 ± 0.05 | 0.136 | ✗ |
| OWT (trained, flat E_app) | 10M-51M | 3 | 2.49 ± 0.01 |  -  | ✗* |

\* OWT summary uses published flat \(E_{\text{app}}\) only (no epoch curves in repo); holdout looks good but sanity/LOO are not a fair test.

**Score (vendored logs only): 7 / 13 corpora pass holdout + LOO + sanity** (excluding OWT as reference-only). With full Step-2 fetched grid (30 CSVs): **7 / 9** on matched-token Meta ladders alone.

### Interpretation

- **Three independent corpus families** validate the method: **Pile (Pythia)**, **Step-2 English web mix (Meta Farseer)**, and **Kempner OLMo** (fineweb / fineweb-edu / smollm on iso-flop ladders).
- Step-2 floors cluster **≈1.55-1.65 nats** across token budgets (ti134698-ti172881)  -  same corpus family, consistent triangulation.
- Pile floor **≈1.48 nats** sits below Step-2, as expected for a different corpus/stack (not comparable as “entropy”).
- Kempner **fineweb-edu ≈ 2.50 nats** aligns with our OWT reference (~2.5)  -  different stacks, similar web-edu difficulty band; not proof they are the same corpus.
- **OLMo fails** when the largest ladder run (13B) has a **truncated** public log (~12k steps).
- **OPT / code iso-flop failures** show the gates working: non-monotonic \(E_{\text{app}}(N)\) breaks the Chinchilla separable ansatz even when holdout Δ is small.

---

## Data sources (where to get logs)

### Already integrated

| Source | Sizes | Access | Script |
|--------|-------|--------|--------|
| EleutherAI Pythia TSV | 14M-410M | [GitHub TSV](https://github.com/EleutherAI/pythia/tree/ccdf77005e27f2b6811d85ca145532cc502180b6/hmm-training-maps/training_losses/data) | `pythia_chinchilla_e_from_logs.py` |
| Pythia 1.4B / 6.9B W&B export | 1.4B, 6.9B | `data/public_logs/pythia/` (vendored CSV) | same |
| Meta Farseer Step-2 | h832-h2176 × multiple `ti*` budgets | `data/public_logs/meta_step2/` (**10 CSVs** vendored: ti134698×3 + ti145166×7) or full `FS-step2v2_*.csv` | `meta_step2_chinchilla_e_from_logs.py` |
| Kempner OLMo iso-flop sweep | 13-14 sizes × 6 corpora | `data/public_logs/kempner/kempner_sweep.csv` | `kempner_chinchilla_e_from_logs.py` |
| Meta OPT PILE trajectories | 125M-175B | `data/public_logs/opt/opt_trajectories.csv` | `opt_chinchilla_e_from_logs.py` |
| OLMo W&B exports | 1B, 7B, 13B | optional `data/public_logs/olmo/` | `olmo_chinchilla_e_from_logs.py` |

### High-value, not yet fully integrated

| Source | Why | Access | Blocker |
|--------|-----|--------|---------|
| **EleutherAI Pythia W&B** (`eleutherai/pythia`) | Full 8-size Pile ladder incl. 1B, 2.8B, 12B | Public W&B project; `download.py` in Pythia repo | Need W&B export or API pull; some sizes have short/fragmented logs |
| **Meta Step Law / Farseer** (`billzid/Farseer`, `billzid/predictable-scale`) | 7+ sizes, 60M-1B on Step-2 mix | Already in fetched `FS-step2v2_*` CSVs | Vendoring ~30 CSVs (~5 MB total) for standalone GitHub |
| **SmolLM2** (`HuggingFaceTB/smolLM2`) | Multi-size Fineweb-edu ladder | Public W&B | Need fetch + param counts; not yet scripted |
| **AllenAI OLMo-2** long runs | Complete 7B/13B curves | W&B `ai2-llm/OLMo-2` | 13B export currently too short |
| **Synthetic validation** | Known H ground truth | Colab `synthetic_chinchilla_e_a100.ipynb` | Copy `triangulation.json` into repo |

### Not suitable (wrong protocol)

| Source | Why skip |
|--------|----------|
| **LR-Transfer-Trajectory** (HF) | Width/LR grid, ~2k steps  -  too shallow for \(C^*\) floor |
| **Marin optimizer sweeps** (`MR-sweep-130m-2B-*`) | Fixed 130M, optimizer ablations  -  not a size ladder |
| **Chinchilla original 400-run grid** | Proprietary MassiveText + no public logs |
| **datablations contour** | Iso-loss contours in (N,D), not matched multi-size ladder |

---

## Recommended next pulls (zero GPU, high leverage)

1. **Vendor remaining Step-2 `ti*` tags** (ti139508, ti153451, ti172881) for standalone 7/9 matched-token score without sibling repo.
2. **Expand Pythia W&B exports** for 1B + 2.8B with **complete** 143k-step curves (filter bad exports like current `Pythia-2.8b.csv` at 22k steps).
3. **SmolLM2 W&B fetch**  -  fourth independent corpus family on Fineweb-edu (may overlap Kempner fineweb-edu band).
4. **Copy synthetic triangulation.json** from Colab  -  closes the known-H validation gate in `chinchilla_e_robustness.py`.

---

## Files

| Path | Role |
|------|------|
| `scripts/public_ladder_sweep.py` | Batch sweep across Pythia + Step-2 + Kempner + OPT |
| `scripts/kempner_chinchilla_e_from_logs.py` | Kempner iso-flop OLMo ladders |
| `scripts/opt_chinchilla_e_from_logs.py` | Meta OPT / PILE at fixed token budget |
| `scripts/chinchilla_e_robustness.py` | Holdout / LOO / sanity gates |
| `results/public_ladder_sweep/sweep_report.json` | Machine-readable sweep output |
| `docs/LOG_ONLY_TRIANGULATION_RESULTS.md` | Primary IV-E write-up |
