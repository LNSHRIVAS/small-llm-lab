# Reproduction Guide

## Requirements

```bash
pip install -r requirements.txt
```

- Python 3.10+
- GPU strongly recommended for WikiText-103; OpenWebText runs benefit from A100-class VRAM at default batch sizes
- Approximately 15 GB disk for the WikiText-103 token cache; the OpenWebText cache is larger

## Data cache

```text
data/cache/wikitext103_gpt2/   # train_tokens.pt, val_tokens.pt, test_tokens.pt
```

The first run tokenizes WikiText-103 via HuggingFace `datasets` and writes tensors locally.

```bash
export SLM_CACHE_DIR=/path/to/cache   # optional
```

---

## Act I: Embedding geometry (v13 through v16)

Runs v13 through v16 are archived in notebooks `01` through `04`. To approximate the v16-class stack locally, use the v32 script family (E4 conventions) or consult the archived notebooks for exact hyperparameters.

**Expected:** v16 epoch-2 test PPL approximately **46.5** (see [`RESULTS.md`](RESULTS.md), Act I).

---

## Act II: Muon + SparseMuon (v31)

```bash
python scripts/v31_sparse_muon_train.py
```

**Configuration:** eight layers, d=224, k=48, vMF head, 9,958,162 parameters.

**Expected @ epoch 2:** test PPL approximately **48.9**, val PPL approximately **53.6** (notebook `05`; [`act2_optimizer_stack.json`](../results/summaries/act2_optimizer_stack.json)).

---

## Act III: vMF vs softmax (head-only comparison)

Only `HEAD_MODE` differs between arms:

```bash
python scripts/v32_standard_softmax_comparison.py
python scripts/v41_vmf_concentrated.py --sota
```

**Expected @ epoch 2:** softmax val approximately **57**, vMF val approximately **53** (7.4% reduction). See [`act3_v32_head_comparison.json`](../results/summaries/act3_v32_head_comparison.json).

Full v32 Zipf suite (eight epochs and H1 through H8 gates):

```bash
python scripts/v32_zipf_diagnostics.py
```

Set `MODE='main'` at the top of the script (default).

---

## Act IV-A: Chinchilla-E on OpenWebText

```bash
python scripts/owt_chinchilla_e.py --prepare-data
python scripts/owt_chinchilla_e.py
```

**Expected:** E_true approximately **2.49 nats**; OWT test PPL **43 / 39 / 31** (notebook `08`; [`Experiments/triangulation.txt`](../Experiments/triangulation.txt)).

**Note:** E_true on OWT is an extrapolation under a Chinchilla ansatz, not a direct measurement of corpus entropy. Validate the pipeline first:

```bash
set SYNTH_PRESET=a100_fast
python scripts/synthetic_chinchilla_e_validation.py
```

Colab: [`colab/synthetic_chinchilla_e_a100.ipynb`](../colab/synthetic_chinchilla_e_a100.ipynb) (upload `scripts/`, GPU runtime).

See [`PREREGISTER_synthetic_chinchilla_e.md`](PREREGISTER_synthetic_chinchilla_e.md).

---

## Act IV-E: Corpus floor from public logs (zero GPU)

Recovers Chinchilla-E floors from **published training ladders** without training any models. Fixed α = 0.34; holdout and leave-one-out gates in `chinchilla_e_robustness.py`.

```bash
pip install numpy scipy pandas

# Pythia / The Pile (downloads TSVs on first run)
python scripts/pythia_chinchilla_e_from_logs.py

# Meta FAIR Step-2 (needs data/public_logs/meta_step2/*.csv - vendored in repo)
python scripts/meta_step2_chinchilla_e_from_logs.py

# Optional: OLMo / Dolma (exploratory; 13B log too short for gates)
python scripts/olmo_chinchilla_e_from_logs.py

# Holdout / LOO / sanity battery across all corpora
python scripts/chinchilla_e_robustness.py
```

**Expected (6-size Pythia ladder):**

| Corpus | Floor estimate (α=0.34) | Holdout | Gate |
|--------|------------------------:|---------|------|
| Pythia / The Pile | **1.48 ± 0.06 nats** (LOO std) | 6.9B Δ=0.003 | PASS |
| Meta Step-2 | **1.65 ± 0.07 nats** | h1280 Δ=0.024 | PASS |
| OWT (Act IV-A, trained) | **≈2.5 nats** | - | reference |
| OLMo / Dolma | ~2.2 nats | - | FAIL (truncated log) |

Outputs: `results/pythia_chinchilla_from_logs/`, `results/meta_step2_chinchilla_from_logs/`, `results/robustness_chinchilla_e/`.

Full write-up: [`LOG_ONLY_TRIANGULATION_RESULTS.md`](LOG_ONLY_TRIANGULATION_RESULTS.md). Narrative: [`PUBLICATION.md`](PUBLICATION.md) §5.

Override log paths: `META_STEP2_LOG_DIR`, `OLMO_LOG_DIR`. See [`data/public_logs/README.md`](../data/public_logs/README.md).

Colab: [`colab/pythia_floor_from_logs.ipynb`](../colab/pythia_floor_from_logs.ipynb).

---

## Act IV-C: V²/T scaling scan (optional)

At the top of `scripts/v32_zipf_diagnostics.py`:

```python
MODE = 'scaling_scan'
```

```bash
python scripts/v32_zipf_diagnostics.py
```

The original scan (slope +0.183) used an invalid protocol. The corrected scan yielded slope **−0.2525**; per-variant A values appear in [`RESULTS.md`](RESULTS.md). See [`VERIFICATION.md`](VERIFICATION.md) for the methodology review.

---

## Archived notebooks

Notebooks `archive/colab_runs/01` through `08` are executed historical records with Colab Drive paths and large stdout. Use them for audit; use `scripts/` for reproduction.

| Notebook | Act |
|----------|-----|
| `01` through `04` | I: embedding geometry |
| `05` | II: v31 Muon |
| `06`, `07` | III: softmax vs vMF |
| `08` | IV: Chinchilla-E |

---

## Result files

- **JSON:** `results/summaries/act1` through `act4`
- **Tables:** [`docs/RESULTS.md`](RESULTS.md)
- **Audit:** [`docs/VERIFICATION.md`](VERIFICATION.md)
