# Archived Colab Runs

Executed notebooks from Google Colab (T4/A100). They preserve original training logs and hypothesis verdicts for audit.

## Usage

- Paths reference `/content/drive/...` and are not runnable locally without modification.
- Stdout from multi-hour GPU runs makes these files large.
- Reproduce experiments with `scripts/`, not these notebooks.

## Index (notebooks 01 through 08)

| File | Act | Primary result |
|------|-----|----------------|
| `01_v13_k96_k64.ipynb` | I | ep2 test PPL **50.20** |
| `02_v14_k128_k64.ipynb` | I | ep2 test PPL **49.61** |
| `03_v15_k96_k48.ipynb` | I | ep2 test PPL **53.41** |
| `04_v16_k96_k96_shape_match.ipynb` | I | ep2 test PPL **46.46** |
| `05_v31_sparse_muon.ipynb` | II | ep2 test PPL **48.93** |
| `06_v32_softmax_baseline.ipynb` | III | ep2 val **57.05**, test **52.80** |
| `07_v32_vmf_ep3-8.ipynb` | III | ep8 test **39.20**; H1/H4 gates |
| `08_chinchilla_E_owt.ipynb` | IV | E_true **~2.49 nats**; OWT PPL 43/39/31 |

Summaries: [`../../results/summaries/`](../../results/summaries/). Tabulated results: [`../../docs/RESULTS.md`](../../docs/RESULTS.md).
