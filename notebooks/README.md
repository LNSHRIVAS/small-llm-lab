# Notebooks

This repository prioritizes scripts and JSON summaries over notebooks.

| Resource | Location |
|----------|----------|
| Runnable training | [`../scripts/`](../scripts/) |
| JSON summaries | [`../results/summaries/`](../results/summaries/) |
| Tabulated results | [`../docs/RESULTS.md`](../docs/RESULTS.md) |
| Executed Colab records | [`../archive/colab_runs/`](../archive/colab_runs/) (notebooks 01 through 08) |
| Synthetic validation (A100) | [`../colab/synthetic_chinchilla_e_a100.ipynb`](../colab/synthetic_chinchilla_e_a100.ipynb) |

## Recommended reading order

1. [`docs/RESULTS.md`](../docs/RESULTS.md)
2. Matching summary JSON (e.g. `act3_v32_head_comparison.json`)
3. Archived notebook for raw log lines

| Summary JSON | Archive notebooks |
|--------------|-------------------|
| `act1_embedding_geometry.json` | `01` through `04` |
| `act2_optimizer_stack.json` | `05` |
| `act3_v32_head_comparison.json` | `06`, `07` |
| `act4_scaling_laws.json` | `08`, `Experiments/triangulation.txt` |

## Script-first layout

1. Notebooks hardcode Colab Drive paths.
2. Executed notebooks are 2 to 8 MB of stdout, which is difficult to review in version control.
3. Scripts are path-fixed for reproduction with `data/cache/`.
