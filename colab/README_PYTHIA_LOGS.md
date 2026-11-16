# Pythia floor from public logs (zero training)

```bash
pip install numpy scipy pandas
python scripts/pythia_chinchilla_e_from_logs.py
```

**Runtime:** ~30-60 seconds (downloads ~12MB TSV once, then pure CPU).

**Output:** `results/pythia_chinchilla_from_logs/validation_report.txt`

## What this proves

EleutherAI already trained the Pythia ladder on **The Pile** and published step-wise `train/lm_loss` curves. This script:

1. Downloads those TSVs from GitHub (no W&B key)
2. Builds pseudo-epoch `C*` curves (same code path as OWT Act IV)
3. Triangulates `E_true` from **70M / 160M / 410M** only

No GPU. No training bill.

## First result (this repo)

| Model | Final train loss @ 300B tok | E_app |
|-------|----------------------------:|------:|
| Pythia-70M | 2.80 | 2.78 |
| Pythia-160M | 2.50 | 2.46 |
| Pythia-410M | 2.18 | 2.10 |

**Triangulated E_true (fixed alpha=0.34): ~1.29 nats** on The Pile (Pythia stack).

Compare:
- Synthetic validation: known H, error **0.002 nats**
- OWT (you trained): E_true **~2.49 nats**
- Pythia logs (this script): E_true **~1.29 nats**

Different corpora, different stacks - the point is the **same cheap machinery** runs everywhere.

## Extend the ladder

Public TSV today: `70m`, `160m`, `410m`. For 1.4B-12B, point W&B export at `eleutherai/pythia` (see EleutherAI `hmm-training-maps/training_losses/download.py`) and drop CSVs in `results/pythia_chinchilla_from_logs/cache/`.

## Colab

Open [`pythia_floor_from_logs.ipynb`](pythia_floor_from_logs.ipynb) - one cell, no GPU.
