# Synthetic Chinchilla-E validation - Colab (A100)

One notebook, ~15-25 minutes on A100 with the fast preset.

## Files

| File | Purpose |
|------|---------|
| [`synthetic_chinchilla_e_a100.ipynb`](synthetic_chinchilla_e_a100.ipynb) | Run all three models + triangulation |
| [`../scripts/synthetic_chinchilla_e_validation.py`](../scripts/synthetic_chinchilla_e_validation.py) | Training script (needs `owt_chinchilla_e.py` in same folder) |
| [`../docs/PREREGISTER_synthetic_chinchilla_e.md`](../docs/PREREGISTER_synthetic_chinchilla_e.md) | Pre-registered gates |

## Quick start

1. Open [`synthetic_chinchilla_e_a100.ipynb`](synthetic_chinchilla_e_a100.ipynb) in Colab.
2. **Runtime → Change runtime type → GPU → A100** (T4 works too, just slower).
3. Upload the **`small-lm-lab/scripts/`** folder to `/content/small-lm-lab/scripts/`  
   (or mount Drive and point `REPO` at your copy).
4. Run all cells.

## Presets (`SYNTH_PRESET`)

| Preset | Tokens/epoch | Epochs | Typical A100 time |
|--------|--------------|--------|-------------------|
| **`a100_fast`** (default in notebook) | 5M | 4 | ~15-25 min |
| `full` | 10M | 6 | ~45-90 min |
| `smoke` | 2M | 3 | ~10 min (sanity check) |

Override any setting with env vars, e.g. `SYNTH_N_EPOCHS=6`.

## Outputs

Written to `/content/synthetic_chinchilla/` (fast local disk):

- `validation_report.txt` - pass/fail vs known H_true
- `triangulation.json` - full numbers
- `S_1M.json`, `S_2M.json`, `S_3M.json`

Optional last cell copies results to Google Drive.
