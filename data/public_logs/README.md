# Public training logs (vendored for standalone GitHub)

These files let **Act IV-E** scripts run from a standalone clone (no sibling repositories required).

## Layout

| Directory | Contents | Script |
|-----------|----------|--------|
| `pythia/` | W&B CSV exports (1.4B, 6.9B, optional 1B) | `pythia_chinchilla_e_from_logs.py` |
| `meta_step2/` | **30** Meta Farseer FS-step2v2 CSVs (9 `ti*` tags) | `meta_step2_chinchilla_e_from_logs.py` |
| `kempner/` | Kempner OLMo iso-flop sweep | `kempner_chinchilla_e_from_logs.py` |
| `opt/` | Meta OPT PILE trajectories | `opt_chinchilla_e_from_logs.py` |
| `olmo/` | optional OLMo W&B exports | `olmo_chinchilla_e_from_logs.py` |

## Batch sweep + FloorDB

```bash
python scripts/public_ladder_sweep.py      # 21 corpora, failure diagnostics
python scripts/floor_db.py --skip-sweep    # floor_db.csv + law probes
```

Latest sweep: **13 / 21 pass** holdout + LOO + sanity (`results/public_ladder_sweep/sweep_report.txt`).

## Pythia / The Pile

| Size | Source |
|------|--------|
| 14M-410M | EleutherAI GitHub TSV (auto-download to cache) |
| 1.4B, 6.9B | `pythia/Pythia-1_4b.csv`, `Pythia-6.9b.csv` |
| 1B (optional) | `pythia/Pythia-1b.csv` - **not** in default 6-size ladder (protocol bump vs 410M) |

## Meta Step-2

Full **30-file** grid vendored under `meta_step2/`. Five token budgets with ≥3 sizes pass gates (ti134698-ti172881).

Override: `META_STEP2_LOG_DIR=/path/to/csvs`

## Kempner + OPT + Cerebras

- Kempner: single `kempner/kempner_sweep.csv` (~529 finished runs)
- OPT: `opt/opt_trajectories.csv` - scan finds monotonic token slice that passes
- Cerebras: hardcoded 7-point PILE eval in `point_ladder_chinchilla_e.py` (no files needed)

Optional override: `META_STEP2_LOG_DIR=data/public_logs/meta_step2`
