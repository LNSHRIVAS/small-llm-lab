# Documentation index

Reading order for GitHub visitors and reviewers.

| Order | Document | Audience | Time |
|------:|----------|----------|------|
| 1 | [`../README.md`](../README.md) | Everyone | 5 min |
| 2 | [`PUBLICATION.md`](PUBLICATION.md) | Reviewers, hiring, collaborators | 20 min |
| 3 | [`LOG_ONLY_TRIANGULATION_RESULTS.md`](LOG_ONLY_TRIANGULATION_RESULTS.md) | Scaling / ML-systems | 10 min |
| 4 | [`RESULTS.md`](RESULTS.md) | Numbers-only | 15 min |
| 5 | [`FINDINGS.md`](FINDINGS.md) | Interpretation + limits | 15 min |
| 6 | [`VERIFICATION.md`](VERIFICATION.md) | Auditors | 10 min |
| 7 | [`REPRO.md`](REPRO.md) | Reproducers | 10 min |

## By topic

| Topic | Documents |
|-------|-----------|
| Architecture (Acts I–III) | `RESULTS.md` §I–III, `FINDINGS.md` §1–3, notebooks `01`–`07` |
| OWT Chinchilla-E (Act IV-A) | `RESULTS.md` §IV-A, `experiments/triangulation.txt`, notebook `08` |
| Public-log floors (Act IV-E) | `LOG_ONLY_TRIANGULATION_RESULTS.md`, `PUBLIC_LADDER_CATALOG.md`, `HYPOTHESIS_log_only_triangulation.md` |
| Open bounded-law sweep | `PREREGISTER_owt_6size_bounded_law.md`, `FINDINGS.md` §8 |
| Synthetic pipeline validation | `PREREGISTER_synthetic_chinchilla_e.md`, `colab/synthetic_chinchilla_e_a100.ipynb` |
| GitHub release | `PUBLISHING.md` |
| Version history / exclusions | `LINEAGE.md` |

## Scripts ↔ docs

| Script | Doc |
|--------|-----|
| `pythia_chinchilla_e_from_logs.py` | `LOG_ONLY_TRIANGULATION_RESULTS.md` §2 |
| `meta_step2_chinchilla_e_from_logs.py` | `LOG_ONLY_TRIANGULATION_RESULTS.md` §3 |
| `chinchilla_e_robustness.py` | `LOG_ONLY_TRIANGULATION_RESULTS.md` §8 |
| `public_ladder_sweep.py` | `PUBLIC_LADDER_CATALOG.md` |
| `owt_chinchilla_e.py` | `RESULTS.md` §IV-A, `REPRO.md` |
| `v41_vmf_concentrated.py` | `RESULTS.md` §III, `FINDINGS.md` §3 |
