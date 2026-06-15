# Training Scripts

Path-fixed copies for local reproduction. WikiText-103 scripts read from `data/cache/wikitext103_gpt2/` under the repository root.

| Script | Act | Purpose |
|--------|-----|---------|
| `v31_sparse_muon_train.py` | II | Muon + SparseMuon stack (epoch-2 test PPL ~48.9) |
| `v32_zipf_diagnostics.py` | I / IV-C | `MODE='main'`: v32 suite and H1–H8; `MODE='scaling_scan'`: V²/T sweep |
| `v32_standard_softmax_comparison.py` | III | Softmax arm of head-only comparison (`HEAD_MODE='linear'`) |
| `v41_vmf_concentrated.py` | III | vMF arm (`--sota`: interleaved, four epochs) |
| `owt_chinchilla_e.py` | IV-A | Chinchilla-E triangulation on OpenWebText |
| `synthetic_chinchilla_e_validation.py` | IV-A validation | Known Zipf corpus + ~1–3M models; tests triangulation vs H_true |
| `pythia_chinchilla_e_from_logs.py` | IV-E | **Zero-GPU** Pythia/Pile floor from EleutherAI W&B TSV exports |
| `meta_step2_chinchilla_e_from_logs.py` | IV-E | Meta FAIR Step-2 floor from vendored CSV ladders (`data/public_logs/meta_step2/`) |
| `olmo_chinchilla_e_from_logs.py` | IV-E | OLMo/Dolma floor (exploratory; truncated 13B log fails gates) |
| `chinchilla_e_robustness.py` | IV-E | Holdout, LOO, sanity gates across Pythia / Meta / OWT / synthetic |

Colab: [`colab/synthetic_chinchilla_e_a100.ipynb`](../colab/synthetic_chinchilla_e_a100.ipynb) | [`colab/pythia_floor_from_logs.ipynb`](../colab/pythia_floor_from_logs.ipynb)

`v41_vmf_concentrated.py` imports `v32_zipf_diagnostics.py`; both files must remain in `scripts/`.

For Act III, run the softmax and vMF scripts separately and compare epoch-2 validation perplexity (approximately 57 vs 53).

Expected metrics: [`docs/RESULTS.md`](../docs/RESULTS.md); JSON [`results/summaries/`](../results/summaries/).
