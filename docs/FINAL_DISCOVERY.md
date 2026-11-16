# The discovery (honest framing)

**Author:** Laksh Shrivas · **Repo:** [github.com/LNSHRIVAS/small-llm-lab](https://github.com/LNSHRIVAS/small-llm-lab)

This is not a new equation. Hoffmann et al. already fit E_app(N) ≈ E_true + A * N^(-alpha). The discovery is a **portable instrument**: you can read a **corpus-specific irreducible training cross-entropy floor** from public scaling ladders (or a small matched ladder you train), with **out-of-sample gates** that pass or fail for named structural reasons.

Every number below traces to a file in this repository. Nothing here claims a universal entropy law across corpora.

---

## One sentence

We built and validated a zero-GPU procedure that estimates **corpus-specific irreducible training CE** from scaling ladders - holdout/LOO passes on **13/21** public corpora - showing the floor is a property of the **data and training protocol**, not the model architecture, within matched ladders; floor **values are not one universal number** across corpora.

---

## Three headline numbers (with sources)

### 1. Same corpus, different stacks (The Pile)

| Stack | E_true (α=0.34) | Source |
|-------|------------------------------|--------|
| EleutherAI Pythia (6-size train C*) | **1.482** nats | `results/floor_db/floor_db.csv` → `pythia_pile_6` |
| Cerebras-GPT (7-size PILE test CE) | **1.420** nats | `results/floor_db/floor_db.csv` → `cerebras_pile` |

**Agreement:** 0.062 nats (4.2% of the mean).

**How to read it:** Two independent labs on The Pile agree to ~0.06 nats - **suggestive** that the floor tracks the corpus, not the architecture. This is **n=2**, different E_app definitions, and Cerebras holdout Δ = **0.122** vs Pythia **0.003** (`results/public_ladder_sweep/sweep_report.txt`). Do not headline “architecture invariance” without those caveats.

### 2. Cheap models predict expensive ones (within a matched ladder)

| Test | Holdout Δ | Source |
|------|----------:|--------|
| Pythia: train on 14M-410M+1.4B, predict 6.9B E_app | **0.0034** nats | `results/robustness_chinchilla_e/robustness_report.txt` |
| OWT (our runs): train on 10M+25M, predict 51M E_app | **0.0027** nats | `results/robustness_chinchilla_e/robustness_report.txt` |
| Meta Step-2 ti139508 (4 widths): holdout on largest | **0.0046** nats | `results/public_ladder_sweep/sweep_report.txt` |
| Meta Step-2 ti145166 (7 widths): LOO std on E_true | **0.0030** nats | `results/public_ladder_sweep/sweep_report.txt` |

**How to read it:** Within-ladder **E_app** prediction can hit ~0.003 nats. The **floor estimate** itself has wider uncertainty (e.g. Pythia LOO std **±0.06** nats). OWT uses flat published E_app - optimistic; see `floor_db.csv` → `owt_trained`.

### 3. Reproducibility (Meta Step-2, same corpus family)

Five independent matched-token budgets (`ti134698`, `ti139508`, `ti145166`, `ti153451`, `ti172881`) on Step-2 English web:

| Budget | E_true | Source |
|--------|--------------------:|--------|
| ti134698 (5 sizes) | 1.649 | `floor_db.csv` |
| ti139508 (4 sizes) | 1.560 | `floor_db.csv` |
| ti145166 (7 sizes) | 1.564 | `floor_db.csv` |
| ti153451 (5 sizes) | 1.554 | `floor_db.csv` |
| ti172881 (3 sizes) | 1.561 | `floor_db.csv` |

**Cross-instance std:** ~**0.040** nats (~2.5% of mean). Spread max-min: **0.095** nats.

---

## What the sweep actually shows

Batch: `python scripts/public_ladder_sweep.py` · Report: `results/public_ladder_sweep/sweep_report.txt`

| Metric | Value |
|--------|------:|
| Corpora tested | 21 |
| Holdout + LOO + sanity **PASS** | **13** |
| Universal single E_true across passes | **REJECTED** (spread 1.52 nats) · `results/floor_db/law_probes.txt` |

Example **corpus-specific** floors (α=0.34, passing gates):

| Corpus | E_true ± LOO |
|--------|-------------------------|
| The Pile (Pythia) | 1.48 ± 0.06 |
| Meta Step-2 | 1.55-1.65 |
| Kempner fineweb-edu | 2.50 ± 0.01 |
| OpenWebText (our trained reference) | ~2.49 |

Failures are **diagnosed**, not noise: truncated logs (OLMo 13B), non-monotonic ladders (code corpora, OPT @10B tokens), protocol bumps (Pythia +1B). Gated-out runs (e.g. starcoder ~1.26) are **not** cited as evidence.

---

## What this is not

- **Not Shannon entropy** of raw text - irreducible **training CE** under a tokenizer, objective, and protocol.
- **Not a universal constant** - `law_probes.txt` rejects one number across corpora.
- **Not proof of architecture invariance** - two Pile stacks agree; Pythia 7-size (+1B) **fails** gates.
- **Not a new scaling law** - the Chinchilla separable ansatz is prior art; we validate it **out of sample on messy public logs**.

---

## Why it matters (practical)

Labs often spend large compute budgets training to convergence to learn what loss floor a corpus supports. When public logs (or a 3-6 size matched ladder) satisfy quality gates, this instrument reads that number **without new training**. It does not replace training for every corpus - **8/21** ladders fail, and each failure names why.

---

## Citation

If you use this work, code, or results, please cite:

```bibtex
@software{shrivas2026smalllmlab,
  author       = {Shrivas, Laksh Shrivas},
  title        = {Small Language Model Architecture Lab},
  year         = {2026},
  publisher    = {GitHub},
  url          = {https://github.com/LNSHRIVAS/small-llm-lab}
}
```

See also [`CITATION.cff`](../CITATION.cff) and [`LICENSE`](../LICENSE).

---

## Deep dives

| Document | Content |
|----------|---------|
| [`LOG_ONLY_TRIANGULATION_RESULTS.md`](LOG_ONLY_TRIANGULATION_RESULTS.md) | IV-E protocol + tables |
| [`FINDINGS.md`](FINDINGS.md) §7 | Interpretation + limits |
| [`PUBLIC_LADDER_CATALOG.md`](PUBLIC_LADDER_CATALOG.md) | Ladder inventory |
| [`PUBLICATION.md`](PUBLICATION.md) | Full research narrative |
