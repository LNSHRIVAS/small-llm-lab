# GitHub publication checklist

Use this before `git push` and `gh repo create`. Goal: a **self-contained, audit-friendly** public repo that tells a clear story without leaking held work or broken reproduction paths.

---

## 1. What to publish (this repo)

**Repository name:** `small-lm-lab`  
**Public flagship** while `chinchilla-slope-one-repro` remains held for paper review.

### Include

| Category | Paths |
|----------|--------|
| Narrative | `README.md`, `docs/PUBLICATION.md`, `docs/FINDINGS.md`, `docs/RESULTS.md` |
| IV-E results | `docs/LOG_ONLY_TRIANGULATION_RESULTS.md`, `results/pythia_*`, `results/meta_step2_*`, `results/robustness_chinchilla_e/` |
| Act I-IV summaries | `results/summaries/act*.json`, `experiments/triangulation.txt` |
| Reproduction | `scripts/`, `requirements.txt`, `docs/REPRO.md`, `docs/VERIFICATION.md` |
| Historical record | `archive/colab_runs/01`-`08` (large; optional slim - see §4) |
| Colab | `colab/` notebooks + READMEs |
| Pre-registers | `docs/PREREGISTER_*.md`, `docs/HYPOTHESIS_log_only_triangulation.md` |

### Exclude (already gitignored)

- `archive/internal/` - bounded-law deep dives, θ sweeps, Zipf v2, etc.
- `data/cache/`, `*.pt`, checkpoints, `.wandb_api_key`
- `scripts/owt_chinchilla/` runtime outputs
- Private monorepo root `transformers-cult/`

### Do not link on resume

- The private `transformers-cult` workspace
- `chinchilla-slope-one-repro` until review decision (double-blind)

---

## 2. Pre-push hygiene

- [ ] Add `LICENSE` (MIT - stated in README)
- [ ] Replace `LNSHRIVAS` in docs and [`../../RESUME_BULLETS.md`](../../RESUME_BULLETS.md)
- [ ] Remove junk files (e.g. `_tmp_*.csv` in repo root)
- [ ] Run CPU validation smoke test:
  ```bash
  pip install -r requirements.txt
  python scripts/pythia_chinchilla_e_from_logs.py
  python scripts/chinchilla_e_robustness.py
  ```
- [ ] Confirm `.gitignore` excludes secrets and large caches
- [ ] Pin Python version in README (3.10+)
- [ ] **Leak grep** (must be clean before push):
  ```powershell
  cd portfolio/small-lm-lab
  rg -i "chinchilla-slope-one-repro|transformers-cult|archive/internal|theta.loop|wandb_api|LNSHRIVAS" --glob "!archive/internal/**" .
  rg "LNSHRIVAS" ../RESUME_BULLETS.md
  ```
  Expected: only `PUBLISHING.md` / `data/public_logs/README.md` mention the held repo as *do not publish*; no API keys; replace placeholders.

---

## 3. Self-contained log data (IV-E)

Scripts default to `data/public_logs/` inside this repo. Vendored files:

```text
data/public_logs/
  pythia/
    Pythia-1_4b.csv          # 1.4B W&B export (~5 MB)
    Pythia-6.9b.csv          # 6.9B W&B export (~4 MB)
  meta_step2/
    FS-step2v2_*_sc_h832_*_ti134698.csv
    FS-step2v2_*_sc_h1024_*_ti134698.csv
    FS-step2v2_*_sc_h1280_*_ti134698.csv
  olmo/                    # optional
    ...
```

Pythia 14M-410M TSVs **download automatically** from EleutherAI GitHub on first run.

Legacy note: `meta_step2` and `olmo` scripts fall back to sibling `chinchilla-slope-one-repro` for local monorepo dev only - **do not link that repo on GitHub**.

---

## 4. Notebook size trade-off

`archive/colab_runs/08_chinchilla_E_owt.ipynb` and others contain **multi-megabyte stdout**. Options:

| Strategy | Pros | Cons |
|----------|------|------|
| **Ship full notebooks** | Maximum auditability | Large clone size |
| **Clear outputs, keep notebooks** | Smaller repo | Less inline proof |
| **Git LFS for notebooks** | Balance | LFS quota |

Recommended for first public push: **ship with outputs** for notebook `08` and `06`-`07` (headline acts); clear outputs for redundant runs if size is an issue.

---

## 5. README structure (landing page)

The root [`README.md`](../README.md) should contain:

1. One-paragraph thesis + badge links to VERIFICATION / RESULTS  
2. Results table (Acts I-IV + IV-E)  
3. **Two-thread scaling box** (confirmed log-only vs open shape sweep)  
4. Quick reproduce commands (CPU + GPU tiers)  
5. Documentation map → `docs/PUBLICATION.md`  
6. Scope / limitations / license  

Do not bury IV-E - it is the most **field-portable** result.

---

## 6. Suggested GitHub metadata

**Description (≤350 chars):**

> Pre-registered small-LM falsification: vMF heads, Muon optimizers, embedding geometry, Chinchilla-E on OWT, and zero-GPU corpus-floor recovery from public training logs (Pythia, Meta Step-2). Log-verified.

**Topics:** `language-model`, `scaling-laws`, `pytorch`, `reproducible-research`, `transformers`, `machine-learning-research`

**Website:** link to `docs/PUBLICATION.md` on GitHub Pages (optional) or raw README.

---

## 7. Release narrative (GitHub Release v0.1.0)

**Title:** `v0.1.0 - Public release: Acts I-IV + log-only floor recovery`

**Body template:**

```markdown
## Highlights
- Head-only vMF vs softmax @ 9.96M params: −7.4% val PPL @ epoch 2 (matched stack)
- Chinchilla-E on OpenWebText: E_true ≈ 2.49 nats; PPL 43/39/31 @ 10M/25M/51M
- **New:** Corpus floor from public logs (zero GPU): Pythia/Pile **1.48 +/- 0.06 nats**, Meta Step-2 **1.55-1.65 nats**, **13/21** public ladders pass holdout/LOO/sanity gates

## Reproduce (CPU, ~1 min)
pip install -r requirements.txt
python scripts/pythia_chinchilla_e_from_logs.py
python scripts/chinchilla_e_robustness.py

## Full narrative
docs/PUBLICATION.md

## Open work
Optional 5-size OWT sweep for within-run shape law - docs/PREREGISTER_owt_6size_bounded_law.md (~25-35 A100-h incremental)
```

---

## 8. GPU budget reference (for readers)

Act IV-A OWT ladder (your Colab `08` timings, A100 40GB, 3B tokens/model):

| Model | GPU-hours |
|-------|----------:|
| 10M | ~4 |
| 25M | ~5.6 |
| 51M | ~8 |
| **Total (3 models)** | **~17.5** |

Incremental **5-size bounded-law test** (+100M, +250M only): **~25-35 GPU-hours** estimated.

Log-only IV-E: **$0 GPU**.

---

## 9. Post-publish

- [ ] Pin repo on GitHub profile  
- [ ] Update resume / LinkedIn with `small-lm-lab` link ([`portfolio/RESUME_BULLETS.md`](../../RESUME_BULLETS.md))  
- [ ] Optional: add arXiv / blog post linking to `docs/PUBLICATION.md`  
- [ ] After paper decision: publish `chinchilla-slope-one-repro` separately  

---

## 10. What not to claim in the GitHub README

- “Universal text entropy = X nats” - floors are **corpus-specific**
- “Proved Chinchilla wrong” - different corpora, different stacks
- “Universal within-run law confirmed” - **open** until 5-size sweep
- Numbers from the held meta-analysis paper
