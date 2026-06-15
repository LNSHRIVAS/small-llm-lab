# Pre-registration: Synthetic Chinchilla-E validation

**Purpose:** Test whether the Chinchilla-style triangulation pipeline recovers a **known** entropy floor, before interpreting E_true on OpenWebText.

**Script:** `scripts/synthetic_chinchilla_e_validation.py`

---

## Corpus (ground truth known before training)

I generate **i.i.d. unigram** token sequences from a fixed Zipf distribution over **2,048 active types** (truncated from rank-ordered GPT-2 vocabulary slots). Full softmax remains over the active set only, so models can be ~1–3M parameters and still learn the distribution.

\[
p_k \propto \frac{1}{k^{s}}, \quad k = 1,\ldots,V, \quad V = 2048 \text{ (default)}, \quad s = 1.0
\]

**Ground-truth entropy (nats):**

\[
H_{\text{true}} = -\sum_{k=1}^{V} p_k \ln p_k
\]

Computed analytically before training. For i.i.d. unigram text, optimal cross-entropy equals \(H_{\text{true}}\); I use that as the triangulation target.

Override vocabulary size: `SYNTH_ACTIVE_VOCAB=2048` (default).

---

## Models (small, fast)

Three matched softmax transformers (same depth family as Act IV OWT, reduced width):

| Name | Target scale | Notes |
|------|--------------|-------|
| S_1M | ~1M params | Smallest |
| S_2M | ~2M params | Middle |
| S_3M | ~3M params | Largest |

Exact parameter counts are logged at train time.

---

## Training protocol (scaled down from OWT Act IV)

| Setting | OWT Act IV | Synthetic validation |
|---------|------------|----------------------|
| Tokens / epoch | 500M | **10M** (override: `SYNTH_TOKENS_PER_EPOCH`) |
| Epochs | 6 | **6** (override: `SYNTH_N_EPOCHS`) |
| Val tokens | 5M | **500k** |
| Seq length | 1024 | 1024 |
| Optimizer | AdamW + cosine | Same |

Per epoch: validation CE, bucket CE, and H8-style `C*` from training CE vs step (same helpers as `owt_chinchilla_e.py`).

---

## Analysis (same pipeline as OWT)

1. Per model: fit `C*(T) = E_app + B·T^{-β}` → `E_app`, `β_rep`
2. Two-point triangulation (smallest + largest), α = 0.34
3. Three-point triangulation (all three), free α and fixed α = 0.34

---

## Pre-registered gates (before seeing results)

### Primary (method validation)

| Gate | Criterion | Pass |
|------|-----------|------|
| **P1** | \|E_true (3-pt free α) − H_true\| | **< 0.15 nats** |
| **P2** | \|E_true (3-pt fixed α=0.34) − H_true\| | **< 0.20 nats** |

### Secondary (sanity)

| Gate | Criterion | Pass |
|------|-----------|------|
| **S1** | Largest model val CE @ final epoch − H_true | **< 0.25 nats** |
| **S2** | E_app decreases with N (monotonic across 3 sizes) | Yes |

### Explicitly not a gate here

- β_rep ∝ √N (already failed on OWT; not part of synthetic validation)
- Match to OWT E ≈ 2.49 nats

---

## Outcomes

| Outcome | Interpretation |
|---------|----------------|
| P1 and P2 pass | Triangulation pipeline is **validated** on controlled data; OWT E_true can be reported as extrapolated with method support |
| P1 fails | Method or protocol bug; **do not** claim OWT floor discovery |
| P1 pass, S1 fail | Models under-trained; extend epochs or capacity, re-run |

---

## Runtime estimate

On a single GPU (T4 or better): roughly **30–90 minutes** for all three models at default 10M tokens/epoch × 6 epochs. CPU is supported but much slower.

Quick smoke test:

```bash
set SYNTH_PRESET=smoke
python scripts/synthetic_chinchilla_e_validation.py
```

Colab A100 (~15–25 min): open [`colab/synthetic_chinchilla_e_a100.ipynb`](../colab/synthetic_chinchilla_e_a100.ipynb), upload `scripts/`, run with `SYNTH_PRESET=a100_fast`.

---

## Outputs

Under `results/synthetic_chinchilla/` (or `SYNTH_CHINCHILLA_DIR`):

- `S_1M.json`, `S_2M.json`, `S_3M.json`
- `triangulation.json`
- `validation_report.txt`
