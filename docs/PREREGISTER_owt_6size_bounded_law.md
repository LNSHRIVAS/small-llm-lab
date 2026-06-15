# Pre-register: OWT 6-size bounded-law sweep

**Status:** not yet run (June 2026)  
**Purpose:** Resolve whether a **shared within-run shape** and **triangulated floor** can both hold at once — a confound that **cannot** be broken at n=3 but **can** at n≥5.

This is **separate** from the log-only floor triangulation in Act IV-E, which is already confirmed on public data. See [`LOG_ONLY_TRIANGULATION_RESULTS.md`](LOG_ONLY_TRIANGULATION_RESULTS.md).

---

## Two questions (do not conflate)

| Question | Status | What would settle it |
|----------|--------|----------------------|
| **A. Cheap floor from public logs** | **Confirmed** (Pythia + Meta Step-2, holdout/LOO pass) | Already done — no sweep required |
| **B. Universal within-run shape** CE = C∞(N) + (H−C∞)(1+t/τ)^(−p) with shared p | **Open** | This sweep |

---

## Model (Question B)

For each size N in a matched ladder on **OpenWebText** (same architecture family, GPT-2 tokenizer, protocol as Act IV-A):

\[
\mathrm{CE}(N,t) = C_\infty(N) + \bigl(H - C_\infty(N)\bigr)\,\bigl(1 + t/\tau(N)\bigr)^{-p}
\]

**Fixed externally:**

- \(H\) = measured corpus unigram entropy on OWT (same as prior bounded-law work)
- \(p\) = **one shared value** across all sizes (not free per curve)
- \(C_\infty(N) = E_{\text{true}} + k\,N^{-\alpha}\) with **α = 0.34** fixed; fit \(E_{\text{true}}\) and per-size τ (and k if needed)

**Compare to baselines:**

- Free \(p_N\) per size (same functional form)
- Triangulation-only \(E_{\text{true}} \approx 2.485\) nats from Act IV-A (independent anchor)

---

## Protocol (locked before run)

1. **Sizes:** 6 models spanning **≥10M to ≥500M** params (e.g. 10M, 25M, 51M, 100M, 250M, 500M — exact configs TBD from Act IV architecture template).
2. **Same stack:** depth style, optimizer, tokenizer, tokens/epoch, and epoch count as Act IV-A three-run ladder; train **long enough** that late trajectory is in hyperbolic tail (same criterion as clean OWT runs — truncated logs invalidate, as OLMo 13B demonstrated).
3. **Fit:** pooled or joint fit with **shared p**; report ΔR² vs free-per-size p.
4. **Floor recovery:** under fixed shared p, regress recovered \(C_\infty(N)\) → \(E_{\text{true}}\) via \(N^{-0.34}\) law.

---

## Pre-registered gates (written before run)

### Confirm (both must pass)

| ID | Gate | Threshold |
|----|------|-----------|
| **S1** | Shared-p vs free-p | ΔR² (free − shared) **< 0.01** on pooled late-curve fit |
| **S2** | Floor recovery | Recovered \(E_{\text{true}}\) under shared p in **[2.39, 2.58]** nats (triangulation band from Act IV-A ±0.10) |

**If S1 and S2 pass:** bounded law with **universal shape exponent p** and **Chinchilla-consistent floor** are **confirmed** at n=6 — the n=3 floor drift (+0.26 nats to 2.744) was rank degeneracy, not physics.

### Reject (either triggers fail)

| ID | Gate | Threshold |
|----|------|-----------|
| **F1** | Floor drift under shared p | \|E_{\text{true}}^{\text{recovered}} − 2.485\| **> 0.10** nats **and** S1 still passes (shape looks universal but floor won't lock) |
| **F2** | Shape not universal | ΔR² (free − shared) **≥ 0.01** — shared p materially worse; shape is size-dependent |

**If F1 or F2:** reject **universal within-run law** as stated (shape and floor are coupled, or p is not shared).

---

## What n=3 already showed (not moving goalposts)

From `archive/internal/LOCKED_OWT_TWO_ANCHOR_INVESTIGATION.md`:

- Shared p ≈ 0.93: ΔR² ≈ **0.003** vs free p → **S1 would pass**
- Fixed-p floor regression: \(E_{\text{true}}\) → **2.744** nats vs triangulation **2.485** → **S2 fails** (+0.26 nats)

At n=3, **universal shape and independent floor cannot both be verified** — they trade off. The sweep exists to break that degeneracy.

---

## What we already know without the sweep

1. **Floors are corpus-specific** (~1.3 Pile, ~1.7 Step-2, ~2.5 OWT) — not one universal constant.
2. **Fixed-α triangulation on public logs** recovers corpus floors at zero training cost (Pythia holdout Δ=0.079, Meta Δ=0.024).
3. **Failures track preconditions:** truncated logs (OLMo), free-α bound hits — not random noise.

The sweep adds (if it lands): *the approach to that corpus-specific floor has a near-universal shape exponent p across sizes on OWT.*

---

## Artifacts (target)

```
results/owt_6size_bounded_law/
  ladder_manifest.json
  per_model_train_logs/
  bounded_law_fit.json
  verdict.txt
```

Script TBD: extend `scripts/owt_chinchilla_e.py` or new `scripts/owt_6size_bounded_law.py`.
