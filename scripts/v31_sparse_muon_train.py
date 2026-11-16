# Auto-extracted from archive/colab_runs/05_v31_sparse_muon.ipynb
# Regenerate: python scripts/build_v31_floor_train.py

# =======================================================================
# v17 — shape-match at the boundary, smaller budget — k_in = k_out = 80
# =======================================================================
#
# Sister experiment to v16. Same hypothesis (shape-match is the Lagrangian
# optimum on the boundary k_in = k_out), held at the v13 parameter budget
# (11.6M) instead of the v14 budget (13.2M). If the reallocation theory
# holds we should beat v13's 50.20 with the same param count by pushing
# both ranks to 80 instead of running 96 / 64.
#
# Configuration:
#   k_in  = 80   (was 96 in v13)
#   k_out = 80   (was 64 in v13)
#   d     = 192, d_head = 48, n_layers = 8  (unchanged)
#
# Params ≈ 11.6M — same as v13. Pure shape comparison against v13's 50.20.
#
# Two independent extrapolations from the measured marginal rates:
#   From v13 (96/64): reallocate 16 k_in → k_out
#     50.20 − 16·(0.201 − 0.018) = 47.27
#   From v15 (96/48): +32 k_out (0.201), −16 k_in (0.018)
#     53.41 − 32·0.201 + 16·0.018 = 47.26
#   Agreement → confidence in the linear extrapolation.
#
# Predictions:
#   Linear: 47.27
#   With concavity: test_ppl_ep2 = 48.3 ± 2
#   κ̄ ≈ 28-35 (still diffuse; k_out/2 = 40)
#   Per-vector vMF interference: cos_max = √(2·ln V / 80) = 0.52
#     (vs 0.58 at k_out=64, 0.67 at k_out=48)
#
# Falsification:
#   test_ppl_ep2 > 50.2 → same-budget shape-match is worse than mismatched
#     v13; reallocation hypothesis dies and the k_in dims at v13 were doing
#     more than 0.018 PPL/dim of work.
#   test_ppl_ep2 < 47 → k_out payoff even steeper than measured rates.
#
# Together v16 and v17 give us the slope of the PPL curve along the
# k_in = k_out diagonal:  slope ≈ (PPL(80) − PPL(96)) / 16.
# That slope tells us whether to push further or whether we have hit the
# next wall (d=192 backbone, d_ff=512, n_layers=8).
#
# Warm-start line copies full B^T (K_IN = K_OUT = 80, so [:K_OUT] is no-op).
# =======================================================================


import os
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'

import gc
import math
import shutil
import time
from pathlib import Path
from types import SimpleNamespace

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from transformers import GPT2TokenizerFast
from datasets import load_dataset


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
if torch.cuda.is_available():
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    USE_BF16 = torch.cuda.is_bf16_supported()
else:
    USE_BF16 = False


# ── v26: redistribute from embeddings into the body ──────────────────────
# v22-v25 post-mortem: four output-side interventions (MoS, asym-MoS-init,
# SiLU-MoS, MTP) produced IDENTICAL CE curves to three sig figs at matching
# steps, versus the v21 baseline. The head is not the bottleneck. The head
# is 12k parameters. In the v25 10M-param model the split was:
#     A_in  + A_out         = 6.43M   (63.9%)
#     Body  (8 * D=192)     = 3.54M   (35.2%)
#     Everything else       = 0.09M
# Two thirds of the model is the V-by-k lookup tables. The 96/96 baseline
# beats 64/64 not because k=96 has higher rank than k=64, but because
# doubling-plus k doubles-plus the embedding tables by 3.2M parameters
# while leaving the body untouched. That 3.2M delta is *the entire
# performance gap*.
#
# v26 inverts the allocation at the same total budget (~10M params):
#    K_IN, K_OUT:  64  -> 48    (A_in+A_out: 6.43M -> 4.82M, saves 1.61M)
#    D_MODEL:      192 -> 224   (attn per layer +33%)
#    D_FF_GATE:    512 -> 640   (SwiGLU FFN per layer +40%)
#    N_LAYERS:     8 (unchanged)
# Net: body grows 3.54M -> 5.04M (+42%); embeddings shrink -1.61M; total
# lands near 9.95M. Body share goes from 35% to ~51% — the "brain" now
# owns the majority of the parameter mass instead of the lookup table.
# k_out = 48 also realizes the compression target stated at project start.
D_MODEL   = 224
N_HEADS   = 4
D_FF_GATE = 640
N_LAYERS  = 8
K_IN      = 48
K_OUT     = 48
DROPOUT   = 0.1
ROPE_BASE = 10000.0
N_POSITIONS = 1024
VOCAB_SIZE  = 50257

BATCH_SIZE   = 8
ACCUM_STEPS  = 8
SEQ_LEN      = 1024
LR           = 3e-4
EPOCHS       = 2
WARMUP_STEPS = 300
GRAD_CLIP    = 1.0

SCHEDULE_HORIZON_EPOCHS = 10
NUM_WORKERS      = 2
VAL_MAX_BATCHES  = None
EVAL_STRIDE      = 512

# ── v19 head-family knob ─────────────────────────────────────────────────
# HEAD_MODE controls how logits are computed from the k-space projection z:
#   'vmf'    — logit_v = κ(h) · cos(ẑ, Ã_out_v) = (z · A_out[v]) / ||A_out[v]||
#              Normalizes A_out per-row. Removes the per-token norm DoF.
#              All token discrimination must live on the sphere S^{k-1}.
#              Welch bound at k=64 is 0.58 — near-saturated for V=50257.
#   'linear' — logit_v = z · A_out[v]
#              No normalization on A_out. Per-token norm ||A_out[v]|| is free,
#              so it can act as a learned unigram prior (common tokens grow,
#              rare tokens shrink). Same param count as vMF head. Same
#              out_to_k projection. This restores the norm DoF that vMF deletes.
HEAD_MODE = 'vmf'

# ── v20: softmax-bottleneck bias slot ─────────────────────────────────────
# Yang et al. (2017): a factored softmax exp(z·A[w]) can represent logit
# matrices of rank ≤ k+1, where the +1 is supplied by a learnable bias b[w].
# Without bias, the unigram component log P(w) must be absorbed inside the
# column span of A, costing one full k-dimension permanently. At k=64,
# V=50257, this is catastrophic: 1/64 of the discrimination budget is burnt
# on information that is rank-1 in context and parameterizable as V scalars.
#
# Cost: +V scalars ≈ +50,257 params on a 10M model (+0.5%), no rank added.
# Effect: CE at step 0 drops from ln(V)=10.82 to H(unigram)≈7.0 before any
# gradient step. Vectors are then freed to encode P(w|c)/P(w) only.
OUT_BIAS_ENABLED       = True
OUT_BIAS_INIT_UNIGRAM  = True   # init b[w] = log(count_w + eps) - log Z
OUT_BIAS_INIT_EPS      = 1.0    # Laplace-smoothing count for unseen tokens

# ── v21: input-token bigram shortcut ──────────────────────────────────────
# Adds logit_v += alpha * (A_in[x_t] . A_out[v]), a rank-64 factorization of
# the log-bigram matrix P(w | prev_token). Gives A_out a coherent early
# attractor (the bigram geometry) instead of relying on random-walk coupling
# through the transformer body. Parameter cost: 1 scalar. Requires k_in == k_out.
# At init alpha = 0, so the model is identical to v20 at step 0 and only uses
# the shortcut to the extent training finds it useful.
SHORTCUT_ENABLED       = True
SHORTCUT_INIT          = 0.0    # alpha_0 — start with the shortcut disabled

# ── v22: Mixture of Softmaxes (Yang et al. 2017) ──────────────────────────
# A single factored softmax exp(z·A[w]) + b has rank ≤ k+1 in the log-prob
# matrix. Wikitext-103's conditional distribution has much higher intrinsic
# rank, so k=64 hits the ceiling early. MoS computes n independent factored
# softmaxes with different context vectors z_i = U_i(h), combines them with
# a context-dependent gate pi_i(c) = softmax(W_g h)_i. The resulting log-prob
# matrix has rank up to n·(k+1), without growing k or A_out.
#
#   log p(v|c) = logsumexp_i [ log pi_i(c) + kappa_i cos(z_i/|z_i|, A_out[v]/|A_out[v]|) + b_v - logZ_i(c) ]
#
# Parameter cost for n=2: n·D·k projection + D·n gate = 192·64·2 + 192·2
#                       ≈ 24,960 params (+0.25% on 10M). No rank added to k.
# Effective rank gain: 1·(k+1) -> n·(k+1). At k=64, 65 -> 130 (beats k=96 head).
MOS_COMPONENTS = 1      # v25: MoS proved net-neutral (v22/v23/v24). Off.

# ── v23: MoS symmetry-breaking ────────────────────────────────────────────
# v22 observation: both MoS projections were warm-started from B^T + 0.02
# noise. With identical-looking init, uniform gate, and symmetric gradient
# descent, the two components collapsed to a duplicate. gate_H stayed at
# ln(2)=0.693 for 3600 steps, meaning effective rank never exceeded 65.
#
# v23 breaks the symmetry explicitly:
#   (a) Component 0 is warm-started from B^T (keeps the subspace-rotation
#       tax paid-off in early training).
#   (b) Components 1..n-1 use default Kaiming init (std ~1/sqrt(D) = 0.072),
#       placing them in a qualitatively different region of weight space.
#   (c) Gate has a learnable bias initialized so pi_0 dominates at step 0
#       (pi = softmax([1,0]) = [0.73, 0.27] for n=2). Component 1's random
#       projection produces near-uniform log p_1 initially; weighting it
#       at 27% costs ~0.3 extra nats on step-0 CE, which is a much smaller
#       tax than losing rank permanently.
#   (d) Diversity bonus: add -lambda * sym-KL(p_0, p_1) to the loss. This
#       is tiny (lambda=1e-3) but keeps a constant gradient pushing the
#       components' softmaxes apart until the data itself separates them.
MOS_ASYMMETRIC_INIT      = True
MOS_GATE_BIAS_INIT_SKEW  = 1.0       # pi_0 = softmax([skew, 0, ...0])
# v23 used 1e-3; diagnostics showed cos_z climbed from 0.115 back to 0.408 by
# step 3200, proving CE gradient (~3e-4/elem) dominated diversity (~6e-5/elem).
# v24: 2e-2 gives diversity grad ~1.2e-3/elem, 4x the CE gradient => sticky.
MOS_DIVERSITY_LAMBDA     = 2e-2

# v24: structural asymmetry between components. Component 0 is a pure linear
# map (warm-started from B^T so early CE is fast). Components 1..n-1 apply a
# SiLU nonlinearity before the vMF decode, so they compute a functionally
# *different* class of map and cannot collapse to component 0's solution even
# if gradient pressure wants them to. SiLU is chosen over tanh because it is
# unbounded (lets kappa = ||z_i|| grow for discriminative predictions) while
# still dampening the negative half so the two components point into
# structurally different regions of S^{k-1}.
MOS_NONLIN_ON_EXTRA_COMPONENTS = True

# ── v25: Multi-Token Prediction (Gloeckle 2024; DeepSeek-V3) ──────────────
# The v22-v24 MoS experiments showed r_head <= k+1 is NOT the binding limit:
# we doubled the log-prob matrix's rank ceiling with MoS, the mixture did
# specialize (gate_H 0.41, k_per_comp [13.9, 7.7], cos_z 0.16), and CE did
# not move versus a single-softmax run. That proves r_body is binding at
# k_out = 64: the body is only self-organizing rank-k_out features because
# that's all the gradient signal it gets from a single rank-k_out head.
#
# MTP fixes this without touching k_out. A second output projection U_mtp
# of shape D -> k_out runs against token t+2 instead of t+1. Its CE gradient
# flows through a *different* rank-k_out subspace of h than the main head,
# so body parameters see update directions of effective rank up to 2*k_out.
# This is isomorphic to the gradient bandwidth gain of going from k_out=64
# to k_out=128, but A_out stays V x 64 and the inference head stays rank-65.
#
# Cost: one extra D x k_out projection = 12,288 params (+0.12%). Zero
# inference-time overhead (U_mtp is only used for the aux loss).
MTP_ENABLED      = True
MTP_DEPTH        = 2       # predict token at position t+2 (t+1 is the main task)
MTP_WEIGHT       = 0.3     # beta in total_loss = CE_main + beta * CE_mtp

# ── v27: Muon optimizer for body matrix parameters (Keller Jordan 2024) ───
# v22-v26 pattern: five architectures, identical CE curves within 0.04 nats
# at matched steps. Head tricks neutral. Body widening neutral. k shrinking
# neutral. That invariance means the binding constraint is NOT architecture
# — it's whatever is shared across all five runs. The primary shared
# element is the optimizer: AdamW at lr=3e-4.
#
# AdamW treats every parameter as a scalar and normalizes its update by
# per-coordinate second-moment estimate. For a weight matrix W in R^{m x n},
# AdamW's effective step concentrates along the singular directions where
# the gradient has large magnitude — i.e. it rotates W fast in a few
# directions and slow in the rest. At 10M params over 14k steps this makes
# most of W_gate/W_up/W_down stay near their init in most directions.
#
# Muon orthogonalizes the momentum buffer of each 2D matrix parameter via
# a 5-step Newton-Schulz iteration before taking the step. This turns the
# matrix update into an approximation of U V^T where G = U S V^T, i.e. it
# updates the matrix uniformly across its singular spectrum. In practice
# (Jordan 2024, modded-nanoGPT speedrun), this is 1.5-2x faster wall-clock
# to a given val loss at 100M-1B scale, and should be even more pronounced
# at our 10M/14k-step regime where per-step rotation speed is the bottleneck.
#
# Routing:
#   A_in.weight, A_out.weight -> AdamW   (sparse gradient, Muon's NS assumes
#                                         dense; this is the canonical split)
#   All other 2D weights      -> Muon    (body Linears, B, out_to_k*, mos_gate)
#   1D params (norms, biases) -> AdamW   (Muon requires ndim==2)
#   shortcut_scale (0D)       -> AdamW
MUON_ENABLED    = True
MUON_LR         = 0.02    # Muon uses a much larger LR than AdamW (orthogonalized updates have fixed norm)
MUON_MOMENTUM   = 0.95
MUON_NESTEROV   = True
MUON_NS_STEPS   = 5       # Newton-Schulz iterations per update. 5 is Jordan's recommended value.

# ── v28: Sparse Muon for embedding tables ────────────────────────────────
# v27 finding: Muon shifted CE by 0.22 nats (constant offset, not a slope
# change). Body grad shrank from 1.14 -> 0.21 (Muon saturated the body),
# while A_out grad GREW from 0.54 -> 0.88 (the body's demand on A_out
# rotated faster than AdamW @ lr=3e-4 could supply). The next bottleneck
# is the embedding tables sitting on a slow optimizer.
#
# Standard Muon excludes embeddings because nn.Embedding.weight.grad is
# "sparse" (most rows are zero — only batch-active tokens have signal).
# But the active rows form a dense (n_active, k) sub-matrix where Muon's
# orthogonalization assumption holds exactly. We extract that sub-matrix
# each step, run NS5 on it, and scatter the orthogonalized update back to
# only the rows that produced it. Inactive rows are not touched.
#
# Why this should work where AdamW can't:
# - AdamW's per-coordinate adaptive denominator gives every active row an
#   update of magnitude ~lr regardless of how rarely the token appears.
#   For a token that fires once per 200 steps, total annual update ~lr.
# - SparseMuon gives an active row an update of magnitude ~1/sqrt(k) per
#   step it is active. For k=48 that's ~0.144 PER OCCURRENCE, vs AdamW's
#   ~3e-4. Per-occurrence step size is ~500x larger.
# - Frequency awareness comes from "how many times you appear", not from
#   "how big each update is". This is the right asymmetry for a Zipfian
#   vocab: each occurrence of a rare token now does real work.
SPARSE_MUON_ENABLED = True
SPARSE_MUON_LR      = 0.005  # smaller than body Muon — embeddings care about row direction, not full matrix rotation
SPARSE_MUON_MOMENTUM = 0.95
SPARSE_MUON_NS_STEPS = 5

# ── v29: tangent-only SparseMuon for A_out ───────────────────────────────
# v28 finding: SparseMuon gave a 0.44-nat initial lift then matched v27's
# slope after step 2000. Root cause is a SparseMuon / vMF interaction:
#   - vMF logit = kappa * cos(z, A_out[v]) is invariant to ||A_out[v]||.
#     Only the *direction* of A_out[v] carries loss-relevant signal.
#   - NS5-orthogonalized updates have fixed per-row magnitude ~1/sqrt(k)
#     regardless of current row norm, and they include a radial component
#     (parallel to A_out[v]) that the loss cannot see.
#   - The radial component is therefore never contested by gradient, but
#     it keeps adding to the row norm. Over 4200 steps ||A_out||_bar grew
#     from 0.138 to 0.408 (2.96x).
#   - Angular update per step = |Δ_tangent| / ||A_out||. As norm grows,
#     angular rotation per step collapses: from ~36 mrad/step at init to
#     ~13 mrad/step at step 4000. That shrinkage is the exact deceleration
#     we see in CE slope (6.1e-4 → 2.0e-4 over the same window).
# Fix: project the orthogonalized update onto the tangent space of the
# current row before applying it. Radial component is discarded. All of
# the per-step update budget goes into rotation, and ||A_out|| stays ~init
# forever. A_in does NOT get tangent projection because A_in is an input
# lookup whose norm DOES affect downstream magnitude through B @ A_in[v].
SPARSE_MUON_TANGENT_A_OUT = False  # v29 disproved this — radial growth was load-bearing
SPARSE_MUON_TANGENT_A_IN  = False

# ── v31: per-row Adagrad-over-occurrence-count LR in SparseMuon ──────────
# v28/v29/v30 all converge to the same ~0.8 nats/step 1/t tail. Four
# different optimizer configurations, same coefficient. That is a signal:
# the tail is NOT set by the optimizer. Decomposition of the grad logs
# shows body gradient decays 102x from init to step 4800 while A_out and
# A_in only decay ~16x. By step 4800 head matrices produce ~2/3 of the
# total first-order signal. The head is the rate limiter, not the body.
#
# Inside the head: 50k A_out rows need to rotate to their optimal angular
# positions on S^(k-1). Each row gets gradient only when its token appears.
# By Zipf: top-100 tokens get ~5000 occurrences over 5000 steps and their
# rows rotate tens of radians — long-since converged directionally. Rank-
# 30000 tokens get ~10 occurrences with per-occurrence angular step ~0.036
# rad at ||A_out||_bar=0.14, so total rotation budget is ~0.36 rad ≈ 20°.
# Starting from a random point on S^47, 20° is nothing. The bottom half
# of the vocabulary is literally unrotated from init by the end of the
# run. The 1/t slope coefficient is locked by Zipf, not by variance.
#
# Cure: equalize the angular budget across token ranks. Per-row LR
# multiplier = scale * (1 + update_count[v])^(-power) with power=0.5.
#   - Most rare tokens (count<=1) get full base LR.
#   - Frequent tokens get attenuated; at count=10000 the multiplier is
#     ~0.01, so their rows effectively stop rotating (they are already
#     done). The compute saved on frequent rows is redirected nowhere —
#     the update magnitude just falls — but their momentum buffers still
#     decay cleanly, so gradient SIGNAL on frequent rows still flows to
#     the body via gradient chaining. We are only killing frequent-row
#     *self-motion*, not frequent-row *teaching of body*.
#   - Net effect predicted: the 1/t A coefficient drops by ~30-40% because
#     the slow-converging rare tokens now accumulate real angular distance
#     per occurrence, shifting CE at matched step DOWN.
#
# This is the Adagrad accumulator restricted to the *count* dimension only
# — we do NOT divide by gradient magnitude, because NS5 already normalizes
# step magnitude. Dividing by count^0.5 is the pure per-row "rarity boost".
# Floor the multiplier at PER_ROW_LR_MIN so frequent tokens never lock
# completely (keeps late drift available if the body recalibrates them).
SPARSE_MUON_PER_ROW_LR_ENABLED = True
# Critical: A_out gradient is dense under full softmax CE (every vocab row gets
# non-zero grad each step). So "update_count from non-zero grad rows" is NOT a
# token-frequency proxy for A_out; it just tracks global step and uniformly
# shrinks all rows, effectively freezing A_out too early. Keep per-row LR only
# where row-activity is truly sparse.
SPARSE_MUON_PER_ROW_LR_A_OUT   = False
SPARSE_MUON_PER_ROW_LR_A_IN    = True
SPARSE_MUON_PER_ROW_LR_POWER   = 0.5   # 0.5 = Adagrad exponent over count
SPARSE_MUON_PER_ROW_LR_SCALE   = 1.0   # multiplier before (1+count)^-power
SPARSE_MUON_PER_ROW_LR_MIN     = 0.02  # floor so frequent rows stay unfrozen
SPARSE_MUON_PER_ROW_LR_MAX     = 1.0   # ceiling (rare row never > base LR)

# ── v30: cosine warm restarts + weight EMA (DISABLED in v31) ─────────────
# v29 post-mortem (per-window slopes for v29):
#   5.87e-4 at t=1500, 3.39e-4 at t=2500, 2.34e-4 at t=3500
# Fits slope = A/(t + t0) with A ≈ 0.80 nats, t0 ≈ -130, residuals < 2%.
# This is exact Robbins-Monro 1/t behavior for constant-stepsize SGD near
# a local quadratic minimum. The tail is not an optimizer bottleneck. It
# is the batch-gradient-variance envelope σ²·lr/(2·t). You cannot beat
# the 1/t rate without either (a) reducing σ², or (b) leaving the basin.
#
# v28 outperformed v29 by ~0.06 nats because radial growth in v28 acted as
# an emergent per-token adaptive damping (frequent tokens grew ||A_out||,
# shrinking their angular rate via the 1/||A_out|| factor in vMF's
# F.normalize Jacobian). v29's tangent projection killed that mechanism
# and caused frequent-token oscillation (||z|| dropped 14.49 -> 12.59
# because body gave up pushing kappa against moving targets). Revert.
#
# v30 attacks the 1/t asymptote by basin-hopping: SGDR-style cosine
# warm restarts. Every RESTART_STEPS the LR re-warms to peak and cosine-
# decays, temporarily destabilizing the current solution. Muon's
# orthogonalized update kicks every singular direction uniformly on
# restart, so the body can jump to structurally different basins rather
# than just retracing the dominant-eigenvector direction. We ALSO track
# an EMA of weights over the full trajectory — as a diagnostic (large
# EMA/inst gap => real variance floor) and as a potential evaluation
# weight set (EMA is typically 0.1-0.3 nats below instantaneous late in
# training).
SGDR_ENABLED        = False    # v30 showed restarts are a null intervention: v30 and v28
SGDR_RESTART_STEPS  = 2000     # CE curves matched to within 0.008 nats at every step
SGDR_WARMUP_STEPS   = 100      # post-restart; basin-hopping theory is falsified.
SGDR_MIN_LR_FRAC    = 0.1

EMA_ENABLED         = True
EMA_DECAY           = 0.999
EMA_EVAL_STRIDE_MUL = 1        # evaluate EMA at same cadence as normal eval

# Kappa cap only applies when HEAD_MODE='vmf'. Left here as-is for revert runs.
KAPPA_CAP_ENABLED = False
KAPPA_CAP_FRAC    = 0.15
KAPPA_CAP_INIT    = 8.0
KAPPA_CAP_FINAL   = 28.0

# Frequency-balanced CE DISABLED — the v18 attempt put 95% of the vocab at the
# 3x clamp (Zipfian tail), causing rare-token BPE fragments to dominate the
# loss. Weighted CE at init was 11.16 > ln(V)=10.82, i.e. above uniform. Do
# not re-enable without a much smaller alpha (~0.05) AND a much tighter clamp
# (~1.3), and only as a second-order experiment after HEAD_MODE is settled.
FREQ_BAL_ENABLED = False
FREQ_ALPHA       = 0.0
FREQ_EPS         = 1.0
FREQ_CLAMP_MAX   = 1.3

GRAD_LOG_EVERY   = 200

DRIVE_CACHE_DIR = Path(os.environ.get('V31_DRIVE_CACHE', '/content/drive/MyDrive/llm_token_cache/wikitext103_gpt2'))
LOCAL_CACHE_DIR = Path('/content/token_cache/wikitext103_gpt2')
TOKENIZER_NAME  = 'gpt2'
TOKENIZE_BATCH_SIZE = 2000

RUN_NAME = 'v31_kin48_kout48_d224_ff640_smuon_perrow_count05'

# ── E4 Diagnosis ─────────────────────────────────────────────────────────────
#
#  THE DISEASE: A is still tied in E3. With k=24, V/k = 2094.
#
#  For any A[v]:
#    OUTPUT path  → touches all 50,257 rows every step, magnitude κ ≈ 17
#    INPUT  path  → touches ~8,192 rows/step (16.3%), magnitude ≈ 0.35
#    AdamW effective update: INPUT contributes ~2% of OUTPUT
#    → A is 97.9% shaped by retrieval geometry, 2.1% by representation quality
#    → 84% of tokens (never in batch) have A[v] chosen purely for retrieval,
#      not for being a useful transformer input
#
#  WHY D/k = 8× understates it:
#    Standard tied head (V×D): output gradient spread over D dims, input path
#    can live in the orthogonal complement. V/D ≈ 262.
#    Factored tied head (V×k): output gradient spread over only k dims. No
#    orthogonal complement to hide in. V/k ≈ 2094. AdamW rounds input path to
#    near-zero in the shared second moment. Absolute harm, not relative.
#
#  THE FIX (E4):
#    Split A → A_in (input path) + A_out (output path). No shared matrix.
#    INPUT:  ids → A_in → B → transformer → h
#    OUTPUT: h → out_to_k → z → normalize(A_out) → logits
#
#  GRADIENT ROUTING (complete):
#    A_in.weight   ← g_in  only  (sparse, batch tokens, free to specialize)
#    A_out.weight  ← g_out only  (dense, all V, free to optimize retrieval)
#    B.weight      ← g_in  only  (unchanged from E2/E3)
#    out_to_k      ← g_out only  (unchanged from E2/E3)
#
#  WARM-START (free fix):
#    Random init: out_to_k and B.T span subspaces with k²/D ≈ 0.5% overlap.
#    First ~1000 steps spent rotating out_to_k to find B's subspace (visible
#    in E3 κ trajectory: slow growth phase 200→1200 before κ plateaus).
#    Fix: warm-start out_to_k from B.T rows to maximize initial overlap.
#    With k_in != k_out, copy B.T[:k_out] into out_to_k.weight.
#
#  PARAMS:
#    E3: A shared = V×k = 1,206,168
#    E4: A_in + A_out = 2×V×k = 2,412,336 (+1,206,168)
#    E4 total ≈ 5,964,528 params (+25.3% vs E3)
#
#  PREDICTIONS:
#    epoch 1 val_recall@1  ≥ 0.34   (vs E3 epoch 1: 0.3169)
#    epoch 2 val_recall@1  ≥ 0.38   (vs E3 epoch 2 est: 0.333)
#    epoch 2 val_ppl       ≤ 65     (vs E3 epoch 2 est: ~75)
#    epoch 1 train_ce drop ≥ 1.3×E3 (decoupled A_in unlocks better h)
#
#  DIAGNOSTIC to log:
#    ‖A_in[v]‖ vs ‖A_out[v]‖ histograms  → do they diverge?
#    cos(A_in[v], A_out[v]) per token     → if → 1: tying was benign
#                                           if → 0: matrices wanted different geometry
#
#  FALSIFICATION:
#    recall < 0.32 at epoch 1 → A-conflict was not the bottleneck
#    ppl > 72 at epoch 2      → gains don't translate to PPL, need per-token scale
#    cos(A_in, A_out) → 1     → E4 rediscovers E3, split was unnecessary
#
# ─────────────────────────────────────────────────────────────────────────────


# ── Dataset ───────────────────────────────────────────────────────────────────
class TokenDataset(Dataset):
    def __init__(self, tokens, seq_len):
        self.tokens  = tokens
        self.seq_len = seq_len
        self.n       = (len(tokens) - 1) // seq_len

    def __len__(self):
        return self.n

    def __getitem__(self, i):
        s = i * self.seq_len
        return self.tokens[s:s + self.seq_len], self.tokens[s + 1:s + self.seq_len + 1]


# ── Building blocks (unchanged) ───────────────────────────────────────────────
class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps    = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        orig_dtype = x.dtype
        x32 = x.to(torch.float32)
        rms = x32.pow(2).mean(dim=-1, keepdim=True).add(self.eps).rsqrt()
        return (x32 * rms).to(orig_dtype) * self.weight


class RotaryEmbedding(nn.Module):
    def __init__(self, dim, max_seq_len, base=10000.0):
        super().__init__()
        assert dim % 2 == 0
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        t        = torch.arange(max_seq_len, dtype=torch.float)
        freqs    = torch.einsum('i,j->ij', t, inv_freq)
        self.register_buffer('cos_cached', freqs.cos()[None, None, :, :], persistent=False)
        self.register_buffer('sin_cached', freqs.sin()[None, None, :, :], persistent=False)

    def apply(self, x, offset=0):
        t   = x.size(-2)
        cos = self.cos_cached[:, :, offset:offset + t, :].to(dtype=x.dtype, device=x.device)
        sin = self.sin_cached[:, :, offset:offset + t, :].to(dtype=x.dtype, device=x.device)
        x_even, x_odd = x[..., ::2], x[..., 1::2]
        out_even = x_even * cos - x_odd * sin
        out_odd  = x_even * sin + x_odd * cos
        return torch.stack((out_even, out_odd), dim=-1).flatten(-2)


# ── E4: FactorizedEmbeddingE4 — THE core change ───────────────────────────────
class FactorizedEmbeddingE4(nn.Module):
    """
    Full path decoupling: A_in ≠ A_out. No tied weights anywhere.

    E3 had a shared A ∈ R^{V×k}. With k=24, V/k=2094, making the output-path
    gradient (dense, magnitude ~17) completely dominate the input-path gradient
    (sparse 16.3% coverage, magnitude ~0.35). AdamW's adaptive denominator
    rounds the input-path contribution to ~2% of the effective update.

    A_in  — optimized to be good transformer inputs (sparse gradient, low mag)
    A_out — optimized for retrieval geometry on S^{k-1} (dense gradient, high mag)
    Both matrices now get 100% of their respective gradient signals.

    Parameter count vs E3: +V×k = +1,206,168 params (+25.3%)
    """
    def __init__(self, vocab_size, d_model, k_in, k_out):
        super().__init__()
        self.A_in  = nn.Embedding(vocab_size, k_in)       # INPUT path only
        self.A_out = nn.Embedding(vocab_size, k_out)      # OUTPUT path only
        self.B     = nn.Linear(k_in, d_model, bias=False) # input upscale
        nn.init.normal_(self.A_in.weight,  0.0, 0.02)
        nn.init.normal_(self.A_out.weight, 0.0, 0.02)
        nn.init.orthogonal_(self.B.weight)

    def embed(self, ids):
        """INPUT path: ids → A_in → B → R^D. Gradient flows to A_in, B only."""
        return self.B(self.A_in(ids))

    def normalized_A_out(self):
        """OUTPUT path: A_out on unit sphere. Gradient flows to A_out only."""
        return F.normalize(self.A_out.weight.float(), dim=-1)   # [V, k]

    def decode_scores(self, z, kappa_cap=None, mode='vmf'):
        """
        mode='vmf'   : logit_v = κ(h) · cos(ẑ, Ã_out_v). Strips ||A_out[v]||.
        mode='linear': logit_v = z · A_out[v]. Keeps ||A_out[v]|| as free DoF.
        kappa_cap    : only honored when mode='vmf'.
        z            : [B, T, k_out] unnormalized from out_to_k.
        """
        z_f = z.float()
        if mode == 'linear':
            # Plain low-rank linear head. No normalization on either side.
            # A_out[v] is free to grow/shrink — learned unigram prior.
            return z_f @ self.A_out.weight.float().T
        # vMF path
        A_n       = self.normalized_A_out()
        kappa     = z_f.norm(dim=-1, keepdim=True).clamp_min(1e-6)
        direction = z_f / kappa
        if kappa_cap is not None:
            kappa = kappa.clamp(max=kappa_cap)
        return kappa * (direction @ A_n.T)


class TransformerBlock(nn.Module):
    def __init__(self, d_model, n_heads, d_ff_gate, dropout, n_layers, rope):
        super().__init__()
        assert d_model % n_heads == 0
        self.n_heads   = n_heads
        self.d_head    = d_model // n_heads
        self.dropout_p = dropout
        self.rope      = rope

        self.norm_attn = RMSNorm(d_model)
        self.norm_ffn  = RMSNorm(d_model)
        self.qkv_proj  = nn.Linear(d_model, 3 * d_model, bias=False)
        self.out_proj  = nn.Linear(d_model, d_model, bias=False)
        self.q_norm    = RMSNorm(self.d_head)
        self.k_norm    = RMSNorm(self.d_head)
        self.w_gate    = nn.Linear(d_model, d_ff_gate, bias=False)
        self.w_up      = nn.Linear(d_model, d_ff_gate, bias=False)
        self.w_down    = nn.Linear(d_ff_gate, d_model, bias=False)
        self.dropout   = nn.Dropout(dropout)

        scale = 1.0 / math.sqrt(2 * n_layers)
        nn.init.normal_(self.qkv_proj.weight, 0.0, 0.02)
        nn.init.normal_(self.out_proj.weight, 0.0, 0.02 * scale)
        nn.init.normal_(self.w_gate.weight,   0.0, 0.02)
        nn.init.normal_(self.w_up.weight,     0.0, 0.02)
        nn.init.normal_(self.w_down.weight,   0.0, 0.02 * scale)

    def attention(self, x):
        b, t, d = x.shape
        q, k, v = self.qkv_proj(x).chunk(3, dim=-1)
        def reshape(z): return z.view(b, t, self.n_heads, self.d_head).transpose(1, 2)
        q, k, v = reshape(q), reshape(k), reshape(v)
        q = self.rope.apply(self.q_norm(q))
        k = self.rope.apply(self.k_norm(k))
        out = F.scaled_dot_product_attention(
            q, k, v, is_causal=True,
            dropout_p=self.dropout_p if self.training else 0.0,
        )
        return self.out_proj(out.transpose(1, 2).contiguous().view(b, t, d))

    def ffn(self, x):
        return self.w_down(self.dropout(F.silu(self.w_gate(x)) * self.w_up(x)))

    def forward(self, x):
        x = x + self.dropout(self.attention(self.norm_attn(x)))
        x = x + self.dropout(self.ffn(self.norm_ffn(x)))
        return x


# ── E4: Main model ─────────────────────────────────────────────────────────────
class BreakthroughMicroTransformerE4(nn.Module):
    """
    E4: Full gradient path decoupling — no shared matrix between input and output.

    All five ingredients now active:

    Ingredient               | Baseline | E2  | E3  | E4
    ─────────────────────────┼──────────┼─────┼─────┼────
    log Z(h) partition fn    |   ✓      |  ✓  |  ✓  |  ✓
    Decoupled B (k→D)        |   ✗      |  ✓  |  ✓  |  ✓
    ‖A_out_v‖ = 1 (sphere)   |   ✗      |  ✓  |  ✓  |  ✓
    Adaptive κ(h) = ‖z‖      |   ✗      |  ✗  |  ✓  |  ✓
    Decoupled A (V×k)        |   ✗      |  ✗  |  ✗  |  ✓  ← E4 addition

    Core disease fixed by E4:
      Shared A with V/k=2094 meant output-path gradient (κ≈17, dense) drowned
      input-path gradient (mag≈0.35, 16.3% sparse) through AdamW's adaptive
      denominator. A was 97.9% a retrieval matrix, 2.1% a representation matrix.
      A_in and A_out can now specialize independently.

    Warm-start:
      out_to_k.weight ← B.weight.T at init.
      Gives 100% subspace alignment vs 0.5% random.
      Eliminates the ~1000-step subspace-rotation tax visible in E3 κ trajectory.
    """
    def __init__(self, unigram_logprob=None):
        super().__init__()
        self.embed      = FactorizedEmbeddingE4(VOCAB_SIZE, D_MODEL, K_IN, K_OUT)
        self.rope       = RotaryEmbedding(D_MODEL // N_HEADS, N_POSITIONS, ROPE_BASE)
        self.in_dropout = nn.Dropout(DROPOUT)
        self.blocks     = nn.ModuleList([
            TransformerBlock(D_MODEL, N_HEADS, D_FF_GATE, DROPOUT, N_LAYERS, self.rope)
            for _ in range(N_LAYERS)
        ])
        self.norm_final = RMSNorm(D_MODEL)

        # MoS: n independent context projections U_i : D -> k_out, shared A_out
        # and shared out_bias. When MOS_COMPONENTS == 1 this collapses to the
        # v21 single-softmax head exactly.
        self.n_mos    = max(1, int(MOS_COMPONENTS))
        self.out_to_k_list = nn.ModuleList([
            nn.Linear(D_MODEL, K_OUT, bias=False) for _ in range(self.n_mos)
        ])
        # Context-dependent mixture gate: h -> R^n  (only meaningful for n > 1).
        # Weight is zero-initialized so the gate starts context-independent;
        # the bias (v23) seeds an asymmetric mixture so component 0 carries
        # the early prediction budget and component 1 can specialize slowly
        # under low mixture weight.
        if self.n_mos > 1:
            self.mos_gate = nn.Linear(D_MODEL, self.n_mos, bias=True)
            nn.init.zeros_(self.mos_gate.weight)
            with torch.no_grad():
                gate_bias_init = torch.zeros(self.n_mos)
                # softmax([skew, 0, 0, ...]) heavily favors component 0:
                #   n=2, skew=1.0 -> pi = [0.731, 0.269]
                gate_bias_init[0] = float(MOS_GATE_BIAS_INIT_SKEW)
                self.mos_gate.bias.copy_(gate_bias_init)

        # Projection init.
        #   v22: all components warm-started from B^T with tiny perturbation
        #        -> both components collapsed to the same local minimum,
        #        gate_H stayed at ln(2) for 3600 steps, effective rank = 65.
        #   v23: component 0 keeps the B^T warm-start (pays subspace-rotation
        #        tax once, fast early CE). Components 1..n-1 use default
        #        Kaiming init, placing them substantially far from B^T in
        #        weight space so they descend to a different solution.
        with torch.no_grad():
            base = self.embed.B.weight.T[:K_OUT, :].clone()
            for i, layer in enumerate(self.out_to_k_list):
                if i == 0 or not MOS_ASYMMETRIC_INIT:
                    layer.weight.copy_(base)
                    if MOS_ASYMMETRIC_INIT is False and i > 0:
                        # v22-compatibility branch (disabled by default)
                        noise = torch.randn_like(layer.weight) * 0.02
                        layer.weight.add_(noise)
                # else: leave layer.weight at its default nn.Linear init
                #       (Kaiming uniform, std ~1/sqrt(D)).

        # v25: Multi-Token Prediction auxiliary head.
        # A second projection U_mtp : D -> k_out whose CE against token t+MTP_DEPTH
        # supplies a *second* rank-k_out gradient subspace to the body. At inference
        # it is never called, so it is training-only capacity. Warm-started from
        # B^T (same rationale as the main head — skips the subspace-rotation tax).
        if MTP_ENABLED:
            self.out_to_k_mtp = nn.Linear(D_MODEL, K_OUT, bias=False)
            with torch.no_grad():
                self.out_to_k_mtp.weight.copy_(self.embed.B.weight.T[:K_OUT, :])

        # Softmax-bottleneck bias slot: fills the rank-1 unigram channel that
        # Yang et al. (2017) identify as the missing +1 in k+1 factored rank.
        # Stored as a buffer of zeros when disabled so the forward path is
        # branch-free.
        if OUT_BIAS_ENABLED:
            self.out_bias = nn.Parameter(torch.zeros(VOCAB_SIZE))
            if unigram_logprob is not None and OUT_BIAS_INIT_UNIGRAM:
                with torch.no_grad():
                    self.out_bias.copy_(unigram_logprob.detach())
                    # Center so the learned bias drifts from a zero-mean prior;
                    # the constant offset has no effect on softmax gradients.
                    self.out_bias.sub_(self.out_bias.mean())
        else:
            self.register_buffer('out_bias', torch.zeros(VOCAB_SIZE), persistent=False)

        # Bigram shortcut scale alpha: 0-d learnable scalar. With alpha=0 at
        # init, v21 step 0 is identical to v20 step 0. AdamW will only push
        # alpha away from 0 if the bigram signal actually reduces loss.
        if SHORTCUT_ENABLED:
            assert K_IN == K_OUT, 'bigram shortcut requires k_in == k_out (dot product A_in[x_t]·A_out[v])'
            self.shortcut_scale = nn.Parameter(torch.tensor(float(SHORTCUT_INIT)))
        else:
            self.register_buffer('shortcut_scale', torch.tensor(0.0), persistent=False)

    def forward_features(self, ids):
        h = self.in_dropout(self.embed.embed(ids))
        for blk in self.blocks:
            h = blk(h)
        return self.norm_final(h)
    def _component_logits(self, h, proj, head_mode, kappa_cap, apply_nonlin=False):
        """Full [B, T, V] logits for a single MoS component, including bias.
        If apply_nonlin, z = SiLU(proj(h)) — used for components 1..n-1 so the
        mixture has a linear and a nonlinear head that cannot share a
        solution under gradient descent."""
        z = proj(h).float()
        if apply_nonlin:
            z = F.silu(z)
        logits = self.embed.decode_scores(z, kappa_cap=kappa_cap, mode=head_mode)
        logits = logits + self.out_bias.float()
        return logits, z

    def forward(self, ids, labels=None, kappa_cap=None, class_weight=None,
                head_mode=HEAD_MODE):
        h = self.forward_features(ids)                      # [B, T, D]

        # Pre-compute the bigram shortcut once (it's shared across MoS components).
        if SHORTCUT_ENABLED:
            a_in_t        = self.embed.A_in(ids).float()          # [B, T, k_in]
            a_out_mat     = self.embed.A_out.weight.float()       # [V, k_out]
            bigram_logits = a_in_t @ a_out_mat.T                  # [B, T, V]
            bigram_scaled = self.shortcut_scale.float() * bigram_logits
        else:
            bigram_scaled = None

        # Single-component fast path (MOS_COMPONENTS == 1): exact v21 behavior.
        if self.n_mos == 1:
            logits, z = self._component_logits(h, self.out_to_k_list[0], head_mode, kappa_cap)
            if bigram_scaled is not None:
                logits = logits + bigram_scaled
            loss = None
            mtp_loss_val = torch.zeros((), device=h.device)
            if labels is not None:
                loss = F.cross_entropy(
                    logits.view(-1, VOCAB_SIZE),
                    labels.view(-1),
                    weight=class_weight,
                )

                # v25: Multi-token auxiliary loss. Training only; we only
                # compute it when (a) it's enabled, (b) we're in training mode
                # (saves the memory during eval), and (c) there's at least one
                # position with a valid t+MTP_DEPTH target.
                if (MTP_ENABLED and self.training and
                        labels.shape[1] > MTP_DEPTH):
                    # Ignore positions t where t + MTP_DEPTH is out of range.
                    # labels is already shifted by 1 from ids, so labels[t] = y_{t+1}.
                    # The MTP target for position t is y_{t+MTP_DEPTH} = labels[t + (MTP_DEPTH - 1)].
                    shift = MTP_DEPTH - 1
                    T_mtp = labels.shape[1] - shift
                    mtp_labels = labels[:, shift:].contiguous()                 # [B, T_mtp]
                    h_mtp      = h[:, :T_mtp, :]                                # [B, T_mtp, D]

                    z_mtp = self.out_to_k_mtp(h_mtp).float()
                    mtp_logits = self.embed.decode_scores(
                        z_mtp, kappa_cap=kappa_cap, mode=head_mode
                    )
                    mtp_logits = mtp_logits + self.out_bias.float()
                    # We DO NOT add the bigram shortcut to the MTP head: the
                    # shortcut models P(y_{t+1} | x_t), not P(y_{t+MTP_DEPTH} | x_t),
                    # so adding it would inject the wrong prior.
                    mtp_loss_val = F.cross_entropy(
                        mtp_logits.view(-1, VOCAB_SIZE),
                        mtp_labels.view(-1),
                    )
                    loss = loss + MTP_WEIGHT * mtp_loss_val

            kappa = z.norm(dim=-1, keepdim=True)
            zero_div = torch.zeros((), device=h.device)
            return SimpleNamespace(loss=loss, z=z, kappa=kappa, logits=logits,
                                   mos_div=zero_div, mtp_loss=mtp_loss_val.detach())

        # ── MoS path (n >= 2) ────────────────────────────────────────────────
        # Gate: log pi_i(c) = log_softmax(W_g h)_i   shape [B, T, n]
        log_pi = F.log_softmax(self.mos_gate(h).float(), dim=-1)

        # Eval callers (evaluate/evaluate_ppl_sliding) pass labels and read
        # out.logits (argmax for recall@1). Route them through the full
        # log-prob path; during training (self.training True) we use the
        # memory-efficient streaming label-only path.
        want_full_logits = (labels is None) or (not self.training)

        if want_full_logits:
            # Eval / inference path: materialize full mixture log-probs.
            per_component_logp = []
            z_first = None
            for i, proj in enumerate(self.out_to_k_list):
                nonlin_i = bool(MOS_NONLIN_ON_EXTRA_COMPONENTS) and (i > 0)
                logits_i, z_i = self._component_logits(
                    h, proj, head_mode, kappa_cap, apply_nonlin=nonlin_i
                )
                if bigram_scaled is not None:
                    logits_i = logits_i + bigram_scaled
                if i == 0:
                    z_first = z_i
                log_p_i = F.log_softmax(logits_i, dim=-1)        # [B, T, V]
                per_component_logp.append(log_pi[..., i:i+1] + log_p_i)
                del logits_i, log_p_i
            log_p = torch.logsumexp(torch.stack(per_component_logp, dim=0), dim=0)
            kappa = z_first.norm(dim=-1, keepdim=True)
            loss = None
            if labels is not None:
                # NLL on the mixture log-probs; class_weight not supported in
                # MoS eval (we never use it at eval anyway).
                loss = F.nll_loss(log_p.view(-1, VOCAB_SIZE), labels.view(-1))
            zero_div = torch.zeros((), device=h.device)
            zero_mtp = torch.zeros((), device=h.device)
            return SimpleNamespace(loss=loss, z=z_first, kappa=kappa, logits=log_p,
                                   mos_div=zero_div, mtp_loss=zero_mtp)

        # Training path: we need log p(y_t | c_t) per (t,y_t), not the full [V]
        # dimension. Compute logZ_i (scalar per (B,T)) and the label logit per
        # component; combine via logsumexp with gate log-weights. Memory stays
        # at one full [B, T, V] tensor at a time, freed between components.
        labels_flat = labels.unsqueeze(-1)                         # [B, T, 1]
        comp_label_logp = []                                       # list of [B, T]
        z_list          = []                                       # list of [B, T, k]
        for i, proj in enumerate(self.out_to_k_list):
            nonlin_i = bool(MOS_NONLIN_ON_EXTRA_COMPONENTS) and (i > 0)
            logits_i, z_i = self._component_logits(
                h, proj, head_mode, kappa_cap, apply_nonlin=nonlin_i
            )
            if bigram_scaled is not None:
                logits_i = logits_i + bigram_scaled
            z_list.append(z_i)
            logZ_i       = torch.logsumexp(logits_i, dim=-1)             # [B, T]
            label_logit  = logits_i.gather(-1, labels_flat).squeeze(-1)  # [B, T]
            log_p_i_y    = label_logit - logZ_i                          # [B, T]
            comp_label_logp.append(log_pi[..., i] + log_p_i_y)
            del logits_i                                                 # free [B,T,V]
        # logsumexp over the mixture dimension
        log_p_y = torch.logsumexp(torch.stack(comp_label_logp, dim=0), dim=0)  # [B, T]
        z_first = z_list[0]

        # class_weight path is rarely used (off by default); apply per-label weight
        # to the NLL if supplied.
        if class_weight is not None:
            w = class_weight[labels]                                     # [B, T]
            loss = -(w * log_p_y).sum() / w.sum().clamp_min(1e-8)
        else:
            loss = -log_p_y.mean()

        # v23 diversity penalty: average |cos(z_i, z_j)| across component pairs.
        # Zero when z vectors are pairwise orthogonal (components producing
        # maximally different predictions). Kept small by MOS_DIVERSITY_LAMBDA.
        div_val = torch.zeros((), device=h.device, dtype=loss.dtype)
        if self.n_mos > 1 and MOS_DIVERSITY_LAMBDA > 0.0:
            pair_cos = []
            for i in range(self.n_mos):
                for j in range(i + 1, self.n_mos):
                    c_ij = F.cosine_similarity(z_list[i], z_list[j], dim=-1).abs().mean()
                    pair_cos.append(c_ij)
            div_val = torch.stack(pair_cos).mean()
            loss = loss + MOS_DIVERSITY_LAMBDA * div_val

        kappa = z_first.norm(dim=-1, keepdim=True)
        # For eval-time 'logits' compatibility (recall@1 etc.), we return None
        # here in training; the training loop never consumes out.logits.
        zero_mtp = torch.zeros((), device=h.device)
        return SimpleNamespace(loss=loss, z=z_first, kappa=kappa, logits=None,
                               mos_div=div_val.detach(), mtp_loss=zero_mtp)

    def count_params(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# ── E4 Diagnostic: cosine similarity between A_in and A_out ──────────────────
@torch.no_grad()
def log_embedding_alignment(model, step, epoch):
    """
    Core E4 falsification diagnostic.

    When k_in == k_out, we compute per-token cosine directly.
    When k_in != k_out, we compute cosine on the shared subspace:
      A_in[:, :k_shared] vs A_out[:, :k_shared], k_shared=min(k_in, k_out).

    This keeps the diagnostic shape-safe while preserving a comparable trend.
    Also logs ‖A_in‖ vs ‖A_out‖ to monitor norm drift.
    """
    A_in  = model.embed.A_in.weight.float()
    A_out = model.embed.A_out.weight.float()

    k_in, k_out = A_in.size(1), A_out.size(1)
    k_shared = min(k_in, k_out)

    A_in_cmp = A_in[:, :k_shared]
    A_out_cmp = A_out[:, :k_shared]

    A_in_n  = F.normalize(A_in_cmp,  dim=-1)
    A_out_n = F.normalize(A_out_cmp, dim=-1)

    # Per-token cosine similarity in shared subspace.
    cos_sim = (A_in_n * A_out_n).sum(dim=-1)   # [V]

    # norms
    norm_in  = A_in.norm(dim=-1)               # [V]
    norm_out = A_out.norm(dim=-1)              # [V]

    print(
        f'[E4 diag] ep{epoch} step{step:>5} | '
        f'cos_shared(A_in,A_out): mean={cos_sim.mean():.4f} std={cos_sim.std():.4f} '
        f'min={cos_sim.min():.4f} max={cos_sim.max():.4f} '
        f'(k_shared={k_shared}, k_in={k_in}, k_out={k_out}) | '
        f'||A_in||: mean={norm_in.mean():.3f} | '
        f'||A_out||: mean={norm_out.mean():.3f}'
    )
    return cos_sim.mean().item()


# ── Head-regularization helpers (kappa cap schedule, freq-balanced CE, grads) ─
def kappa_cap_at_step(global_step: int, total_opt_steps: int):
    """Linear ramp of the ||z|| ceiling across the first KAPPA_CAP_FRAC of
    optimizer steps. Returns None once the warm-cap window is over or when
    the cap is disabled."""
    if not KAPPA_CAP_ENABLED:
        return None
    window = max(1, int(total_opt_steps * KAPPA_CAP_FRAC))
    if global_step >= window:
        return None
    frac = global_step / float(window)
    return float(KAPPA_CAP_INIT + frac * (KAPPA_CAP_FINAL - KAPPA_CAP_INIT))


@torch.no_grad()
def build_unigram_logprob(train_tokens, vocab_size=VOCAB_SIZE, eps=OUT_BIAS_INIT_EPS):
    """Empirical log P(w) over the training corpus with Laplace smoothing.
    Returns a tensor of shape [V] suitable for initializing the output bias.

    H(unigram) = -sum_w p(w) * log p(w) is the CE floor this bias buys us:
    a model that outputs softmax(b) with b = log p(w) alone already achieves
    this CE at step 0, before a single gradient update.
    """
    counts = torch.bincount(train_tokens.to(torch.long), minlength=vocab_size).float()
    p = (counts + eps) / (counts.sum() + eps * vocab_size)
    logp = p.log()
    H = -(p * logp).sum().item()
    present = int((counts > 0).sum().item())
    print(
        f'[unigram] H(unigram)={H:.3f} nats  '
        f'(uniform baseline ln(V)={math.log(vocab_size):.3f}, '
        f'headroom={math.log(vocab_size) - H:.3f} nats) | '
        f'vocab_present={present}/{vocab_size} | '
        f'logp range=[{logp.min().item():.2f}, {logp.max().item():.2f}]'
    )
    return logp


@torch.no_grad()
def build_freq_class_weights(train_tokens, vocab_size=VOCAB_SIZE):
    """Inverse-frequency class weights for CE, normalized to mean 1, clamped.
    Rare tokens get larger weight so the output head must place them carefully
    instead of being driven entirely by common-token gradients."""
    if not FREQ_BAL_ENABLED or FREQ_ALPHA <= 0.0:
        return None
    counts = torch.bincount(train_tokens.to(torch.long), minlength=vocab_size).float()
    raw    = (counts + FREQ_EPS).pow(-FREQ_ALPHA)
    # normalize so mean weight = 1 over the observed vocabulary
    w = raw / raw.mean().clamp_min(1e-12)
    w = w.clamp(max=FREQ_CLAMP_MAX)
    present = int((counts > 0).sum().item())
    print(
        f'[freq-bal] vocab_present={present}/{vocab_size} '
        f'weights: min={w.min().item():.3f} mean={w.mean().item():.3f} '
        f'max={w.max().item():.3f} p95={w.quantile(0.95).item():.3f}'
    )
    return w


@torch.no_grad()
def log_grad_norms(model, step, epoch):
    """Per-component gradient L2 norms, taken just before optimizer.step().
    Tells us whether A_in is actually being trained vs coasting on warm-start,
    and whether any path is dominating the update."""
    def gnorm(p):
        return float(p.grad.detach().float().norm().item()) if (p.grad is not None) else 0.0

    e      = model.embed
    g_ain  = gnorm(e.A_in.weight)
    g_aout = gnorm(e.A_out.weight)
    g_B    = gnorm(e.B.weight)
    # Aggregate grad norm across all MoS projection components (sqrt of sum of squares).
    g_ok_sq = 0.0
    for layer in model.out_to_k_list:
        if layer.weight.grad is not None:
            g_ok_sq += float(layer.weight.grad.detach().float().pow(2).sum().item())
    g_ok = g_ok_sq ** 0.5
    g_gate = gnorm(model.mos_gate.weight) if hasattr(model, 'mos_gate') else 0.0
    g_mtp  = gnorm(model.out_to_k_mtp.weight) if hasattr(model, 'out_to_k_mtp') else 0.0
    body_sq = 0.0
    for n, p in model.named_parameters():
        if not p.requires_grad or p.grad is None:
            continue
        if n.startswith('blocks.') or n.startswith('norm_final') or n.startswith('rope'):
            body_sq += float(p.grad.detach().float().pow(2).sum().item())
    g_body = body_sq ** 0.5
    print(
        f'[grad] ep{epoch} step{step:>5} | '
        f'A_in={g_ain:.4f} A_out={g_aout:.4f} B={g_B:.4f} '
        f'out_to_k={g_ok:.4f} mtp={g_mtp:.4f} gate={g_gate:.4f} body={g_body:.4f}'
    )


@torch.no_grad()
def log_sparse_muon_counts(optimizer, step, epoch):
    """v31: log the per-row update-count distribution for A_in / A_out.

    This is the direct diagnostic for the Zipfian-rotation-budget hypothesis.
    If the theory is right:
      - p50 count rises much slower than p99 count (long tail of rare tokens
        that almost never get updates).
      - p99 - p50 gap widens over training.
      - n_touched grows but saturates well below V (many tokens never seen).
    The multiplier range printed ("mul[min..p50..max]") shows what per-row
    LR scaling is active. If min=floor and max=1.0 the system is in the
    Adagrad regime; if max < 1.0 nothing is rare anymore and the policy is
    already saturated.
    """
    if not isinstance(optimizer, _OptCombo):
        return
    smuon = None
    for o, n in zip(optimizer.optimizers, optimizer.names):
        if n == 'sparse_muon':
            smuon = o
            break
    if smuon is None:
        return
    parts = []
    for group in smuon.param_groups:
        nm = group.get('name', '?')
        power = group['per_row_lr_power']
        scale = group['per_row_lr_scale']
        pr_min = group['per_row_lr_min']
        pr_max = group['per_row_lr_max']
        for p in group['params']:
            state = smuon.state.get(p, {})
            counts = state.get('update_count', None)
            if counts is None:
                continue
            c = counts.float()
            total_touched = int((c > 0).sum().item())
            c_nz = c[c > 0]
            if c_nz.numel() == 0:
                parts.append(f'{nm}[none touched]')
                continue
            p50 = float(c_nz.quantile(0.50).item())
            p90 = float(c_nz.quantile(0.90).item())
            p99 = float(c_nz.quantile(0.99).item())
            cmin = float(c_nz.min().item())
            cmax = float(c_nz.max().item())
            # Multiplier distribution (including untouched rows at count=0).
            mul = (scale / (1.0 + c).pow(power)).clamp(min=pr_min, max=pr_max)
            mul_min = float(mul.min().item())
            mul_p50 = float(mul.median().item())
            mul_max = float(mul.max().item())
            parts.append(
                f'{nm}: V_touched={total_touched}/{counts.numel()} '
                f'count[min={cmin:.0f} p50={p50:.0f} p90={p90:.0f} '
                f'p99={p99:.0f} max={cmax:.0f}] '
                f'mul[min={mul_min:.3f} p50={mul_p50:.3f} max={mul_max:.3f}]'
            )
    if parts:
        print(
            f'[smuon-count] ep{epoch} step{step:>5} | ' + ' || '.join(parts)
        )


# ── Data loading (unchanged) ──────────────────────────────────────────────────
def _copy_if_needed(src, dst):
    if not src.exists():
        return False
    dst.parent.mkdir(parents=True, exist_ok=True)
    if not dst.exists():
        print(f'Copying cache: {src} -> {dst}')
        shutil.copy2(src, dst)
    return True


def _tokenize_split(tokenizer, texts, split_name, batch_size=TOKENIZE_BATCH_SIZE):
    texts   = [t for t in texts if t and t.strip()]
    all_ids = []
    print(f'[{split_name}] tokenizing {len(texts):,} non-empty documents')
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        enc   = tokenizer(batch, add_special_tokens=False, truncation=False)
        for ids in enc['input_ids']:
            if ids:
                all_ids.extend(ids)
                all_ids.append(tokenizer.eos_token_id)
        if (i // batch_size) % 20 == 0:
            print(f'[{split_name}] {min(i + batch_size, len(texts)):,}/{len(texts):,} docs')
    tokens = torch.tensor(all_ids, dtype=torch.long)
    print(f'[{split_name}] tokens={tokens.numel():,}')
    return tokens


def load_wikitext103():
    tokenizer = GPT2TokenizerFast.from_pretrained(TOKENIZER_NAME)
    tokenizer.pad_token = tokenizer.eos_token
    LOCAL_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    drive_files = {s: DRIVE_CACHE_DIR / f'{s}_tokens.pt' for s in ['train', 'val', 'test']}
    local_files = {s: LOCAL_CACHE_DIR  / f'{s}_tokens.pt' for s in ['train', 'val', 'test']}
    for split in ['train', 'val', 'test']:
        _copy_if_needed(drive_files[split], local_files[split])
    if all(local_files[s].exists() for s in ['train', 'val', 'test']):
        print('Loading cached token tensors from local cache...')
        train_tokens = torch.load(local_files['train'], map_location='cpu')
        val_tokens   = torch.load(local_files['val'],   map_location='cpu')
        test_tokens  = torch.load(local_files['test'],  map_location='cpu')
        print(f'train: {train_tokens.numel():,} | val: {val_tokens.numel():,} | test: {test_tokens.numel():,}')
        return tokenizer, train_tokens, val_tokens, test_tokens
    print('Cache missing — tokenizing WikiText-103...')
    raw          = load_dataset('wikitext', 'wikitext-103-raw-v1')
    train_tokens = _tokenize_split(tokenizer, raw['train']['text'],      'train')
    val_tokens   = _tokenize_split(tokenizer, raw['validation']['text'], 'val')
    test_tokens  = _tokenize_split(tokenizer, raw['test']['text'],       'test')
    for s, t in [('train', train_tokens), ('val', val_tokens), ('test', test_tokens)]:
        torch.save(t, local_files[s])
    try:
        DRIVE_CACHE_DIR.mkdir(parents=True, exist_ok=True)
        for s in ['train', 'val', 'test']:
            shutil.copy2(local_files[s], drive_files[s])
    except Exception as e:
        print(f'Warning: could not copy to Drive: {e}')
    return tokenizer, train_tokens, val_tokens, test_tokens


def build_loaders(train_tokens, val_tokens):
    train_loader = DataLoader(
        TokenDataset(train_tokens, SEQ_LEN),
        batch_size=BATCH_SIZE, shuffle=True,
        num_workers=NUM_WORKERS, pin_memory=True, drop_last=True,
    )
    val_loader = DataLoader(
        TokenDataset(val_tokens, SEQ_LEN),
        batch_size=BATCH_SIZE, shuffle=False,
        num_workers=NUM_WORKERS, pin_memory=True, drop_last=False,
    )
    return train_loader, val_loader


# ── Evaluation ────────────────────────────────────────────────────────────────
def evaluate(model, loader, max_batches=None, split_name='val'):
    model.eval()
    correct, total, loss_sum, steps = 0, 0, 0.0, 0
    kappa_mean_sum, kappa_min_sum, kappa_max_sum = 0.0, 0.0, 0.0
    with torch.no_grad():
        for step, batch in enumerate(loader):
            if max_batches is not None and step >= max_batches:
                break
            x, y  = [t.to(device, non_blocking=True) for t in batch]
            dtype = torch.bfloat16 if USE_BF16 else torch.float32
            with torch.amp.autocast('cuda', dtype=dtype, enabled=USE_BF16):
                out = model(x, labels=y)
            preds          = out.logits.argmax(dim=-1)
            correct       += (preds == y).sum().item()
            total         += y.numel()
            loss_sum      += out.loss.item()
            kappa_mean_sum += out.kappa.mean().item()
            kappa_min_sum  += out.kappa.min().item()
            kappa_max_sum  += out.kappa.max().item()
            steps += 1
    recall1  = correct   / max(total, 1)
    avg_loss = loss_sum  / max(steps, 1)
    ppl      = math.exp(min(avg_loss, 20))
    k_mean   = kappa_mean_sum / max(steps, 1)
    k_min    = kappa_min_sum  / max(steps, 1)
    k_max    = kappa_max_sum  / max(steps, 1)
    print(
        f'{split_name} recall@1={recall1:.6f} | ce={avg_loss:.6f} | ppl={ppl:.2f} | '
        f'κ mean={k_mean:.3f} min={k_min:.3f} max={k_max:.3f}'
    )
    return {'recall@1': recall1, 'ce_loss': avg_loss, 'ppl': ppl,
            'kappa_mean': k_mean, 'kappa_min': k_min, 'kappa_max': k_max}


def evaluate_ppl_sliding(model, tokens, stride=EVAL_STRIDE, seq_len=SEQ_LEN):
    model.eval()
    n = tokens.numel()
    nll_sum, ntok, prev_end = 0.0, 0, 0
    with torch.no_grad():
        for s in range(0, n - 1, stride):
            e  = min(s + seq_len, n - 1)
            x  = tokens[s:e].unsqueeze(0).to(device)
            y  = tokens[s + 1:e + 1].unsqueeze(0).to(device)
            pred_start = max(prev_end - s, 0) if s > 0 else 0
            if pred_start >= x.size(1):
                prev_end = e
                continue
            y_in = y.clone()
            if pred_start > 0:
                y_in[:, :pred_start] = -100
            dtype = torch.bfloat16 if USE_BF16 else torch.float32
            with torch.amp.autocast('cuda', dtype=dtype, enabled=USE_BF16):
                out = model(x, labels=y_in)
            valid     = (y_in != -100).sum().item()
            nll_sum  += out.loss.item() * valid
            ntok     += valid
            prev_end  = e
            if e >= n - 1:
                break
    return math.exp(min(nll_sum / max(ntok, 1), 20))


# ── Optimizer & scheduler ─────────────────────────────────────────────────────
#
# v27: Muon (Keller Jordan, 2024) for body matrix params + AdamW for the rest.
#
# Newton-Schulz 5-step quintic iteration computes an approximation of the
# "zeroth power" of G, i.e. an orthogonal factor U V^T from the SVD G = U S V^T.
# This transforms the raw momentum-accumulated gradient into an orthogonalized
# update that hits all singular directions of the weight matrix with roughly
# equal step size, rather than concentrating in the top-k directions the way
# AdamW's per-coordinate preconditioning does.
#
# The coefficients (a, b, c) = (3.4445, -4.7750, 2.0315) are the quintic NS
# coefficients Jordan tuned to converge to the orthogonal factor in 5 steps
# on bf16 inputs without spectral overshoot.
class WeightEMA:
    """
    Exponential moving average of model weights. EMA tracks the time-average
    of the weight trajectory, which for SGD near a quadratic minimum equals
    the minimum (oscillations cancel). The gap ||w - w_ema|| is a direct
    readout of the noise-floor the training is operating against.

    Kept in fp32 regardless of model dtype so the tiny per-step update doesn't
    lose precision at long horizons.
    """
    def __init__(self, model, decay=0.999):
        self.decay  = decay
        self.shadow = {}
        self.backup = {}
        for name, p in model.named_parameters():
            if p.requires_grad:
                self.shadow[name] = p.detach().float().clone()

    @torch.no_grad()
    def update(self, model):
        d = self.decay
        for name, p in model.named_parameters():
            if p.requires_grad and name in self.shadow:
                self.shadow[name].mul_(d).add_(p.detach().float(), alpha=1.0 - d)

    @torch.no_grad()
    def gap_stats(self, model):
        """Returns (global relative gap, matrix-only mean relative gap).

        v30 used per-tensor relative gap averaged uniformly; that metric was
        dominated by scalars (e.g. shortcut_scale) whose EMA norm is near
        zero, giving meaningless 100x relative gaps that swamped the signal
        from the actual weight matrices. v31 reports two cleaner numbers:

          global = ||concat(w - w_ema)|| / ||concat(w_ema)||
                   (one number across ALL parameters, weighted by size —
                    the 2D matrices dominate and the scalar jitter washes
                    out. Direct noise-floor readout.)

          mat_mean = mean over matrix (ndim>=2) tensors of relative gap
                   (scale-free per-tensor view, but restricted to the
                    weight matrices where the learning actually lives;
                    ignores bias vectors and RMSNorm scalars.)
        """
        diff_sq_sum  = 0.0
        base_sq_sum  = 0.0
        mat_gaps     = []
        for name, p in model.named_parameters():
            if p.requires_grad and name in self.shadow:
                w     = p.detach().float()
                w_ema = self.shadow[name]
                d2 = (w - w_ema).pow(2).sum().item()
                b2 = w_ema.pow(2).sum().item()
                diff_sq_sum += d2
                base_sq_sum += b2
                if p.ndim >= 2:
                    mat_gaps.append(math.sqrt(d2) / (math.sqrt(b2) + 1e-12))
        global_gap = math.sqrt(diff_sq_sum) / (math.sqrt(base_sq_sum) + 1e-12)
        mat_mean   = sum(mat_gaps) / len(mat_gaps) if mat_gaps else 0.0
        return global_gap, mat_mean

    @torch.no_grad()
    def swap_to_ema(self, model):
        """Swap model weights with EMA weights, saving originals for swap-back."""
        self.backup = {}
        for name, p in model.named_parameters():
            if p.requires_grad and name in self.shadow:
                self.backup[name] = p.detach().clone()
                p.data.copy_(self.shadow[name].to(p.dtype))

    @torch.no_grad()
    def swap_back(self, model):
        for name, p in model.named_parameters():
            if name in self.backup:
                p.data.copy_(self.backup[name])
        self.backup = {}


def zeropower_via_newtonschulz5(G, steps=5, eps=1e-7):
    assert G.ndim == 2
    a, b, c = (3.4445, -4.7750, 2.0315)
    X = G.to(torch.bfloat16)
    transpose = G.size(0) > G.size(1)
    if transpose:
        X = X.T
    # Normalize so spectral norm <= 1 (precondition for NS convergence).
    X = X / (X.norm() + eps)
    for _ in range(steps):
        A = X @ X.T
        B = b * A + c * (A @ A)
        X = a * X + B @ X
    if transpose:
        X = X.T
    return X.to(G.dtype)


class Muon(torch.optim.Optimizer):
    """
    Muon: Momentum Orthogonalized via Newton-Schulz.
    Updates each 2D matrix parameter by orthogonalizing the momentum buffer
    before subtracting it. Equivalent to steepest descent on the matrix
    manifold under the spectral-norm-bounded trust region.
    Expects params to all be 2D (matrix) tensors with dense gradients.
    """
    def __init__(self, params, lr=0.02, momentum=0.95, nesterov=True, ns_steps=5):
        defaults = dict(lr=lr, momentum=momentum, nesterov=nesterov, ns_steps=ns_steps)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        for group in self.param_groups:
            lr       = group['lr']
            mu       = group['momentum']
            nesterov = group['nesterov']
            ns_steps = group['ns_steps']
            for p in group['params']:
                if p.grad is None:
                    continue
                g = p.grad
                assert g.ndim == 2, 'Muon only handles 2D matrix params'
                state = self.state[p]
                if 'momentum_buffer' not in state:
                    state['momentum_buffer'] = torch.zeros_like(g)
                buf = state['momentum_buffer']
                buf.mul_(mu).add_(g)
                update = g.add(buf, alpha=mu) if nesterov else buf
                update = zeropower_via_newtonschulz5(update, steps=ns_steps)
                # Jordan's rectangular scaling: sqrt(max(1, m/n)) so tall matrices
                # get a slightly larger step to match the effective norm of square ones.
                scale = max(1.0, update.size(-2) / update.size(-1)) ** 0.5
                p.add_(update, alpha=-lr * scale)
        return loss


class SparseMuon(torch.optim.Optimizer):
    """
    Muon variant for embedding-table-style parameters with sparse gradients.
    For each step:
      1. Detect which rows of the parameter received non-zero gradient.
      2. Stack those rows into an (n_active, k) dense sub-matrix.
      3. Apply momentum to a per-row buffer (only active rows accumulate).
      4. Orthogonalize the active sub-buffer via Newton-Schulz.
      5. Subtract the orthogonalized update from those rows. Untouched
         rows of the parameter are not modified.
    The momentum buffer is full (V, k) so an inactive row's old momentum
    decays naturally over steps where it was inactive. When it next fires
    its buffer state still reflects (decayed) prior gradient direction.
    """
    def __init__(self, params, lr=0.005, momentum=0.95, ns_steps=5,
                 tangent_only=False, eps=1e-10,
                 per_row_lr=False, per_row_lr_power=0.5,
                 per_row_lr_scale=1.0, per_row_lr_min=0.02,
                 per_row_lr_max=1.0):
        defaults = dict(lr=lr, momentum=momentum, ns_steps=ns_steps,
                        tangent_only=tangent_only, eps=eps,
                        per_row_lr=per_row_lr,
                        per_row_lr_power=per_row_lr_power,
                        per_row_lr_scale=per_row_lr_scale,
                        per_row_lr_min=per_row_lr_min,
                        per_row_lr_max=per_row_lr_max)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        for group in self.param_groups:
            lr              = group['lr']
            mu              = group['momentum']
            ns_steps        = group['ns_steps']
            tangent_only    = group['tangent_only']
            eps             = group['eps']
            per_row_lr      = group['per_row_lr']
            pr_power        = group['per_row_lr_power']
            pr_scale        = group['per_row_lr_scale']
            pr_min          = group['per_row_lr_min']
            pr_max          = group['per_row_lr_max']
            for p in group['params']:
                if p.grad is None:
                    continue
                g = p.grad
                assert g.ndim == 2, 'SparseMuon expects (V, k) embedding-style params'
                state = self.state[p]
                if 'momentum_buffer' not in state:
                    state['momentum_buffer'] = torch.zeros_like(p)
                if 'update_count' not in state:
                    # int32 is plenty (max ~2B updates) and half the memory of int64.
                    state['update_count'] = torch.zeros(
                        p.size(0), dtype=torch.int32, device=p.device)
                buf = state['momentum_buffer']
                # Decay all rows' momentum, then add this step's gradient.
                # Inactive rows simply add 0; their buffer keeps decaying.
                buf.mul_(mu).add_(g)
                # Determine active rows from the *gradient* this step
                # (not the buffer — buffer can be non-zero even when row is
                # inactive this step due to prior momentum).
                row_grad_norm = g.norm(dim=-1)
                active_mask = row_grad_norm > eps
                n_active = int(active_mask.sum().item())
                if n_active == 0:
                    continue
                active_idx = torch.where(active_mask)[0]
                # Increment per-row occurrence count BEFORE computing LR scale so
                # the first time a row fires it sees count=1 (not count=0).
                counts = state['update_count']
                counts.index_add_(0, active_idx,
                                  torch.ones_like(active_idx, dtype=counts.dtype))
                # Pull the active sub-matrix of the buffer and orthogonalize.
                buf_active = buf.index_select(0, active_idx)  # (n_active, k)
                update = zeropower_via_newtonschulz5(buf_active, steps=ns_steps)
                scale = max(1.0, update.size(-2) / update.size(-1)) ** 0.5
                if tangent_only:
                    # Project update onto the tangent space of the current row.
                    # For a parameter whose "length" is invariant under the loss
                    # (e.g. A_out used via cos(z, A_out[v])), the radial component
                    # of the update is invisible to the loss and wastes the
                    # per-step update budget by inflating ||A_out|| — which in
                    # turn shrinks angular change per step (∝ 1 / ||A_out||).
                    # Strip the radial component so 100% of the step goes into
                    # rotation and norms stay approximately fixed.
                    p_active = p.index_select(0, active_idx)              # (n_active, k)
                    row_norm_sq = (p_active * p_active).sum(dim=-1, keepdim=True).clamp_min(eps)
                    radial_coeff = (update * p_active).sum(dim=-1, keepdim=True) / row_norm_sq
                    update = update - radial_coeff * p_active              # perpendicular to p_active
                if per_row_lr:
                    # Adagrad-over-occurrence-count: each row's per-update step
                    # magnitude is attenuated by (1 + count)^(-power). Rare tokens
                    # keep near-unit multiplier and accumulate real angular
                    # distance per rare occurrence; frequent tokens' self-rotation
                    # is damped once they are already directionally converged.
                    # Floor at pr_min so frequent rows never fully freeze.
                    active_counts = counts.index_select(0, active_idx).to(update.dtype)
                    row_alpha = pr_scale / (1.0 + active_counts).pow(pr_power)
                    row_alpha = row_alpha.clamp(min=pr_min, max=pr_max)
                    update = update * row_alpha.unsqueeze(-1)              # (n_active, k)
                # Scatter the (possibly tangent-projected / row-scaled) update back.
                p.index_add_(0, active_idx, update, alpha=-lr * scale)
        return loss


class _OptCombo:
    """Thin wrapper that forwards step/zero_grad to a list of optimizers and
    exposes a merged param_groups property so scheduler wrappers still work."""
    def __init__(self, optimizers, names=None):
        self.optimizers = list(optimizers)
        self.names      = list(names) if names is not None else [f'opt{i}' for i in range(len(optimizers))]

    def step(self, closure=None):
        for o in self.optimizers:
            o.step()

    def zero_grad(self, set_to_none=True):
        for o in self.optimizers:
            o.zero_grad(set_to_none=set_to_none)

    @property
    def param_groups(self):
        merged = []
        for o in self.optimizers:
            merged.extend(o.param_groups)
        return merged

    def state_dict(self):
        return {n: o.state_dict() for n, o in zip(self.names, self.optimizers)}

    def load_state_dict(self, sd):
        for n, o in zip(self.names, self.optimizers):
            if n in sd:
                o.load_state_dict(sd[n])


class _SchedCombo:
    """Same pattern for schedulers. get_last_lr returns a flat list in opt order."""
    def __init__(self, schedulers, names=None):
        self.schedulers = list(schedulers)
        self.names      = list(names) if names is not None else [f'sched{i}' for i in range(len(schedulers))]

    def step(self):
        for s in self.schedulers:
            s.step()

    def get_last_lr(self):
        out = []
        for s in self.schedulers:
            out.extend(s.get_last_lr())
        return out


def _partition_params_for_muon(model):
    """
    Returns (muon_params, sparse_muon_groups, adamw_decay, adamw_nodecay, routing_info).

    Routing:
    - Embedding tables (A_in, A_out): SparseMuon if enabled, else AdamW (decay).
      A_in and A_out go into separate param groups inside SparseMuon so they
      can carry different settings (e.g. tangent_only on for A_out, off for A_in).
    - All other 2D weights (body Linears, B, out_to_k*, mos_gate): Muon.
    - 1D params (RMSNorm weights, out_bias) and 0D params (shortcut_scale):
      AdamW (no decay).
    """
    muon_params, adamw_decay, adamw_nodecay = [], [], []
    muon_names,  adamw_decay_names, adamw_nodecay_names = [], [], []
    sparse_muon_groups = []  # list of dicts: {'params': [...], 'tangent_only': bool, 'name': str}
    sparse_muon_names  = []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if name == 'embed.A_out.weight':
            if SPARSE_MUON_ENABLED:
                sparse_muon_groups.append({
                    'params': [p],
                    'tangent_only':     SPARSE_MUON_TANGENT_A_OUT,
                    'per_row_lr':       (SPARSE_MUON_PER_ROW_LR_ENABLED and SPARSE_MUON_PER_ROW_LR_A_OUT),
                    'per_row_lr_power': SPARSE_MUON_PER_ROW_LR_POWER,
                    'per_row_lr_scale': SPARSE_MUON_PER_ROW_LR_SCALE,
                    'per_row_lr_min':   SPARSE_MUON_PER_ROW_LR_MIN,
                    'per_row_lr_max':   SPARSE_MUON_PER_ROW_LR_MAX,
                    'name': name,
                })
                sparse_muon_names.append(name)
            else:
                adamw_decay.append(p);   adamw_decay_names.append(name)
        elif name == 'embed.A_in.weight':
            if SPARSE_MUON_ENABLED:
                sparse_muon_groups.append({
                    'params': [p],
                    'tangent_only':     SPARSE_MUON_TANGENT_A_IN,
                    'per_row_lr':       (SPARSE_MUON_PER_ROW_LR_ENABLED and SPARSE_MUON_PER_ROW_LR_A_IN),
                    'per_row_lr_power': SPARSE_MUON_PER_ROW_LR_POWER,
                    'per_row_lr_scale': SPARSE_MUON_PER_ROW_LR_SCALE,
                    'per_row_lr_min':   SPARSE_MUON_PER_ROW_LR_MIN,
                    'per_row_lr_max':   SPARSE_MUON_PER_ROW_LR_MAX,
                    'name': name,
                })
                sparse_muon_names.append(name)
            else:
                adamw_decay.append(p);   adamw_decay_names.append(name)
        elif p.ndim == 2:
            muon_params.append(p);   muon_names.append(name)
        elif p.ndim >= 2:
            adamw_decay.append(p);   adamw_decay_names.append(name)
        else:
            adamw_nodecay.append(p); adamw_nodecay_names.append(name)
    routing = {
        'muon':          muon_names,
        'sparse_muon':   sparse_muon_names,
        'adamw_decay':   adamw_decay_names,
        'adamw_nodecay': adamw_nodecay_names,
    }
    return muon_params, sparse_muon_groups, adamw_decay, adamw_nodecay, routing


def make_optimizer(model):
    if not MUON_ENABLED:
        decay, no_decay = [], []
        for _, p in model.named_parameters():
            if not p.requires_grad:
                continue
            (decay if p.ndim >= 2 else no_decay).append(p)
        opt = torch.optim.AdamW([
            {'params': decay,    'weight_decay': 0.1},
            {'params': no_decay, 'weight_decay': 0.0},
        ], lr=LR, betas=(0.9, 0.95), eps=1e-8)
        opt._muon_routing = None
        return opt

    muon_params, sparse_muon_groups, adamw_decay, adamw_nodecay, routing = _partition_params_for_muon(model)
    optimizers, names = [], []
    muon = Muon(muon_params, lr=MUON_LR, momentum=MUON_MOMENTUM,
                nesterov=MUON_NESTEROV, ns_steps=MUON_NS_STEPS)
    optimizers.append(muon); names.append('muon')
    if SPARSE_MUON_ENABLED and len(sparse_muon_groups) > 0:
        # Per-param-group tangent_only flag lets A_out (vMF head, norm-invariant)
        # and A_in (input lookup, norm-sensitive) carry different geometry.
        sparse_muon = SparseMuon(sparse_muon_groups, lr=SPARSE_MUON_LR,
                                 momentum=SPARSE_MUON_MOMENTUM, ns_steps=SPARSE_MUON_NS_STEPS)
        optimizers.append(sparse_muon); names.append('sparse_muon')
    # AdamW only gets non-empty groups (avoid empty param-group warnings)
    adamw_groups = []
    if len(adamw_decay) > 0:
        adamw_groups.append({'params': adamw_decay,   'weight_decay': 0.1})
    if len(adamw_nodecay) > 0:
        adamw_groups.append({'params': adamw_nodecay, 'weight_decay': 0.0})
    if len(adamw_groups) > 0:
        adamw = torch.optim.AdamW(adamw_groups, lr=LR, betas=(0.9, 0.95), eps=1e-8)
        optimizers.append(adamw); names.append('adamw')
    combo = _OptCombo(optimizers, names=names)
    combo._muon_routing = routing
    return combo


def make_scheduler(optimizer, steps_per_epoch):
    total_steps = SCHEDULE_HORIZON_EPOCHS * steps_per_epoch

    def lr_lambda_cosine(step):
        if step < WARMUP_STEPS:
            return float(step + 1) / float(max(1, WARMUP_STEPS))
        progress = (step - WARMUP_STEPS) / float(max(1, total_steps - WARMUP_STEPS))
        return 0.5 * (1.0 + math.cos(math.pi * min(max(progress, 0.0), 1.0)))

    def lr_lambda_sgdr(step):
        # Initial global warmup is preserved so the first cycle doesn't spike
        # from zero. After that, every RESTART_STEPS steps we re-warm briefly
        # and cosine-decay to SGDR_MIN_LR_FRAC of peak before snapping back.
        if step < WARMUP_STEPS:
            return float(step + 1) / float(max(1, WARMUP_STEPS))
        local = (step - WARMUP_STEPS) % SGDR_RESTART_STEPS
        if local < SGDR_WARMUP_STEPS:
            # mini-warmup at the start of each cycle, peak at end of mini-warmup
            return SGDR_MIN_LR_FRAC + (1.0 - SGDR_MIN_LR_FRAC) * (local + 1) / float(max(1, SGDR_WARMUP_STEPS))
        progress = (local - SGDR_WARMUP_STEPS) / float(max(1, SGDR_RESTART_STEPS - SGDR_WARMUP_STEPS))
        cosine = 0.5 * (1.0 + math.cos(math.pi * min(max(progress, 0.0), 1.0)))
        return SGDR_MIN_LR_FRAC + (1.0 - SGDR_MIN_LR_FRAC) * cosine

    lr_lambda = lr_lambda_sgdr if SGDR_ENABLED else lr_lambda_cosine

    if isinstance(optimizer, _OptCombo):
        scheds = [torch.optim.lr_scheduler.LambdaLR(o, lr_lambda) for o in optimizer.optimizers]
        return _SchedCombo(scheds, names=optimizer.names)
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


# ── Training loop ─────────────────────────────────────────────────────────────
def train_model(model, train_loader, val_loader, test_tokens, class_weight=None):
    print(f'Training {RUN_NAME} | params={model.count_params():,}')
    print(
        'E4 gradient routing:\n'
        '  A_in.weight   <- g_in  only  (sparse, free to specialize as transformer input)\n'
        '  A_out.weight  <- g_out only  (dense,  free to specialize as retrieval geometry)\n'
        '  B.weight      <- g_in  only  (upscale, unchanged from E3)\n'
        '  out_to_k      <- g_out only  (projection, unchanged from E3)\n'
        '  Warm-start: out_to_k.weight <- B.weight.T[:k_out]\n'
        f'  head-mode: {HEAD_MODE}  '
        f"(vmf: logit = kappa*cos(z,A_out);  linear: logit = z.A_out[v])\n"
        f'  out-bias: enabled={OUT_BIAS_ENABLED} '
        f'init_unigram={OUT_BIAS_INIT_UNIGRAM}  '
        f'(softmax-bottleneck rank +1 slot: b_v captures log P(w))\n'
        f'  bigram shortcut: enabled={SHORTCUT_ENABLED} '
        f'alpha_0={SHORTCUT_INIT}  '
        f'(logit += alpha * A_in[x_t] . A_out[v]; coherent early attractor for A_out)\n'
        f'  MoS: n={MOS_COMPONENTS}  '
        f'(rank cap 1*(k+1)={K_OUT+1} -> {MOS_COMPONENTS}*(k+1)={MOS_COMPONENTS*(K_OUT+1)}; '
        f'shared A_out, shared bias, {MOS_COMPONENTS} projections + gate)\n'
        f'  MoS init: asymmetric={MOS_ASYMMETRIC_INIT} '
        f'gate_bias_skew={MOS_GATE_BIAS_INIT_SKEW} '
        f'diversity_lambda={MOS_DIVERSITY_LAMBDA} '
        f'nonlin_on_extra_components={MOS_NONLIN_ON_EXTRA_COMPONENTS}\n'
        f'  MTP: enabled={MTP_ENABLED} depth={MTP_DEPTH} weight={MTP_WEIGHT}  '
        f'(aux CE on token t+{MTP_DEPTH}; +{K_OUT * D_MODEL} params, training only)\n'
        f'  kappa-cap: enabled={KAPPA_CAP_ENABLED} frac={KAPPA_CAP_FRAC} '
        f'[{KAPPA_CAP_INIT} -> {KAPPA_CAP_FINAL}]  (only active for vmf)\n'
        f'  freq-bal CE: enabled={FREQ_BAL_ENABLED} alpha={FREQ_ALPHA} '
        f'clamp_max={FREQ_CLAMP_MAX}\n'
        f'  optimizer: '
        f'{("Muon(lr=" + str(MUON_LR) + ") body 2D + " + ("SparseMuon(lr=" + str(SPARSE_MUON_LR) + ") A_in/A_out + " if SPARSE_MUON_ENABLED else "") + "AdamW(lr=" + f"{LR:.0e}" + ") for 1D/scalars" + ("" if SPARSE_MUON_ENABLED else " and embeddings")) if MUON_ENABLED else ("AdamW(lr=" + f"{LR:.0e}" + ") for all params")}\n'
    )

    log_embedding_alignment(model, 0, 0)

    optimizer = make_optimizer(model)
    scheduler = make_scheduler(optimizer, len(train_loader))

    routing = getattr(optimizer, '_muon_routing', None)
    if routing is not None:
        muon_set        = set(routing['muon'])
        sparse_muon_set = set(routing.get('sparse_muon', []))
        adamw_set       = set(routing['adamw_decay']) | set(routing['adamw_nodecay'])
        muon_params_total        = sum(p.numel() for n_, p in model.named_parameters() if n_ in muon_set)
        sparse_muon_params_total = sum(p.numel() for n_, p in model.named_parameters() if n_ in sparse_muon_set)
        adamw_params_total       = sum(p.numel() for n_, p in model.named_parameters() if n_ in adamw_set)
        print(
            f'  muon         -> {len(muon_set):>3d} tensors, {muon_params_total:>10,} params '
            f'(body Linear weights, B, out_to_k*, mos_gate)\n'
            + (f'  sparse_muon  -> {len(sparse_muon_set):>3d} tensors, {sparse_muon_params_total:>10,} params '
               f'(A_in, A_out — orthogonalize active rows only)\n'
               if len(sparse_muon_set) > 0 else '')
            + f'  adamw        -> {len(adamw_set):>3d} tensors, {adamw_params_total:>10,} params '
            f'(RMSNorm weights, out_bias, shortcut_scale)\n'
            f'  muon example names: {routing["muon"][:3]}\n'
            + (f'  sparse_muon names: {routing["sparse_muon"]}\n' if len(sparse_muon_set) > 0 else '')
            + f'  adamw example names: {(routing["adamw_decay"] + routing["adamw_nodecay"])[:4]}'
        )

    # Total optimizer steps (for kappa-cap schedule).
    opt_steps_per_epoch = len(train_loader) // ACCUM_STEPS
    total_opt_steps     = opt_steps_per_epoch * EPOCHS

    if class_weight is not None:
        class_weight = class_weight.to(device)

    ema = WeightEMA(model, decay=EMA_DECAY) if EMA_ENABLED else None

    global_opt_step = 0

    for epoch in range(1, EPOCHS + 1):
        model.train()
        t0                = time.time()
        running_loss      = 0.0   # total loss incl. MTP aux (for optimizer accounting)
        running_main_ce   = 0.0   # pure main-head CE (comparable to v21-v24 logs)
        optimizer.zero_grad(set_to_none=True)

        for step, batch in enumerate(train_loader, start=1):
            x, y  = [t.to(device, non_blocking=True) for t in batch]
            dtype = torch.bfloat16 if USE_BF16 else torch.float32

            # kappa cap is only meaningful for vmf; skip in linear mode.
            k_cap = kappa_cap_at_step(global_opt_step, total_opt_steps) if HEAD_MODE == 'vmf' else None
            with torch.amp.autocast('cuda', dtype=dtype, enabled=USE_BF16):
                out  = model(x, labels=y, kappa_cap=k_cap, class_weight=class_weight,
                             head_mode=HEAD_MODE)
                loss = out.loss / ACCUM_STEPS
            loss.backward()
            running_loss += out.loss.item()
            # Subtract MTP contribution to recover main-head CE for logging.
            mtp_contrib = (MTP_WEIGHT * float(out.mtp_loss.item())) \
                          if (MTP_ENABLED and hasattr(out, 'mtp_loss')) else 0.0
            running_main_ce += out.loss.item() - mtp_contrib

            if step % ACCUM_STEPS == 0:
                # Log grad norms *before* clipping/step so we see raw signal.
                if (global_opt_step % GRAD_LOG_EVERY) == 0:
                    log_grad_norms(model, step, epoch)
                    log_sparse_muon_counts(optimizer, step, epoch)
                torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
                optimizer.step()
                scheduler.step()
                if ema is not None:
                    ema.update(model)
                optimizer.zero_grad(set_to_none=True)
                global_opt_step += 1

            if step % 200 == 0:
                kappa_mean = out.kappa.mean().item()
                kappa_std  = out.kappa.std().item()
                cap_str    = f'{k_cap:.2f}' if k_cap is not None else 'off'
                with torch.no_grad():
                    aout_norm_mean = model.embed.A_out.weight.float().norm(dim=-1).mean().item()
                    aout_norm_std  = model.embed.A_out.weight.float().norm(dim=-1).std().item()
                    b = model.out_bias.float().detach()
                    b_std, b_min, b_max = b.std().item(), b.min().item(), b.max().item()
                    alpha_val = float(model.shortcut_scale.detach().item())
                    # MoS diagnostics: look at the gate logits on this batch
                    mos_str = ''
                    if model.n_mos > 1:
                        h_dbg = model.forward_features(x)
                        gate_logits_dbg = model.mos_gate(h_dbg).float()
                        pi = F.softmax(gate_logits_dbg, dim=-1)            # [B, T, n]
                        pi_mean = pi.mean(dim=(0, 1))                       # [n]
                        gate_H = -(pi * pi.clamp_min(1e-8).log()).sum(dim=-1).mean().item()
                        pi_str = ' '.join(f'{p.item():.3f}' for p in pi_mean)
                        gb = model.mos_gate.bias.detach().float().cpu().tolist() \
                             if model.mos_gate.bias is not None else None
                        gb_str = ('[' + ' '.join(f'{v:+.2f}' for v in gb) + ']') if gb else '-'
                        div_val = float(out.mos_div.item()) if hasattr(out, 'mos_div') else 0.0
                        # Per-component z-norm ( = kappa in vMF) — shows whether the
                        # SiLU-damped component is actually producing sharp logits.
                        z_norms = []
                        for j, proj in enumerate(model.out_to_k_list):
                            z_j = proj(h_dbg).float()
                            if MOS_NONLIN_ON_EXTRA_COMPONENTS and j > 0:
                                z_j = F.silu(z_j)
                            z_norms.append(z_j.norm(dim=-1).mean().item())
                        zn_str = '[' + ' '.join(f'{v:.2f}' for v in z_norms) + ']'
                        mos_str = (f' pi_mean=[{pi_str}] gate_H={gate_H:.3f} '
                                   f'gate_bias={gb_str} cos_z={div_val:.3f} '
                                   f'k_per_comp={zn_str}')
                mtp_str = ''
                if MTP_ENABLED and hasattr(out, 'mtp_loss') and out.mtp_loss is not None:
                    mtp_v = float(out.mtp_loss.item())
                    mtp_str = f' mtp_ce={mtp_v:.3f}'
                lrs = scheduler.get_last_lr()
                if MUON_ENABLED and SPARSE_MUON_ENABLED and len(lrs) >= 3:
                    lr_str = f'lr_muon={lrs[0]:.2e} lr_smuon={lrs[1]:.2e} lr_adamw={lrs[-1]:.2e}'
                elif MUON_ENABLED and len(lrs) >= 2:
                    lr_str = f'lr_muon={lrs[0]:.2e} lr_adamw={lrs[-1]:.2e}'
                else:
                    lr_str = f'lr {lrs[0]:.2e}'
                ema_str = ''
                if ema is not None:
                    g_all, g_mat = ema.gap_stats(model)
                    ema_str = f' ema_gap[global={g_all:.3e} mat_mean={g_mat:.3e}]'
                print(
                    f'epoch {epoch} step {step}/{len(train_loader)} '
                    f'ce_loss {running_main_ce / step:.6f} '
                    f'{lr_str} '
                    f'||z||_bar={kappa_mean:.3f} ||z||_std={kappa_std:.3f} '
                    f'kappa_cap={cap_str} '
                    f'||A_out||_bar={aout_norm_mean:.3f} ||A_out||_std={aout_norm_std:.3f} '
                    f'bias[std={b_std:.3f} min={b_min:.2f} max={b_max:.2f}] '
                    f'alpha={alpha_val:+.4f}{mtp_str}{mos_str}{ema_str}'
                )

            # Log A_in vs A_out alignment at key checkpoints
            # Step 200: should show cos ≈ 0 (random inits diverging or converging?)
            # Step 1000: has warm-start eliminated the subspace rotation phase?
            # Step 2000: is the split stabilizing?
            if step in (200, 1000, 2000, 4000, 8000):
                model.eval()
                log_embedding_alignment(model, step, epoch)
                model.train()

        val_metrics = evaluate(model, val_loader, max_batches=VAL_MAX_BATCHES, split_name='val')
        test_ppl    = evaluate_ppl_sliding(model, test_tokens)
        train_ce    = running_main_ce / len(train_loader)

        # EMA evaluation: swap to EMA weights, re-run val, swap back. If the
        # 1/t theory is right, EMA val PPL should be meaningfully lower than
        # instantaneous because EMA averages out the SGD noise ball.
        ema_val_metrics = None
        if ema is not None:
            ema.swap_to_ema(model)
            ema_val_metrics = evaluate(model, val_loader, max_batches=VAL_MAX_BATCHES, split_name='val(ema)')
            ema.swap_back(model)

        # End-of-epoch full diagnostic
        cos_mean = log_embedding_alignment(model, len(train_loader), epoch)

        ema_str = ''
        if ema_val_metrics is not None:
            delta = val_metrics["ppl"] - ema_val_metrics["ppl"]
            ema_str = (f' ema_val_recall@1={ema_val_metrics["recall@1"]:.6f} '
                       f'ema_val_ppl={ema_val_metrics["ppl"]:.2f} (Δppl={delta:+.2f})')
        print(
            f'\nEpoch {epoch}: '
            f'train_ce={train_ce:.4f} '
            f'val_recall@1={val_metrics["recall@1"]:.6f} '
            f'val_ppl={val_metrics["ppl"]:.2f} '
            f'test_ppl={test_ppl:.2f}{ema_str} '
            f'cos_shared(A_in,A_out)={cos_mean:.4f} '
            f'time={time.time() - t0:.1f}s\n'
        )

        # Falsification check
        if epoch == 1:
            if val_metrics['recall@1'] < 0.32:
                print('[FALSIFIED] Epoch 1 recall < 0.32: A-conflict was not the bottleneck.')
            elif val_metrics['recall@1'] >= 0.34:
                print('[CONFIRMED] Epoch 1 recall ≥ 0.34: A-conflict diagnosis correct.')
            if cos_mean > 0.8:
                print('[FALSIFIED] cos_shared(A_in,A_out) > 0.8: tying appears close in shared subspace.')
            elif cos_mean < 0.3:
                print('[CONFIRMED] cos_shared(A_in,A_out) < 0.3: matrices diverged in shared subspace.')

        torch.cuda.empty_cache()
        gc.collect()


# ── Entry point ───────────────────────────────────────────────────────────────
def main():
    print(f'Device: {device}')
    if torch.cuda.is_available():
        print(f'GPU: {torch.cuda.get_device_name(0)} | BF16: {USE_BF16}')
    print(f'Run: {RUN_NAME}\n')
    print(
        '[DIAGNOSIS] v30 FALSIFIED the basin-hopping hypothesis. SGDR warm\n'
        '  restarts (cycle=2000, min_frac=0.1) produced a CE curve IDENTICAL\n'
        '  to v28 within +/-0.008 nats at every matched step (1000, 2000,\n'
        '  3000, 4000, 4200). The LR dropping from 1.84e-2 to 2.18e-3 at the\n'
        '  restart did not perturb CE trajectory — post-restart, the curve\n'
        '  snapped right back onto the same 1/t line. One basin, tight.\n'
        '\n'
        '  What the four-run 1/t fit actually says, re-read with cold eyes:\n'
        '    v26 (AdamW body):          A=1.69  t0=+2625\n'
        '    v27 (Muon body):           A=1.17  t0=+1038\n'
        '    v28 (Muon + SparseMuon):   A=0.87  t0=-67\n'
        '    v29 (tangent-only):        A=0.80  t0=-132\n'
        '    v30 (SGDR):                A=0.77  t0=+47\n'
        '  Each optimizer transition reduced A. But v28 -> v29 -> v30 (three\n'
        '  runs, three variants of the same optimizer family) plateau at A\n'
        '  ~= 0.8. Within Muon-family we are squeezed out. The tail is not\n'
        '  set by the optimizer any more.\n'
        '\n'
        '  Where is it set? Decompose the grad logs:\n'
        '    step    body  A_in  A_out  out_to_k  B\n'
        '    init   16.35  2.09   1.34      0.53  0.67\n'
        '    1608    0.49  0.15   0.18      0.17  0.036\n'
        '    3208    0.34  0.13   0.12      0.18  0.020\n'
        '    4808    0.16  0.12   0.083     0.11  0.009\n'
        '  Body gradient decays 102x. A_in 17x. A_out 16x. By step 4800 the\n'
        '  head produces ~2/3 of total first-order learning signal. The\n'
        '  BODY is done. The HEAD is the rate limiter.\n'
        '\n'
        '  Inside the head: 50k rows of A_out each need to rotate from random\n'
        '  init to their optimal angular position on S^47. Per-occurrence\n'
        '  angular step ~= lr/||A_out[v]||. Zipf means:\n'
        '    top-100 tokens: ~5000 occurrences over 5000 steps =>\n'
        '      total rotation ~=  5000 * 0.01 = 50 rad => MANY wrappings,\n'
        '      directionally converged long ago.\n'
        '    rank-30000 tokens: ~10 occurrences =>\n'
        '      total rotation ~= 10 * 0.036 = 0.36 rad = 20 deg.\n'
        '      Starting random on S^47, 20 deg is nothing. UNROTATED.\n'
        '  The 1/t slope coefficient is locked by Zipf, not by variance.\n'
        '  The bottom half of the vocabulary literally cannot reach its\n'
        '  optimal direction at this data budget. That is the deamon.\n'
        '\n'
        '[FIX] v31 equalizes per-row angular budget only on sparse rows.\n'
        '  In SparseMuon, multiply each active row\'s update by\n'
        '  (1 + update_count[v])^(-0.5), floored at 0.02. Under full\n'
        '  softmax CE, A_out gradients are dense (all rows active), so\n'
        '  grad-based row count is NOT a frequency proxy for A_out. Therefore\n'
        '  per-row LR is enabled for A_in and disabled for A_out.\n'
        '  This remains an Adagrad accumulator over COUNT dimension where\n'
        '  sparsity semantics are valid:\n'
        '    count=0    => mul=1.00 (first-ever update: full base LR)\n'
        '    count=10   => mul=0.30 (mid-freq)\n'
        '    count=100  => mul=0.10\n'
        '    count=1000 => mul=0.03\n'
        '    count>=2500 => mul=0.02 (floor; frequent rows only micro-move)\n'
        '  SGDR disabled (it was a null intervention).\n'
        '  Also fixed the EMA gap metric: v30\'s per-tensor-averaged gap was\n'
        '  dominated by shortcut_scale (near-zero EMA norm). v31 reports\n'
        '  ||concat(w-w_ema)||/||concat(w_ema)|| (global, matrix-dominated)\n'
        '  and the matrix-only mean as a second number.\n'
        '\n'
        '[PREDICT] If the Zipf rotation-budget theory is right, v31 diverges\n'
        '  from v28 starting around step 2000 — when frequent-row multipliers\n'
        '  first drop below 0.1 and rare-row signal starts dominating the\n'
        '  A_out update direction. Slope coefficient A drops from 0.80 to\n'
        '  0.50-0.55 nats. At matched step 4000, v31 lands 0.15-0.30 nats\n'
        '  below v28 (target CE 5.27-5.42). [smuon-count] logs will show\n'
        '  ||A_out||_bar growth SLOWING relative to v28, because frequent\n'
        '  rows\' radial drift is now damped by per-row multipliers.\n'
        '  Falsifier A: v31 within +/-0.03 nats of v28 at step 4000 AND\n'
        '  ||A_out||_bar trajectory matches v28 => rotation budget is NOT\n'
        '  the bottleneck; rare-token signal is genuinely gone and redistrib\n'
        '  doesn\'t help. In that case v32 is variance reduction (batch=16384\n'
        '  via ACCUM_STEPS=16) as the last cheap lever before we commit to\n'
        '  architectural changes (PQ codebook for A_out).\n'
        '  Falsifier B: v31 early CE worse than v28 by >0.1 nats at step 500\n'
        '  AND does not recover => the frequent-row LR damping was too\n'
        '  aggressive; raise the floor to 0.1 and re-run.\n'
    )
    _, train_tokens, val_tokens, test_tokens = load_wikitext103()
    train_loader, val_loader = build_loaders(train_tokens, val_tokens)
    class_weight    = build_freq_class_weights(train_tokens, vocab_size=VOCAB_SIZE)
    unigram_logprob = build_unigram_logprob(train_tokens, vocab_size=VOCAB_SIZE)
    model = BreakthroughMicroTransformerE4(unigram_logprob=unigram_logprob).to(device)
    print(f'Params: {model.count_params():,}  (E3 was 4,758,360; +{model.count_params()-4_758_360:,})\n')
    train_model(model, train_loader, val_loader, test_tokens, class_weight=class_weight)


if __name__ == '__main__':
    main()
