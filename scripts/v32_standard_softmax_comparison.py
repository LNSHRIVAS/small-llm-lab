# =============================================================================
# v32 — Standard softmax head (fair comparison to vMF)
# =============================================================================
#
# ONLY intentional differences vs v32_zipf_diagnostics.py:
#   • HEAD_MODE = 'linear'  (plain logits z @ W_out^T; decode_scores linear-only)
#   • + log_wout_norm_by_frequency() each epoch (H4-softmax; for the paper table)
#   • RESUME_ENABLED = False by default (fresh baseline; never load vMF ckpt)
#   • Storage: repo-local data/cache/wikitext103_gpt2 (portfolio copy).
#     Override with WKT103_CACHE_DIR or
#     V32_ARTIFACT_DIR if you want checkpoints elsewhere.  Share the same ARTIFACT_DIR
#     as v32 so train/val/test *_tokens.pt are identical.
#
# Training hyperparameters, DIAG_FREQ, EVAL_STRIDE, BATCH/ACCUM, and mid-epoch
# diagnostics match v32_zipf_diagnostics.py.  Default EPOCHS=2 here (comparison);
# v32_zipf uses 8 for phase transition.  Log field `kappa` in
# forward/eval is still ||z|| (linear head has no vMF concentration).
#
# Two run modes:
#
#   MODE = 'main'          — training + full diagnostic suite
#
#   MODE = 'scaling_scan'  — V²/T scan (same as v32; head is still linear)
#
# HYPOTHESES (MODE='main'):
#
#   H1  ZIPF-GEOMETRY MAIN CLAIM
#       Pearson(angle(A_out[v], init), -log rank(v))  >  0.5    → CONFIRMED
#
#   H2  TWO-LEARNING-PROBLEMS
#       Per-bucket val CE: top-bucket CE drops while tail-bucket CE flat
#       (or grows) → CONFIRMED. Compares ratio across epochs.
#
#   H3  GRADIENT INCONSISTENCY
#       cos(grad[v, t], grad[v, t-1]) by frequency bucket.
#       gap(top_100 - tail_30K+)  >  0.2  → CONFIRMED.
#
#   H4  RADIAL AUTO-DAMPING
#       Pearson(||A_out[v]||, -log rank(v))  >  0.5  → CONFIRMED.
#
#   H5  A_IN PERPETUAL LEARNING
#       A_in grad norm by bucket should stay flat (not decay) across the run.
#
#   H6  BODY SATURATION
#       head/body gradient ratio  >  2× after warmup → CONFIRMED.
#
#   H7  FROZEN A_IN ABLATION  (toggle FREEZE_A_IN=True)
#       Compare final PPL to the FREEZE_A_IN=False baseline.
#
#   H8  1/t FIT
#       CE(t) = A/t + C* fit to per-opt-step instantaneous CE on t ≥ one full
#       epoch of opt steps (skip warmup-dominated epoch 1); min_step = opt_steps/epoch.
#       A reported per epoch; stability across epochs argues for an
#       optimizer-independent floor.
#
# SCALING SCAN (MODE='scaling_scan') tests the prediction:
#
#       A_floor  ∝  V² / T
#
#   Halve V at fixed T  → A drops ~4×
#   Double T at fixed V → A drops ~2×
#   Equivalent perturbations under the law.
#
#   Verdict: log-log slope of A vs V²/T should be ≈ +1.0.
#            CONFIRMED if slope ∈ [0.8, 1.2].
#            FALSIFIED if slope outside [0.5, 1.5].
#
# CORRECTNESS GUARANTEES (vs the original v32 sketch):
#
#   - Gradient-consistency hook is GPU-resident; no per-microbatch CPU sync.
#   - log_ce_by_frequency uses NON-OVERLAPPING windows (no double-counting).
#   - 1/t fit uses per-opt-step instantaneous CE (not a cumulative mean), with
#     min_step = opt_steps_per_epoch (skip epoch-1 transient).
#   - Hooks are paused during eval / per-bucket probes (no contamination).
#   - All randomness seeded with SEED for cross-variant comparability.
#
# =============================================================================

import os
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'

import gc
import math
import shutil
import sys
import time
import numpy as np
from pathlib import Path
from types import SimpleNamespace

# Force UTF-8 stdout so that box-drawing and math glyphs render on Windows
# consoles (cp1252). Colab/Linux already default to UTF-8 — this is a no-op
# there. Wrapped in try/except so it never crashes the run.
try:
    if sys.stdout.encoding and 'utf' not in sys.stdout.encoding.lower():
        sys.stdout.reconfigure(encoding='utf-8', errors='replace')
        sys.stderr.reconfigure(encoding='utf-8', errors='replace')
except Exception:
    pass

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from transformers import GPT2TokenizerFast
from datasets import load_dataset


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
if torch.cuda.is_available():
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32       = True
    USE_BF16 = torch.cuda.is_bf16_supported()
    USE_FP16 = not USE_BF16   # fall back to FP16 on T4 (no BF16 support)
else:
    USE_BF16 = False
    USE_FP16 = False

SEED = 42
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)
np.random.seed(SEED)


def amp_autocast():
    """One source of truth for mixed-precision context."""
    if USE_BF16:
        return torch.amp.autocast('cuda', dtype=torch.bfloat16, enabled=True)
    if USE_FP16:
        return torch.amp.autocast('cuda', dtype=torch.float16, enabled=True)
    return torch.amp.autocast('cuda', enabled=False)


# =============================================================================
# HYPERPARAMETERS  (match v32_zipf_diagnostics.py — do not drift for comparability)
# =============================================================================
#
# Optional: on large GPUs, BATCH_SIZE=32 & ACCUM_STEPS=2 keeps 65,536 tokens/opt-step.
#
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
# Softmax comparison run: 2 epochs (H2 / H4 / C* at ep2 vs v32 at same epoch).
# Full v32_zipf job uses EPOCHS=8; increase here only if you want a longer softmax baseline.
EPOCHS       = 2
WARMUP_STEPS = 300
GRAD_CLIP    = 1.0

SCHEDULE_HORIZON_EPOCHS = 10
NUM_WORKERS      = 2
VAL_MAX_BATCHES  = None
EVAL_STRIDE      = 512

HEAD_MODE              = 'linear'   # standard softmax logits (fair vs vMF)
OUT_BIAS_ENABLED       = True
OUT_BIAS_INIT_UNIGRAM  = True
OUT_BIAS_INIT_EPS      = 1.0
SHORTCUT_ENABLED       = True
SHORTCUT_INIT          = 0.0
MOS_COMPONENTS         = 1
MOS_ASYMMETRIC_INIT    = True
MOS_GATE_BIAS_INIT_SKEW = 1.0
MOS_DIVERSITY_LAMBDA   = 2e-2
MOS_NONLIN_ON_EXTRA_COMPONENTS = True
MTP_ENABLED  = True
MTP_DEPTH    = 2
MTP_WEIGHT   = 0.3
MUON_ENABLED    = True
MUON_LR         = 0.02
MUON_MOMENTUM   = 0.95
MUON_NESTEROV   = True
MUON_NS_STEPS   = 5
SPARSE_MUON_ENABLED  = True
SPARSE_MUON_LR       = 0.005
SPARSE_MUON_MOMENTUM = 0.95
SPARSE_MUON_NS_STEPS = 5
SPARSE_MUON_TANGENT_A_OUT = False
SPARSE_MUON_TANGENT_A_IN  = False
SPARSE_MUON_PER_ROW_LR_ENABLED = True
SPARSE_MUON_PER_ROW_LR_A_OUT   = False
SPARSE_MUON_PER_ROW_LR_A_IN    = True
SPARSE_MUON_PER_ROW_LR_POWER   = 0.5
SPARSE_MUON_PER_ROW_LR_SCALE   = 1.0
SPARSE_MUON_PER_ROW_LR_MIN     = 0.02
SPARSE_MUON_PER_ROW_LR_MAX     = 1.0
SGDR_ENABLED = False
EMA_ENABLED  = True
EMA_DECAY    = 0.999
KAPPA_CAP_ENABLED = False
KAPPA_CAP_FRAC    = 0.15
KAPPA_CAP_INIT    = 8.0
KAPPA_CAP_FINAL   = 28.0
FREQ_BAL_ENABLED = False
FREQ_ALPHA       = 0.0
FREQ_EPS         = 1.0
FREQ_CLAMP_MAX   = 1.3

# ── Storage: repo-local cache (portfolio copy; no Colab Drive paths) ─────────
_REPO_ROOT = Path(__file__).resolve().parents[1]
LOCAL_CACHE_DIR = Path(os.environ.get(
    'WKT103_CACHE_DIR',
    str(_REPO_ROOT / 'data' / 'cache' / 'wikitext103_gpt2')))
DRIVE_CACHE_DIR = LOCAL_CACHE_DIR
if os.environ.get('V32_ARTIFACT_DIR'):
    ARTIFACT_DIR = Path(os.environ['V32_ARTIFACT_DIR'])
else:
    ARTIFACT_DIR = LOCAL_CACHE_DIR
try:
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
except OSError as ex:
    print(f'[warn] could not create ARTIFACT_DIR={ARTIFACT_DIR}: {ex}')

TOKENIZER_NAME  = 'gpt2'
TOKENIZE_BATCH_SIZE = 2000

RUN_NAME = 'v32_standard_softmax_comparison'

# =============================================================================
# MODE DISPATCH
# =============================================================================
#
# 'main'          → full training with H1..H8 diagnostics (same schedule as v32_zipf).
#                   Time budget: ~8–9h per 2 epochs on T4; scales with EPOCHS.
#
# 'scaling_scan'  → multi-variant short trainings to test the
#                   A_floor ∝ V² / T law (separate Colab session recommended).
#                   Time budget: ~3-4h on T4 with default SCAN config.
#                   Tests one prediction in isolation; does NOT replace 'main'.
#
# Pick ONE per Colab session.  Run main first, then scaling_scan separately.
# =============================================================================

MODE = 'main'   # 'main' or 'scaling_scan'

# ── Diagnostic flags (used in 'main' mode) ───────────────────────────────────

FREEZE_A_IN      = False   # H7: set True to run the frozen-A_in ablation
DIAG_FREQ        = 200     # opt steps between H3/H5/H6 grad-diagnostics (same as v32)
SAVE_CHECKPOINTS = True    # save per-epoch checkpoints for post-hoc analysis
RESUME_ENABLED = False     # True only to resume *this* softmax run from its own ckpt
RESUME_FROM    = ARTIFACT_DIR / f'{RUN_NAME}_ep2.pt'  # example path if RESUME_ENABLED

# Frequency buckets used throughout all diagnostics
FREQ_BUCKETS = [
    ('top_100',    0,      100),
    ('100_1K',     100,    1000),
    ('1K_10K',     1000,   10000),
    ('10K_30K',    10000,  30000),
    ('tail_30K+',  30000,  50257),
]

# ── Scaling-scan config (used in 'scaling_scan' mode) ───────────────────────
#
# Predictions to falsify:
#   Halve V at fixed T  → A drops ~4×  (V² scaling)
#   Double T at fixed V → A drops ~2×  (1/T scaling)
#   These are equivalent perturbations under A ∝ V²/T.
#
# Each variant: V_eff-way head on top types; T_trained ∝ T_corpus.  See v32_zipf
# scaling-scan docstring for full protocol (this file mirrors it).

SCAN_VARIANTS = [
    # name           V_eff   T_frac
    ('V50K_T1.0',    50257,  1.00),   # baseline reference
    ('V25K_T1.0',    25000,  1.00),   # halve V
    ('V12K_T1.0',    12500,  1.00),   # quarter V
    ('V50K_T0.5',    50257,  0.50),   # halve T
    ('V50K_T0.25',   50257,  0.25),   # quarter T
]
SCAN_ACCUM      = 2            # smaller than main ACCUM_STEPS for faster scan
SCAN_BATCH      = 8
SCAN_LOG_EVERY  = 100          # opt-step interval for scan progress log
SCAN_FIT_FROM   = 200          # floor for 1/t fit (reduced-V)
SCAN_FIT_FROM_FULL_V = 500     # floor for full-V

SCAN_TOKEN_PASS_FRAC   = 0.28
SCAN_MIN_OPT_STEPS     = 400
SCAN_MAX_OPT_STEPS_CAP = 8000


def _scan_tokens_per_opt_step():
    return int(SCAN_BATCH * SCAN_ACCUM * SEQ_LEN)


def _scan_compute_max_steps(T_corpus: int) -> int:
    tpo = _scan_tokens_per_opt_step()
    steps = int(math.ceil(SCAN_TOKEN_PASS_FRAC * max(T_corpus, 1) / tpo))
    return max(SCAN_MIN_OPT_STEPS, min(steps, SCAN_MAX_OPT_STEPS_CAP))


def _scan_compute_fit_from(V_eff: int, max_steps: int) -> int:
    base = SCAN_FIT_FROM_FULL_V if V_eff >= VOCAB_SIZE - 1 else SCAN_FIT_FROM
    lo = max(base, int(0.20 * max_steps))
    return min(lo, max_steps - 50)


# =============================================================================
# DATASET
# =============================================================================

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


# =============================================================================
# MODEL (identical to v31 — no changes)
# =============================================================================

class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.eps    = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        orig = x.dtype
        x32  = x.to(torch.float32)
        rms  = x32.pow(2).mean(-1, keepdim=True).add(self.eps).rsqrt()
        return (x32 * rms).to(orig) * self.weight


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
        cos = self.cos_cached[:, :, offset:offset+t, :].to(dtype=x.dtype, device=x.device)
        sin = self.sin_cached[:, :, offset:offset+t, :].to(dtype=x.dtype, device=x.device)
        xe, xo = x[..., ::2], x[..., 1::2]
        return torch.stack((xe*cos - xo*sin, xe*sin + xo*cos), dim=-1).flatten(-2)


class FactorizedEmbeddingE4(nn.Module):
    def __init__(self, vocab_size, d_model, k_in, k_out):
        super().__init__()
        self.A_in  = nn.Embedding(vocab_size, k_in)
        self.A_out = nn.Embedding(vocab_size, k_out)
        self.B     = nn.Linear(k_in, d_model, bias=False)
        nn.init.normal_(self.A_in.weight,  0.0, 0.02)
        nn.init.normal_(self.A_out.weight, 0.0, 0.02)
        nn.init.orthogonal_(self.B.weight)

        # H7: optionally freeze A_in (announced once in the run banner; do not
        # print here so we don't spam the scaling scan, which builds many
        # fresh models).
        if FREEZE_A_IN:
            self.A_in.weight.requires_grad = False

    def embed(self, ids):
        return self.B(self.A_in(ids))

    def normalized_A_out(self):
        return F.normalize(self.A_out.weight.float(), dim=-1)

    def decode_scores(self, z, kappa_cap=None, mode='vmf'):
        # Standard softmax: logits = z @ W_out^T (no vMF; kappa_cap ignored).
        return z.float() @ self.A_out.weight.float().T


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
        self.qkv_proj  = nn.Linear(d_model, 3*d_model, bias=False)
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
        out = F.scaled_dot_product_attention(q, k, v, is_causal=True,
              dropout_p=self.dropout_p if self.training else 0.0)
        return self.out_proj(out.transpose(1,2).contiguous().view(b, t, d))

    def ffn(self, x):
        return self.w_down(self.dropout(F.silu(self.w_gate(x)) * self.w_up(x)))

    def forward(self, x):
        x = x + self.dropout(self.attention(self.norm_attn(x)))
        x = x + self.dropout(self.ffn(self.norm_ffn(x)))
        return x


class BreakthroughMicroTransformerE4(nn.Module):
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
        self.n_mos      = max(1, int(MOS_COMPONENTS))
        self.out_to_k_list = nn.ModuleList([
            nn.Linear(D_MODEL, K_OUT, bias=False) for _ in range(self.n_mos)
        ])
        if self.n_mos > 1:
            self.mos_gate = nn.Linear(D_MODEL, self.n_mos, bias=True)
            nn.init.zeros_(self.mos_gate.weight)
            with torch.no_grad():
                b = torch.zeros(self.n_mos)
                b[0] = float(MOS_GATE_BIAS_INIT_SKEW)
                self.mos_gate.bias.copy_(b)
        with torch.no_grad():
            base = self.embed.B.weight.T[:K_OUT, :].clone()
            for i, layer in enumerate(self.out_to_k_list):
                if i == 0 or not MOS_ASYMMETRIC_INIT:
                    layer.weight.copy_(base)
        if MTP_ENABLED:
            self.out_to_k_mtp = nn.Linear(D_MODEL, K_OUT, bias=False)
            with torch.no_grad():
                self.out_to_k_mtp.weight.copy_(self.embed.B.weight.T[:K_OUT, :])
        if OUT_BIAS_ENABLED:
            self.out_bias = nn.Parameter(torch.zeros(VOCAB_SIZE))
            if unigram_logprob is not None and OUT_BIAS_INIT_UNIGRAM:
                with torch.no_grad():
                    self.out_bias.copy_(unigram_logprob.detach())
                    self.out_bias.sub_(self.out_bias.mean())
        else:
            self.register_buffer('out_bias', torch.zeros(VOCAB_SIZE), persistent=False)
        if SHORTCUT_ENABLED:
            assert K_IN == K_OUT
            self.shortcut_scale = nn.Parameter(torch.tensor(float(SHORTCUT_INIT)))
        else:
            self.register_buffer('shortcut_scale', torch.tensor(0.0), persistent=False)

    def forward_features(self, ids):
        h = self.in_dropout(self.embed.embed(ids))
        for blk in self.blocks:
            h = blk(h)
        return self.norm_final(h)

    def _component_logits(self, h, proj, head_mode, kappa_cap, apply_nonlin=False):
        z = proj(h).float()
        if apply_nonlin:
            z = F.silu(z)
        logits = self.embed.decode_scores(z, kappa_cap=kappa_cap, mode=head_mode)
        logits = logits + self.out_bias.float()
        return logits, z

    def forward(self, ids, labels=None, kappa_cap=None, class_weight=None,
                head_mode=HEAD_MODE):
        h = self.forward_features(ids)
        if SHORTCUT_ENABLED:
            a_in_t    = self.embed.A_in(ids).float()
            a_out_mat = self.embed.A_out.weight.float()
            bigram_scaled = self.shortcut_scale.float() * (a_in_t @ a_out_mat.T)
        else:
            bigram_scaled = None

        if self.n_mos == 1:
            logits, z = self._component_logits(h, self.out_to_k_list[0],
                                                head_mode, kappa_cap)
            if bigram_scaled is not None:
                logits = logits + bigram_scaled
            loss = None
            mtp_loss_val = torch.zeros((), device=h.device)
            if labels is not None:
                loss = F.cross_entropy(logits.view(-1, VOCAB_SIZE),
                                       labels.view(-1), weight=class_weight)
                if MTP_ENABLED and self.training and labels.shape[1] > MTP_DEPTH:
                    shift      = MTP_DEPTH - 1
                    T_mtp      = labels.shape[1] - shift
                    mtp_labels = labels[:, shift:].contiguous()
                    h_mtp      = h[:, :T_mtp, :]
                    z_mtp      = self.out_to_k_mtp(h_mtp).float()
                    mtp_logits = self.embed.decode_scores(z_mtp, kappa_cap=kappa_cap,
                                                          mode=head_mode)
                    mtp_logits = mtp_logits + self.out_bias.float()
                    mtp_loss_val = F.cross_entropy(mtp_logits.view(-1, VOCAB_SIZE),
                                                   mtp_labels.view(-1))
                    loss = loss + MTP_WEIGHT * mtp_loss_val
            # Linear head: `kappa` is ||z|| for logging (not vMF concentration).
            kappa = z.norm(dim=-1, keepdim=True)
            return SimpleNamespace(loss=loss, z=z, kappa=kappa, logits=logits,
                                   mos_div=torch.zeros((), device=h.device),
                                   mtp_loss=mtp_loss_val.detach())

        # MoS path (unchanged from v31, not used by default)
        log_pi = F.log_softmax(self.mos_gate(h).float(), dim=-1)
        per_component_logp = []
        z_first = None
        for i, proj in enumerate(self.out_to_k_list):
            nonlin_i = bool(MOS_NONLIN_ON_EXTRA_COMPONENTS) and (i > 0)
            logits_i, z_i = self._component_logits(h, proj, head_mode, kappa_cap,
                                                    apply_nonlin=nonlin_i)
            if bigram_scaled is not None:
                logits_i = logits_i + bigram_scaled
            if i == 0: z_first = z_i
            per_component_logp.append(log_pi[..., i:i+1] + F.log_softmax(logits_i, dim=-1))
        log_p = torch.logsumexp(torch.stack(per_component_logp, dim=0), dim=0)
        kappa = z_first.norm(dim=-1, keepdim=True)  # ||z|| when using linear head
        loss  = F.nll_loss(log_p.view(-1, VOCAB_SIZE), labels.view(-1)) if labels is not None else None
        return SimpleNamespace(loss=loss, z=z_first, kappa=kappa, logits=log_p,
                               mos_div=torch.zeros((), device=h.device),
                               mtp_loss=torch.zeros((), device=h.device))

    def count_params(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


class BreakthroughMicroTransformerE4Scan(nn.Module):
    """
    Scaling-scan LM: full input embedding, V_eff-way output (linear softmax here).
    See v32_zipf_diagnostics.BreakthroughMicroTransformerE4Scan for details.
    """
    def __init__(self, head_token_ids: torch.Tensor, unigram_logprob: torch.Tensor):
        super().__init__()
        h = head_token_ids.long().contiguous()
        self.V_eff = int(h.numel())
        self.register_buffer('head_token_ids', h.clone())
        g2l = torch.full((VOCAB_SIZE,), -100, dtype=torch.long)
        g2l[h] = torch.arange(self.V_eff, dtype=torch.long)
        self.register_buffer('global_to_local', g2l)

        self.embed = FactorizedEmbeddingE4(VOCAB_SIZE, D_MODEL, K_IN, K_OUT)
        self.rope = RotaryEmbedding(D_MODEL // N_HEADS, N_POSITIONS, ROPE_BASE)
        self.in_dropout = nn.Dropout(DROPOUT)
        self.blocks = nn.ModuleList([
            TransformerBlock(D_MODEL, N_HEADS, D_FF_GATE, DROPOUT, N_LAYERS, self.rope)
            for _ in range(N_LAYERS)
        ])
        self.norm_final = RMSNorm(D_MODEL)
        self.out_to_k_list = nn.ModuleList([
            nn.Linear(D_MODEL, K_OUT, bias=False)
        ])
        with torch.no_grad():
            base = self.embed.B.weight.T[:K_OUT, :].clone()
            self.out_to_k_list[0].weight.copy_(base)

        if OUT_BIAS_ENABLED:
            self.out_bias_scan = nn.Parameter(torch.zeros(self.V_eff))
            if unigram_logprob is not None and OUT_BIAS_INIT_UNIGRAM:
                sl = unigram_logprob[h].float().detach()
                with torch.no_grad():
                    self.out_bias_scan.copy_(sl)
                    self.out_bias_scan.sub_(self.out_bias_scan.mean())
        else:
            self.register_buffer('out_bias_scan', torch.zeros(self.V_eff), persistent=False)

        if SHORTCUT_ENABLED:
            assert K_IN == K_OUT
            self.shortcut_scale = nn.Parameter(torch.tensor(float(SHORTCUT_INIT)))
        else:
            self.register_buffer('shortcut_scale', torch.tensor(0.0), persistent=False)

    def forward_features(self, ids):
        h = self.in_dropout(self.embed.embed(ids))
        for blk in self.blocks:
            h = blk(h)
        return self.norm_final(h)

    def _logits_head(self, z):
        w = self.embed.A_out.weight[self.head_token_ids]
        if HEAD_MODE == 'linear':
            return z.float() @ w.float().T
        A_n = F.normalize(w.float(), dim=-1)
        kappa = z.float().norm(dim=-1, keepdim=True).clamp_min(1e-6)
        direction = z.float() / kappa
        return kappa * (direction @ A_n.T)

    def forward(self, ids, labels=None, kappa_cap=None, class_weight=None,
                head_mode=HEAD_MODE):
        del kappa_cap, head_mode
        h = self.forward_features(ids)
        z = self.out_to_k_list[0](h).float()
        logits = self._logits_head(z)
        logits = logits + self.out_bias_scan.float().view(1, 1, -1)

        if SHORTCUT_ENABLED:
            a_in_t = self.embed.A_in(ids).float()
            a_s = self.embed.A_out.weight[self.head_token_ids].float()
            logits = logits + self.shortcut_scale.float() * (a_in_t @ a_s.T)

        loss = None
        if labels is not None:
            y_flat = labels.view(-1)
            y_cls = self.global_to_local.to(y_flat.device)[y_flat]
            loss = F.cross_entropy(logits.view(-1, self.V_eff), y_cls,
                                   weight=class_weight, ignore_index=-100)

        kappa = z.norm(dim=-1, keepdim=True)
        return SimpleNamespace(
            loss=loss, z=z, kappa=kappa, logits=logits,
            mos_div=torch.zeros((), device=h.device),
            mtp_loss=torch.zeros((), device=h.device))

    def count_params(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# =============================================================================
# DIAGNOSTIC SUITE
# =============================================================================

class ZipfDiagnostics:
    """
    All hypothesis measurements in one object.
    Accumulators live on the parameter device to avoid per-microbatch
    CPU<->GPU traffic (critical for T4 PCIe throughput).

    Initialise once after model creation; call log_* at the appropriate points
    in the training loop.
    """

    def __init__(self, model, train_tokens, vocab_size=VOCAB_SIZE):
        self.vocab_size = vocab_size
        param_dev = model.embed.A_out.weight.device

        # ── Token frequency and rank (CPU; cheap, used only for bucketing) ──
        freq = torch.bincount(train_tokens.long(), minlength=vocab_size).float()
        self.freq = freq                                              # [V] CPU
        self.rank = freq.argsort(descending=True).argsort()           # [V] CPU
        # rank[v] = position of token v in the descending-frequency ordering.

        # ── Bucket masks (CPU bool; used for slicing per-bucket outputs) ────
        self.buckets = {}
        for label, lo, hi in FREQ_BUCKETS:
            self.buckets[label] = (self.rank >= lo) & (self.rank < hi)

        # ── Initial weight snapshots for angular-distance measurement (H1) ─
        self.W_aout_init = model.embed.A_out.weight.detach().cpu().float().clone()
        self.W_ain_init  = model.embed.A_in.weight.detach().cpu().float().clone()

        # ── Gradient-consistency accumulators (H3, H5) ──────────────────────
        # Kept on the parameter's device so the backward hook never has to
        # synchronise. Memory: 2 × V × k × fp32 ≈ 2 × 50257 × 48 × 4 B ≈ 19 MB
        # per matrix; total ≈ 38 MB on GPU. Trivial vs activation memory.
        k_out = model.embed.A_out.weight.shape[1]
        k_in  = model.embed.A_in.weight.shape[1]
        self.aout_prev_dir    = torch.zeros(vocab_size, k_out, device=param_dev)
        self.aout_cons_sum    = torch.zeros(vocab_size, device=param_dev)
        self.aout_cons_count  = torch.zeros(vocab_size, device=param_dev)
        self.aout_initialized = torch.zeros(vocab_size, dtype=torch.bool, device=param_dev)

        self.ain_prev_dir    = torch.zeros(vocab_size, k_in, device=param_dev)
        self.ain_cons_sum    = torch.zeros(vocab_size, device=param_dev)
        self.ain_cons_count  = torch.zeros(vocab_size, device=param_dev)
        self.ain_initialized = torch.zeros(vocab_size, dtype=torch.bool, device=param_dev)

        # ── 1/t fit history (H8) ────────────────────────────────────────────
        # Each entry is (opt_step, instantaneous_ce). instantaneous_ce is the
        # main-CE for the opt step (averaged across its microbatches), NOT
        # the cumulative running mean — otherwise the fit is biased.
        self.ce_history   = []

        # Lowest opt_step included in H8 / P1 1/t fits (= opt_steps_per_epoch
        # after train_loader is known; skips epoch 1 transient).
        self.h8_fit_min_step = None

        # ── Per-epoch results (cross-epoch comparison in epoch_summary) ────
        self.epoch_results = {}

        # ── Banner ──────────────────────────────────────────────────────────
        H = -(freq/freq.sum() * (freq/freq.sum()).clamp(1e-12).log()).sum().item()
        sep = '═' * 78
        print('\n' + sep)
        print('  ZIPF-GEOMETRY DIAGNOSTIC SUITE  —  initialised')
        print(sep)
        print(f'  vocab={vocab_size}   H(unigram)={H:.3f} nats   uniform={math.log(vocab_size):.3f} nats')
        print(f'  device={param_dev}   accumulators on GPU (no per-microbatch CPU sync)')
        if FREEZE_A_IN:
            print(f'  *** H7 ABLATION: A_in is FROZEN — compare PPL to FREEZE_A_IN=False run ***')
        print(f'\n  Frequency buckets (rank-defined, fixed for the whole run):')
        print(f'    {"label":<12} {"types":>8} {"token_mass":>12}')
        print(f'    {"-"*36}')
        total = freq.sum().item()
        for label, mask in self.buckets.items():
            n = int(mask.sum().item())
            f_sum = freq[mask].sum().item() / max(total, 1.0)
            print(f'    {label:<12} {n:>8} {f_sum:>11.1%}')
        print(sep + '\n')

    # ── H3 / H5: gradient-direction consistency hook ────────────────────────

    def register_grad_hooks(self, model):
        """
        Register backward hooks on A_out and A_in to track per-row gradient
        direction consistency across consecutive backward calls.

        Notes:
        - Hook fires per-microbatch (i.e. on every loss.backward() call),
          which means we measure consistency between successive microbatch
          gradient contributions.  This is the right granularity: if two
          microbatches push token v in opposite directions, that's noise
          the optimizer has to average out.
        - All work is on-GPU; the hook returns None to leave grad unmodified.
        - Gating via .hooks_enabled lets us pause the hooks during eval and
          per-bucket loss probes that run forward+backward but shouldn't
          contaminate the consistency stats.
        """
        self.hooks_enabled = True

        def make_hook(prev_dir, cons_sum, cons_count, initialized):
            def hook(grad):
                if grad is None or not self.hooks_enabled:
                    return None
                with torch.no_grad():
                    g     = grad.detach().float()
                    gn    = g.norm(dim=-1)
                    active = gn > 1e-10
                    if not active.any():
                        return None
                    g_dir = g / gn.unsqueeze(-1).clamp_min(1e-10)

                    both = active & initialized
                    if both.any():
                        cos = (g_dir[both] * prev_dir[both]).sum(dim=-1).clamp_(-1.0, 1.0)
                        cons_sum[both]   += cos
                        cons_count[both] += 1.0

                    prev_dir[active]    = g_dir[active]
                    initialized[active] = True
                return None
            return hook

        self._handle_aout = model.embed.A_out.weight.register_hook(
            make_hook(self.aout_prev_dir, self.aout_cons_sum,
                      self.aout_cons_count, self.aout_initialized)
        )
        if not FREEZE_A_IN:
            self._handle_ain = model.embed.A_in.weight.register_hook(
                make_hook(self.ain_prev_dir, self.ain_cons_sum,
                          self.ain_cons_count, self.ain_initialized)
            )
        else:
            self._handle_ain = None

    def remove_hooks(self):
        if getattr(self, '_handle_aout', None):
            self._handle_aout.remove()
            self._handle_aout = None
        if getattr(self, '_handle_ain', None):
            self._handle_ain.remove()
            self._handle_ain = None

    # ── H3 / H5: report gradient-direction consistency ──────────────────────

    @torch.no_grad()
    def log_grad_consistency(self, step, epoch):
        """
        H3: are A_out gradients inconsistent for rare tokens?
        H5: is A_in gradient behaviour different from A_out?

        Reads the GPU accumulators once (single sync), prints a single
        compact table for each of A_out and A_in.
        """
        def report(cons_sum, cons_count, name):
            cs = cons_sum.detach().to('cpu')
            cc = cons_count.detach().to('cpu')
            seen    = cc > 0
            avg_cos = torch.where(seen, cs / cc.clamp(1.0), torch.zeros_like(cs))
            print(f'  [{name}] ep{epoch} step{step:>5}   '
                  f'cos(g_t , g_{{t-1}})  per-bucket')
            print(f'    {"bucket":<12} {"mean_cos":>10} {"n_obs":>10}')
            print(f'    {"-"*36}')
            results = {}
            for label, mask in self.buckets.items():
                m = mask & seen
                n_obs = float(cc[mask].sum().item())
                if m.sum() == 0:
                    print(f'    {label:<12} {"--":>10} {n_obs:>10.0f}')
                    continue
                c = float(avg_cos[m].mean().item())
                results[label] = c
                print(f'    {label:<12} {c:>10.4f} {n_obs:>10.0f}')
            if 'top_100' in results and 'tail_30K+' in results:
                gap = results['top_100'] - results['tail_30K+']
                verdict = ('H3 evidence' if gap > 0.2 else 'similar across buckets')
                print(f'    gap(top-tail)={gap:+.4f}  →  {verdict}')
            return results

        r_aout = report(self.aout_cons_sum, self.aout_cons_count, 'A_out grad-consistency')
        if not FREEZE_A_IN:
            r_ain = report(self.ain_cons_sum, self.ain_cons_count, 'A_in  grad-consistency')
        else:
            r_ain = {}
        return r_aout, r_ain

    # ── H1 / H4: Angular distance and norm by frequency ──────────────────────

    @torch.no_grad()
    def log_angular_distance(self, model, epoch):
        """
        H1: does angle(A_out[v], init) scale as -log(rank(v))?
        H4: does ||A_out[v]|| scale with token frequency?
        """
        W_now  = model.embed.A_out.weight.detach().cpu().float()
        W_init = self.W_aout_init.detach().cpu().float()

        cos   = (F.normalize(W_init, dim=-1) * F.normalize(W_now, dim=-1)).sum(dim=-1)
        cos   = cos.clamp(-1.0 + 1e-6, 1.0 - 1e-6)
        angle = torch.acos(cos)
        norm_now  = W_now.norm(dim=-1)
        norm_init = W_init.norm(dim=-1)

        print(f'\n  [H1 / H4]  angular distance + radial norm  ·  epoch {epoch}')
        print(f'    {"bucket":<12} {"angle(rad)":>11} {"angle(deg)":>11} '
              f'{"|A_out|_now":>12} {"|A_out|_init":>13}')
        print(f'    {"-"*64}')

        ang_results, norm_results = {}, {}
        for label, mask in self.buckets.items():
            if mask.sum() == 0:
                continue
            a  = angle[mask].mean().item()
            n  = norm_now[mask].mean().item()
            ni = norm_init[mask].mean().item()
            ang_results[label]  = a
            norm_results[label] = n
            print(f'    {label:<12} {a:>11.4f} {math.degrees(a):>10.2f}° '
                  f'{n:>12.4f} {ni:>13.4f}')

        # ── H1 / H4 correlations on a fixed deterministic sample ───────────
        # 2 000 tokens drawn uniformly is biased toward the tail by Zipf
        # (most types ARE rare); that's the population we care about.
        gen   = torch.Generator().manual_seed(SEED + epoch)
        samp  = torch.randperm(self.vocab_size, generator=gen)[:2000]
        log_r = -(self.rank[samp].float() + 1.0).log()

        corr_h1 = torch.corrcoef(torch.stack([angle[samp],     log_r]))[0, 1].item()
        corr_h4 = torch.corrcoef(torch.stack([norm_now[samp],  log_r]))[0, 1].item()

        verdict_h1 = '✓ CONFIRMED' if corr_h1 > 0.5 else '✗ WEAK'
        verdict_h4 = '✓ CONFIRMED' if corr_h4 > 0.5 else '✗ WEAK'
        print(f'    Pearson(angle ,  -log(rank))  = {corr_h1:>+.4f}   {verdict_h1}  [H1]')
        print(f'    Pearson(|A_out|, -log(rank))  = {corr_h4:>+.4f}   {verdict_h4}  [H4]')

        if 'top_100' in ang_results and 'tail_30K+' in ang_results:
            ratio = ang_results['top_100'] / max(ang_results['tail_30K+'], 1e-6)
            print(f'    rotation ratio (top_100 / tail_30K+) = {ratio:>5.1f}×')

        # A_in angular distance for comparison
        W_ain_now  = model.embed.A_in.weight.detach().cpu().float()
        W_ain_init = self.W_ain_init.detach().cpu().float()
        cos_ain    = (F.normalize(W_ain_init, dim=-1) *
                      F.normalize(W_ain_now, dim=-1)).sum(dim=-1).clamp(-1.0 + 1e-6, 1.0 - 1e-6)
        angle_ain  = torch.acos(cos_ain)
        corr_ain   = torch.corrcoef(torch.stack([angle_ain[samp], log_r]))[0, 1].item()
        ratio_aiao = angle.mean().item() / max(angle_ain.mean().item(), 1e-6)
        print(f'    A_in:  mean angle = {angle_ain.mean().item():.4f} rad '
              f'({math.degrees(angle_ain.mean().item()):.1f}°)   '
              f'Pearson(angle, -log r) = {corr_ain:+.4f}')
        print(f'    A_out rotates {ratio_aiao:.1f}× more than A_in (mean)')

        return ang_results, norm_results, corr_h1, corr_h4

    # ── H2: Per-frequency-bucket CE on validation set ────────────────────────

    @torch.no_grad()
    def log_ce_by_frequency(self, model, val_tokens, epoch):
        """
        H2: does rare-token CE stay flat while frequent-token CE improves?

        Implementation notes (correctness-critical):
        - Uses NON-OVERLAPPING windows (stride = SEQ_LEN). The original
          stride=512 version double-counted every token between offsets,
          which biases per-bucket means (especially the tail).
        - Each per-position NLL is a true per-token value because we sum
          NLLs into per_token_nll and divide by per_token_count.
        - Reports coverage per bucket — if coverage of the tail bucket is
          tiny (e.g. <10 % of types seen), the tail-CE is statistical noise
          and should be flagged.
        """
        model.eval()
        per_token_nll   = torch.zeros(self.vocab_size)
        per_token_count = torch.zeros(self.vocab_size)

        n = len(val_tokens)
        # Non-overlapping windows; advances by exactly SEQ_LEN each step.
        for s in range(0, n - SEQ_LEN - 1, SEQ_LEN):
            e = s + SEQ_LEN
            x = val_tokens[s    :e    ].unsqueeze(0).to(device)
            y = val_tokens[s + 1:e + 1].unsqueeze(0).to(device)
            T = x.shape[1]
            with amp_autocast():
                out = model(x, labels=None)
            logits   = out.logits[0].float()                          # [T, V]
            targets  = y[0]                                           # [T]
            # MoS path (MOS_COMPONENTS>1) returns log-probs; n_mos==1 returns logits.
            logprobs = (logits if MOS_COMPONENTS > 1
                        else F.log_softmax(logits, dim=-1))
            nll      = -logprobs.gather(1, targets.unsqueeze(1)).squeeze(1)
            t_cpu    = targets.detach().cpu()
            per_token_nll.scatter_add_  (0, t_cpu, nll.detach().cpu())
            per_token_count.scatter_add_(0, t_cpu, torch.ones(T))

        seen    = per_token_count > 0
        avg_nll = torch.where(seen, per_token_nll / per_token_count.clamp(1.0),
                              torch.zeros_like(per_token_nll))

        print(f'\n  [H2]  per-bucket val CE  ·  epoch {epoch}   (non-overlapping windows)')
        print(f'    {"bucket":<12} {"CE":>9} {"PPL":>9} '
              f'{"tokens":>10} {"types_seen":>12}')
        print(f'    {"-"*56}')

        results = {}
        for label, mask in self.buckets.items():
            m = mask & seen
            n_types_total = int(mask.sum().item())
            n_types_seen  = int((per_token_count[mask] > 0).sum().item())
            tok_mass      = float(per_token_count[mask].sum().item())
            if m.sum() == 0:
                print(f'    {label:<12} {"--":>9} {"--":>9} '
                      f'{tok_mass:>10.0f} {0:>5d}/{n_types_total:<6d}')
                continue
            ce  = float(avg_nll[m].mean().item())
            ppl = math.exp(min(ce, 20.0))
            results[label] = ce
            print(f'    {label:<12} {ce:>9.4f} {ppl:>9.2f} '
                  f'{tok_mass:>10.0f} {n_types_seen:>5d}/{n_types_total:<6d}')

        # Coverage warning: tail buckets often have <50 % type coverage in val.
        tail_cov = (per_token_count[self.buckets['tail_30K+']] > 0).float().mean().item()
        if tail_cov < 0.10:
            print(f'    !! tail_30K+ coverage = {tail_cov:.1%}  '
                  f'(CE on tail is dominated by a few outliers)')

        if 'top_100' in results and 'tail_30K+' in results:
            ratio = results['top_100'] / max(results['tail_30K+'], 1e-6)
            print(f'    ratio CE(top_100) / CE(tail_30K+) = {ratio:.4f}')

        return results

    # ── H6: Body vs head gradient ratio ──────────────────────────────────────

    @torch.no_grad()
    def log_body_head_ratio(self, model, step, epoch):
        """
        H6: does the head gradient persist while the body gradient collapses?

        Returns (g_body, head_total, ratio).  Single compact line of output.
        """
        def gnorm(p):
            return float(p.grad.detach().float().norm()) if p.grad is not None else 0.0

        e      = model.embed
        g_ain  = gnorm(e.A_in.weight)
        g_aout = gnorm(e.A_out.weight)
        g_B    = gnorm(e.B.weight)
        g_ok   = math.sqrt(sum(gnorm(l.weight) ** 2 for l in model.out_to_k_list))
        g_mtp  = gnorm(model.out_to_k_mtp.weight) if hasattr(model, 'out_to_k_mtp') else 0.0

        body_sq = sum(
            p.grad.detach().float().pow(2).sum().item()
            for n, p in model.named_parameters()
            if p.requires_grad and p.grad is not None and
               (n.startswith('blocks.') or n.startswith('norm_final'))
        )
        g_body     = math.sqrt(body_sq)
        head_total = math.sqrt(g_aout ** 2 + g_ok ** 2 + g_mtp ** 2)
        ratio      = head_total / max(g_body, 1e-10)
        flag       = '  HEAD>>BODY' if ratio > 2.0 else ''
        print(f'  [H6] ep{epoch} step{step:>5}   '
              f'body={g_body:>7.4f}  head={head_total:>7.4f}  '
              f'(A_out={g_aout:.4f} out_k={g_ok:.4f} A_in={g_ain:.4f} B={g_B:.4f} mtp={g_mtp:.4f})  '
              f'h/b={ratio:>5.2f}×{flag}')
        return g_body, head_total, ratio

    # ── H8: 1/t coefficient fit ──────────────────────────────────────────────

    def record_ce(self, opt_step, ce_value):
        """
        Append one (opt_step, instantaneous_ce) sample.

        IMPORTANT: ce_value MUST be the CE for THIS opt step (averaged across
        the microbatches that contributed to the step) — not a cumulative
        running mean. The original code passed the running mean, which is
        dominated by old high losses and biases A upward.
        """
        self.ce_history.append((int(opt_step), float(ce_value)))

    def fit_1_over_t(self, min_step=2000, max_step=None):
        """
        Fit CE(t) = A/t + C*  by ordinary least squares.
        Returns (A, C_star, residual_std, n_points)  or  (None, None, None, 0).
        """
        data = [(s, c) for s, c in self.ce_history if s >= min_step
                and (max_step is None or s <= max_step)]
        if len(data) < 10:
            return None, None, None, 0
        steps  = np.array([d[0] for d in data], dtype=np.float64)
        ce_arr = np.array([d[1] for d in data], dtype=np.float64)
        X      = np.column_stack([1.0 / steps, np.ones_like(steps)])
        coeffs, *_ = np.linalg.lstsq(X, ce_arr, rcond=None)
        A, C_star = float(coeffs[0]), float(coeffs[1])
        res_std = float(np.std(ce_arr - X @ coeffs))
        return A, C_star, res_std, len(data)

    def log_1_over_t_fit(self, epoch):
        """Print the 1/t fit result for this epoch."""
        ms = self.h8_fit_min_step if self.h8_fit_min_step is not None else 1817
        A, C_star, res_std, n = self.fit_1_over_t(min_step=ms)
        if A is None:
            print(f'\n  [H8]  not enough post-warmup data for 1/t fit yet '
                  f'(have {len(self.ce_history)} points)')
            return None, None
        print(f'\n  [H8]  CE(t) = A/t + C*    epoch {epoch}    fit on {n} points (t≥{ms})')
        print(f'    A           = {A:>9.4f} nats')
        print(f'    C*          = {C_star:>9.4f} nats')
        print(f'    residual σ  = {res_std:>9.5f}')
        if epoch > 1:
            prev_A = self.epoch_results.get(epoch - 1, {}).get('A')
            if prev_A is not None:
                print(f'    ΔA vs ep{epoch-1}= {A - prev_A:>+9.4f}   '
                      f'(stable A → optimizer-independent floor)')
        return A, C_star

    # ── H5: A_in gradient norm per bucket over time ───────────────────────────

    @torch.no_grad()
    def log_ain_grad_by_frequency(self, model, step, epoch):
        """
        H5: does the A_in gradient norm stay constant per bucket
        (i.e. perpetual learning of rare tokens)?

        Reads the current accumulated A_in.grad once and prints one line.
        """
        if model.embed.A_in.weight.grad is None:
            return
        g     = model.embed.A_in.weight.grad.detach().float()
        gn    = g.norm(dim=-1).cpu()
        active = gn > 0
        parts = []
        for label, mask in self.buckets.items():
            m = mask & active
            if m.sum() == 0:
                continue
            parts.append(f'{label}={gn[m].mean().item():.4f}')
        if parts:
            print(f'  [H5] ep{epoch} step{step:>5}   '
                  f'|grad A_in|/row by bucket :  ' + '  '.join(parts))

    # ── End-of-epoch summary ────────────────────────────────────────────────

    def epoch_summary(self, epoch, A, freq_ce, ang_results, corr_h1,
                       corr_h4, val_ppl, test_ppl):
        """Print a structured pass/fail verdict on all hypotheses."""
        sep = '═' * 78
        print('\n' + sep)
        print(f'  HYPOTHESIS VERDICT  ·  epoch {epoch}')
        print(sep)

        def badge(cond, ok='✓ CONFIRMED', no='✗ WEAK', maybe='? INCONCLUSIVE'):
            if cond is None:
                return maybe
            return ok if cond else no

        # H1
        print(f'    H1  Zipf-geometry main claim       '
              f'Pearson(angle, -log r) = {corr_h1:>+.4f}    '
              f'{badge(corr_h1 > 0.5)}')

        # H2
        h2_msg = 'NO DATA'
        freq_ce_ratio = None
        if freq_ce and 'top_100' in freq_ce and 'tail_30K+' in freq_ce:
            freq_ce_ratio = freq_ce['top_100'] / max(freq_ce['tail_30K+'], 1e-6)
            prev_ratio = self.epoch_results.get(epoch - 1, {}).get('freq_ce_ratio')
            if prev_ratio is None:
                h2_msg = f'ratio={freq_ce_ratio:.3f}    {badge(None)} (need ≥2 epochs)'
            elif freq_ce_ratio < prev_ratio - 0.02:
                h2_msg = (f'ratio={freq_ce_ratio:.3f} (was {prev_ratio:.3f})  '
                          f'rare-tokens-learning    ✓ CONFIRMED')
            elif freq_ce_ratio > prev_ratio + 0.02:
                h2_msg = (f'ratio={freq_ce_ratio:.3f} (was {prev_ratio:.3f})  '
                          f'rare-tokens-falling-behind   ✓ CONFIRMED')
            else:
                h2_msg = (f'ratio={freq_ce_ratio:.3f} (was {prev_ratio:.3f})  '
                          f'stable ratio    ✗ NULL')
        print(f'    H2  Two-learning-problems          {h2_msg}')

        # H4
        print(f'    H4  Radial auto-damping            '
              f'Pearson(|A_out|, -log r) = {corr_h4:>+.4f}   '
              f'{badge(corr_h4 > 0.5)}')
        print(f'    H4  W_out norm vs rank             '
              f'see [H4-softmax] block above for Pearson + buckets')

        # H8
        if A is not None:
            print(f'    H8  1/t floor coefficient          A = {A:.4f} nats')
        else:
            print(f'    H8  1/t floor coefficient          insufficient data')

        # Final perplexities
        print(f'\n    val PPL = {val_ppl:>7.2f}     test PPL = {test_ppl:>7.2f}')
        if FREEZE_A_IN:
            print(f'    H7  A_in FROZEN — compare PPL to FREEZE_A_IN=False baseline.')
        print(sep + '\n')

        # Persist for cross-epoch comparison
        self.epoch_results[epoch] = {
            'A':              A,
            'corr_h1':        corr_h1,
            'corr_h4':        corr_h4,
            'freq_ce_ratio':  freq_ce_ratio,
            'val_ppl':        val_ppl,
        }


# =============================================================================
# EXISTING DIAGNOSTICS (unchanged from v31)
# =============================================================================

@torch.no_grad()
def log_embedding_alignment(model, step, epoch):
    A_in  = model.embed.A_in.weight.float()
    A_out = model.embed.A_out.weight.float()
    k_shared = min(A_in.size(1), A_out.size(1))
    A_in_n  = F.normalize(A_in[:, :k_shared],  dim=-1)
    A_out_n = F.normalize(A_out[:, :k_shared], dim=-1)
    cos_sim = (A_in_n * A_out_n).sum(dim=-1)
    norm_in  = A_in.norm(dim=-1)
    norm_out = A_out.norm(dim=-1)
    print(f'  [align] ep{epoch} step{step:>5}   '
          f'cos(A_in,A_out): {cos_sim.mean():+.4f}±{cos_sim.std():.4f}  '
          f'[{cos_sim.min():+.4f}, {cos_sim.max():+.4f}]   '
          f'|A_in|={norm_in.mean():.3f}   |A_out|={norm_out.mean():.3f}')
    return cos_sim.mean().item()


@torch.no_grad()
def log_grad_norms(model, step, epoch):
    def gnorm(p):
        return float(p.grad.detach().float().norm()) if p.grad is not None else 0.0
    e      = model.embed
    g_ain  = gnorm(e.A_in.weight)
    g_aout = gnorm(e.A_out.weight)
    g_B    = gnorm(e.B.weight)
    g_ok   = sum(gnorm(l.weight)**2 for l in model.out_to_k_list)**0.5
    g_mtp  = gnorm(model.out_to_k_mtp.weight) if hasattr(model, 'out_to_k_mtp') else 0.0
    body_sq = sum(
        p.grad.detach().float().pow(2).sum().item()
        for n, p in model.named_parameters()
        if p.requires_grad and p.grad is not None and
           (n.startswith('blocks.') or n.startswith('norm_final'))
    )
    g_body = body_sq ** 0.5
    print(f'  [grad] ep{epoch} step{step:>5}   '
          f'A_in={g_ain:.4f}  A_out={g_aout:.4f}  B={g_B:.4f}  '
          f'out_k={g_ok:.4f}  mtp={g_mtp:.4f}  body={g_body:.4f}')


# =============================================================================
# OPTIMIZERS  (identical to v31)
# =============================================================================

class WeightEMA:
    def __init__(self, model, decay=0.999):
        self.decay  = decay
        self.shadow = {n: p.detach().float().clone()
                       for n, p in model.named_parameters() if p.requires_grad}
        self.backup = {}

    @torch.no_grad()
    def update(self, model):
        d = self.decay
        for n, p in model.named_parameters():
            if p.requires_grad and n in self.shadow:
                self.shadow[n].mul_(d).add_(p.detach().float(), alpha=1-d)

    @torch.no_grad()
    def gap_stats(self, model):
        diff_sq, base_sq, mat_gaps = 0.0, 0.0, []
        for n, p in model.named_parameters():
            if p.requires_grad and n in self.shadow:
                w, we = p.detach().float(), self.shadow[n]
                d2 = (w-we).pow(2).sum().item()
                b2 = we.pow(2).sum().item()
                diff_sq += d2; base_sq += b2
                if p.ndim >= 2:
                    mat_gaps.append(math.sqrt(d2) / (math.sqrt(b2)+1e-12))
        return (math.sqrt(diff_sq)/(math.sqrt(base_sq)+1e-12),
                sum(mat_gaps)/len(mat_gaps) if mat_gaps else 0.0)

    @torch.no_grad()
    def swap_to_ema(self, model):
        self.backup = {}
        for n, p in model.named_parameters():
            if p.requires_grad and n in self.shadow:
                self.backup[n] = p.detach().clone()
                p.data.copy_(self.shadow[n].to(p.dtype))

    @torch.no_grad()
    def swap_back(self, model):
        for n, p in model.named_parameters():
            if n in self.backup:
                p.data.copy_(self.backup[n])
        self.backup = {}


def zeropower_via_newtonschulz5(G, steps=5, eps=1e-7):
    assert G.ndim == 2
    a, b, c = (3.4445, -4.7750, 2.0315)
    X = G.to(torch.bfloat16)
    transpose = G.size(0) > G.size(1)
    if transpose: X = X.T
    X = X / (X.norm() + eps)
    for _ in range(steps):
        A = X @ X.T
        B = b*A + c*(A@A)
        X = a*X + B@X
    if transpose: X = X.T
    return X.to(G.dtype)


class Muon(torch.optim.Optimizer):
    def __init__(self, params, lr=0.02, momentum=0.95, nesterov=True, ns_steps=5):
        super().__init__(params, dict(lr=lr, momentum=momentum,
                                       nesterov=nesterov, ns_steps=ns_steps))

    @torch.no_grad()
    def step(self, closure=None):
        for group in self.param_groups:
            lr, mu = group['lr'], group['momentum']
            for p in group['params']:
                if p.grad is None: continue
                g   = p.grad
                buf = self.state[p].setdefault('momentum_buffer',
                                                torch.zeros_like(g))
                buf.mul_(mu).add_(g)
                upd = g.add(buf, alpha=mu) if group['nesterov'] else buf
                upd = zeropower_via_newtonschulz5(upd, steps=group['ns_steps'])
                p.add_(upd, alpha=-lr * max(1., upd.size(-2)/upd.size(-1))**0.5)


class SparseMuon(torch.optim.Optimizer):
    def __init__(self, params, lr=0.005, momentum=0.95, ns_steps=5,
                 tangent_only=False, eps=1e-10,
                 per_row_lr=False, per_row_lr_power=0.5,
                 per_row_lr_scale=1.0, per_row_lr_min=0.02,
                 per_row_lr_max=1.0):
        defaults = dict(lr=lr, momentum=momentum, ns_steps=ns_steps,
                        tangent_only=tangent_only, eps=eps,
                        per_row_lr=per_row_lr, per_row_lr_power=per_row_lr_power,
                        per_row_lr_scale=per_row_lr_scale,
                        per_row_lr_min=per_row_lr_min,
                        per_row_lr_max=per_row_lr_max)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        for group in self.param_groups:
            lr, mu = group['lr'], group['momentum']
            for p in group['params']:
                if p.grad is None: continue
                g   = p.grad
                st  = self.state[p]
                buf = st.setdefault('momentum_buffer', torch.zeros_like(p))
                cnt = st.setdefault('update_count',
                                     torch.zeros(p.size(0), dtype=torch.int32,
                                                  device=p.device))
                buf.mul_(mu).add_(g)
                active_mask = g.norm(dim=-1) > group['eps']
                n_active    = int(active_mask.sum())
                if n_active == 0: continue
                idx = torch.where(active_mask)[0]
                cnt.index_add_(0, idx,
                                torch.ones_like(idx, dtype=cnt.dtype))
                buf_a  = buf.index_select(0, idx)
                update = zeropower_via_newtonschulz5(buf_a, steps=group['ns_steps'])
                scale  = max(1., update.size(-2)/update.size(-1))**0.5
                if group['per_row_lr']:
                    ac    = cnt.index_select(0, idx).to(update.dtype)
                    ra    = (group['per_row_lr_scale'] /
                             (1. + ac).pow(group['per_row_lr_power']))
                    ra    = ra.clamp(group['per_row_lr_min'],
                                     group['per_row_lr_max'])
                    update = update * ra.unsqueeze(-1)
                p.index_add_(0, idx, update, alpha=-lr*scale)


class _OptCombo:
    def __init__(self, optimizers, names=None):
        self.optimizers = list(optimizers)
        self.names      = names or [f'opt{i}' for i in range(len(optimizers))]

    def step(self):
        for o in self.optimizers: o.step()

    def zero_grad(self, set_to_none=True):
        for o in self.optimizers: o.zero_grad(set_to_none=set_to_none)

    @property
    def param_groups(self):
        out = []
        for o in self.optimizers: out.extend(o.param_groups)
        return out


class _SchedCombo:
    def __init__(self, schedulers):
        self.schedulers = list(schedulers)

    def step(self):
        for s in self.schedulers: s.step()

    def get_last_lr(self):
        out = []
        for s in self.schedulers: out.extend(s.get_last_lr())
        return out


def _partition_params(model):
    muon, sparse_groups, adamw_d, adamw_nd = [], [], [], []
    for name, p in model.named_parameters():
        if not p.requires_grad: continue
        if name == 'embed.A_out.weight':
            if SPARSE_MUON_ENABLED:
                sparse_groups.append({
                    'params': [p], 'name': name,
                    'tangent_only':     SPARSE_MUON_TANGENT_A_OUT,
                    'per_row_lr':       SPARSE_MUON_PER_ROW_LR_ENABLED and SPARSE_MUON_PER_ROW_LR_A_OUT,
                    'per_row_lr_power': SPARSE_MUON_PER_ROW_LR_POWER,
                    'per_row_lr_scale': SPARSE_MUON_PER_ROW_LR_SCALE,
                    'per_row_lr_min':   SPARSE_MUON_PER_ROW_LR_MIN,
                    'per_row_lr_max':   SPARSE_MUON_PER_ROW_LR_MAX,
                })
            else:
                adamw_d.append(p)
        elif name == 'embed.A_in.weight':
            if SPARSE_MUON_ENABLED and not FREEZE_A_IN:
                sparse_groups.append({
                    'params': [p], 'name': name,
                    'tangent_only':     SPARSE_MUON_TANGENT_A_IN,
                    'per_row_lr':       SPARSE_MUON_PER_ROW_LR_ENABLED and SPARSE_MUON_PER_ROW_LR_A_IN,
                    'per_row_lr_power': SPARSE_MUON_PER_ROW_LR_POWER,
                    'per_row_lr_scale': SPARSE_MUON_PER_ROW_LR_SCALE,
                    'per_row_lr_min':   SPARSE_MUON_PER_ROW_LR_MIN,
                    'per_row_lr_max':   SPARSE_MUON_PER_ROW_LR_MAX,
                })
            else:
                adamw_d.append(p)
        elif p.ndim == 2:
            muon.append(p)
        elif p.ndim >= 2:
            adamw_d.append(p)
        else:
            adamw_nd.append(p)
    return muon, sparse_groups, adamw_d, adamw_nd


def make_optimizer(model):
    muon_p, sparse_groups, adamw_d, adamw_nd = _partition_params(model)
    opts, names = [], []
    opts.append(Muon(muon_p, lr=MUON_LR, momentum=MUON_MOMENTUM,
                     nesterov=MUON_NESTEROV, ns_steps=MUON_NS_STEPS))
    names.append('muon')
    if SPARSE_MUON_ENABLED and sparse_groups:
        opts.append(SparseMuon(sparse_groups, lr=SPARSE_MUON_LR,
                               momentum=SPARSE_MUON_MOMENTUM,
                               ns_steps=SPARSE_MUON_NS_STEPS))
        names.append('sparse_muon')
    ag = []
    if adamw_d:  ag.append({'params': adamw_d,  'weight_decay': 0.1})
    if adamw_nd: ag.append({'params': adamw_nd, 'weight_decay': 0.0})
    if ag:
        opts.append(torch.optim.AdamW(ag, lr=LR, betas=(0.9, 0.95), eps=1e-8))
        names.append('adamw')
    combo = _OptCombo(opts, names)
    combo._names = names
    return combo


def make_scheduler(optimizer, steps_per_epoch):
    total = SCHEDULE_HORIZON_EPOCHS * steps_per_epoch

    def lr_lambda(step):
        if step < WARMUP_STEPS:
            return float(step+1) / float(max(1, WARMUP_STEPS))
        prog = (step - WARMUP_STEPS) / float(max(1, total - WARMUP_STEPS))
        return 0.5 * (1.0 + math.cos(math.pi * min(max(prog, 0.), 1.)))

    scheds = [torch.optim.lr_scheduler.LambdaLR(o, lr_lambda)
              for o in optimizer.optimizers]
    return _SchedCombo(scheds)


# =============================================================================
# EVALUATION
# =============================================================================

def evaluate(model, loader, max_batches=None, split_name='val'):
    model.eval()
    correct = total = loss_sum = steps = 0
    km, kmi, kmx = 0., 0., 0.
    with torch.no_grad():
        for i, batch in enumerate(loader):
            if max_batches and i >= max_batches: break
            x, y = [t.to(device, non_blocking=True) for t in batch]
            with amp_autocast():
                out = model(x, labels=y)
            correct   += (out.logits.argmax(-1) == y).sum().item()
            total     += y.numel()
            loss_sum  += out.loss.item()
            km        += out.kappa.mean().item()
            kmi       += out.kappa.min().item()
            kmx       += out.kappa.max().item()
            steps     += 1
    ppl = math.exp(min(loss_sum/max(steps,1), 20))
    print(f'{split_name} recall@1={correct/max(total,1):.6f} | '
          f'ce={loss_sum/max(steps,1):.6f} | ppl={ppl:.2f} | '
          f'κ mean={km/max(steps,1):.3f} min={kmi/max(steps,1):.3f} '
          f'max={kmx/max(steps,1):.3f}')
    return {'recall@1': correct/max(total,1), 'ppl': ppl,
            'ce_loss': loss_sum/max(steps,1)}


def evaluate_ppl_sliding(model, tokens, stride=EVAL_STRIDE, seq_len=SEQ_LEN):
    model.eval()
    nll_sum = ntok = prev_end = 0
    with torch.no_grad():
        for s in range(0, len(tokens)-1, stride):
            e  = min(s+seq_len, len(tokens)-1)
            x  = tokens[s:e].unsqueeze(0).to(device)
            y  = tokens[s+1:e+1].unsqueeze(0).to(device)
            ps = max(prev_end-s, 0) if s > 0 else 0
            if ps >= x.size(1): prev_end=e; continue
            yi = y.clone()
            if ps > 0: yi[:, :ps] = -100
            with amp_autocast():
                out = model(x, labels=yi)
            valid    = (yi != -100).sum().item()
            nll_sum += out.loss.item() * valid
            ntok    += valid
            prev_end = e
            if e >= len(tokens)-1: break
    return math.exp(min(nll_sum/max(ntok,1), 20))


# =============================================================================
# DATA LOADING
# =============================================================================

def _tokenize_split(tokenizer, texts, name, batch_size=TOKENIZE_BATCH_SIZE):
    texts   = [t for t in texts if t and t.strip()]
    all_ids = []
    print(f'[{name}] tokenizing {len(texts):,} docs')
    for i in range(0, len(texts), batch_size):
        enc = tokenizer(texts[i:i+batch_size],
                        add_special_tokens=False, truncation=False)
        for ids in enc['input_ids']:
            if ids:
                all_ids.extend(ids)
                all_ids.append(tokenizer.eos_token_id)
    tokens = torch.tensor(all_ids, dtype=torch.long)
    print(f'[{name}] {tokens.numel():,} tokens')
    return tokens


def load_wikitext103():
    tokenizer = GPT2TokenizerFast.from_pretrained(TOKENIZER_NAME)
    tokenizer.pad_token = tokenizer.eos_token
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    tok = {s: ARTIFACT_DIR / f'{s}_tokens.pt' for s in ['train', 'val', 'test']}

    # One-time migrate: legacy local cache → ARTIFACT_DIR when Drive is primary.
    if ARTIFACT_DIR.resolve() != LOCAL_CACHE_DIR.resolve():
        LOCAL_CACHE_DIR.mkdir(parents=True, exist_ok=True)
        for s in ['train', 'val', 'test']:
            if not tok[s].exists():
                leg = LOCAL_CACHE_DIR / f'{s}_tokens.pt'
                if leg.exists():
                    print(f'[data] migrating {leg} → {tok[s]}')
                    shutil.copy2(leg, tok[s])

    def _load_pt(path):
        try:
            return torch.load(path, map_location='cpu', weights_only=False)
        except TypeError:
            return torch.load(path, map_location='cpu')

    if all(tok[s].exists() for s in ['train', 'val', 'test']):
        print(f'[data] loading cached tokens from {ARTIFACT_DIR}')
        tr = _load_pt(tok['train'])
        va = _load_pt(tok['val'])
        te = _load_pt(tok['test'])
        print(f'  train:{tr.numel():,}  val:{va.numel():,}  test:{te.numel():,}')
        return tokenizer, tr, va, te

    print('[data] tokenising WikiText-103 (first run; saving to ARTIFACT_DIR)...')
    raw = load_dataset('wikitext', 'wikitext-103-raw-v1')
    tr  = _tokenize_split(tokenizer, raw['train']['text'],      'train')
    va  = _tokenize_split(tokenizer, raw['validation']['text'], 'val')
    te  = _tokenize_split(tokenizer, raw['test']['text'],       'test')
    for s, t in [('train', tr), ('val', va), ('test', te)]:
        torch.save(t, tok[s])
        print(f'[data] saved {tok[s]}')
    return tokenizer, tr, va, te


def build_loaders(train_tokens, val_tokens):
    tl = DataLoader(TokenDataset(train_tokens, SEQ_LEN),
                    batch_size=BATCH_SIZE, shuffle=True,
                    num_workers=NUM_WORKERS, pin_memory=True, drop_last=True)
    vl = DataLoader(TokenDataset(val_tokens, SEQ_LEN),
                    batch_size=BATCH_SIZE, shuffle=False,
                    num_workers=NUM_WORKERS, pin_memory=True)
    return tl, vl


def build_unigram_logprob(train_tokens, vocab_size=VOCAB_SIZE,
                           eps=OUT_BIAS_INIT_EPS):
    counts = torch.bincount(train_tokens.long(), minlength=vocab_size).float()
    p      = (counts + eps) / (counts.sum() + eps*vocab_size)
    logp   = p.log()
    H      = -(p * logp).sum().item()
    print(f'[unigram] H={H:.3f} (uniform={math.log(vocab_size):.3f})')
    return logp


def _tokens_seen_per_epoch(train_loader):
    """Microbatches that participate in a full optimizer step, × tokens/microbatch."""
    m = len(train_loader) - (len(train_loader) % ACCUM_STEPS)
    return m * BATCH_SIZE * SEQ_LEN


def log_cstar_vs_T(diag, epoch, train_loader):
    """
    Priority 1 analysis: track C* as T grows.
    One line per epoch. This is the phase transition measurement.
    """
    ms = diag.h8_fit_min_step if diag.h8_fit_min_step is not None else 1817
    A, C_star, res_std, n = diag.fit_1_over_t(min_step=ms)
    if C_star is None:
        return

    T = epoch * _tokens_seen_per_epoch(train_loader)
    V = VOCAB_SIZE
    V2_over_T = (V ** 2) / T

    # Prediction from c=0.1263 fitted at epoch 2
    H_language = 2.698
    c_fitted    = 0.1263
    C_star_pred = H_language + c_fitted * V2_over_T

    print(f'\n[P1 C*-vs-T] epoch={epoch:>2}  '
          f'T={T:>13,}  '
          f'V²/T={V2_over_T:>8.4f}  '
          f'C*_measured={C_star:.4f}  '
          f'C*_predicted={C_star_pred:.4f}  '
          f'delta={C_star - C_star_pred:+.4f}  '
          f'A={A:.2f}  '
          f'residual_σ={res_std:.5f}  '
          f'n={n}')

    # Phase transition check
    if C_star - H_language < 0.15:
        print(f'  *** APPROACHING SATURATION: '
              f'C* - H_language = {C_star - H_language:.4f} nats ***')
    elif C_star - H_language < 0.50:
        print(f'  *** SATURATION ZONE: '
              f'C* - H_language = {C_star - H_language:.4f} nats ***')


@torch.no_grad()
def log_wout_norm_by_frequency(model, diag, epoch):
    """
    Standard-softmax analog of H4: ||W_out|| vs Zipf rank (W_out = A_out rows).
    Pearson on the same 2000-token subsample scheme as log_angular_distance.
    """
    W    = model.embed.A_out.weight.detach().cpu().float()
    norm = W.norm(dim=-1)
    rank = diag.rank
    gen  = torch.Generator().manual_seed(SEED + epoch)
    samp = torch.randperm(diag.vocab_size, generator=gen)[:2000]
    log_r = -(rank[samp].float() + 1.0).log()
    corr = torch.corrcoef(torch.stack([norm[samp], log_r]))[0, 1].item()

    print(f'\n  [H4-softmax] epoch {epoch}  ||W_out|| vs Zipf rank')
    print(f'    Pearson(||W_out||, -log(rank)) = {corr:+.4f}')
    print(f'    {"bucket":<12} {"mean||W_out||":>14}')
    print(f'    {"-"*28}')
    for label, mask in diag.buckets.items():
        if mask.sum() == 0:
            continue
        print(f'    {label:<12} {norm[mask].mean().item():>14.4f}')
    return corr


# =============================================================================
# MAIN TRAINING LOOP WITH FULL DIAGNOSTICS
# =============================================================================

def train_model(model, train_loader, val_loader, test_tokens,
                train_tokens, val_tokens,
                resume_state=None, start_epoch=1):

    sep = '═' * 78
    print('\n' + sep)
    print(f'  TRAIN  ·  {RUN_NAME}')
    print(sep)
    print(f'  params={model.count_params():,}   FREEZE_A_IN={FREEZE_A_IN}   '
          f'DIAG_FREQ={DIAG_FREQ}   SAVE_CKPT={SAVE_CHECKPOINTS}')

    diag = ZipfDiagnostics(model, train_tokens)

    # Restore history if resuming
    if resume_state is not None:
        diag.ce_history    = resume_state['ce_history']
        diag.epoch_results = resume_state['epoch_results']
        diag.W_aout_init   = resume_state['W_aout_init'].detach().cpu().float().clone()
        diag.W_ain_init    = resume_state['W_ain_init'].detach().cpu().float().clone()
        print(f'[resume] restored {len(diag.ce_history)} CE points')
        print(f'[resume] restored epoch results for epochs: '
              f'{sorted(diag.epoch_results.keys())}')

    diag.register_grad_hooks(model)

    if start_epoch == 1:
        log_embedding_alignment(model, 0, 0)

    optimizer = make_optimizer(model)
    # Scheduler steps once per *optimizer* step, not per microbatch.
    opt_steps_per_epoch = len(train_loader) // ACCUM_STEPS
    if len(train_loader) % ACCUM_STEPS != 0:
        print(f'  [warn] len(train_loader)={len(train_loader)} not divisible by '
              f'ACCUM_STEPS={ACCUM_STEPS}; last {len(train_loader) % ACCUM_STEPS} '
              f'microbatches are trained but never stepped (gradients dropped).')
    scheduler = make_scheduler(optimizer, opt_steps_per_epoch)
    ema       = WeightEMA(model, decay=EMA_DECAY) if EMA_ENABLED else None
    total_opt_steps     = opt_steps_per_epoch * EPOCHS

    # Skip epoch 1 in 1/t fit (same index as first opt step of epoch 2).
    diag.h8_fit_min_step = opt_steps_per_epoch
    print(f'  microbatches/epoch = {len(train_loader)}   '
          f'opt_steps/epoch = {opt_steps_per_epoch}   '
          f'total_opt_steps = {total_opt_steps}   '
          f'H8 min_step = {opt_steps_per_epoch}')
    print(sep + '\n')

    if resume_state is not None and resume_state.get('optimizer') is not None:
        opt_states = resume_state['optimizer']
        if len(opt_states) == len(optimizer.optimizers):
            for i, o in enumerate(optimizer.optimizers):
                o.load_state_dict(opt_states[i])
            print('[resume] restored optimizer state (Muon + SparseMuon + AdamW)')
        else:
            print(f'[resume] checkpoint optimizer length {len(opt_states)} '
                  f'!= current {len(optimizer.optimizers)}; using fresh optimizer')

    # Align LR schedule with resumed CE timeline: fresh LambdaLR starts at
    # last_epoch=-1; without fast-forward, LR would replay warmup after resume.
    if resume_state is not None and diag.ce_history:
        n_sync = len(diag.ce_history)
        print(f'[resume] LR scheduler sync: fast-forward {n_sync} optimizer steps')
        for _ in range(n_sync):
            scheduler.step()
        global_opt_step = n_sync
    else:
        global_opt_step = 0

    for epoch in range(start_epoch, EPOCHS + 1):
        model.train()
        t0              = time.time()
        running_main_ce = 0.0      # cumulative CE of the EPOCH (for log only)
        step_main_ce    = 0.0      # CE accumulated within the CURRENT opt step
        optimizer.zero_grad(set_to_none=True)

        for step, batch in enumerate(train_loader, start=1):
            x, y = [t.to(device, non_blocking=True) for t in batch]
            with amp_autocast():
                out  = model(x, labels=y)
                loss = out.loss / ACCUM_STEPS
            loss.backward()

            mtp_contrib      = (MTP_WEIGHT * float(out.mtp_loss.item())
                                if MTP_ENABLED else 0.0)
            main_ce_micro    = float(out.loss.item()) - mtp_contrib
            step_main_ce    += main_ce_micro
            running_main_ce += main_ce_micro

            if step % ACCUM_STEPS == 0:
                # Pre-step diagnostics (single coherent block per DIAG_FREQ opt steps).
                if global_opt_step % DIAG_FREQ == 0:
                    print(f'\n  ── diagnostics  ·  ep{epoch}  ·  '
                          f'opt_step {global_opt_step}  ──')
                    log_grad_norms(model, step, epoch)
                    diag.log_body_head_ratio(model, step, epoch)
                    diag.log_grad_consistency(step, epoch)
                    if not FREEZE_A_IN:
                        diag.log_ain_grad_by_frequency(model, step, epoch)
                    print()

                torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
                optimizer.step()
                scheduler.step()
                if ema: ema.update(model)
                optimizer.zero_grad(set_to_none=True)

                # Record per-opt-step instantaneous CE for the H8 1/t fit.
                instant_ce = step_main_ce / float(ACCUM_STEPS)
                diag.record_ce(global_opt_step, instant_ce)
                step_main_ce = 0.0
                global_opt_step += 1

            if step % 200 == 0:
                avg_ce_epoch = running_main_ce / step
                lrs          = scheduler.get_last_lr()
                if len(lrs) >= 3:
                    lr_str = (f'lr[muon={lrs[0]:.2e} '
                              f'smuon={lrs[1]:.2e} '
                              f'adamw={lrs[-1]:.2e}]')
                else:
                    lr_str = f'lr={lrs[0]:.2e}'
                kappa_m = out.kappa.mean().item()
                kappa_s = out.kappa.std().item()
                aout    = model.embed.A_out.weight.float().norm(dim=-1)
                aout_m  = aout.mean().item()
                aout_s  = aout.std().item()
                ema_str = ''
                if ema:
                    g_all, g_mat = ema.gap_stats(model)
                    ema_str = f' ema_gap[g={g_all:.2e} m={g_mat:.2e}]'
                print(f'  ep{epoch} step {step:>5}/{len(train_loader)}   '
                      f'ce_avg={avg_ce_epoch:.4f}   {lr_str}   '
                      f'κ={kappa_m:.2f}±{kappa_s:.2f}   '
                      f'|A_out|={aout_m:.3f}±{aout_s:.3f}{ema_str}')

            if step in (200, 1000, 2000, 4000, 8000):
                model.eval()
                log_embedding_alignment(model, step, epoch)
                model.train()

        # ──────────────────────────────────────────────────────────────────
        # END-OF-EPOCH MEASUREMENTS
        # ──────────────────────────────────────────────────────────────────
        # Pause the consistency hooks during eval / per-bucket probe so
        # that downstream backward passes (none, but defensively gated) do
        # not pollute the H3/H5 statistics.
        diag.hooks_enabled = False
        try:
            print('\n' + '─' * 78)
            print(f'  END OF EPOCH {epoch}  ·  measurements')
            print('─' * 78)

            val_metrics = evaluate(model, val_loader, split_name='val')
            test_ppl    = evaluate_ppl_sliding(model, test_tokens)
            train_ce    = running_main_ce / len(train_loader)

            ema_val = None
            if ema:
                ema.swap_to_ema(model)
                ema_val = evaluate(model, val_loader, split_name='val(ema)')
                ema.swap_back(model)

            freq_ce = diag.log_ce_by_frequency(model, val_tokens, epoch)
            ang_results, norm_results, corr_h1, corr_h4 = \
                diag.log_angular_distance(model, epoch)
            log_wout_norm_by_frequency(model, diag, epoch)
            A, C_star = diag.log_1_over_t_fit(epoch)
            log_cstar_vs_T(diag, epoch, train_loader)
            diag.log_grad_consistency(len(train_loader), epoch)
            cos_mean = log_embedding_alignment(model, len(train_loader), epoch)
        finally:
            diag.hooks_enabled = True

        ema_str = ''
        if ema_val:
            delta   = val_metrics['ppl'] - ema_val['ppl']
            ema_str = (f'   ema_val_ppl={ema_val["ppl"]:.2f} (Δ={delta:+.2f})')

        print(f'\n  Epoch {epoch} summary  ·  train_ce={train_ce:.4f}  '
              f'val_ppl={val_metrics["ppl"]:.2f}  '
              f'test_ppl={test_ppl:.2f}{ema_str}  '
              f'cos(A_in,A_out)={cos_mean:.4f}  '
              f'time={time.time()-t0:.1f}s')

        # Full verdict
        diag.epoch_summary(epoch, A, freq_ce, ang_results,
                            corr_h1, corr_h4,
                            val_metrics['ppl'], test_ppl)

        if SAVE_CHECKPOINTS:
            ckpt_path = ARTIFACT_DIR / f'{RUN_NAME}_ep{epoch}.pt'
            torch.save({
                'epoch':        epoch,
                'model':        model.state_dict(),
                'optimizer':    [o.state_dict() for o in optimizer.optimizers],
                'W_aout_init':  diag.W_aout_init,
                'W_ain_init':   diag.W_ain_init,
                'freq':         diag.freq,
                'rank':         diag.rank,
                'ce_history':   diag.ce_history,
                'epoch_results':diag.epoch_results,
                'val_ppl':      val_metrics['ppl'],
                'test_ppl':     test_ppl,
            }, ckpt_path)
            print(f'  [checkpoint] saved → {ckpt_path}')

        torch.cuda.empty_cache()
        gc.collect()

    # ── End of training: remove hooks ────────────────────────────────────────
    diag.remove_hooks()
    print('\n[DONE] All diagnostic hooks removed.')
    return model, diag


# =============================================================================
# SCALING SCAN  —  test  A_floor ∝ V² / T_trained
# =============================================================================
#
#   A_floor  ∝  V² / T   with  T = tokens actually optimized (T_trained).
#
# IMPLEMENTATION (revised — matches v32_zipf_diagnostics.py):
#   • T_corpus = T_frac × |train_tokens|; train_subset = first T_corpus tokens.
#   • Head = V_eff most frequent types on train_subset → BreakthroughMicroTransformerE4Scan
#     (V_eff-way logits; not full-V softmax + masking).
#   • max_opt_steps = ceil(SCAN_TOKEN_PASS_FRAC × T_corpus / tokens_per_opt_step)
#     so T_trained ≈ SCAN_TOKEN_PASS_FRAC × T_corpus across variants.
#   • Regression uses T_trained = max_opt_steps × SCAN_BATCH × SCAN_ACCUM × SEQ_LEN.
#
# =============================================================================

def _train_scan_variant(name, V_eff, T_frac, train_tokens):
    """
    Train one scan variant.

    Returns (A_fit, C_star, n_pts, T_corpus, T_trained, r2_1over_t,
             max_opt_steps, fit_from).
    """
    print('\n' + '─' * 78)
    print(f'  SCAN variant  ·  {name}   V_eff={V_eff}   T_frac={T_frac:.2f}')
    print('─' * 78)

    T_corpus     = int(len(train_tokens) * T_frac)
    train_subset = train_tokens[:T_corpus].contiguous()
    max_opt_steps = _scan_compute_max_steps(T_corpus)
    fit_from      = _scan_compute_fit_from(V_eff, max_opt_steps)
    tok_per_opt   = _scan_tokens_per_opt_step()
    T_trained     = int(max_opt_steps * tok_per_opt)

    freq = torch.bincount(train_subset.long(), minlength=VOCAB_SIZE).float()
    head_token_ids = torch.argsort(freq, descending=True)[:V_eff].long()
    uni = build_unigram_logprob(train_subset)

    print(f'  T_corpus={T_corpus:,}   T_trained≈{T_trained:,}   '
          f'opt_steps={max_opt_steps}   fit_from={fit_from}')
    print(f'  V_eff={V_eff}   V²/T_trained = {(V_eff**2)/max(T_trained,1):.6e}')

    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)
    model = BreakthroughMicroTransformerE4Scan(head_token_ids, uni).to(device)

    optimizer = make_optimizer(model)

    def lr_lambda(opt_step_inner):
        warm = max(1, WARMUP_STEPS // 4)
        if opt_step_inner < warm:
            return float(opt_step_inner + 1) / float(warm)
        prog = (opt_step_inner - warm) / float(max(1, max_opt_steps - warm))
        return 0.5 * (1.0 + math.cos(math.pi * min(max(prog, 0.0), 1.0)))

    scheds = [torch.optim.lr_scheduler.LambdaLR(o, lr_lambda)
              for o in optimizer.optimizers]
    scheduler = _SchedCombo(scheds)

    loader = DataLoader(TokenDataset(train_subset, SEQ_LEN),
                        batch_size=SCAN_BATCH, shuffle=True,
                        num_workers=NUM_WORKERS, pin_memory=True,
                        drop_last=True)
    micros_per_opt = SCAN_ACCUM
    opt_step       = 0
    step_main_ce   = 0.0
    ce_history     = []

    model.train()
    optimizer.zero_grad(set_to_none=True)
    t_start = time.time()

    micro_step = 0
    while opt_step < max_opt_steps:
        for batch in loader:
            if opt_step >= max_opt_steps:
                break
            x, y = [t.to(device, non_blocking=True) for t in batch]

            with amp_autocast():
                out  = model(x, labels=y)
                loss = out.loss / micros_per_opt
            loss.backward()
            step_main_ce += float(out.loss.item())
            micro_step   += 1

            if micro_step % micros_per_opt == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)

                instant_ce = step_main_ce / float(micros_per_opt)
                ce_history.append((opt_step, instant_ce))
                step_main_ce = 0.0
                opt_step += 1

                if opt_step % SCAN_LOG_EVERY == 0:
                    elapsed = time.time() - t_start
                    eta     = elapsed * (max_opt_steps - opt_step) / max(1, opt_step)
                    print(f'    opt_step {opt_step:>5}/{max_opt_steps}   '
                          f'ce={instant_ce:.4f}   '
                          f'lr={scheduler.get_last_lr()[0]:.2e}   '
                          f'elapsed={elapsed:.0f}s   eta={eta:.0f}s')

    data = [(s, c) for (s, c) in ce_history if s >= fit_from]
    if len(data) < 10:
        print(f'    [scan] insufficient post-warmup data for {name} '
              f'(t≥{fit_from}, have {len(data)} points)')
        del model, optimizer, scheduler, loader
        torch.cuda.empty_cache(); gc.collect()
        return None, None, 0, T_corpus, T_trained, float('nan'), max_opt_steps, fit_from

    steps_arr = np.array([d[0] for d in data], dtype=np.float64)
    ce_arr    = np.array([d[1] for d in data], dtype=np.float64)
    X = np.column_stack([1.0 / steps_arr, np.ones_like(steps_arr)])
    coeffs, *_ = np.linalg.lstsq(X, ce_arr, rcond=None)
    A_fit, C_star = float(coeffs[0]), float(coeffs[1])
    pred    = X @ coeffs
    res_std = float(np.std(ce_arr - pred))
    n_pts   = len(data)
    ss_res  = float(np.sum((ce_arr - pred) ** 2))
    ss_tot  = float(np.sum((ce_arr - ce_arr.mean()) ** 2))
    r2_fit  = 1.0 - ss_res / ss_tot if ss_tot > 1e-12 else float('nan')

    print(f'  fit on {n_pts} points (t≥{fit_from}):  '
          f'A={A_fit:.4f}   C*={C_star:.4f}   '
          f'residual σ={res_std:.5f}   R²(1/t)={r2_fit:.4f}')

    del model, optimizer, scheduler, loader
    torch.cuda.empty_cache(); gc.collect()

    return A_fit, C_star, n_pts, T_corpus, T_trained, r2_fit, max_opt_steps, fit_from


def run_scaling_scan(train_tokens):
    """
    Run all SCAN_VARIANTS, fit A per variant, regress log(A) vs log(V²/T_trained).
    Saves a summary .pt to ARTIFACT_DIR.
    """
    sep = '═' * 78
    print('\n' + sep)
    print('  SCALING SCAN  ·  testing   A_floor  ∝  V² / T_trained')
    print(sep)
    print(f'  {len(SCAN_VARIANTS)} variants   '
          f'T_trained ≈ {SCAN_TOKEN_PASS_FRAC:.2f} × T_corpus   '
          f'tok/opt={_scan_tokens_per_opt_step()}')
    print(f'  ACCUM={SCAN_ACCUM}   BATCH={SCAN_BATCH}   '
          f'artifacts → {ARTIFACT_DIR}')
    print(sep)

    results = []
    for (name, V_eff, T_frac) in SCAN_VARIANTS:
        A, Cs, n, T_corpus, T_trained, r2_1t, horizon, ffrom = _train_scan_variant(
            name, V_eff, T_frac, train_tokens)
        results.append({
            'name':           name,
            'V_eff':          V_eff,
            'T_frac':         T_frac,
            'T_corpus':       T_corpus,
            'T_trained':      T_trained,
            'A':              A,
            'C_star':         Cs,
            'n_pts':          n,
            'r2_1over_t':     r2_1t,
            'max_opt_steps':  horizon,
            'fit_from':       ffrom,
        })

    print('\n' + sep)
    print('  SCALING SCAN  ·  per-variant results  (T = T_trained)')
    print(sep)
    print(f'  {"variant":<14} {"V_eff":>7} {"T_corpus":>11} {"T_train":>11} '
          f'{"V²/T":>12} {"A":>10} {"C*":>9} {"n":>6} {"R²(1/t)":>8}')
    print('  ' + '-' * 92)
    for r in results:
        tt = max(r['T_trained'], 1)
        v2t = (r['V_eff'] ** 2) / tt
        a_str = f'{r["A"]:>10.4f}' if r['A'] is not None else f'{"--":>10}'
        c_str = f'{r["C_star"]:>9.4f}' if r['C_star'] is not None else f'{"--":>9}'
        r2v = r['r2_1over_t']
        r2s = f'{r2v:>8.4f}' if r2v == r2v and not math.isinf(r2v) else f'{"--":>8}'
        print(f'  {r["name"]:<14} {r["V_eff"]:>7} {r["T_corpus"]:>11,} '
              f'{r["T_trained"]:>11,} {v2t:>12.4e} {a_str} {c_str} '
              f'{r["n_pts"]:>6} {r2s:>8}')

    valid = [r for r in results if r['A'] is not None and r['A'] > 0]
    regression = None
    verdict    = 'insufficient successful fits (need ≥3 variants with A>0)'

    if len(valid) >= 3:
        log_A  = np.array([math.log(r['A']) for r in valid])
        log_VT = np.array([math.log((r['V_eff'] ** 2) / max(r['T_trained'], 1))
                           for r in valid])
        slope, intercept = np.polyfit(log_VT, log_A, 1)
        pred   = slope * log_VT + intercept
        ss_res = float(np.sum((log_A - pred) ** 2))
        ss_tot = float(np.sum((log_A - log_A.mean()) ** 2))
        r2     = 1.0 - ss_res / ss_tot if ss_tot > 0 else float('nan')
        regression = {'slope': slope, 'intercept': intercept, 'r2_loglog': r2}

        base   = valid[0]
        A_base = base['A']
        V_base = base['V_eff']
        T_base = max(base['T_trained'], 1)

        print('\n' + sep)
        print('  V² / T_trained  LAW  VERIFICATION')
        print(sep)
        print(f'  reference variant : {base["name"]}   A_base={A_base:.4f}   '
              f'V_base={V_base}   T_trained={T_base:,}')
        print(f'  predicted A_v = A_base × (V_v / V_base)² × (T_base / T_v)')
        print()
        print(f'  {"variant":<14} {"A_meas":>10} {"A_pred":>10} '
              f'{"meas/pred":>10} {"log err":>10}')
        print('  ' + '-' * 60)
        for r in valid:
            Tt = max(r['T_trained'], 1)
            A_pred = A_base * (r['V_eff'] / V_base) ** 2 * (T_base / Tt)
            ratio   = r['A'] / A_pred if A_pred > 0 else float('inf')
            log_err = math.log(ratio)
            print(f'  {r["name"]:<14} {r["A"]:>10.4f} {A_pred:>10.4f} '
                  f'{ratio:>9.3f}× {log_err:>+10.4f}')

        print()
        print(f'  log-log regression :   log A  =  {slope:+.4f} · log(V²/T_trained)  +  {intercept:+.4f}')
        print(f'  predicted slope    :   +1.0000')
        print(f'  R² (log-log)       :   {r2:.4f}')
        print(f'  per-variant R²(1/t) above — if V50K rows are low, trust slope less.')

        if 0.8 <= slope <= 1.2:
            verdict = '✓ V²/T LAW CONFIRMED   (slope within ±20% of 1.0)'
        elif 0.5 <= slope <= 1.5:
            verdict = '~ V²/T LAW PARTIAL     (slope inside [0.5, 1.5] but outside [0.8, 1.2])'
        else:
            verdict = '✗ V²/T LAW FALSIFIED   (slope outside [0.5, 1.5])'
        print(f'  verdict            :   {verdict}')
        regression['verdict'] = verdict
    else:
        print('\n  [scan] not enough successful fits to test the V²/T law')

    print(sep)

    payload = {
        'run_name':               RUN_NAME,
        'artifact_dir':           str(ARTIFACT_DIR),
        'mode':                   'scaling_scan',
        'SCAN_TOKEN_PASS_FRAC':   SCAN_TOKEN_PASS_FRAC,
        'SCAN_MIN_OPT_STEPS':     SCAN_MIN_OPT_STEPS,
        'SCAN_MAX_OPT_STEPS_CAP': SCAN_MAX_OPT_STEPS_CAP,
        'SCAN_FIT_FROM':          SCAN_FIT_FROM,
        'SCAN_FIT_FROM_FULL_V':   SCAN_FIT_FROM_FULL_V,
        'variants':               results,
        'regression':             regression,
        'verdict':                verdict,
    }
    out_pt = ARTIFACT_DIR / f'{RUN_NAME}_scaling_scan.pt'
    torch.save(payload, out_pt)
    print(f'\n  [scan] saved summary → {out_pt}\n')

    return results


# =============================================================================
# ENTRY POINT
# =============================================================================

def _print_environment():
    sep = '═' * 78
    print('\n' + sep)
    print(f'  v32  ·  standard softmax comparison  ·  MODE = {MODE!r}')
    print(sep)
    print(f'  device      : {device}')
    if torch.cuda.is_available():
        print(f'  gpu         : {torch.cuda.get_device_name(0)}')
        free, total = torch.cuda.mem_get_info()
        print(f'  gpu memory  : {free/2**30:.1f} / {total/2**30:.1f} GB free')
    print(f'  precision   : {"BF16" if USE_BF16 else "FP16" if USE_FP16 else "FP32"}')
    print(f'  seed        : {SEED}')
    print(f'  run_name    : {RUN_NAME}')
    print(f'  artifacts   : {ARTIFACT_DIR}')
    if MODE == 'main':
        print(f'  HEAD_MODE   : {HEAD_MODE!r}   '
              f'(κ in train/eval logs = ||z|| pre-softmax norm, not vMF concentration)')
        print(f'  FREEZE_A_IN : {FREEZE_A_IN}   '
              f'(set True to run H7 frozen-A_in ablation)')
    elif MODE == 'scaling_scan':
        print(f'  scan        : {len(SCAN_VARIANTS)} variants   '
              f'T_pass≈{SCAN_TOKEN_PASS_FRAC:.2f}×T_corpus   '
              f'ACCUM={SCAN_ACCUM}   BATCH={SCAN_BATCH}')
    print(sep + '\n')


def main():
    if MODE not in ('main', 'scaling_scan'):
        raise ValueError(f"MODE must be 'main' or 'scaling_scan', got {MODE!r}")

    _print_environment()
    _, train_tokens, val_tokens, test_tokens = load_wikitext103()

    if MODE == 'main':
        train_loader, val_loader = build_loaders(train_tokens, val_tokens)
        unigram_logprob          = build_unigram_logprob(train_tokens)
        model = BreakthroughMicroTransformerE4(
            unigram_logprob=unigram_logprob).to(device)
        print(f'  model params : {model.count_params():,}\n')

        resume_state  = None
        start_epoch   = 1
        if RESUME_ENABLED and RESUME_FROM.is_file():
            print(f'[resume] loading checkpoint: {RESUME_FROM}')
            ckpt = torch.load(RESUME_FROM, map_location=device)
            model.load_state_dict(ckpt['model'])
            resume_state = {
                'ce_history':    ckpt['ce_history'],
                'epoch_results': ckpt['epoch_results'],
                'W_aout_init':   ckpt['W_aout_init'],
                'W_ain_init':    ckpt['W_ain_init'],
                'optimizer':     ckpt.get('optimizer'),
            }
            start_epoch = ckpt['epoch'] + 1
            print(f'[resume] model loaded from epoch {ckpt["epoch"]}')
            print(f'[resume] starting from epoch {start_epoch}')
            print(f'[resume] ce_history has '
                  f'{len(ckpt["ce_history"])} points')
            if ckpt.get('optimizer'):
                print('[resume] checkpoint includes optimizer state')
            else:
                print('[resume] checkpoint has no optimizer state (older format); '
                      'Muon/Adam momentum cold — phase-transition C* may be noisier')
        elif RESUME_ENABLED:
            print(f'[resume] RESUME_ENABLED but missing file {RESUME_FROM} — '
                  f'starting from scratch')
        else:
            print('[resume] RESUME_ENABLED=False — fresh softmax baseline '
                  '(no vMF checkpoint)')

        train_model(model, train_loader, val_loader,
                    test_tokens, train_tokens, val_tokens,
                    resume_state=resume_state,
                    start_epoch=start_epoch)
    else:  # MODE == 'scaling_scan'
        run_scaling_scan(train_tokens)

    print('\n[DONE]')


if __name__ == '__main__':
    main()
