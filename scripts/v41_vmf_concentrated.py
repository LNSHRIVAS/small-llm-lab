"""
v41 — vMF k48 + concentrated attention (attention at end of depth)
==================================================================
Default: NNNNFFFF (4 FFN-only → 4 full attn+FFN), rw=0, full bucket eval.
Same training stack as v32/v40: Muon, SparseMuon, MTP, EMA, shortcut, unigram bias.

SOTA run (50K, 9.96M, vMF interleaved, 4 ep — recommended):
  python v41_vmf_concentrated.py --sota

  python v41_vmf_concentrated.py --interleaved --reweight 0 --epochs 4

OWT (500M tok/epoch, compare to Experiments/A_10M.json):
  python v41_vmf_concentrated.py --sota --owt
  # tokens: data/cache/owt_chinchilla_runs/owt_{train,val}_tokens.npy (or OWT_TOKEN_DIR env)

Other:
  python v41_vmf_concentrated.py                              # NNNNFFFF 2 ep
  python v41_vmf_concentrated.py --sota --per-row-aout-lr     # + rare-row SparseMuon LR
  python v41_vmf_concentrated.py --eval-only --checkpoint PATH
  python v41_vmf_concentrated.py --prepare-owt-data           # tokenize OWT only
"""
import os
import gc
import math
import time
import json
import argparse
from pathlib import Path
from types import SimpleNamespace
from datetime import datetime, timezone

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, IterableDataset

import v32_zipf_diagnostics as v32

v32.K_IN = 48
v32.K_OUT = 48
v32.D_MODEL = 224
v32.N_LAYERS = 8
v32.HEAD_MODE = 'vmf'
v32.MTP_ENABLED = True
v32.SHORTCUT_ENABLED = True
v32.OUT_BIAS_ENABLED = True
v32.MUON_ENABLED = True
v32.SPARSE_MUON_ENABLED = True
v32.EMA_ENABLED = True

from v32_zipf_diagnostics import (  # noqa: E402
    TransformerBlock,
    FactorizedEmbeddingE4,
    RMSNorm,
    RotaryEmbedding,
    WeightEMA,
    make_optimizer,
    make_scheduler,
    amp_autocast,
    device,
    USE_BF16,
    D_MODEL,
    N_HEADS,
    D_FF_GATE,
    N_LAYERS,
    DROPOUT,
    ROPE_BASE,
    N_POSITIONS,
    K_IN,
    K_OUT,
    VOCAB_SIZE,
    HEAD_MODE,
    MTP_ENABLED,
    MTP_DEPTH,
    MTP_WEIGHT,
    SHORTCUT_ENABLED,
    SHORTCUT_INIT,
    OUT_BIAS_ENABLED,
    OUT_BIAS_INIT_UNIGRAM,
    MOS_COMPONENTS,
    MOS_ASYMMETRIC_INIT,
    MOS_GATE_BIAS_INIT_SKEW,
    MOS_NONLIN_ON_EXTRA_COMPONENTS,
)

from v39_contrastive_head import (  # noqa: E402
    Tokens,
    load_data,
    build_metadata,
    bucket_dict,
)

# ═══════════════════════════════════════════════════════════════════════
# v41 CONFIG
# ═══════════════════════════════════════════════════════════════════════
LAYER_PATTERN = 'NNNNFFFF'   # N=FFN-only, F=full (attn+FFN)
RARE_REWEIGHT_POWER = 0.0
PER_ROW_AOUT_LR = False      # SparseMuon count-weighted step on embed.A_out rows
MTP_WEIGHT = v32.MTP_WEIGHT
SEQ_LEN = v32.SEQ_LEN
GRAD_CLIP = v32.GRAD_CLIP
WARMUP_STEPS = 300
EPOCHS = 2
LOG_EVERY = 400

BATCH_SIZE = v32.BATCH_SIZE
ACCUM_STEPS = v32.ACCUM_STEPS
EVAL_BATCH_SIZE = 8
NUM_WORKERS = 2

RUN_NAME = 'v41_vmf_concentrated'
OWT_MODE = False
OWT_TOKENS_PER_EPOCH = 500_000_000
OWT_SEED = 42

_SCRIPT_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _SCRIPT_DIR.parent
# Repo-local cache (portfolio copy; no Colab Drive paths)
LOCAL_CACHE_DIR = Path(os.environ.get(
    'WKT103_CACHE_DIR', str(_REPO_ROOT / 'data' / 'cache' / 'wikitext103_gpt2')))
DRIVE_CACHE_DIR = LOCAL_CACHE_DIR
OWT_LOCAL_CACHE_DIR = Path(
    os.environ.get('OWT_V41_LOCAL_CACHE',
                   str(_REPO_ROOT / 'data' / 'cache' / 'owt_vmf_v41'))
)
OWT_DRIVE_CACHE_DIR = Path(
    os.environ.get('OWT_V41_DRIVE_CACHE', str(OWT_LOCAL_CACHE_DIR))
)
# Pre-tokenized OWT cache (from owt_chinchilla_e / A_10M runs)
OWT_TOKEN_DIR = Path(os.environ.get(
    'OWT_TOKEN_DIR', str(_REPO_ROOT / 'data' / 'cache' / 'owt_chinchilla_runs')))
OWT_TRAIN_TOKENS_PATH = OWT_TOKEN_DIR / 'owt_train_tokens.npy'
OWT_VAL_TOKENS_PATH = OWT_TOKEN_DIR / 'owt_val_tokens.npy'


class OWTTokenIterDataset(IterableDataset):
    """Random-window OWT stream (same protocol as owt_chinchilla_e.TokenIterDataset)."""

    def __init__(self, tokens, seq_len, seed=0):
        self.tokens = tokens
        self.seq_len = seq_len
        self.seed = seed

    def __iter__(self):
        rng = np.random.default_rng(self.seed)
        n = len(self.tokens) - self.seq_len - 1
        while True:
            i = int(rng.integers(0, n))
            x = np.asarray(self.tokens[i:i + self.seq_len], dtype=np.int64)
            y = np.asarray(self.tokens[i + 1:i + 1 + self.seq_len], dtype=np.int64)
            yield torch.from_numpy(x), torch.from_numpy(y)


def _assert_owt_train_sufficient(train_tokens):
    n = len(train_tokens)
    if n < OWT_TOKENS_PER_EPOCH:
        raise ValueError(
            f'OWT train buffer too small: {n:,} < {OWT_TOKENS_PER_EPOCH:,} tokens/epoch'
        )
    if n < SEQ_LEN + 8:
        raise ValueError(f'OWT train buffer too short for seq_len={SEQ_LEN}')

V32_INTERLEAVED_PARAMS = 9_958_162

REF_BASELINES = {
    'A_10M_ep4_OWT': {
        'source': 'A_10M.json ep4 (OWT, std softmax d=144, ~2B tok)',
        'corpus': 'OWT',
        'params': 10_165_681,
        'pattern': 'FFFFFFFF',
        'k_out': 0, 'reweight': 0.0,
        'fair': False,
        'val': {
            'top_100': 2.116, '100_1K': 4.529, '1K_10K': 5.880,
            '10K_30K': 7.112, 'tail_30K+': 8.172, 'ppl': 57.75,
        },
    },
    'A_10M_ep1_OWT': {
        'source': 'A_10M.json ep1 (OWT, std softmax, ~500M tok)',
        'corpus': 'OWT',
        'params': 10_165_681,
        'pattern': 'FFFFFFFF',
        'k_out': 0, 'reweight': 0.0,
        'fair': False,
        'val': {
            'top_100': 2.195, '100_1K': 4.696, '1K_10K': 6.210,
            '10K_30K': 7.690, 'tail_30K+': 9.004, 'ppl': 71.33,
        },
    },
    'v41_FFFFFFFF_ep4_WT103': {
        'source': 'v41_vmf_FFFFFFFF_k48_rw0_d224_ep4 (WT103 champion)',
        'corpus': 'WT103',
        'params': 9_958_162,
        'pattern': 'FFFFFFFF',
        'k_out': 48, 'reweight': 0.0,
        'fair': True,
        'val': {
            'top_100': 1.89, '100_1K': 4.46, '1K_10K': 5.76,
            '10K_30K': 7.36, 'tail_30K+': 8.52, 'ppl': 46.84,
        },
        'test_ppl': 46.16,
    },
    'v32_std_ep2': {
        'source': 'standard_softmax.ipynb ep2 (WT103 linear head, FFFFFFFF)',
        'corpus': 'WT103',
        'params': 9_958_162,
        'pattern': 'FFFFFFFF',
        'k_out': 48, 'reweight': 0.0,
        'fair': True,
        'val': {
            'top_100': 3.15, '100_1K': 5.03, '1K_10K': 6.79,
            '10K_30K': 8.35, 'tail_30K+': 10.42, 'ppl': 57.05,
        },
        'test_ppl': 52.80,
    },
    'v41_NNNNFFFF_ep2': {
        'source': 'v41_vmf_NNNNFFFF_k48_rw0_d224 ep2',
        'params': 9_154_002,
        'pattern': 'NNNNFFFF',
        'k_out': 48, 'reweight': 0.0,
        'fair': False,
        'val': {
            'top_100': 1.97, '100_1K': 4.63, '1K_10K': 6.02,
            '10K_30K': 7.71, 'tail_30K+': 8.99, 'ppl': 55.24,
        },
        'test_ppl': 53.88,
    },
    'vmf_interleaved_ep8_ref': {
        'source': 'vmf_exp_ep3_ep8.ipynb ep8 (NOT ep2 — reference only)',
        'params': 9_958_162,
        'pattern': 'FFFFFFFF',
        'k_out': 48, 'reweight': 0.0,
        'fair': False,
        'val': {'top_100': None, 'tail_30K+': None, 'ppl': 42.72},
        'test_ppl': 39.20,
    },
    'v39_k48_rw0p5': {
        'source': 'v39 flat k48 rw0.5 (no vMF stack)',
        'params': 9_947_411,
        'pattern': 'flat',
        'k_out': 48, 'reweight': 0.5,
        'fair': True,
        'val': {'top_100': None, 'tail_30K+': 6.45, 'ppl': 65.77},
    },
    'v40_vmf_banded_k48_rw0p5': {
        'source': 'v40 ep2 banded rw (interleaved vMF)',
        'params': 9_958_162,
        'pattern': 'FFFFFFFF',
        'k_out': 48, 'reweight': 0.5,
        'fair': True,
        'val': {'top_100': 2.36, 'tail_30K+': 6.98, 'ppl': 75.61},
    },
}

BUCKET_DEFS = [
    ('top_100', lambda r: r < 100),
    ('100_1K', lambda r: (r >= 100) & (r < 1000)),
    ('1K_10K', lambda r: (r >= 1000) & (r < 10000)),
    ('10K_30K', lambda r: (r >= 10000) & (r < 30000)),
    ('tail_30K+', lambda r: r >= 30000),
]

WIN_PPL = 50.0
WIN_TAIL = 6.5


def active_cache_dirs():
    if OWT_MODE:
        return OWT_DRIVE_CACHE_DIR, OWT_LOCAL_CACHE_DIR
    return DRIVE_CACHE_DIR, LOCAL_CACHE_DIR


def unigram_counts_mmap(tokens, max_tokens=None):
    """Bincount unigram frequencies from a numpy mmap without loading all RAM."""
    counts = np.zeros(VOCAB_SIZE, dtype=np.int64)
    n = len(tokens) if max_tokens is None else min(len(tokens), max_tokens)
    chunk = 50_000_000
    for i in range(0, n, chunk):
        sl = np.asarray(tokens[i:i + chunk], dtype=np.int64)
        counts += np.bincount(sl, minlength=VOCAB_SIZE)
    return torch.from_numpy(counts)


def rank_and_logp_from_counts(counts):
    sorted_ids = torch.argsort(counts, descending=True)
    rank = torch.empty(VOCAB_SIZE, dtype=torch.long)
    rank[sorted_ids] = torch.arange(VOCAB_SIZE)
    eps = 1.0
    p = (counts.float() + eps) / (counts.sum() + eps * VOCAB_SIZE)
    return rank, p.log()


def load_owt_data():
    train_path = Path(os.environ.get('OWT_TRAIN_TOKENS', OWT_TRAIN_TOKENS_PATH))
    val_path = Path(os.environ.get('OWT_VAL_TOKENS', OWT_VAL_TOKENS_PATH))

    if not train_path.is_file():
        raise FileNotFoundError(
            f'OWT train tokens missing: {train_path}\n'
            f'  Expected default: {OWT_TRAIN_TOKENS_PATH}'
        )
    if not val_path.is_file():
        raise FileNotFoundError(
            f'OWT val tokens missing: {val_path}\n'
            f'  Expected alongside train: {OWT_VAL_TOKENS_PATH}'
        )

    train_np = np.load(train_path, mmap_mode='r')
    val_np = np.load(val_path, mmap_mode='r')
    _assert_owt_train_sufficient(train_np)
    print(f'[owt] train={train_path}  ({len(train_np):,} tokens)')
    print(f'[owt] val={val_path}  ({len(val_np):,} tokens)')
    print(f'[owt] tokens/epoch={OWT_TOKENS_PER_EPOCH:,}')
    counts = unigram_counts_mmap(train_np)
    rank, unigram_logp = rank_and_logp_from_counts(counts)
    val_t = torch.from_numpy(np.asarray(val_np, dtype=np.int64))
    return train_np, val_t, None, rank, unigram_logp


def load_corpus_data():
    if OWT_MODE:
        return load_owt_data()
    _, train, val, test = load_data()
    _, _, _, rank, unigram_logp = build_metadata(train)
    return train, val, test, rank, unigram_logp


def make_owt_train_loader(train_tokens, epoch):
    ds = OWTTokenIterDataset(train_tokens, SEQ_LEN, seed=OWT_SEED + epoch)
    return DataLoader(
        ds, batch_size=BATCH_SIZE, num_workers=NUM_WORKERS,
        pin_memory=torch.cuda.is_available(), drop_last=True,
        persistent_workers=NUM_WORKERS > 0,
    )


def opt_steps_per_epoch_for_corpus(train_tokens=None, train_loader=None):
    tok_per_step = BATCH_SIZE * ACCUM_STEPS * SEQ_LEN
    if OWT_MODE:
        return OWT_TOKENS_PER_EPOCH // tok_per_step
    return len(train_loader) // ACCUM_STEPS


def microbatches_per_epoch(opt_steps):
    return opt_steps * ACCUM_STEPS


class FFNOnlyBlock(nn.Module):
    """FFN-only layer (no cross-token attention)."""

    def __init__(self, d_model, d_ff_gate, dropout, n_layers):
        super().__init__()
        self.norm_ffn = RMSNorm(d_model)
        self.w_gate = nn.Linear(d_model, d_ff_gate, bias=False)
        self.w_up = nn.Linear(d_model, d_ff_gate, bias=False)
        self.w_down = nn.Linear(d_ff_gate, d_model, bias=False)
        self.dropout = nn.Dropout(dropout)
        scale = 1.0 / math.sqrt(2 * n_layers)
        nn.init.normal_(self.w_gate.weight, 0.0, 0.02)
        nn.init.normal_(self.w_up.weight, 0.0, 0.02)
        nn.init.normal_(self.w_down.weight, 0.0, 0.02 * scale)

    def ffn(self, x):
        return self.w_down(self.dropout(F.silu(self.w_gate(x)) * self.w_up(x)))

    def forward(self, x):
        return x + self.dropout(self.ffn(self.norm_ffn(x)))


def parse_layer_pattern(pattern):
    p = pattern.strip().upper()
    if len(p) != N_LAYERS:
        raise ValueError(f'layer-pattern must have {N_LAYERS} chars, got {len(p)}: {pattern!r}')
    for ch in p:
        if ch not in ('N', 'F'):
            raise ValueError(f'layer-pattern chars must be N (FFN-only) or F (full); got {ch!r}')
    return p


def build_blocks(layer_pattern, rope):
    blocks = []
    for ch in layer_pattern:
        if ch == 'N':
            blocks.append(FFNOnlyBlock(D_MODEL, D_FF_GATE, DROPOUT, N_LAYERS))
        else:
            blocks.append(
                TransformerBlock(D_MODEL, N_HEADS, D_FF_GATE, DROPOUT, N_LAYERS, rope)
            )
    return nn.ModuleList(blocks)


class ConcentratedMicroTransformerE4(nn.Module):
    """v32 BreakthroughMicroTransformerE4 with configurable per-layer block type."""

    def __init__(self, unigram_logprob=None, layer_pattern=LAYER_PATTERN):
        super().__init__()
        self.layer_pattern = parse_layer_pattern(layer_pattern)
        self.embed = FactorizedEmbeddingE4(VOCAB_SIZE, D_MODEL, K_IN, K_OUT)
        self.rope = RotaryEmbedding(D_MODEL // N_HEADS, N_POSITIONS, ROPE_BASE)
        self.in_dropout = nn.Dropout(DROPOUT)
        self.blocks = build_blocks(self.layer_pattern, self.rope)
        self.norm_final = RMSNorm(D_MODEL)
        self.n_mos = max(1, int(MOS_COMPONENTS))
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

    def forward_features(self, ids, is_causal=True):
        h = self.in_dropout(self.embed.embed(ids))
        for blk in self.blocks:
            if isinstance(blk, TransformerBlock):
                h = blk(h, is_causal=is_causal)
            else:
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
            a_in_t = self.embed.A_in(ids).float()
            a_out_mat = self.embed.A_out.weight.float()
            bigram_scaled = self.shortcut_scale.float() * (a_in_t @ a_out_mat.T)
        else:
            bigram_scaled = None

        if self.n_mos == 1:
            logits, z = self._component_logits(
                h, self.out_to_k_list[0], head_mode, kappa_cap,
            )
            if bigram_scaled is not None:
                logits = logits + bigram_scaled
            loss = None
            mtp_loss_val = torch.zeros((), device=h.device)
            if labels is not None:
                loss = F.cross_entropy(
                    logits.view(-1, VOCAB_SIZE), labels.view(-1), weight=class_weight,
                )
                if MTP_ENABLED and self.training and labels.shape[1] > MTP_DEPTH:
                    shift = MTP_DEPTH - 1
                    t_mtp = labels.shape[1] - shift
                    mtp_labels = labels[:, shift:].contiguous()
                    h_mtp = h[:, :t_mtp, :]
                    z_mtp = self.out_to_k_mtp(h_mtp).float()
                    mtp_logits = self.embed.decode_scores(
                        z_mtp, kappa_cap=kappa_cap, mode=head_mode,
                    )
                    mtp_logits = mtp_logits + self.out_bias.float()
                    mtp_loss_val = F.cross_entropy(
                        mtp_logits.view(-1, VOCAB_SIZE), mtp_labels.view(-1),
                    )
                    loss = loss + MTP_WEIGHT * mtp_loss_val
            kappa = z.norm(dim=-1, keepdim=True)
            return SimpleNamespace(
                loss=loss, z=z, kappa=kappa, logits=logits,
                mos_div=torch.zeros((), device=h.device),
                mtp_loss=mtp_loss_val.detach(),
            )

        log_pi = F.log_softmax(self.mos_gate(h).float(), dim=-1)
        per_component_logp = []
        z_first = None
        for i, proj in enumerate(self.out_to_k_list):
            nonlin_i = bool(MOS_NONLIN_ON_EXTRA_COMPONENTS) and (i > 0)
            logits_i, z_i = self._component_logits(
                h, proj, head_mode, kappa_cap, apply_nonlin=nonlin_i,
            )
            if bigram_scaled is not None:
                logits_i = logits_i + bigram_scaled
            if i == 0:
                z_first = z_i
            per_component_logp.append(
                log_pi[..., i:i + 1] + F.log_softmax(logits_i, dim=-1),
            )
        log_p = torch.logsumexp(torch.stack(per_component_logp, dim=0), dim=0)
        kappa = z_first.norm(dim=-1, keepdim=True)
        loss = (
            F.nll_loss(log_p.view(-1, VOCAB_SIZE), labels.view(-1))
            if labels is not None else None
        )
        return SimpleNamespace(
            loss=loss, z=z_first, kappa=kappa, logits=log_p,
            mos_div=torch.zeros((), device=h.device),
            mtp_loss=torch.zeros((), device=h.device),
        )

    def count_params(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


def tune_for_gpu(fast=True):
    global BATCH_SIZE, ACCUM_STEPS, EVAL_BATCH_SIZE, NUM_WORKERS, LOG_EVERY
    if not fast or not torch.cuda.is_available():
        return
    torch.backends.cudnn.benchmark = True
    mem_gb = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
    if mem_gb >= 38:
        BATCH_SIZE, ACCUM_STEPS = 16, 4
        EVAL_BATCH_SIZE = 32
        NUM_WORKERS = 4
        LOG_EVERY = 400
        print(f'[gpu] A100 tune: train_bs={BATCH_SIZE}×accum={ACCUM_STEPS} '
              f'eval_bs={EVAL_BATCH_SIZE} workers={NUM_WORKERS}')


def run_name_for(layer_pattern, k_out, reweight, per_row_aout_lr=False, epochs=2, owt=False):
    pat = layer_pattern.upper()
    rw = f'{reweight:g}'.replace('.', 'p')
    rw_tag = 'rw0' if reweight <= 0.0 else f'rw{rw}'
    name = f'v41_vmf_{pat}_k{k_out}_{rw_tag}_d{v32.D_MODEL}'
    if epochs != 2:
        name += f'_ep{epochs}'
    if owt:
        name += '_owt'
    if per_row_aout_lr:
        name += '_aoutpr'
    return name


def apply_run_config(layer_pattern=None, k_in=None, k_out=None,
                     reweight=None, run_name=None, per_row_aout_lr=None,
                     epochs=None, owt=None):
    global LAYER_PATTERN, RARE_REWEIGHT_POWER, RUN_NAME, PER_ROW_AOUT_LR, EPOCHS, OWT_MODE
    if layer_pattern is not None:
        LAYER_PATTERN = parse_layer_pattern(layer_pattern)
    if k_out is not None:
        v32.K_OUT = k_out
        v32.K_IN = k_in if k_in is not None else k_out
    elif k_in is not None:
        v32.K_IN = k_in
        v32.K_OUT = k_in
    if reweight is not None:
        RARE_REWEIGHT_POWER = reweight
    if per_row_aout_lr is not None:
        PER_ROW_AOUT_LR = bool(per_row_aout_lr)
        v32.SPARSE_MUON_PER_ROW_LR_A_OUT = PER_ROW_AOUT_LR
    if epochs is not None:
        EPOCHS = int(epochs)
    if owt is not None:
        OWT_MODE = bool(owt)
    if run_name:
        RUN_NAME = run_name
    else:
        RUN_NAME = run_name_for(
            LAYER_PATTERN, v32.K_OUT, RARE_REWEIGHT_POWER,
            per_row_aout_lr=PER_ROW_AOUT_LR, epochs=EPOCHS, owt=OWT_MODE,
        )


def count_params_configured():
    return ConcentratedMicroTransformerE4(
        unigram_logprob=None, layer_pattern=LAYER_PATTERN,
    ).count_params()


def build_model(unigram_logp):
    return ConcentratedMicroTransformerE4(
        unigram_logprob=unigram_logp, layer_pattern=LAYER_PATTERN,
    ).to(device)


def print_config_banner():
    tok_per_step = BATCH_SIZE * ACCUM_STEPS * SEQ_LEN
    n_ffn = LAYER_PATTERN.count('N')
    n_full = LAYER_PATTERN.count('F')
    n_params = count_params_configured()
    print('\n' + '═' * 78)
    print('  v41 — vMF + concentrated attention')
    print('═' * 78)
    print(f'  arch        : {N_LAYERS}L  d={D_MODEL}  k_in={K_IN}  k_out={K_OUT}')
    print(f'  pattern     : {LAYER_PATTERN}  ({n_ffn} FFN-only + {n_full} full attn+FFN)')
    print(f'  head        : vMF  rw={RARE_REWEIGHT_POWER}  (standard CE)')
    aout_lr = 'on' if PER_ROW_AOUT_LR else 'off'
    print(f'  optimizer   : Muon(body) + SparseMuon(A_in/A_out) + AdamW(bias/norm)')
    print(f'  A_out per-row LR (SparseMuon): {aout_lr}')
    print(f'  extras      : MTP d={MTP_DEPTH} w={MTP_WEIGHT}  EMA={v32.EMA_DECAY}  '
          f'shortcut={SHORTCUT_ENABLED}  unigram_bias={OUT_BIAS_ENABLED}')
    print(f'  schedule    : cosine  warmup={WARMUP_STEPS}  epochs={EPOCHS}')
    corpus = 'OWT' if OWT_MODE else 'WikiText-103'
    if OWT_MODE:
        print(f'  corpus      : {corpus}  ({OWT_TOKENS_PER_EPOCH:,} tok/epoch)')
        print(f'  owt tokens  : {OWT_TRAIN_TOKENS_PATH}')
    else:
        print(f'  corpus      : {corpus}')
    print(f'  batch       : {BATCH_SIZE}×accum={ACCUM_STEPS}  seq={SEQ_LEN}  '
          f'→ {tok_per_step:,} tok/opt-step')
    print(f'  params      : {n_params:,}  (v32 interleaved ref={V32_INTERLEAVED_PARAMS:,})')
    delta = n_params - V32_INTERLEAVED_PARAMS
    if delta:
        print(f'  param delta : {delta:+,} vs interleaved FFFFFFFF')
    print('═' * 78 + '\n')


def print_param_breakdown(model):
    parts = {}
    n_ffn_only = sum(1 for blk in model.blocks if isinstance(blk, FFNOnlyBlock))
    n_full = len(model.blocks) - n_ffn_only
    for n, p in model.named_parameters():
        if not p.requires_grad:
            continue
        key = n.split('.')[0]
        parts[key] = parts.get(key, 0) + p.numel()
    total = sum(parts.values())
    print(f'  [param-breakdown] total={total:,}  blocks: {n_ffn_only} FFN-only, {n_full} full')
    for k in sorted(parts, key=lambda x: -parts[x]):
        print(f'    {k}: {parts[k]:,}')
    return parts


@torch.no_grad()
def diag_head(model, rank):
    W = model.embed.A_out.weight.detach().float().cpu()
    norm = W.norm(dim=-1)
    rank_cpu = rank.cpu() if rank.is_cuda else rank
    samp = torch.randperm(VOCAB_SIZE, generator=torch.Generator().manual_seed(42))[:4000]
    log_r = -(rank_cpu[samp].float() + 1).log()
    corr = torch.corrcoef(torch.stack([norm[samp], log_r]))[0, 1].item()
    buckets = {}
    for name, mask_fn in BUCKET_DEFS:
        m = mask_fn(rank_cpu)
        if m.any():
            buckets[name] = norm[m].mean().item()
    top = buckets.get('top_100', float('nan'))
    tail = buckets.get('tail_30K+', float('nan'))
    ratio = tail / top if top > 0 else float('nan')
    print(f'  [head-diag] Pearson(||A_out||, -log rank) = {corr:.4f}')
    print(f'  [head-diag] ||A_out|| by bucket:')
    for name, v in buckets.items():
        print(f'    {name}: {v:.4f}')
    print(f'  [head-diag] tail/top norm ratio = {ratio:.3f}')
    return {'pearson_norm_rank': corr, 'norm_buckets': buckets, 'tail_top_norm_ratio': ratio}


@torch.no_grad()
def eval_split(model, tokens, rank, split_name='val', batch_size=None):
    model.eval()
    bs = batch_size or EVAL_BATCH_SIZE
    rank_dev = rank.to(device) if not rank.is_cuda else rank
    bucket_ce = torch.zeros(len(BUCKET_DEFS))
    bucket_n = torch.zeros(len(BUCKET_DEFS))
    total_nll = total_n = 0.0
    loader = DataLoader(
        Tokens(tokens, SEQ_LEN), bs, shuffle=False,
        num_workers=NUM_WORKERS, pin_memory=True, drop_last=False,
        persistent_workers=NUM_WORKERS > 0,
    )
    for x, y in loader:
        x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
        with amp_autocast():
            out = model(x, labels=None)
        logits = out.logits.float().reshape(-1, VOCAB_SIZE)
        yf = y.reshape(-1)
        nll = F.cross_entropy(logits, yf, reduction='none')
        total_nll += nll.sum().item()
        total_n += yf.numel()
        for i, (_, mask_fn) in enumerate(BUCKET_DEFS):
            m = mask_fn(rank_dev[yf])
            if m.any():
                bucket_ce[i] += nll[m].sum().item()
                bucket_n[i] += m.sum().item()

    print(f'\n  [{split_name}] per-bucket CE  (n={int(total_n):,} positions)')
    for i, (name, _) in enumerate(BUCKET_DEFS):
        ce = bucket_ce[i] / bucket_n[i].clamp_min(1)
        print(f'    {name}: CE={ce:.4f} (n={int(bucket_n[i])})')
    agg_ce = total_nll / max(total_n, 1)
    top_ce = (bucket_ce[0] / bucket_n[0].clamp_min(1)).item()
    tail_ce = (bucket_ce[-1] / bucket_n[-1].clamp_min(1)).item()
    tt_ratio = tail_ce / top_ce if top_ce > 0 else float('nan')
    print(f'    AGG: CE={agg_ce:.4f}  PPL={math.exp(agg_ce):.2f}  '
          f'tail/top CE ratio={tt_ratio:.2f}×')
    ce_vec = bucket_ce / bucket_n.clamp_min(1)
    return {
        'ce': agg_ce,
        'ppl': math.exp(agg_ce),
        'n_positions': int(total_n),
        'tail_top_ce_ratio': tt_ratio,
        'buckets': bucket_dict(ce_vec, bucket_n),
    }


def print_comparison_table(result):
    corpus = 'OWT' if OWT_MODE else 'WT103'
    if OWT_MODE:
        baseline_order = [
            'A_10M_ep4_OWT', 'A_10M_ep1_OWT',
            'v41_FFFFFFFF_ep4_WT103', 'v32_std_ep2',
        ]
    else:
        baseline_order = [
            'v32_std_ep2', 'v41_FFFFFFFF_ep4_WT103',
            'A_10M_ep4_OWT', 'A_10M_ep1_OWT',
        ]
    ordered_keys = [k for k in baseline_order if k in REF_BASELINES]
    ordered_keys += [k for k in REF_BASELINES if k not in ordered_keys]

    rows = []
    for key in ordered_keys:
        b = REF_BASELINES[key]
        v = b['val']
        tail, top, ppl = v.get('tail_30K+'), v.get('top_100'), v.get('ppl')
        ratio = (tail / top) if (tail and top) else None
        fair = '✓' if b.get('fair', True) else '~'
        rows.append((
            key, b.get('corpus', '—'), b.get('pattern', '—'), b['k_out'], b['reweight'],
            b.get('params'), fair, top, tail, ratio, ppl,
        ))

    v = result['val']
    tail = v['buckets']['tail_30K+']['ce']
    top = v['buckets']['top_100']['ce']
    rows.append((
        result['run_name'], corpus, result['layer_pattern'], result['k_out'],
        result['reweight'], result.get('n_params'), '✓',
        top, tail, tail / top, v['ppl'],
    ))

    print('\n' + '=' * 126)
    print(f'  v41 vs baselines  ({corpus}, full val buckets)')
    print('=' * 126)
    print(f'  {"run":<36} {"corp":<6} {"pat":<10} {"k":>3} {"rw":>4} {"params":>10} {"ok":>3} '
          f'{"top":>6} {"tail":>6} {"t/t":>5} {"valPPL":>7}')
    print('  ' + '-' * 112)
    for name, corp, pat, k, rw, params, fair, top, tail, ratio, ppl in rows:
        top_s = f'{top:.2f}' if top is not None else '—'
        tail_s = f'{tail:.2f}' if tail is not None else '—'
        ratio_s = f'{ratio:.2f}' if ratio is not None else '—'
        ppl_s = f'{ppl:.2f}' if ppl is not None else '—'
        p_s = f'{params/1e6:.2f}M' if params else '—'
        print(f'  {name:<36} {corp:<6} {pat:<10} {k!s:>3} {rw!s:>4} {p_s:>10} {fair:>3} '
              f'{top_s:>6} {tail_s:>6} {ratio_s:>5} {ppl_s:>7}')
    print('=' * 126)


def check_win(result):
    tail = result['val']['buckets']['tail_30K+']['ce']
    ppl = result['val']['ppl']
    test_block = result.get('test') or {}
    test_ppl = test_block.get('ppl')
    ok = tail <= WIN_TAIL and ppl <= WIN_PPL
    print(f'\n  [win-check] tail_30K+={tail:.2f} (≤{WIN_TAIL})  '
          f'val PPL={ppl:.2f} (≤{WIN_PPL})  →  {"PASS ✓" if ok else "MISS"}')
    if test_ppl is not None:
        test_tail = test_block['buckets']['tail_30K+']['ce']
        print(f'  [test]      tail_30K+={test_tail:.2f}  test PPL={test_ppl:.2f}')
    return ok


def _batch_tail_ce(logits, y_flat, rank):
    """Instantaneous tail_30K+ CE on current microbatch (training monitor)."""
    rank_dev = rank.to(logits.device) if rank.device != logits.device else rank
    m = rank_dev[y_flat] >= 30000
    if not m.any():
        return None
    return F.cross_entropy(logits[m], y_flat[m]).item()


def ckpt_path(run_name, epoch=2):
    for d in active_cache_dirs():
        p = d / f'{run_name}_ep{epoch}.pt'
        if p.exists():
            return p
    drive, _ = active_cache_dirs()
    return drive / f'{run_name}_ep{epoch}.pt'


def train_run(train_tokens, val_tokens, test_tokens, rank, unigram_logp):
    opt_steps_ep = None
    if OWT_MODE:
        opt_steps_ep = opt_steps_per_epoch_for_corpus()
        micro_per_ep = microbatches_per_epoch(opt_steps_ep)
        print(f'  [train] OWT fixed epoch: {opt_steps_ep} opt-steps  '
              f'({micro_per_ep} microbatches)  '
              f'→ {opt_steps_ep * BATCH_SIZE * ACCUM_STEPS * SEQ_LEN:,} tok/ep')
    else:
        train_loader = DataLoader(
            Tokens(train_tokens, SEQ_LEN), BATCH_SIZE, shuffle=True,
            num_workers=NUM_WORKERS, pin_memory=True, drop_last=True,
            persistent_workers=NUM_WORKERS > 0,
        )
        opt_steps_ep = opt_steps_per_epoch_for_corpus(train_loader=train_loader)
        micro_per_ep = len(train_loader)
        print(f'  [train] microbatches/epoch={len(train_loader)}  opt_steps/epoch={opt_steps_ep}')

    model = build_model(unigram_logp)
    n_params = model.count_params()
    print_param_breakdown(model)

    optimizer = make_optimizer(model)
    scheduler = make_scheduler(optimizer, opt_steps_ep)
    ema = WeightEMA(model, decay=v32.EMA_DECAY) if v32.EMA_ENABLED else None

    epoch_times = []
    val_metrics = None
    epoch_summaries = []
    t_train0 = time.time()
    global_opt_step = 0
    tok_per_step = BATCH_SIZE * ACCUM_STEPS * SEQ_LEN

    for epoch in range(1, EPOCHS + 1):
        model.train()
        t0 = time.time()
        running = 0.0
        optimizer.zero_grad(set_to_none=True)

        if OWT_MODE:
            loader = make_owt_train_loader(train_tokens, epoch)
            batch_iter = iter(loader)
            step_range = range(1, micro_per_ep + 1)
        else:
            batch_iter = iter(train_loader)
            step_range = range(1, micro_per_ep + 1)

        for step in step_range:
            x, y = next(batch_iter)
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            with amp_autocast():
                out = model(x, labels=y)
                loss = out.loss / ACCUM_STEPS
            loss.backward()
            running += out.loss.item()

            if step % ACCUM_STEPS == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
                optimizer.step()
                scheduler.step()
                if ema:
                    ema.update(model)
                optimizer.zero_grad(set_to_none=True)
                global_opt_step += 1

            if step % LOG_EVERY == 0:
                mtp = getattr(out, 'mtp_loss', None)
                mtp_v = mtp.item() if mtp is not None else 0.0
                kappa_m = out.kappa.mean().item()
                lrs = scheduler.get_last_lr()
                lr_str = f'muon={lrs[0]:.2e}' if lrs else ''
                logits_flat = out.logits.float().reshape(-1, VOCAB_SIZE)
                y_flat = y.reshape(-1)
                tail_b = _batch_tail_ce(logits_flat, y_flat, rank)
                tail_s = f' batch_tail={tail_b:.3f}' if tail_b is not None else ''
                print(f'  step {step:>5}/{micro_per_ep} ep{epoch} opt={global_opt_step} '
                      f'ce={running/step:.4f} mtp={mtp_v:.3f} κ={kappa_m:.2f} {lr_str}{tail_s}')

        epoch_times.append(time.time() - t0)
        t_tokens = epoch * opt_steps_ep * tok_per_step
        print(f'\n  END EP{epoch}  train_time={epoch_times[-1]:.0f}s  T_tokens={t_tokens:,}')
        val_metrics = eval_split(model, val_tokens, rank, split_name='val')

        ema_val = None
        if ema:
            ema.swap_to_ema(model)
            print('  [ema] val eval with EMA weights:')
            ema_val = eval_split(model, val_tokens, rank, split_name='val(ema)')
            ema.swap_back(model)

        head_diag = diag_head(model, rank)
        ep_summary = {
            'epoch': epoch,
            'T_tokens': t_tokens,
            'train_time_s': epoch_times[-1],
            'val': val_metrics,
            'val_ema': ema_val,
            'head_diag': head_diag,
        }
        epoch_summaries.append(ep_summary)

        ckpt = {
            'epoch': epoch,
            'model': model.state_dict(),
            'val_ce': val_metrics['ce'],
            'val_tail_ce': val_metrics['buckets']['tail_30K+']['ce'],
            'val_ppl': val_metrics['ppl'],
            'head_diag': head_diag,
            'run_name': RUN_NAME,
            'layer_pattern': LAYER_PATTERN,
            'k_in': v32.K_IN,
            'k_out': v32.K_OUT,
            'rare_reweight_power': RARE_REWEIGHT_POWER,
            'per_row_aout_lr': PER_ROW_AOUT_LR,
            'head_mode': 'vmf',
            'n_params': n_params,
            'corpus': 'OWT' if OWT_MODE else 'WT103',
            'T_tokens': t_tokens,
        }
        if ema_val:
            ckpt['val_ema_ppl'] = ema_val['ppl']
        for d in active_cache_dirs():
            d.mkdir(parents=True, exist_ok=True)
            path = d / f'{RUN_NAME}_ep{epoch}.pt'
            torch.save(ckpt, path)
            print(f'  [checkpoint] → {path}')

    test_metrics = None
    if test_tokens is not None:
        test_metrics = eval_split(model, test_tokens, rank, split_name='test')
    else:
        print('\n  [test] skipped (no held-out test split for OWT)')
    head_diag_final = diag_head(model, rank)
    total_s = time.time() - t_train0
    print(f'\n  [timing] total={total_s/3600:.2f}h  epochs={[f"{t:.0f}s" for t in epoch_times]}')

    result = {
        'run_name': RUN_NAME,
        'corpus': 'OWT' if OWT_MODE else 'WT103',
        'layer_pattern': LAYER_PATTERN,
        'k_in': v32.K_IN,
        'k_out': v32.K_OUT,
        'reweight': RARE_REWEIGHT_POWER,
        'per_row_aout_lr': PER_ROW_AOUT_LR,
        'head_mode': 'vmf',
        'n_params': n_params,
        'val': val_metrics,
        'test': test_metrics,
        'head_diag': head_diag_final,
        'epoch_summaries': epoch_summaries,
        'epoch_times_s': epoch_times,
        'total_time_s': total_s,
        'trained': True,
    }
    check_win(result)
    print_comparison_table(result)
    return result


def eval_only(checkpoint=None, owt=None):
    path = Path(checkpoint) if checkpoint else ckpt_path(RUN_NAME)
    if not path.exists():
        raise FileNotFoundError(f'Checkpoint not found: {path}')
    ckpt = torch.load(path, map_location=device, weights_only=False)
    ckpt_owt = ckpt.get('corpus') == 'OWT'
    apply_run_config(
        layer_pattern=ckpt.get('layer_pattern', LAYER_PATTERN),
        k_in=ckpt.get('k_in'),
        k_out=ckpt.get('k_out'),
        reweight=ckpt.get('rare_reweight_power', RARE_REWEIGHT_POWER),
        per_row_aout_lr=ckpt.get('per_row_aout_lr', PER_ROW_AOUT_LR),
        run_name=ckpt.get('run_name', RUN_NAME),
        owt=ckpt_owt if owt is None else owt,
    )
    _, val_tokens, test_tokens, rank, unigram_logp = load_corpus_data()
    model = build_model(unigram_logp).to(device)
    model.load_state_dict(ckpt['model'])
    n_params = model.count_params()
    print(f'[checkpoint] loaded {path}  epoch={ckpt.get("epoch", "?")}  '
          f'corpus={"OWT" if OWT_MODE else "WT103"}  params={n_params:,}')
    head_diag = diag_head(model, rank)
    result = {
        'run_name': RUN_NAME,
        'corpus': 'OWT' if OWT_MODE else 'WT103',
        'layer_pattern': LAYER_PATTERN,
        'k_out': ckpt.get('k_out', v32.K_OUT),
        'reweight': ckpt.get('rare_reweight_power', RARE_REWEIGHT_POWER),
        'n_params': n_params,
        'val': eval_split(model, val_tokens, rank, 'val'),
        'head_diag': head_diag,
    }
    if test_tokens is not None:
        result['test'] = eval_split(model, test_tokens, rank, 'test')
    check_win(result)
    print_comparison_table(result)
    return result


def main():
    p = argparse.ArgumentParser(description='v41 vMF + concentrated attention')
    p.add_argument('--eval-only', action='store_true')
    p.add_argument('--checkpoint', type=str, default=None)
    p.add_argument('--skip-existing', action='store_true')
    p.add_argument('--force-retrain', action='store_true')
    p.add_argument('--no-fast', action='store_true')
    p.add_argument('--owt', action='store_true',
                   help='Train on OpenWebText (500M tok/epoch); primary ref = A_10M.json')
    p.add_argument('--prepare-owt-data', action='store_true',
                   help='Tokenize/cache OWT only (CPU ok; run before --owt on GPU)')
    p.add_argument('--sota', action='store_true',
                   help='Preset: FFFFFFFF interleaved, rw=0, 4 epochs (~9.96M, 50K SOTA run)')
    p.add_argument('--layer-pattern', type=str, default=LAYER_PATTERN,
                   help='Per-layer type: N=FFN-only, F=full attn+FFN (default NNNNFFFF)')
    p.add_argument('--interleaved', action='store_true',
                   help='Shortcut for --layer-pattern FFFFFFFF')
    p.add_argument('--reweight', type=float, default=None,
                   help='Rare reweight power (default 0 = standard CE)')
    p.add_argument('--per-row-aout-lr', action='store_true',
                   help='SparseMuon per-row LR on embed.A_out (rare rows step larger)')
    p.add_argument('--k-in', type=int, default=None)
    p.add_argument('--k-out', type=int, default=48)
    p.add_argument('--epochs', type=int, default=None)
    p.add_argument('--run-name', type=str, default=None)
    args = p.parse_args()

    if args.prepare_owt_data:
        owt_script = _SCRIPT_DIR / 'owt_chinchilla_e.py'
        if not owt_script.is_file():
            raise FileNotFoundError(
                f'--prepare-owt-data needs {_SCRIPT_DIR / "owt_chinchilla_e.py"}\n'
                f'  Run from the Experiments/ folder, or tokenize manually into:\n'
                f'    {OWT_TRAIN_TOKENS_PATH}\n'
                f'    {OWT_VAL_TOKENS_PATH}'
            )
        import sys
        if str(_SCRIPT_DIR) not in sys.path:
            sys.path.insert(0, str(_SCRIPT_DIR))
        from owt_chinchilla_e import prepare_owt_tokens
        os.environ.setdefault('OWT_TOKEN_CACHE_DIR', str(OWT_TOKEN_DIR))
        os.environ.setdefault('OWT_CHINCHILLA_DIR', str(OWT_TOKEN_DIR))
        print('=' * 72)
        print('  PREPARE OWT DATA (no training)')
        print(f'  Output dir: {OWT_TOKEN_DIR}')
        print(f'    {OWT_TRAIN_TOKENS_PATH.name}')
        print(f'    {OWT_VAL_TOKENS_PATH.name}')
        print('=' * 72)
        prepare_owt_tokens()
        print('\n[done] Next on GPU:')
        print('  python v41_vmf_concentrated.py --sota --owt')
        return

    if args.sota:
        pattern = 'FFFFFFFF'
        reweight = 0.0 if args.reweight is None else args.reweight
        epochs = 4 if args.epochs is None else args.epochs
    else:
        pattern = 'FFFFFFFF' if args.interleaved else args.layer_pattern
        reweight = RARE_REWEIGHT_POWER if args.reweight is None else args.reweight
        epochs = EPOCHS if args.epochs is None else args.epochs

    apply_run_config(
        layer_pattern=pattern,
        k_in=args.k_in,
        k_out=args.k_out,
        reweight=reweight,
        per_row_aout_lr=args.per_row_aout_lr,
        epochs=epochs,
        run_name=args.run_name,
        owt=args.owt,
    )

    if not args.no_fast:
        tune_for_gpu(fast=True)

    print(f'\n[v41] {RUN_NAME}  pattern={LAYER_PATTERN}  corpus={"OWT" if OWT_MODE else "WT103"}  '
          f'epochs={EPOCHS}  rw={RARE_REWEIGHT_POWER}  k={v32.K_OUT}')
    print_config_banner()

    if args.eval_only:
        eval_only(args.checkpoint, owt=args.owt if args.owt else None)
        return

    ep_final = ckpt_path(RUN_NAME, epoch=EPOCHS)
    if args.skip_existing and not args.force_retrain and ep_final.exists():
        print(f'[skip] ep{EPOCHS} exists: {ep_final} — eval only (use --force-retrain to retrain)')
        eval_only(str(ep_final))
        return

    train_tokens, val_tokens, test_tokens, rank, unigram_logp = load_corpus_data()
    gc.collect()

    model_smoke = build_model(unigram_logp)
    print(f'  [smoke] built model  params={model_smoke.count_params():,}')
    del model_smoke
    gc.collect()

    result = train_run(train_tokens, val_tokens, test_tokens, rank, unigram_logp)

    drive_cache, local_cache = active_cache_dirs()
    out_json = drive_cache / f'{RUN_NAME}_results.json'
    out_json.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        'created': datetime.now(timezone.utc).isoformat(),
        'config': {
            'run_name': RUN_NAME,
            'corpus': 'OWT' if OWT_MODE else 'WT103',
            'layer_pattern': LAYER_PATTERN,
            'head': 'vmf',
            'k_in': v32.K_IN,
            'k_out': v32.K_OUT,
            'reweight': RARE_REWEIGHT_POWER,
            'per_row_aout_lr': PER_ROW_AOUT_LR,
            'epochs': EPOCHS,
            'tokens_per_epoch': OWT_TOKENS_PER_EPOCH if OWT_MODE else None,
            'batch_size': BATCH_SIZE,
            'accum_steps': ACCUM_STEPS,
            'n_params': result['n_params'],
            'v32_interleaved_params': V32_INTERLEAVED_PARAMS,
            'mtp_depth': MTP_DEPTH,
            'mtp_weight': MTP_WEIGHT,
            'ema_decay': v32.EMA_DECAY,
            'stack': 'v32 compound (Muon+MTP+EMA+shortcut) + layer-pattern blocks',
        },
        'result': result,
        'baselines': REF_BASELINES,
        'win_criteria': {'max_val_ppl': WIN_PPL, 'max_tail_ce': WIN_TAIL},
    }
    with open(out_json, 'w') as f:
        json.dump(payload, f, indent=2, default=str)
    local_cache.mkdir(parents=True, exist_ok=True)
    local_json = local_cache / f'{RUN_NAME}_results.json'
    with open(local_json, 'w') as f:
        json.dump(payload, f, indent=2, default=str)
    print(f'[results] → {out_json}')


if __name__ == '__main__':
    main()
