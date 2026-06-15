"""
================================================================================
Chinchilla-E estimate from three small GPT-2–tokenized models on OpenWebText.
================================================================================

Default = full Chinchilla-E protocol (matches WT103-style analysis):

    TOKENS_PER_EPOCH = 500_000_000
    N_EPOCHS         = 6

so each model yields enough (T, C*) points to fit C*(T) = E_app + B·T^{-β} (the
h8 tail uses steps ≥ one epoch, so you need several epochs — not 3).

    pip install torch transformers datasets tiktoken numpy scipy
    python owt_chinchilla_e.py

Micro-batches are sized for a single A100 40GB (65536 tokens/step via batch×accum).

Prepare token cache on CPU so the GPU session only trains (same folder for both):

    python owt_chinchilla_e.py --prepare-data

What it does:
1) Streams OpenWebText, tokenizes with GPT-2 BPE, caches train+val tokens.
2) Trains three transformers (10M / 25M / 51M params), same depth/optimizer style.
3) Each epoch: val CE, bucket CE, and a C* fit from training CE vs step.
4) Saves JSON after each model; at the end: two-point and three-point E fits
   and a beta_rep vs sqrt(N) check.

Outputs under OWT_CHINCHILLA_DIR: A_10M.json, C_25M.json, B_51M.json, backups, manifest.json,
triangulation.txt, triangulation_three.json. Token .npy can live under OWT_TOKEN_CACHE_DIR
(fast local disk) while checkpoints stay on Drive — see Colab helper.
"""

import os, sys, math, json, time, gc
from types import SimpleNamespace
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import IterableDataset, DataLoader

# ============================================================================
# 1. CONFIG
# ============================================================================
SEED   = 42
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
DTYPE  = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16

if DEVICE == 'cuda':
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = True

# Colab / IterableDataset: extra workers often cause hangs; override with OWT_DATALOADER_WORKERS=2 if needed.
_DATALOADER_WORKERS = max(0, int(os.environ.get('OWT_DATALOADER_WORKERS', '0')))
# Eval builds fp32 logits [chunk, seq, vocab]; keep chunk small on 40GB after training peak.
_EVAL_FORWARD_CHUNK = max(1, int(os.environ.get('OWT_EVAL_FORWARD_CHUNK', '8')))

# All checkpoints, JSON, triangulation → OWT_CHINCHILLA_DIR (use persistent storage on Colab).
# Optional: OWT_TOKEN_CACHE_DIR = fast local path with owt_*_tokens.npy only (training I/O);
#           if unset, token caches live next to other outputs under OUT_DIR.
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
OUT_DIR = os.environ.get(
    'OWT_CHINCHILLA_DIR',
    os.path.join(_SCRIPT_DIR, 'owt_chinchilla'),
)
TOKEN_CACHE_DIR = (os.environ.get('OWT_TOKEN_CACHE_DIR') or '').strip() or OUT_DIR
os.makedirs(OUT_DIR, exist_ok=True)
if TOKEN_CACHE_DIR != OUT_DIR:
    os.makedirs(TOKEN_CACHE_DIR, exist_ok=True)

VOCAB_SIZE       = 50257
SEQ_LEN          = 1024
N_VAL_TOKENS     =   5_000_000
TOKENS_PER_EPOCH = 500_000_000
N_EPOCHS         = 6

# Same 65536 tokens/optimizer step for all models. Sizes below fit fp32 logits + training on one A100 40GB.
MODEL_CONFIGS = {
    'A_10M': dict(d_model=144, n_heads=4, n_layers=8, d_ff=640,
                  batch=32, accum=2,
                  lr=3e-4),
    'C_25M': dict(d_model=288, n_heads=4, n_layers=8, d_ff=1152,
                  batch=16, accum=4,
                  lr=2.8e-4),
    'B_51M': dict(d_model=448, n_heads=8, n_layers=8, d_ff=2048,
                  batch=8, accum=8,
                  lr=2.5e-4),
}


def _token_cache_names():
    return 'owt_train_tokens.npy', 'owt_val_tokens.npy'


def _h8_fit_t_min(steps_per_epoch):
    """Require steps ≥ one full epoch so early transient is excluded (ep2–ep6 give usable C* points)."""
    return steps_per_epoch

# Pre-registered sqrt(N) prediction for beta_rep (derived from WT103 runs)
# beta_rep(N) = C_SQRT * sqrt(N) where C_SQRT is anchored at 10M
_C_SQRT_ANCHOR_N    = 10_114_848   # actual param count of A_10M
_C_SQRT_ANCHOR_BETA = 0.5135       # fitted beta from WT103 vMF 10M run
BETA_SQRT_C = _C_SQRT_ANCHOR_BETA / math.sqrt(_C_SQRT_ANCHOR_N)

DROPOUT      = 0.1
WARMUP_STEPS = 300
GRAD_CLIP    = 1.0

# Frequency buckets (rank-defined) — same as WT103 paper
BUCKETS = [
    ('top_100',   0,     100),
    ('100_1K',    100,   1000),
    ('1K_10K',    1000,  10000),
    ('10K_30K',   10000, 30000),
    ('tail_30K+', 30000, VOCAB_SIZE),
]

# ============================================================================
# 1b. IO HELPERS
# ============================================================================
def _json_dump_atomic(path, obj):
    """Write JSON via temp file + replace to avoid half-written files on crash."""
    tmp = path + '.tmp'
    with open(tmp, 'w', encoding='utf-8') as f:
        json.dump(obj, f, indent=2)
    os.replace(tmp, path)


def _assert_train_tokens_sufficient(train_tokens):
    """Catch cache mismatch before hours of training."""
    n = len(train_tokens)
    if n < TOKENS_PER_EPOCH:
        raise ValueError(
            f"Train token buffer too small: len={n:,}, need >= {TOKENS_PER_EPOCH:,}. "
            f"Rebuild cache (500M train tokens in owt_train_tokens.npy)."
        )
    if n < SEQ_LEN + 8:
        raise ValueError(f"Train buffer too short for seq_len={SEQ_LEN}.")


# ============================================================================
# 2. TOKENIZE OPENWEBTEXT
# ============================================================================
def prepare_owt_tokens():
    """Tokenize train+val OWT slice and cache to disk. Re-uses cache if present."""
    train_name, val_name = _token_cache_names()
    train_path = os.path.join(TOKEN_CACHE_DIR, train_name)
    val_path   = os.path.join(TOKEN_CACHE_DIR, val_name)
    if os.path.exists(train_path) and os.path.exists(val_path):
        train = np.load(train_path, mmap_mode='r')
        val   = np.load(val_path,   mmap_mode='r')
        print(f"[data] loaded cached tokens (no HuggingFace download):")
        print(f"       {train_path}")
        print(f"       {val_path}")
        print(f"       train={len(train):,}  val={len(val):,}")
        return train, val

    print("[data] cache missing; will stream OpenWebText with GPT-2 BPE (~20–40 min)…")
    print(f"       expected after save:\n       {train_path}\n       {val_path}")
    from datasets import load_dataset
    import tiktoken
    enc = tiktoken.get_encoding("gpt2")
    eot = enc.eot_token
    ds  = load_dataset("openwebtext", split="train", streaming=True)

    def stream_tokens(needed):
        out = np.zeros(needed, dtype=np.uint16)
        i = 0; t0 = time.time()
        for ex in ds:
            ids = enc.encode_ordinary(ex['text'])
            ids.append(eot)
            n = len(ids)
            if i + n > needed:
                out[i:needed] = ids[:needed-i]
                i = needed; break
            out[i:i+n] = ids
            i += n
            if i % 10_000_000 < n:
                print(f"  {i/1e6:.0f}M / {needed/1e6:.0f}M tokens  "
                      f"({i/needed*100:.1f}%, {(time.time()-t0)/60:.1f}min)")
        return out[:i]

    total_needed = TOKENS_PER_EPOCH + N_VAL_TOKENS
    all_tokens   = stream_tokens(total_needed)
    if len(all_tokens) < total_needed:
        raise RuntimeError(
            f"OpenWebText stream ended early: got {len(all_tokens):,} tokens, "
            f"need {total_needed:,}. Retry or check HuggingFace connectivity."
        )
    val_tokens   = all_tokens[:N_VAL_TOKENS]
    train_tokens = all_tokens[N_VAL_TOKENS:]
    if len(train_tokens) < TOKENS_PER_EPOCH:
        raise RuntimeError(
            f"Train slice shorter than TOKENS_PER_EPOCH: {len(train_tokens):,} < {TOKENS_PER_EPOCH:,}"
        )
    np.save(train_path, train_tokens)
    np.save(val_path,   val_tokens)
    print(f"[data] saved: train={len(train_tokens):,}  val={len(val_tokens):,}")
    return train_tokens, val_tokens

# ============================================================================
# 3. MODEL: Standard softmax transformer (RoPE, RMSNorm, SwiGLU, tied emb)
# ============================================================================
class RMSNorm(nn.Module):
    def __init__(self, d, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(d))
        self.eps = eps
    def forward(self, x):
        return self.weight * x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

class RoPE(nn.Module):
    def __init__(self, dim, max_len=4096, base=10000.0):
        super().__init__()
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        t = torch.arange(max_len).float()
        freqs = torch.einsum('i,j->ij', t, inv_freq)
        self.register_buffer('cos', freqs.cos()[None, None, :, :], persistent=False)
        self.register_buffer('sin', freqs.sin()[None, None, :, :], persistent=False)
    def forward(self, x):
        T = x.size(-2)
        cos = self.cos[:, :, :T, :].to(x.dtype)
        sin = self.sin[:, :, :T, :].to(x.dtype)
        xe, xo = x[..., ::2], x[..., 1::2]
        return torch.stack((xe*cos - xo*sin, xe*sin + xo*cos), -1).flatten(-2)

class Block(nn.Module):
    def __init__(self, d_model, n_heads, d_ff, n_layers, rope):
        super().__init__()
        self.n_heads = n_heads
        self.d_head  = d_model // n_heads
        self.rope    = rope
        self.norm1   = RMSNorm(d_model)
        self.norm2   = RMSNorm(d_model)
        self.qkv     = nn.Linear(d_model, 3*d_model, bias=False)
        self.proj    = nn.Linear(d_model, d_model, bias=False)
        self.q_norm  = RMSNorm(self.d_head)
        self.k_norm  = RMSNorm(self.d_head)
        self.gate    = nn.Linear(d_model, d_ff, bias=False)
        self.up      = nn.Linear(d_model, d_ff, bias=False)
        self.down    = nn.Linear(d_ff, d_model, bias=False)
        self.drop    = nn.Dropout(DROPOUT)
        s = 1.0 / math.sqrt(2 * n_layers)
        nn.init.normal_(self.qkv.weight,  0, 0.02)
        nn.init.normal_(self.proj.weight, 0, 0.02 * s)
        nn.init.normal_(self.gate.weight, 0, 0.02)
        nn.init.normal_(self.up.weight,   0, 0.02)
        nn.init.normal_(self.down.weight, 0, 0.02 * s)

    def forward(self, x):
        b, t, d = x.shape
        h = self.norm1(x)
        q, k, v = self.qkv(h).chunk(3, -1)
        def r(z): return z.view(b, t, self.n_heads, self.d_head).transpose(1, 2)
        q, k, v = r(q), r(k), r(v)
        q = self.rope(self.q_norm(q))
        k = self.rope(self.k_norm(k))
        a = F.scaled_dot_product_attention(q, k, v, is_causal=True,
                dropout_p=DROPOUT if self.training else 0.0)
        x = x + self.drop(self.proj(a.transpose(1, 2).contiguous().view(b, t, d)))
        x = x + self.drop(self.down(F.silu(self.gate(self.norm2(x))) * self.up(self.norm2(x))))
        return x

class TransformerLM(nn.Module):
    def __init__(self, cfg, unigram_logp=None):
        super().__init__()
        d, h, L, df = cfg['d_model'], cfg['n_heads'], cfg['n_layers'], cfg['d_ff']
        self.emb    = nn.Embedding(VOCAB_SIZE, d)
        nn.init.normal_(self.emb.weight, 0, 0.02)
        self.rope    = RoPE(d // h, SEQ_LEN)
        self.in_drop = nn.Dropout(DROPOUT)
        self.blocks  = nn.ModuleList([Block(d, h, df, L, self.rope) for _ in range(L)])
        self.norm_f  = RMSNorm(d)
        # Tied output: out_W shares weight with embedding (matches GPT-2 / Chinchilla)
        self.out_bias = nn.Parameter(torch.zeros(VOCAB_SIZE))
        if unigram_logp is not None:
            with torch.no_grad():
                self.out_bias.copy_(unigram_logp - unigram_logp.mean())

    def forward(self, ids, labels=None):
        x = self.in_drop(self.emb(ids))
        for blk in self.blocks:
            x = blk(x)
        x = self.norm_f(x)
        logits = (x.float() @ self.emb.weight.float().T) + self.out_bias.float()
        if labels is not None:
            loss = F.cross_entropy(logits.view(-1, VOCAB_SIZE), labels.view(-1))
            return SimpleNamespace(loss=loss, logits=logits)
        return SimpleNamespace(logits=logits, loss=None)

# ============================================================================
# 4. DATA LOADING
# ============================================================================
class TokenIterDataset(IterableDataset):
    def __init__(self, tokens, seq_len, seed=0):
        self.tokens  = tokens
        self.seq_len = seq_len
        self.seed    = seed

    def __iter__(self):
        rng = np.random.default_rng(self.seed)
        n = len(self.tokens) - self.seq_len - 1
        while True:
            i = rng.integers(0, n)
            x = self.tokens[i:i+self.seq_len].astype(np.int64)
            y = self.tokens[i+1:i+1+self.seq_len].astype(np.int64)
            yield torch.from_numpy(x), torch.from_numpy(y)

def make_loader(tokens, batch, seq_len, num_workers=None, seed=0):
    if num_workers is None:
        num_workers = _DATALOADER_WORKERS
    ds = TokenIterDataset(tokens, seq_len, seed=seed)
    return DataLoader(
        ds, batch_size=batch, num_workers=num_workers,
        pin_memory=(DEVICE == 'cuda'), persistent_workers=(num_workers > 0),
        drop_last=True,
    )

# ============================================================================
# 5. EVALUATION + DIAGNOSTICS
# ============================================================================
def get_unigram_counts(tokens):
    return np.bincount(tokens.astype(np.int64), minlength=VOCAB_SIZE).astype(np.float64)

def assign_buckets_by_rank(unigram_counts):
    order    = np.argsort(-unigram_counts)
    rank_of  = np.zeros(VOCAB_SIZE, dtype=np.int64)
    for r, v in enumerate(order):
        rank_of[v] = r
    bucket_of = np.full(VOCAB_SIZE, -1, dtype=np.int8)
    for bi, (_, lo, hi) in enumerate(BUCKETS):
        bucket_of[(rank_of >= lo) & (rank_of < hi)] = bi
    return bucket_of

def _autocast_ctx():
    if DEVICE != 'cuda':
        import contextlib
        return contextlib.nullcontext()
    return torch.amp.autocast('cuda', dtype=DTYPE)


@torch.no_grad()
def evaluate(model, val_tokens, batch, seq_len, bucket_of):
    model.eval()
    n_seqs       = len(val_tokens) // (seq_len + 1)
    bucket_loss  = np.zeros(len(BUCKETS), dtype=np.float64)
    bucket_count = np.zeros(len(BUCKETS), dtype=np.int64)
    total_loss = 0.0; total_count = 0
    chunk = _EVAL_FORWARD_CHUNK
    for i in range(0, n_seqs, batch):
        ks = min(batch, n_seqs - i)
        xb = np.zeros((ks, seq_len), dtype=np.int64)
        yb = np.zeros((ks, seq_len), dtype=np.int64)
        for j in range(ks):
            base = (i+j) * (seq_len + 1)
            xb[j] = val_tokens[base:base+seq_len]
            yb[j] = val_tokens[base+1:base+1+seq_len]
        # Sub-chunk on dim0 so peak fp32 logits stay small (avoids OOM right after training).
        for i0 in range(0, ks, chunk):
            kc = min(chunk, ks - i0)
            x = torch.from_numpy(xb[i0:i0+kc]).to(DEVICE)
            y = torch.from_numpy(yb[i0:i0+kc]).to(DEVICE)
            with _autocast_ctx():
                out = model(x)
            logits = out.logits
            logp   = F.log_softmax(logits, -1)
            nll    = -logp.gather(-1, y.unsqueeze(-1)).squeeze(-1)
            flat_nll = nll.reshape(-1).cpu().numpy()
            flat_y   = y.reshape(-1).cpu().numpy()
            flat_b   = bucket_of[flat_y]
            for bi in range(len(BUCKETS)):
                mask = flat_b == bi
                bucket_loss[bi]  += flat_nll[mask].sum()
                bucket_count[bi] += int(mask.sum())
            total_loss  += flat_nll.sum()
            total_count += len(flat_nll)
            del out, logits, logp, nll, x, y
    model.train()
    overall    = total_loss / max(1, total_count)
    per_bucket = [bucket_loss[bi] / max(1, bucket_count[bi]) for bi in range(len(BUCKETS))]
    weights    = bucket_count / max(1, total_count)
    return float(overall), [float(v) for v in per_bucket], weights.tolist()

# ============================================================================
# 6. H8 1/t fit (shared helper)
# ============================================================================
def h8_fit(steps, ces, t_min):
    """Fit CE(t) = A/t + C* over points with t >= t_min."""
    s = np.asarray(steps); c = np.asarray(ces)
    mask = s >= t_min
    s, c = s[mask], c[mask]
    if len(s) < 5:
        return None
    M = np.column_stack([1.0 / s, np.ones_like(s)])
    A, Cstar = np.linalg.lstsq(M, c, rcond=None)[0]
    pred = A / s + Cstar
    res  = c - pred
    return dict(A=float(A), Cstar=float(Cstar), n=int(mask.sum()),
                resid_sigma=float(res.std(ddof=1)),
                R2=float(1 - res.var() / max(c.var(), 1e-12)))

# ============================================================================
# 7. SHARED FIT HELPERS (used by both triangulation functions)
# ============================================================================
def _fit3_T_C(T_list, C_list):
    """
    Fit C*(T) = E_app + B * T^{-beta} via nonlinear least squares.
    Returns (E_app, beta, rss) — best over multiple restarts.
    """
    from scipy.optimize import least_squares
    T = np.asarray(T_list, dtype=np.float64)
    C = np.asarray(C_list, dtype=np.float64)

    def r(p):
        return np.exp(p[0]) + np.exp(p[1]) / np.power(T, p[2]) - C

    best = (1e9, None)
    for E0 in np.log([0.5, 1.0, 2.0, 3.0, 4.0]):
        for b0 in [0.1, 0.5, 1.0, 1.5]:
            try:
                res = least_squares(r, [E0, np.log(100.0), b0],
                                    bounds=([np.log(0.01), np.log(1e-3), 0.01],
                                            [np.log(15),   np.log(1e15), 3.0]),
                                    max_nfev=5000)
                rss = float(np.sum(res.fun**2))
                if rss < best[0]:
                    best = (rss, res.x)
            except Exception:
                pass
    if best[1] is None:
        raise RuntimeError(
            "_fit3_T_C: all least_squares restarts failed — check T/C data")
    p = best[1]
    E_app = float(np.exp(p[0]))
    beta  = float(p[2])
    return E_app, beta, best[0]

def _collect_T_C(result_dict):
    """Extract (T_tokens, C*_or_val_ce) lists from a model result dict."""
    T = [e['T_tokens'] for e in result_dict['epochs']]
    C = [e['h8']['Cstar'] if e['h8'] else e['val_ce'] for e in result_dict['epochs']]
    return T, C

# ============================================================================
# 8. CHECKPOINT AFTER EACH MODEL
# ============================================================================
def save_checkpoint_after_model(name, result_dict, all_results_so_far):
    """Save per-model backup JSON + manifest so crashes are recoverable."""
    final_path = os.path.join(OUT_DIR, f"{name}_final_backup.json")
    _json_dump_atomic(final_path, result_dict)
    manifest = dict(
        out_dir=OUT_DIR,
        completed=list(all_results_so_far.keys()),
        last_completed=name,
        time_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
    )
    _json_dump_atomic(os.path.join(OUT_DIR, "manifest.json"), manifest)
    print(f"[checkpoint] saved {final_path} and manifest.json")

# ============================================================================
# 9. TRAIN ONE MODEL
# ============================================================================
def train_one(name, cfg, train_tokens, val_tokens, bucket_of, unigram_logp):
    print(f"\n{'='*70}\nTRAINING {name}\n{'='*70}")
    torch.manual_seed(SEED); np.random.seed(SEED)

    model    = TransformerLM(cfg, unigram_logp=unigram_logp.to(DEVICE)).to(DEVICE)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  params: {n_params:,}")

    batch, accum, lr = cfg['batch'], cfg['accum'], cfg['lr']
    tokens_per_step  = batch * accum * SEQ_LEN
    steps_per_epoch  = TOKENS_PER_EPOCH // tokens_per_step
    total_steps      = steps_per_epoch * N_EPOCHS
    print(f"  tokens/step={tokens_per_step}  steps/epoch={steps_per_epoch}  total={total_steps}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, betas=(0.9, 0.95),
                                  weight_decay=0.1, eps=1e-8)

    def lr_at(step):
        if step < WARMUP_STEPS:
            return step / WARMUP_STEPS
        prog = (step - WARMUP_STEPS) / max(1, total_steps - WARMUP_STEPS)
        return max(0.1, 0.5 * (1 + math.cos(math.pi * prog)))

    loader = make_loader(train_tokens, batch, SEQ_LEN, seed=SEED)
    it     = iter(loader)
    scaler = torch.amp.GradScaler('cuda', enabled=(DTYPE == torch.float16))

    log_steps, log_ce = [], []
    epoch_results     = []
    t_start = time.time(); t_log = t_start

    for ep in range(1, N_EPOCHS + 1):
        running = 0.0; running_n = 0
        for s in range(1, steps_per_epoch + 1):
            global_step = (ep - 1) * steps_per_epoch + s
            for g in optimizer.param_groups:
                g['lr'] = lr * lr_at(global_step)
            optimizer.zero_grad(set_to_none=True)
            for _ in range(accum):
                xb, yb = next(it)
                xb = xb.to(DEVICE, non_blocking=True)
                yb = yb.to(DEVICE, non_blocking=True)
                with _autocast_ctx():
                    out = model(xb, yb)
                loss = out.loss / accum
                if scaler.is_enabled():
                    scaler.scale(loss).backward()
                else:
                    loss.backward()
                running   += float(out.loss.detach())
                running_n += 1
            if scaler.is_enabled():
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
                scaler.step(optimizer); scaler.update()
            else:
                torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
                optimizer.step()
            if global_step % 200 == 0:
                ce_avg = running / max(1, running_n)
                log_steps.append(global_step); log_ce.append(ce_avg)
                running = 0.0; running_n = 0
                if time.time() - t_log > 60:
                    t_log = time.time()
                    elapsed = (time.time() - t_start) / 60
                    print(f"  step {global_step:>6}  ep{ep}  ce={ce_avg:.4f}  "
                          f"elapsed={elapsed:.1f}min")

        # End of epoch — free allocator cache before eval (training leaves GPU nearly full).
        if DEVICE == 'cuda':
            gc.collect()
            torch.cuda.empty_cache()

        T_seen   = ep * steps_per_epoch * tokens_per_step
        # Eval I/O batch; forward is sub-chunked inside evaluate() for peak VRAM.
        if name == 'B_51M':
            eval_bs = min(batch * 2, 16)
        else:
            eval_bs = batch * 2
        val_ce, bucket_ce, bucket_w = evaluate(
            model, val_tokens, batch=eval_bs, seq_len=SEQ_LEN, bucket_of=bucket_of)
        fit = h8_fit(log_steps, log_ce, t_min=_h8_fit_t_min(steps_per_epoch))
        rec = dict(epoch=ep, T_tokens=T_seen, val_ce=val_ce,
                   bucket_ce=bucket_ce, bucket_w=bucket_w, h8=fit)
        epoch_results.append(rec)
        cstar_str = f"{fit['Cstar']:.4f}" if fit else "NA"
        print(f"  END EP{ep}: T={T_seen/1e6:.0f}M  val_ce={val_ce:.4f}  C*={cstar_str}")

        # Save partial after every epoch
        _json_dump_atomic(
            os.path.join(OUT_DIR, f"{name}.json"),
            dict(name=name, cfg=cfg, n_params=n_params,
                 epochs=epoch_results,
                 train_log=dict(steps=log_steps, ce=log_ce)),
        )

    del model; gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return dict(name=name, cfg=cfg, n_params=n_params, epochs=epoch_results)

# ============================================================================
# 10. TWO-POINT TRIANGULATION (A + B only, replicates WT103 method)
# ============================================================================
def chinchilla_triangulate(results_A, results_B, alpha=0.34):
    """
    Two-point Chinchilla-E extraction.
    Step 1: fit C*(T) = E_app + B/T^beta per model.
    Step 2: with alpha prior, solve E_true + A*N^{-alpha} = E_app.
    """
    T_A, C_A = _collect_T_C(results_A)
    T_B, C_B = _collect_T_C(results_B)

    E_app_A, beta_A, _ = _fit3_T_C(T_A, C_A)
    E_app_B, beta_B, _ = _fit3_T_C(T_B, C_B)

    N_A = results_A['n_params']
    N_B = results_B['n_params']

    A_amp  = (E_app_A - E_app_B) / (N_A**(-alpha) - N_B**(-alpha))
    E_true = E_app_A - A_amp * N_A**(-alpha)

    return dict(
        N_A=N_A, N_B=N_B, alpha_prior=alpha,
        E_app_A=float(E_app_A), beta_A=float(beta_A),
        E_app_B=float(E_app_B), beta_B=float(beta_B),
        A_amp=float(A_amp), E_true=float(E_true),
    )

# ============================================================================
# 11. THREE-POINT TRIANGULATION (A + C + B, overdetermined OLS)
# ============================================================================
def chinchilla_triangulate_three(results_A, results_C, results_B, alpha=None):
    """
    Three-point Chinchilla-E extraction.

    If alpha is None: fit E_true, A, alpha jointly via OLS on the three
    E_app values (overdetermined with 3 equations, 3 unknowns).

    If alpha is provided: use it as a prior and do OLS on E_true, A only.

    Also performs the pre-registered sqrt(N) beta_rep test.
    """
    models = [results_A, results_C, results_B]
    rows = []
    for res in models:
        T, C = _collect_T_C(res)
        E_app, beta, _ = _fit3_T_C(T, C)
        rows.append(dict(res=res, N=res['n_params'], E_app=E_app, beta=beta))
    rows.sort(key=lambda r: r['N'])
    N_list     = [r['N'] for r in rows]
    E_app_list = [r['E_app'] for r in rows]
    beta_list  = [r['beta'] for r in rows]

    N   = np.array(N_list,     dtype=np.float64)
    Ea  = np.array(E_app_list, dtype=np.float64)

    if alpha is None:
        # Nonlinear: E_app(N) = E + A * N^{-alpha}
        # Linearise by searching over alpha
        from scipy.optimize import minimize_scalar
        def sse_alpha(a):
            x = N**(-a)
            # OLS for E, A given alpha
            M = np.column_stack([np.ones(3), x])
            p, _, _, _ = np.linalg.lstsq(M, Ea, rcond=None)
            pred = p[0] + p[1] * x
            return float(np.sum((Ea - pred)**2))

        res_opt = minimize_scalar(sse_alpha, bounds=(0.20, 0.60), method='bounded')
        alpha_fit = float(res_opt.x)
        x  = N**(-alpha_fit)
        M  = np.column_stack([np.ones(3), x])
        p, _, _, _ = np.linalg.lstsq(M, Ea, rcond=None)
        E_true = float(p[0]); A_amp = float(p[1])
        alpha_used = alpha_fit
    else:
        alpha_used = alpha
        x  = N**(-alpha)
        M  = np.column_stack([np.ones(3), x])
        p, _, _, _ = np.linalg.lstsq(M, Ea, rcond=None)
        E_true = float(p[0]); A_amp = float(p[1])
        alpha_fit = alpha

    # Residuals from the 3-point fit
    Ea_pred = E_true + A_amp * N**(-alpha_used)
    residuals = (Ea - Ea_pred).tolist()

    # beta_rep ~ sqrt(N) pre-registered test
    N_small  = N_list[0]; beta_small  = beta_list[0]
    N_mid    = N_list[1]; beta_mid    = beta_list[1]
    N_large  = N_list[2]; beta_large  = beta_list[2]

    beta_pred_mid   = beta_small * math.sqrt(N_mid   / N_small)
    beta_pred_large = beta_small * math.sqrt(N_large / N_small)

    delta_mid   = beta_mid   - beta_pred_mid
    delta_large = beta_large - beta_pred_large

    # Fitted exponent: beta ~ N^exponent
    if beta_small <= 0 or N_large <= N_small:
        exponent = float('nan')
    else:
        exponent = math.log(beta_large / beta_small) / math.log(N_large / N_small)

    return dict(
        models=[r['res']['name'] for r in rows],
        N_list=[int(n) for n in N_list],
        E_app_list=E_app_list,
        beta_list=beta_list,
        alpha_used=alpha_used,
        alpha_fitted=(alpha_fit if alpha is None else None),
        E_true=E_true,
        A_amp=A_amp,
        residuals=residuals,
        beta_sqrt_N=dict(
            C_sqrt=BETA_SQRT_C,
            beta_pred_mid=beta_pred_mid,
            beta_actual_mid=beta_mid,
            delta_mid=delta_mid,
            beta_pred_large=beta_pred_large,
            beta_actual_large=beta_large,
            delta_large=delta_large,
            fitted_exponent=exponent,
            pass_mid=(abs(delta_mid) < 0.05),
            pass_large=(abs(delta_large) < 0.05),
        ),
    )

# ============================================================================
# 11b. N-POINT LADDER FIT (overdetermined when n > 2)
# ============================================================================
def chinchilla_fit_eapp_ladder(n_params_list, e_app_list, alpha=0.34):
    """
    Fit E_app(N) = E_true + A * N^{-alpha} via OLS on an arbitrary ladder.

    With fixed alpha this has 2 parameters (E_true, A) and n - 2 degrees of
    freedom when n >= 3. Use holdout / LOO for validation — not the in-sample
    RMSE alone.
    """
    N = np.asarray(n_params_list, dtype=np.float64)
    Ea = np.asarray(e_app_list, dtype=np.float64)
    if len(N) < 2:
        raise ValueError("Need at least 2 ladder points")
    if len(N) != len(Ea):
        raise ValueError("n_params and e_app length mismatch")

    x = N ** (-alpha)
    M = np.column_stack([np.ones(len(N)), x])
    p, _, _, _ = np.linalg.lstsq(M, Ea, rcond=None)
    E_true = float(p[0])
    A_amp = float(p[1])
    pred = E_true + A_amp * x
    resid = Ea - pred
    rmse = float(np.sqrt(np.mean(resid ** 2)))
    dof = int(len(N) - 2)
    return dict(
        alpha_prior=alpha,
        n_points=int(len(N)),
        dof=dof,
        E_true=E_true,
        A_amp=A_amp,
        rmse=rmse,
        residuals=resid.tolist(),
        E_app_pred=pred.tolist(),
    )


def chinchilla_predict_e_app(e_true, a_amp, n_params, alpha=0.34):
    """Predict E_app at size n_params from fitted floor + amplitude."""
    return float(e_true + a_amp * (float(n_params) ** (-alpha)))


# ============================================================================
# 12. MAIN
# ============================================================================
def main():
    print(f"OUT_DIR:  {OUT_DIR}")
    print(f"  (checkpoints + JSON + triangulation — keep on persistent disk on Colab)")
    if TOKEN_CACHE_DIR != OUT_DIR:
        print(f"TOKEN_CACHE_DIR:  {TOKEN_CACHE_DIR}")
        print(f"  (read-only token .npy here; fast local SSD is OK)")
    print(f"TRAIN:    {TOKENS_PER_EPOCH/1e6:.0f}M tok/epoch × {N_EPOCHS} epochs/model")
    print(f"DEVICE:   {DEVICE}  |  DTYPE: {DTYPE}")
    if DEVICE == 'cuda':
        p = torch.cuda.get_device_properties(0)
        print(f"GPU:      {torch.cuda.get_device_name(0)}  |  VRAM {p.total_memory / (1024**3):.1f} GiB")
    print(f"VOCAB:    {VOCAB_SIZE}  SEQ_LEN: {SEQ_LEN}")
    print(f"DataLoader workers: {_DATALOADER_WORKERS}  (set OWT_DATALOADER_WORKERS to override)")
    print()

    train_tokens, val_tokens = prepare_owt_tokens()
    _assert_train_tokens_sufficient(train_tokens)
    counts       = get_unigram_counts(train_tokens)
    bucket_of    = assign_buckets_by_rank(counts)
    p            = counts / counts.sum()
    p            = np.maximum(p, 1e-12)
    unigram_logp = torch.from_numpy(np.log(p).astype(np.float32))

    # Train in order: small → medium → large (fail-safe: each saves independently)
    train_order = ['A_10M', 'C_25M', 'B_51M']
    res = {}
    for name in train_order:
        res[name] = train_one(name, MODEL_CONFIGS[name],
                              train_tokens, val_tokens, bucket_of, unigram_logp)
        save_checkpoint_after_model(name, res[name], res)

    # ── Two-point triangulation (replicates WT103 methodology) ──────────────
    tri2_034  = chinchilla_triangulate(res['A_10M'], res['B_51M'], alpha=0.34)
    tri2_384  = chinchilla_triangulate(res['A_10M'], res['B_51M'], alpha=0.384)

    # ── Three-point triangulation (all three models) ─────────────────────────
    tri3_free = chinchilla_triangulate_three(res['A_10M'], res['C_25M'], res['B_51M'],
                                             alpha=None)
    tri3_034  = chinchilla_triangulate_three(res['A_10M'], res['C_25M'], res['B_51M'],
                                             alpha=0.34)

    # Save machine-readable three-point results
    _json_dump_atomic(
        os.path.join(OUT_DIR, "triangulation_three.json"),
        dict(free_alpha=tri3_free, fixed_alpha_034=tri3_034),
    )

    # ── Human-readable summary ───────────────────────────────────────────────
    N_A = res['A_10M']['n_params']
    N_C = res['C_25M']['n_params']
    N_B = res['B_51M']['n_params']

    bsN = tri3_free['beta_sqrt_N']

    msg = f"""
================================================================================
CHINCHILLA-E TRIANGULATION ON OPENWEBTEXT
================================================================================

THREE MODELS (matched softmax architecture, GPT-2 tokenizer):
  Model A: {N_A:>14,} params  E_app={tri2_034['E_app_A']:.4f}  β_rep={tri2_034['beta_A']:.4f}
  Model C: {N_C:>14,} params  E_app={tri3_free['E_app_list'][1]:.4f}  β_rep={tri3_free['beta_list'][1]:.4f}
  Model B: {N_B:>14,} params  E_app={tri2_034['E_app_B']:.4f}  β_rep={tri2_034['beta_B']:.4f}

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
TWO-POINT TRIANGULATION (A + B only, replicates WT103 result)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

  With α = 0.34 (Chinchilla prior):   E_true = {tri2_034['E_true']:.4f} nats
  With α = 0.384 (WT103-implied):     E_true = {tri2_384['E_true']:.4f} nats

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
THREE-POINT TRIANGULATION (A + C + B, overdetermined OLS)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

  Free α (OLS-fitted):  α = {tri3_free['alpha_used']:.4f}   E_true = {tri3_free['E_true']:.4f} nats
  Fixed α = 0.34:                      E_true = {tri3_034['E_true']:.4f} nats

  Fit residuals (E_app_actual - E_app_predicted):
    A_10M: {tri3_free['residuals'][0]:+.4f} nats
    C_25M: {tri3_free['residuals'][1]:+.4f} nats
    B_51M: {tri3_free['residuals'][2]:+.4f} nats

  GPT-2 medium upper-bound on OWT (~70K steps): 2.855 nats
  Expected range (E_OWT > E_WT103 = 2.68, E_OWT < 2.855):
    PASS if {min(tri3_free['E_true'], tri3_034['E_true']):.3f} < E < 2.855

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
PRE-REGISTERED β_rep ~ √N TEST
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

  Prediction: β_rep(N) = {BETA_SQRT_C:.6f} × √N   (anchored at WT103 10M β=0.5135)

  25M model:  β_pred = {bsN['beta_pred_mid']:.4f}   β_actual = {bsN['beta_actual_mid']:.4f}
              delta  = {bsN['delta_mid']:+.4f}   PASS (|Δ|<0.05): {bsN['pass_mid']}

  51M model:  β_pred = {bsN['beta_pred_large']:.4f}   β_actual = {bsN['beta_actual_large']:.4f}
              delta  = {bsN['delta_large']:+.4f}   PASS (|Δ|<0.05): {bsN['pass_large']}

  Fitted exponent: β_rep ~ N^{bsN['fitted_exponent']:.4f}
  Pre-registered:  β_rep ~ N^0.5000

  Overall β_rep scaling PASS: {bsN['pass_mid'] and bsN['pass_large']}

================================================================================
"""
    print(msg)
    tri_path = os.path.join(OUT_DIR, "triangulation.txt")
    tmp = tri_path + '.tmp'
    with open(tmp, 'w', encoding='utf-8') as f:
        f.write(msg)
    os.replace(tmp, tri_path)
    print(f"[done] Results in {OUT_DIR}/")
    print(f"  triangulation.txt")
    print(f"  triangulation_three.json")
    print(f"  A_10M.json  C_25M.json  B_51M.json")


def run_prepare_data_only():
    """
    Build owt_train_tokens.npy + owt_val_tokens.npy (500M + 5M tokens) under OUT_DIR and exit.
    CPU-only (streaming + BPE). Use the same OWT_CHINCHILLA_DIR when training.
    """
    tr_name, va_name = _token_cache_names()
    print("=" * 72)
    print("PREPARE DATA ONLY  (no model training)")
    print(f"  OUT_DIR:  {OUT_DIR}")
    print(f"  Will create (if missing): {tr_name}, {va_name}")
    print(f"  Approx size: ~{2 * (TOKENS_PER_EPOCH + N_VAL_TOKENS) / 1e9:.2f} GB uint16 on disk")
    print("  GPU: not used for this step.")
    print("=" * 72)
    prepare_owt_tokens()
    print("\n[prepare-data] Done. On the GPU machine:")
    print(f"  export OWT_CHINCHILLA_DIR={OUT_DIR!r}")
    print("  python owt_chinchilla_e.py")


if __name__ == '__main__':
    if len(sys.argv) > 1 and sys.argv[1] in ('--prepare-data', '--tokenize-only'):
        run_prepare_data_only()
    else:
        main()