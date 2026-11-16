"""
Synthetic Chinchilla-E validation — known corpus entropy, small models, fast run.

Generates i.i.d. Zipf-unigram token streams with analytically known H_true,
trains three tiny softmax LMs (~1M / ~2M / ~3M params), then runs the same
triangulation pipeline as owt_chinchilla_e.py.

Pre-registration: docs/PREREGISTER_synthetic_chinchilla_e.md

    pip install torch numpy scipy
    python scripts/synthetic_chinchilla_e_validation.py

Quick smoke (CPU or GPU):
    set SYNTH_PRESET=smoke
    python scripts/synthetic_chinchilla_e_validation.py

Colab A100 (~15–25 min):
    set SYNTH_PRESET=a100_fast
    python scripts/synthetic_chinchilla_e_validation.py
    # or open colab/synthetic_chinchilla_e_a100.ipynb
"""

from __future__ import annotations

import gc
import json
import math
import os
import sys
import time

import numpy as np

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if _SCRIPT_DIR not in sys.path:
    sys.path.insert(0, _SCRIPT_DIR)

import owt_chinchilla_e as oce  # noqa: E402

# ---------------------------------------------------------------------------
# Synthetic protocol (override OWT scale)
# ---------------------------------------------------------------------------
ZIPF_EXPONENT = float(os.environ.get("SYNTH_ZIPF_S", "1.0"))
ACTIVE_VOCAB = int(os.environ.get("SYNTH_ACTIVE_VOCAB", "2048"))
TOKENS_PER_EPOCH = int(os.environ.get("SYNTH_TOKENS_PER_EPOCH", "10000000"))
N_VAL_TOKENS = int(os.environ.get("SYNTH_N_VAL_TOKENS", "500000"))
N_EPOCHS = int(os.environ.get("SYNTH_N_EPOCHS", "6"))
SEED = int(os.environ.get("SYNTH_SEED", "42"))

OUT_DIR = os.environ.get(
    "SYNTH_CHINCHILLA_DIR",
    os.path.join(_SCRIPT_DIR, "..", "results", "synthetic_chinchilla"),
)
OUT_DIR = os.path.abspath(OUT_DIR)
os.makedirs(OUT_DIR, exist_ok=True)

# Gates from PREREGISTER_synthetic_chinchilla_e.md
GATE_P1 = 0.15  # |E_true free alpha - H_true|
GATE_P2 = 0.20  # |E_true fixed alpha - H_true|
GATE_S1 = 0.25  # |val_ce largest - H_true|

_DEFAULT_MODEL_CONFIGS = {
    "S_1M": dict(d_model=96, n_heads=4, n_layers=4, d_ff=384, batch=64, accum=1, lr=3e-4),
    "S_2M": dict(d_model=128, n_heads=4, n_layers=6, d_ff=512, batch=32, accum=2, lr=3e-4),
    "S_3M": dict(d_model=160, n_heads=4, n_layers=6, d_ff=640, batch=32, accum=2, lr=2.8e-4),
}
_A100_FAST_CONFIGS = {
    "S_1M": dict(d_model=96, n_heads=4, n_layers=4, d_ff=384, batch=128, accum=1, lr=3e-4),
    "S_2M": dict(d_model=128, n_heads=4, n_layers=6, d_ff=512, batch=64, accum=2, lr=3e-4),
    "S_3M": dict(d_model=160, n_heads=4, n_layers=6, d_ff=640, batch=64, accum=2, lr=2.8e-4),
}
MODEL_CONFIGS = dict(_DEFAULT_MODEL_CONFIGS)
TRAIN_ORDER = ["S_1M", "S_2M", "S_3M"]


def _apply_preset():
    """Optional presets: a100_fast (Colab), smoke, full (default)."""
    global TOKENS_PER_EPOCH, N_EPOCHS, N_VAL_TOKENS, MODEL_CONFIGS
    preset = os.environ.get("SYNTH_PRESET", "").strip().lower()
    if preset in ("a100_fast", "fast", "colab"):
        TOKENS_PER_EPOCH = int(os.environ.get("SYNTH_TOKENS_PER_EPOCH", "5000000"))
        N_EPOCHS = int(os.environ.get("SYNTH_N_EPOCHS", "4"))
        N_VAL_TOKENS = int(os.environ.get("SYNTH_N_VAL_TOKENS", "250000"))
        MODEL_CONFIGS = dict(_A100_FAST_CONFIGS)
    elif preset == "smoke":
        TOKENS_PER_EPOCH = int(os.environ.get("SYNTH_TOKENS_PER_EPOCH", "2000000"))
        N_EPOCHS = int(os.environ.get("SYNTH_N_EPOCHS", "3"))
        N_VAL_TOKENS = int(os.environ.get("SYNTH_N_VAL_TOKENS", "100000"))
        MODEL_CONFIGS = dict(_DEFAULT_MODEL_CONFIGS)
    elif preset in ("full", ""):
        pass
    else:
        raise ValueError(f"Unknown SYNTH_PRESET={preset!r}; use a100_fast, smoke, or full")


def _apply_active_vocab(vocab_size: int):
    """Shrink softmax/embedding so ~1–3M param models are meaningful."""
    oce.VOCAB_SIZE = vocab_size
    if vocab_size <= 30000:
        bounds = [
            ("top_100", 0, min(100, vocab_size)),
            ("100_1K", 100, min(1000, vocab_size)),
            ("1K_plus", 1000, vocab_size),
        ]
        oce.BUCKETS = [(n, lo, hi) for n, lo, hi in bounds if lo < hi]


def zipf_probs(vocab_size: int, exponent: float) -> np.ndarray:
    ranks = np.arange(1, vocab_size + 1, dtype=np.float64)
    w = ranks ** (-exponent)
    p = w / w.sum()
    return p


def entropy_nats(p: np.ndarray) -> float:
    p = np.maximum(p, 1e-300)
    return float(-np.sum(p * np.log(p)))


def sample_tokens(n: int, p: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Sample n i.i.d. token ids from categorical(p)."""
    cdf = np.cumsum(p)
    u = rng.random(n)
    ids = np.searchsorted(cdf, u).astype(np.uint16)
    return ids


def _cache_matches_protocol(meta: dict, train_len: int, val_len: int) -> tuple[bool, str]:
    """Return (ok, reason) for reusing cached token files."""
    if meta.get("active_vocab") != ACTIVE_VOCAB:
        return False, f"active_vocab {meta.get('active_vocab')} != {ACTIVE_VOCAB}"
    if meta.get("zipf_s") != ZIPF_EXPONENT:
        return False, f"zipf_s {meta.get('zipf_s')} != {ZIPF_EXPONENT}"
    if meta.get("tokens_per_epoch") != TOKENS_PER_EPOCH:
        return False, (
            f"tokens_per_epoch {meta.get('tokens_per_epoch')} != {TOKENS_PER_EPOCH}"
        )
    if meta.get("n_val_tokens") != N_VAL_TOKENS:
        return False, f"n_val_tokens {meta.get('n_val_tokens')} != {N_VAL_TOKENS}"
    if meta.get("seed") != SEED:
        return False, f"seed {meta.get('seed')} != {SEED}"
    if train_len < TOKENS_PER_EPOCH:
        return False, f"train buffer {train_len:,} < {TOKENS_PER_EPOCH:,}"
    if val_len < N_VAL_TOKENS:
        return False, f"val buffer {val_len:,} < {N_VAL_TOKENS:,}"
    return True, ""


def prepare_synthetic_tokens():
    cache_dir = os.path.join(OUT_DIR, "cache")
    os.makedirs(cache_dir, exist_ok=True)
    train_path = os.path.join(cache_dir, "synth_train_tokens.npy")
    val_path = os.path.join(cache_dir, "synth_val_tokens.npy")
    meta_path = os.path.join(cache_dir, "synth_meta.json")

    if (
        os.path.exists(train_path)
        and os.path.exists(val_path)
        and os.path.exists(meta_path)
    ):
        with open(meta_path, encoding="utf-8") as f:
            meta = json.load(f)
        train = np.load(train_path, mmap_mode="r")
        val = np.load(val_path, mmap_mode="r")
        ok, reason = _cache_matches_protocol(meta, len(train), len(val))
        if ok:
            print(f"[data] loaded cache: train={len(train):,} val={len(val):,}")
            print(f"       H_true={meta['H_true_nats']:.6f} nats (Zipf s={meta['zipf_s']})")
            return train, val, meta
        print(f"[data] cache stale ({reason}); regenerating")

    p = zipf_probs(ACTIVE_VOCAB, ZIPF_EXPONENT)
    h_true = entropy_nats(p)
    rng = np.random.default_rng(SEED)
    print(
        f"[data] generating i.i.d. Zipf tokens (V={ACTIVE_VOCAB}, s={ZIPF_EXPONENT}, "
        f"H_true={h_true:.6f} nats)"
    )
    val = sample_tokens(N_VAL_TOKENS, p, rng)
    train = sample_tokens(TOKENS_PER_EPOCH, p, rng)
    np.save(train_path, train)
    np.save(val_path, val)
    meta = dict(
        zipf_s=ZIPF_EXPONENT,
        H_true_nats=h_true,
        vocab_size=ACTIVE_VOCAB,
        tokens_per_epoch=TOKENS_PER_EPOCH,
        n_val_tokens=N_VAL_TOKENS,
        seed=SEED,
        active_vocab=ACTIVE_VOCAB,
    )
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)
    print(f"[data] saved {train_path}")
    return train, val, meta


def _patch_oce_globals():
    """Align imported OWT module with synthetic scale."""
    oce.TOKENS_PER_EPOCH = TOKENS_PER_EPOCH
    oce.N_EPOCHS = N_EPOCHS
    oce.OUT_DIR = OUT_DIR
    oce.SEED = SEED


def train_one_synthetic(name, cfg, train_tokens, val_tokens, bucket_of, unigram_logp):
    """Same training loop as owt_chinchilla_e.train_one (imports helpers)."""
    import torch

    print(f"\n{'=' * 70}\nTRAINING {name}\n{'=' * 70}")
    torch.manual_seed(SEED)
    np.random.seed(SEED)

    model = oce.TransformerLM(cfg, unigram_logp=unigram_logp.to(oce.DEVICE)).to(oce.DEVICE)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  params: {n_params:,}")

    batch, accum, lr = cfg["batch"], cfg["accum"], cfg["lr"]
    tokens_per_step = batch * accum * oce.SEQ_LEN
    steps_per_epoch = TOKENS_PER_EPOCH // tokens_per_step
    if steps_per_epoch < 10:
        raise ValueError(
            f"{name}: only {steps_per_epoch} steps/epoch; increase SYNTH_TOKENS_PER_EPOCH "
            f"or reduce batch*accum*seq_len"
        )
    total_steps = steps_per_epoch * N_EPOCHS
    print(f"  tokens/step={tokens_per_step}  steps/epoch={steps_per_epoch}  total={total_steps}")

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=lr, betas=(0.9, 0.95), weight_decay=0.1, eps=1e-8
    )

    warmup = min(oce.WARMUP_STEPS, max(10, total_steps // 8))

    def lr_at(step):
        if step < warmup:
            return step / max(1, warmup)
        prog = (step - warmup) / max(1, total_steps - warmup)
        return max(0.1, 0.5 * (1 + math.cos(math.pi * prog)))

    loader = oce.make_loader(train_tokens, batch, oce.SEQ_LEN, seed=SEED)
    it = iter(loader)
    scaler = torch.amp.GradScaler("cuda", enabled=(oce.DTYPE == torch.float16))

    log_steps, log_ce = [], []
    epoch_results = []
    t_start = time.time()

    for ep in range(1, N_EPOCHS + 1):
        running = 0.0
        running_n = 0
        for s in range(1, steps_per_epoch + 1):
            global_step = (ep - 1) * steps_per_epoch + s
            for g in optimizer.param_groups:
                g["lr"] = lr * lr_at(global_step)
            optimizer.zero_grad(set_to_none=True)
            for _ in range(accum):
                xb, yb = next(it)
                xb = xb.to(oce.DEVICE, non_blocking=True)
                yb = yb.to(oce.DEVICE, non_blocking=True)
                with oce._autocast_ctx():
                    out = model(xb, yb)
                loss = out.loss / accum
                if scaler.is_enabled():
                    scaler.scale(loss).backward()
                else:
                    loss.backward()
                running += float(out.loss.detach())
                running_n += 1
            if scaler.is_enabled():
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), oce.GRAD_CLIP)
                scaler.step(optimizer)
                scaler.update()
            else:
                torch.nn.utils.clip_grad_norm_(model.parameters(), oce.GRAD_CLIP)
                optimizer.step()
            if global_step % 50 == 0:
                ce_avg = running / max(1, running_n)
                log_steps.append(global_step)
                log_ce.append(ce_avg)
                running = 0.0
                running_n = 0

        if oce.DEVICE == "cuda":
            gc.collect()
            torch.cuda.empty_cache()

        t_seen = ep * steps_per_epoch * tokens_per_step
        eval_bs = min(batch * 2, 64)
        val_ce, bucket_ce, bucket_w = oce.evaluate(
            model, val_tokens, batch=eval_bs, seq_len=oce.SEQ_LEN, bucket_of=bucket_of
        )
        fit = oce.h8_fit(log_steps, log_ce, t_min=oce._h8_fit_t_min(steps_per_epoch))
        rec = dict(
            epoch=ep,
            T_tokens=t_seen,
            val_ce=val_ce,
            bucket_ce=bucket_ce,
            bucket_w=bucket_w,
            h8=fit,
        )
        epoch_results.append(rec)
        cstar_str = f"{fit['Cstar']:.4f}" if fit else "NA"
        print(
            f"  END EP{ep}: T={t_seen/1e6:.1f}M  val_ce={val_ce:.4f}  "
            f"C*={cstar_str}  (elapsed={(time.time()-t_start)/60:.1f} min)"
        )

        oce._json_dump_atomic(
            os.path.join(OUT_DIR, f"{name}.json"),
            dict(
                name=name,
                cfg=cfg,
                n_params=n_params,
                epochs=epoch_results,
                train_log=dict(steps=log_steps, ce=log_ce),
            ),
        )

    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return dict(name=name, cfg=cfg, n_params=n_params, epochs=epoch_results)


def run_validation(h_true: float, tri3_free: dict, tri3_034: dict, results: dict) -> dict:
    e_free = float(tri3_free["E_true"])
    e_fixed = float(tri3_034["E_true"])
    largest = results[TRAIN_ORDER[-1]]
    final_val = float(largest["epochs"][-1]["val_ce"])
    e_apps = [tri3_free["E_app_list"][i] for i in range(3)]

    report = dict(
        H_true_nats=h_true,
        E_true_free_alpha=e_free,
        E_true_fixed_alpha_034=e_fixed,
        delta_free=abs(e_free - h_true),
        delta_fixed=abs(e_fixed - h_true),
        final_val_ce_largest=final_val,
        delta_val=abs(final_val - h_true),
        E_app_monotonic=all(e_apps[i] > e_apps[i + 1] for i in range(2)),
        gates=dict(
            P1_pass=abs(e_free - h_true) < GATE_P1,
            P2_pass=abs(e_fixed - h_true) < GATE_P2,
            S1_pass=abs(final_val - h_true) < GATE_S1,
            S2_pass=all(e_apps[i] > e_apps[i + 1] for i in range(2)),
        ),
        gate_thresholds=dict(P1=GATE_P1, P2=GATE_P2, S1=GATE_S1),
    )
    report["primary_pass"] = report["gates"]["P1_pass"] and report["gates"]["P2_pass"]
    return report


def write_report(
    meta: dict,
    tri2: dict,
    tri3_free: dict,
    tri3_034: dict,
    validation: dict,
    results: dict,
) -> str:
    h = meta["H_true_nats"]
    lines = [
        "=" * 72,
        "SYNTHETIC CHINCHILLA-E VALIDATION",
        "=" * 72,
        "",
        f"Corpus: i.i.d. Zipf unigram (V={ACTIVE_VOCAB}, s={meta['zipf_s']})",
        f"H_true (analytic): {h:.6f} nats",
        f"Tokens/epoch: {TOKENS_PER_EPOCH:,}  Epochs: {N_EPOCHS}",
        "",
        "MODELS:",
    ]
    for name in TRAIN_ORDER:
        r = results[name]
        ep = r["epochs"][-1]
        t, c = oce._collect_T_C(r)
        e_app, beta, _ = oce._fit3_T_C(t, c)
        lines.append(
            f"  {name}: {r['n_params']:,} params  E_app={e_app:.4f}  "
            f"beta={beta:.4f}  val_ce={ep['val_ce']:.4f}"
        )
    lines.extend(
        [
            "",
            "TRIANGULATION:",
            f"  Two-point (alpha=0.34): E_true = {tri2['E_true']:.4f} nats",
            f"  Three-point free alpha: E_true = {tri3_free['E_true']:.4f} nats  "
            f"(alpha={tri3_free['alpha_used']:.4f})",
            f"  Three-point fixed alpha=0.34: E_true = {tri3_034['E_true']:.4f} nats",
            "",
            "VALIDATION vs H_true:",
            f"  |E_true free - H_true|  = {validation['delta_free']:.4f}  "
            f"(gate P1 < {GATE_P1}: {validation['gates']['P1_pass']})",
            f"  |E_true fixed - H_true| = {validation['delta_fixed']:.4f}  "
            f"(gate P2 < {GATE_P2}: {validation['gates']['P2_pass']})",
            f"  |val_ce largest - H_true| = {validation['delta_val']:.4f}  "
            f"(gate S1 < {GATE_S1}: {validation['gates']['S1_pass']})",
            f"  E_app monotonic in N: {validation['E_app_monotonic']}  "
            f"(gate S2: {validation['gates']['S2_pass']})",
            "",
            f"PRIMARY PASS (P1 and P2): {validation['primary_pass']}",
            "",
            "=" * 72,
        ]
    )
    return "\n".join(lines)


def main():
    _apply_preset()
    _apply_active_vocab(ACTIVE_VOCAB)
    _patch_oce_globals()
    preset = os.environ.get("SYNTH_PRESET", "full").strip().lower() or "full"
    print(f"OUT_DIR: {OUT_DIR}")
    print(f"DEVICE: {oce.DEVICE}")
    print(f"Preset: {preset}")
    print(f"Protocol: {TOKENS_PER_EPOCH/1e6:.1f}M tok/epoch x {N_EPOCHS} epochs")
    print(f"Pre-registration: docs/PREREGISTER_synthetic_chinchilla_e.md")
    print()

    train_tokens, val_tokens, meta = prepare_synthetic_tokens()
    h_true = float(meta["H_true_nats"])

    counts = oce.get_unigram_counts(train_tokens.astype(np.int64))
    bucket_of = oce.assign_buckets_by_rank(counts)
    p = counts / counts.sum()
    p = np.maximum(p, 1e-12)
    unigram_logp = __import__("torch").from_numpy(np.log(p).astype(np.float32))

    results = {}
    for name in TRAIN_ORDER:
        results[name] = train_one_synthetic(
            name,
            MODEL_CONFIGS[name],
            train_tokens,
            val_tokens,
            bucket_of,
            unigram_logp,
        )
        oce.save_checkpoint_after_model(name, results[name], results)

    tri2 = oce.chinchilla_triangulate(results["S_1M"], results["S_3M"], alpha=0.34)
    tri3_free = oce.chinchilla_triangulate_three(
        results["S_1M"], results["S_2M"], results["S_3M"], alpha=None
    )
    tri3_034 = oce.chinchilla_triangulate_three(
        results["S_1M"], results["S_2M"], results["S_3M"], alpha=0.34
    )

    validation = run_validation(h_true, tri3_free, tri3_034, results)

    tri_path = os.path.join(OUT_DIR, "triangulation.json")
    oce._json_dump_atomic(
        tri_path,
        dict(
            meta=meta,
            two_point_alpha_034=tri2,
            three_point_free=tri3_free,
            three_point_fixed_034=tri3_034,
            validation=validation,
        ),
    )

    msg = write_report(meta, tri2, tri3_free, tri3_034, validation, results)
    print(msg)
    report_path = os.path.join(OUT_DIR, "validation_report.txt")
    with open(report_path + ".tmp", "w", encoding="utf-8") as f:
        f.write(msg)
    os.replace(report_path + ".tmp", report_path)
    print(f"[done] {report_path}")
    print(f"[done] {tri_path}")


if __name__ == "__main__":
    main()
