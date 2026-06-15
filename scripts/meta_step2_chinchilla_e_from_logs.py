"""
Zero-GPU Chinchilla-E triangulation from Meta FAIR Step-2 scaling logs (public CSV).

Uses matched ladder CSVs vendored under data/public_logs/meta_step2/
(FS-step2v2_* with identical token budget ti134698). Corpus is the Step-2
English web mix used in those scaling runs (not The Pile / Dolma).

    pip install numpy scipy pandas
    python scripts/meta_step2_chinchilla_e_from_logs.py

Override log directory:
    META_STEP2_LOG_DIR=/path/to/fetched python scripts/meta_step2_chinchilla_e_from_logs.py
"""

from __future__ import annotations

import argparse
import glob
import json
import math
import os
import re
import sys
from typing import Iterable

import numpy as np
import pandas as pd

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.abspath(os.path.join(_SCRIPT_DIR, ".."))
if _SCRIPT_DIR not in sys.path:
    sys.path.insert(0, _SCRIPT_DIR)

import owt_chinchilla_e as oce  # noqa: E402

VOCAB_SIZE = 50257
LATE_TOKEN_FRAC = 0.35  # ignore earliest ~35% of tokens for C* bands
N_PSEUDO_EPOCHS = 6
TOKEN_BUDGET_TAG = "ti134698"

# Matched ladder on the same Step-2 token budget (filename suffix ti134698).
DEFAULT_TRIPLET = ("h832", "h1024", "h1280")

LADDER = {
    "h832": dict(
        glob=f"FS-step2v2_*_sc_h832_*_{TOKEN_BUDGET_TAG}.csv",
        label="Step2-h832-L12",
    ),
    "h1024": dict(
        glob=f"FS-step2v2_*_sc_h1024_*_{TOKEN_BUDGET_TAG}.csv",
        label="Step2-h1024-L16",
    ),
    "h1280": dict(
        glob=f"FS-step2v2_*_sc_h1280_*_{TOKEN_BUDGET_TAG}.csv",
        label="Step2-h1280-L20",
    ),
    "h896": dict(
        glob=f"FS-step2v2_*_sc_h896_*_{TOKEN_BUDGET_TAG}.csv",
        label="Step2-h896-L14",
    ),
    "h1152": dict(
        glob=f"FS-step2v2_*_sc_h1152_*_{TOKEN_BUDGET_TAG}.csv",
        label="Step2-h1152-L18",
    ),
}

_DEFAULT_LOG_DIR = os.path.join(_REPO, "data", "public_logs", "meta_step2")
_FALLBACK_LOG_DIR = os.path.abspath(
    os.path.join(_REPO, "..", "chinchilla-slope-one-repro", "data", "external", "fetched")
)
LOG_DIR = os.environ.get("META_STEP2_LOG_DIR") or (
    _DEFAULT_LOG_DIR if os.path.isdir(_DEFAULT_LOG_DIR) else _FALLBACK_LOG_DIR
)
OUT_DIR = os.environ.get(
    "META_STEP2_CHINCHILLA_DIR",
    os.path.join(_REPO, "results", "meta_step2_chinchilla_from_logs"),
)
OUT_DIR = os.path.abspath(OUT_DIR)


def _estimate_params(filename: str) -> int:
    m = re.search(r"sc_h(\d+)_ffnh(\d+)_numh(\d+)_numl(\d+)", filename)
    if not m:
        raise ValueError(f"Cannot parse architecture from {filename}")
    d_model, ffn, _n_heads, n_layers = map(int, m.groups())
    per_layer = 12 * d_model * d_model + 2 * d_model * ffn
    return int(n_layers * per_layer + VOCAB_SIZE * d_model * 2)


def _resolve_csv(size_key: str) -> str:
    meta = LADDER[size_key]
    matches = sorted(glob.glob(os.path.join(LOG_DIR, meta["glob"])))
    if not matches:
        raise FileNotFoundError(
            f"No CSV for {size_key} under {LOG_DIR} (glob {meta['glob']}). "
            f"Copy FS-step2v2 CSVs into data/public_logs/meta_step2/ or set META_STEP2_LOG_DIR."
        )
    return matches[0]


def load_curve(size_key: str) -> tuple[pd.DataFrame, str]:
    path = _resolve_csv(size_key)
    df = pd.read_csv(path)
    needed = {"step", "ce", "tokens"}
    if not needed.issubset(df.columns):
        raise ValueError(f"{path} missing columns; need {needed}, got {list(df.columns)}")
    df = df.sort_values("tokens").drop_duplicates("tokens", keep="last")
    df = df.rename(columns={"ce": "loss"})
    return df, path


def _pseudo_epoch_bounds(token_min: float, token_max: float, n_epochs: int) -> list[tuple[float, float]]:
    edges = np.linspace(token_min, token_max, n_epochs + 1)
    bounds = []
    for i in range(n_epochs):
        lo, hi = float(edges[i]), float(edges[i + 1])
        if hi > lo:
            bounds.append((lo, hi))
    return bounds


def build_result_from_log(name: str, size_key: str, df: pd.DataFrame, csv_path: str) -> dict:
    token_max = float(df["tokens"].max())
    token_min = token_max * LATE_TOKEN_FRAC
    bounds = _pseudo_epoch_bounds(token_min, token_max, N_PSEUDO_EPOCHS)

    epoch_results = []
    for ep, (lo, hi) in enumerate(bounds, start=1):
        band = df[(df["tokens"] >= lo) & (df["tokens"] <= hi)]
        if len(band) < 5:
            continue
        steps = band["step"].to_numpy(dtype=np.float64)
        losses = band["loss"].to_numpy(dtype=np.float64)
        tokens_x = band["tokens"].to_numpy(dtype=np.float64)
        end_tokens = float(hi)
        val_ce = float(band["loss"].iloc[-1])
        fit = oce.h8_fit(tokens_x, losses, t_min=max(lo, float(tokens_x.min())))
        epoch_results.append(
            dict(
                epoch=ep,
                T_tokens=end_tokens,
                val_ce=val_ce,
                step_end=float(band["step"].iloc[-1]),
                h8=fit,
            )
        )

    if len(epoch_results) < 3:
        raise ValueError(f"{name}: only {len(epoch_results)} pseudo-epochs; need >= 3")

    n_params = _estimate_params(os.path.basename(csv_path))
    return dict(
        name=name,
        size_key=size_key,
        label=LADDER[size_key]["label"],
        n_params=n_params,
        corpus="Meta FAIR Step-2 English mix (ti134698 matched ladder)",
        source_csv=os.path.basename(csv_path),
        total_tokens=int(token_max),
        final_train_loss=float(df["loss"].iloc[-1]),
        epochs=epoch_results,
    )


def _e_app_last_cstar(result: dict) -> float:
    last = result["epochs"][-1]
    h8 = last.get("h8") or {}
    return float(h8["Cstar"])


def _flat_result(n_params: int, e_app: float, name: str = "flat") -> dict:
    """OWT-compatible dict with flat C* so _fit3_T_C returns E_app ≈ e_app."""
    return dict(
        name=name,
        n_params=n_params,
        epochs=[
            dict(T_tokens=(i + 1) * 100_000_000_000, h8=dict(Cstar=e_app), val_ce=e_app)
            for i in range(N_PSEUDO_EPOCHS)
        ],
    )


def triangulate_ladder(results: dict[str, dict]) -> dict:
    keys = sorted(results.keys(), key=lambda k: results[k]["n_params"])
    if len(keys) < 3:
        raise ValueError("Need at least 3 models for triangulation")
    small, mid, large = keys[0], keys[len(keys) // 2], keys[-1]

    flat = {
        k: _flat_result(results[k]["n_params"], _e_app_last_cstar(results[k]), results[k]["name"])
        for k in keys
    }
    r_s, r_m, r_l = flat[small], flat[mid], flat[large]

    tri2 = oce.chinchilla_triangulate(r_s, r_l, alpha=0.34)
    tri3_free = oce.chinchilla_triangulate_three(r_s, r_m, r_l, alpha=None)
    tri3_034 = oce.chinchilla_triangulate_three(r_s, r_m, r_l, alpha=0.34)

    rows = []
    for k in keys:
        r = results[k]
        e_app = _e_app_last_cstar(r)
        t, c = oce._collect_T_C(r)
        _, beta, _ = oce._fit3_T_C(t, c)
        rows.append(
            dict(
                size=k,
                label=r["label"],
                n_params=r["n_params"],
                E_app=e_app,
                beta=beta,
                final_loss=r["final_train_loss"],
            )
        )

    return dict(
        triplet=dict(small=small, mid=mid, large=large),
        per_model=rows,
        two_point_alpha_034=tri2,
        three_point_free=tri3_free,
        three_point_fixed_034=tri3_034,
        note="E_app = last pseudo-epoch C* (runs are >> knee; inner T-sweep fit hits floor bound)",
    )


def write_report(tri: dict) -> str:
    t3 = tri["three_point_free"]
    t34 = tri["three_point_fixed_034"]
    lines = [
        "=" * 72,
        "META STEP-2 CHINCHILLA-E FROM PUBLIC LOGS (ZERO TRAINING)",
        "=" * 72,
        "",
        "Corpus: Meta FAIR Step-2 English web mix (matched ti134698 ladder)",
        f"Metric: train CE vs tokens; E_app = late C* (runs at ~256B tok, past knee)",
        "",
        "LADDER:",
    ]
    for row in tri["per_model"]:
        lines.append(
            f"  {row['label']:18s}  {row['n_params']/1e6:6.1f}M params  "
            f"E_app={row['E_app']:.4f}  beta={row['beta']:.4f}  "
            f"final_loss={row['final_loss']:.4f}"
        )
    lines.extend(
        [
            "",
            "TRIANGULATION (E_app(N) = E_true + A*N^{-alpha}):",
            f"  Two-point (alpha=0.34):          E_true = {tri['two_point_alpha_034']['E_true']:.4f} nats",
            f"  Three-point free alpha:          E_true = {t3['E_true']:.4f} nats  "
            f"(alpha={t3['alpha_used']:.4f})",
            f"  Three-point fixed alpha=0.34:    E_true = {t34['E_true']:.4f} nats",
            "",
            "NOTE: Independent corpus/stack from Pythia/The Pile — compare floors qualitatively.",
            "",
            "=" * 72,
        ]
    )
    return "\n".join(lines)


def parse_sizes(s: str) -> list[str]:
    keys = [x.strip().lower() for x in s.split(",") if x.strip()]
    for k in keys:
        if k not in LADDER:
            raise ValueError(f"Unknown size {k!r}; choose from {list(LADDER)}")
    if len(keys) < 3:
        raise ValueError("Need at least 3 sizes for triangulation")
    return keys


def main(argv: Iterable[str] | None = None):
    if hasattr(sys.stdout, "reconfigure"):
        try:
            sys.stdout.reconfigure(encoding="utf-8")
        except Exception:
            pass

    ap = argparse.ArgumentParser(description="Meta Step-2 log-only Chinchilla-E triangulation")
    ap.add_argument(
        "--sizes",
        default=",".join(DEFAULT_TRIPLET),
        help="Comma-separated ladder (default: h832,h1024,h1280)",
    )
    args = ap.parse_args(list(argv) if argv is not None else None)

    os.makedirs(OUT_DIR, exist_ok=True)
    size_keys = parse_sizes(args.sizes)

    print(f"OUT_DIR: {OUT_DIR}")
    print(f"LOG_DIR: {LOG_DIR}")
    print(f"Ladder: {size_keys}")
    print()

    results = {}
    for sk in size_keys:
        df, path = load_curve(sk)
        name = f"step2_{sk}"
        results[sk] = build_result_from_log(name, sk, df, path)
        ep = results[sk]["epochs"][-1]
        cstar = ep["h8"]["Cstar"] if ep.get("h8") else float("nan")
        print(
            f"[{LADDER[sk]['label']}] tokens={results[sk]['total_tokens']/1e9:.2f}B  "
            f"params={results[sk]['n_params']/1e6:.1f}M  "
            f"final_loss={results[sk]['final_train_loss']:.4f}  "
            f"last C*={cstar:.4f}"
        )
        oce._json_dump_atomic(os.path.join(OUT_DIR, f"{name}.json"), results[sk])

    tri = triangulate_ladder(results)
    report = write_report(tri)
    print()
    print(report)

    oce._json_dump_atomic(
        os.path.join(OUT_DIR, "triangulation.json"),
        dict(
            protocol=dict(
                corpus="Meta FAIR Step-2",
                source="FS-step2v2 public CSV (data/public_logs/meta_step2)",
                token_budget_tag=TOKEN_BUDGET_TAG,
                late_token_frac=LATE_TOKEN_FRAC,
                n_pseudo_epochs=N_PSEUDO_EPOCHS,
            ),
            results={
                k: dict(n_params=v["n_params"], final_train_loss=v["final_train_loss"])
                for k, v in results.items()
            },
            triangulation=tri,
        ),
    )
    report_path = os.path.join(OUT_DIR, "validation_report.txt")
    with open(report_path + ".tmp", "w", encoding="utf-8") as f:
        f.write(report)
    os.replace(report_path + ".tmp", report_path)
    print(f"[done] {report_path}")


if __name__ == "__main__":
    main()
