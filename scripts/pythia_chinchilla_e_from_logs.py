"""
Zero-GPU Chinchilla-E triangulation from public Pythia training logs.

EleutherAI published step-wise train/lm_loss curves (W&B export on GitHub).
This script downloads them, builds the same (T, C*) records as owt_chinchilla_e.py,
and triangulates E_true for **The Pile** under the Pythia/GPT-NeoX stack.

No training. No GPU. No W&B API key required (uses public TSV cache).

    pip install numpy scipy pandas
    python scripts/pythia_chinchilla_e_from_logs.py

Optional W&B refresh (needs wandb login):
    python scripts/pythia_chinchilla_e_from_logs.py --source wandb

Data: EleutherAI/pythia PR branch training_losses/data/*.tsv
      https://github.com/EleutherAI/pythia/tree/ccdf77005e27f2b6811d85ca145532cc502180b6/hmm-training-maps/training_losses
"""

from __future__ import annotations

import argparse
import io
import json
import math
import os
import sys
import urllib.request
from typing import Iterable

import numpy as np
import pandas as pd

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if _SCRIPT_DIR not in sys.path:
    sys.path.insert(0, _SCRIPT_DIR)

import owt_chinchilla_e as oce  # noqa: E402

# ---------------------------------------------------------------------------
# Pythia protocol (matched ladder on The Pile)
# ---------------------------------------------------------------------------
TOKENS_PER_STEP = 2_097_152  # 2M tokens / optimizer step (Pythia v1)
TOTAL_STEPS = 143_000
LATE_STEP_MIN = 50_000  # exclude early transient for C* bands
N_PSEUDO_EPOCHS = 6

PYTHIA_TSV_BASE = (
    "https://raw.githubusercontent.com/EleutherAI/pythia/"
    "ccdf77005e27f2b6811d85ca145532cc502180b6/hmm-training-maps/training_losses/data"
)

# Non-deduped Pile ladder; n_params from HF model cards (GPT-NeoX tokenizer, vocab 50304)
# TSV: EleutherAI GitHub (14m–410m). CSV: W&B exports in data/public_logs/pythia/ (1.4b, 6.9b).
_REPO = os.path.abspath(os.path.join(_SCRIPT_DIR, ".."))
_PUBLIC_PYTHIA = os.path.join(_REPO, "data", "public_logs", "pythia")
_FALLBACK_PYTHIA = os.path.abspath(
    os.path.join(_REPO, "..", "chinchilla-slope-one-repro", "data", "external", "fetched")
)

PYTHIA_LADDER = {
    "14m": dict(n_params=14_015_104, tsv="14m-seed1.tsv", label="Pythia-14M"),
    "70m": dict(n_params=70_415_744, tsv="70m-seed1.tsv", label="Pythia-70M"),
    "160m": dict(n_params=157_092_864, tsv="160m-seed5.tsv", label="Pythia-160M"),
    "410m": dict(n_params=405_736_448, tsv="410m-seed1.tsv", label="Pythia-410M"),
    "1.4b": dict(n_params=1_451_786_752, csv="Pythia-1_4b.csv", label="Pythia-1.4B"),
    "6.9b": dict(n_params=6_857_302_016, csv="Pythia-6.9b.csv", label="Pythia-6.9B"),
    # Optional — incomplete or mismatched step protocols; not in default ladder
    "1b": dict(n_params=1_010_560_000, csv="Pythia-1b.csv", label="Pythia-1B"),
    "2.8b": dict(n_params=2_807_892_992, csv="Pythia-2.8b.csv", label="Pythia-2.8B"),
    "12b": dict(n_params=11_827_680_256, csv="Pythia-12b.csv", label="Pythia-12B"),
}

# Six-point overdetermined ladder (4 GitHub TSV + 2 vendored W&B CSV)
DEFAULT_LADDER = ("14m", "70m", "160m", "410m", "1.4b", "6.9b")
DEFAULT_TRIPLET = DEFAULT_LADDER[:3]  # backward compat

OUT_DIR = os.environ.get(
    "PYTHIA_CHINCHILLA_DIR",
    os.path.join(_SCRIPT_DIR, "..", "results", "pythia_chinchilla_from_logs"),
)
OUT_DIR = os.path.abspath(OUT_DIR)


def _download(url: str, dest: str) -> str:
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    if os.path.isfile(dest):
        return dest
    print(f"[fetch] {url}")
    with urllib.request.urlopen(url, timeout=120) as resp:
        data = resp.read()
    tmp = dest + ".tmp"
    with open(tmp, "wb") as f:
        f.write(data)
    os.replace(tmp, dest)
    return dest


def _pythia_csv_path(csv_name: str) -> str:
    for base in (_PUBLIC_PYTHIA, _FALLBACK_PYTHIA):
        path = os.path.join(base, csv_name)
        if os.path.isfile(path):
            return path
    return os.path.join(_PUBLIC_PYTHIA, csv_name)


def load_loss_tsv(size_key: str, cache_dir: str) -> pd.DataFrame:
    """Load (step, loss) for a Pythia size from GitHub TSV or vendored W&B CSV."""
    meta = PYTHIA_LADDER[size_key]
    tsv_name = meta.get("tsv")
    csv_name = meta.get("csv")
    if tsv_name:
        path = os.path.join(cache_dir, tsv_name)
        _download(f"{PYTHIA_TSV_BASE}/{tsv_name}", path)
        df = pd.read_csv(path, sep="\t")
        df = df.rename(columns={"train/lm_loss": "loss"})
    elif csv_name:
        path = _pythia_csv_path(csv_name)
        if not os.path.isfile(path):
            raise FileNotFoundError(
                f"No CSV for {size_key}: expected {path}. "
                f"Copy {csv_name} into data/public_logs/pythia/ — see data/public_logs/README.md"
            )
        df = pd.read_csv(path)
        df = df.rename(columns={"ce": "loss"})
    else:
        raise ValueError(f"No public log path for {size_key}")
    if "step" not in df.columns or "loss" not in df.columns:
        raise ValueError(f"Unexpected columns in {path}: {list(df.columns)}")
    df = df.sort_values("step").drop_duplicates("step", keep="last")
    return df


def _pseudo_epoch_bounds(step_min: int, step_max: int, n_epochs: int) -> list[tuple[int, int]]:
    edges = np.linspace(step_min, step_max, n_epochs + 1, dtype=int)
    bounds = []
    for i in range(n_epochs):
        lo = int(edges[i])
        hi = int(edges[i + 1])
        if hi > lo:
            bounds.append((lo, hi))
    return bounds


def build_result_from_log(
    name: str,
    size_key: str,
    df: pd.DataFrame,
    n_pseudo_epochs: int = N_PSEUDO_EPOCHS,
) -> dict:
    """Map public (step, train/lm_loss) → OWT-compatible result dict."""
    meta = PYTHIA_LADDER[size_key]
    step_max = int(df["step"].max())
    step_min = max(LATE_STEP_MIN, int(df["step"].min()))
    bounds = _pseudo_epoch_bounds(step_min, step_max, n_pseudo_epochs)

    epoch_results = []
    for ep, (lo, hi) in enumerate(bounds, start=1):
        band = df[(df["step"] >= lo) & (df["step"] <= hi)]
        if len(band) < 5:
            continue
        steps = band["step"].to_numpy(dtype=np.float64)
        losses = band["loss"].to_numpy(dtype=np.float64)
        end_step = int(hi)
        t_tokens = end_step * TOKENS_PER_STEP
        val_ce = float(df.loc[df["step"] == end_step, "loss"].iloc[0])
        fit = oce.h8_fit(steps, losses, t_min=max(lo, step_min))
        epoch_results.append(
            dict(
                epoch=ep,
                T_tokens=t_tokens,
                val_ce=val_ce,
                step_end=end_step,
                h8=fit,
            )
        )

    if len(epoch_results) < 3:
        raise ValueError(f"{name}: only {len(epoch_results)} pseudo-epochs; need ≥3")

    return dict(
        name=name,
        size_key=size_key,
        label=meta["label"],
        n_params=meta["n_params"],
        corpus="The Pile (Pythia non-deduped ladder)",
        tokens_per_step=TOKENS_PER_STEP,
        total_steps=int(df["step"].max()),
        final_train_loss=float(df["loss"].iloc[-1]),
        epochs=epoch_results,
        train_log=dict(
            steps=df["step"].tolist()[:: max(1, len(df) // 500)],
            ce=df["loss"].tolist()[:: max(1, len(df) // 500)],
        ),
    )


def _e_app_from_result(r: dict) -> float:
    """Prefer late-epoch C* (stable asymptote); fall back to full-curve fit."""
    if r.get("epochs"):
        ep = r["epochs"][-1]
        h8 = ep.get("h8")
        if h8 and h8.get("Cstar") is not None:
            return float(h8["Cstar"])
    t, c = oce._collect_T_C(r)
    e_app, _, _ = oce._fit3_T_C(t, c)
    return float(e_app)


def triangulate_ladder(results: dict[str, dict]) -> dict:
    keys = sorted(results.keys(), key=lambda k: results[k]["n_params"])
    if len(keys) < 3:
        raise ValueError("Need at least 3 models for triangulation")
    small, mid, large = keys[0], keys[len(keys) // 2], keys[-1]
    r_s, r_m, r_l = results[small], results[mid], results[large]

    tri2 = oce.chinchilla_triangulate(r_s, r_l, alpha=0.34)
    tri3_free = oce.chinchilla_triangulate_three(r_s, r_m, r_l, alpha=None)
    tri3_034 = oce.chinchilla_triangulate_three(r_s, r_m, r_l, alpha=0.34)

    rows = []
    n_list, e_list = [], []
    for k in keys:
        r = results[k]
        t, c = oce._collect_T_C(r)
        e_app = _e_app_from_result(r)
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
        n_list.append(r["n_params"])
        e_list.append(e_app)

    n_point = oce.chinchilla_fit_eapp_ladder(n_list, e_list, alpha=0.34)

    return dict(
        ladder_sizes=keys,
        triplet=dict(small=small, mid=mid, large=large),
        per_model=rows,
        two_point_alpha_034=tri2,
        three_point_free=tri3_free,
        three_point_fixed_034=tri3_034,
        n_point_fixed_034=n_point,
    )


def write_report(tri: dict, results: dict) -> str:
    t3 = tri["three_point_free"]
    t34 = tri["three_point_fixed_034"]
    npf = tri["n_point_fixed_034"]
    lines = [
        "=" * 72,
        "PYTHIA CHINCHILLA-E FROM PUBLIC LOGS (ZERO TRAINING)",
        "=" * 72,
        "",
        "Corpus: The Pile (EleutherAI Pythia matched ladder, same token order)",
        f"Metric: train/lm_loss -> pseudo-epoch C* (steps >= {LATE_STEP_MIN:,})",
        f"Tokens/step: {TOKENS_PER_STEP:,}  Ladder: {len(tri['ladder_sizes'])} sizes",
        "",
        "LADDER:",
    ]
    for row in tri["per_model"]:
        lines.append(
            f"  {row['label']:14s}  {row['n_params']/1e6:6.1f}M params  "
            f"E_app={row['E_app']:.4f}  beta={row['beta']:.4f}  "
            f"final_loss={row['final_loss']:.4f}"
        )
    lines.extend(
        [
            "",
            "TRIANGULATION (E_app(N) = E_true + A*N^{-alpha}):",
            f"  N-point OLS (alpha=0.34, n={npf['n_points']}, dof={npf['dof']}):  "
            f"E_true = {npf['E_true']:.4f} nats  RMSE = {npf['rmse']:.4f}",
            f"  Three-point fixed alpha=0.34 (legacy):    E_true = {t34['E_true']:.4f} nats",
            f"  Two-point (alpha=0.34):                   E_true = {tri['two_point_alpha_034']['E_true']:.4f} nats",
            f"  Three-point free alpha (do not claim):    E_true = {t3['E_true']:.4f} nats  "
            f"(alpha={t3['alpha_used']:.4f})",
            "",
            "NOTE: Primary estimate uses overdetermined N-point fit when n>=4.",
            "Report uncertainty from holdout/LOO (chinchilla_e_robustness.py), not in-sample RMSE.",
            "",
            "=" * 72,
        ]
    )
    return "\n".join(lines)


def parse_sizes(s: str) -> list[str]:
    keys = [x.strip().lower() for x in s.split(",") if x.strip()]
    for k in keys:
        if k not in PYTHIA_LADDER:
            raise ValueError(f"Unknown size {k!r}; choose from {list(PYTHIA_LADDER)}")
    if len(keys) < 3:
        raise ValueError("Need at least 3 sizes for triangulation")
    return keys


def main(argv: Iterable[str] | None = None):
    if hasattr(sys.stdout, "reconfigure"):
        try:
            sys.stdout.reconfigure(encoding="utf-8")
        except Exception:
            pass
    ap = argparse.ArgumentParser(description="Pythia log-only Chinchilla-E triangulation")
    ap.add_argument(
        "--sizes",
        default=",".join(DEFAULT_LADDER),
        help="Comma-separated ladder (default: 14m,70m,160m,410m,1.4b,6.9b)",
    )
    ap.add_argument(
        "--source",
        choices=("github", "wandb"),
        default="github",
        help="github = public TSV (no API key); wandb = refresh via W&B (needs login)",
    )
    args = ap.parse_args(list(argv) if argv is not None else None)

    os.makedirs(OUT_DIR, exist_ok=True)
    cache_dir = os.path.join(OUT_DIR, "cache")
    os.makedirs(cache_dir, exist_ok=True)

    size_keys = parse_sizes(args.sizes)
    if args.source == "wandb":
        raise NotImplementedError(
            "W&B refresh: pip install wandb && wandb login, then use EleutherAI "
            "hmm-training-maps/training_losses/download.py; re-run with --source github"
        )

    print(f"OUT_DIR: {OUT_DIR}")
    print(f"Source: public GitHub TSV ({PYTHIA_TSV_BASE})")
    print(f"Ladder: {size_keys}")
    print()

    results = {}
    for sk in size_keys:
        df = load_loss_tsv(sk, cache_dir)
        name = f"pythia_{sk}"
        results[sk] = build_result_from_log(name, sk, df)
        ep = results[sk]["epochs"][-1]
        print(
            f"[{PYTHIA_LADDER[sk]['label']}] steps={results[sk]['total_steps']:,}  "
            f"final_loss={results[sk]['final_train_loss']:.4f}  "
            f"last C*={ep['h8']['Cstar']:.4f} @ T={ep['T_tokens']/1e9:.2f}B tok"
        )
        oce._json_dump_atomic(os.path.join(OUT_DIR, f"{name}.json"), results[sk])

    tri = triangulate_ladder(results)
    report = write_report(tri, results)
    print()
    print(report)

    oce._json_dump_atomic(
        os.path.join(OUT_DIR, "triangulation.json"),
        dict(
            protocol=dict(
                corpus="The Pile",
                source="EleutherAI pythia training_losses TSV",
                tokens_per_step=TOKENS_PER_STEP,
                late_step_min=LATE_STEP_MIN,
                n_pseudo_epochs=N_PSEUDO_EPOCHS,
            ),
            results={k: dict(n_params=v["n_params"], final_train_loss=v["final_train_loss"]) for k, v in results.items()},
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
