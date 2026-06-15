"""
Zero-GPU Chinchilla-E triangulation from public OLMo training logs (Allen AI / Dolma).

Primary source: bundled W&B export CSVs under `data/public_logs/olmo/`.

Corpus: Dolma v1 / OLMo pretraining mix (not The Pile).

    pip install numpy scipy pandas
    python scripts/olmo_chinchilla_e_from_logs.py

Optional refresh (needs WANDB_API_KEY in environment — not read from disk):
    python scripts/olmo_chinchilla_e_from_logs.py --source wandb
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import sys
from typing import Iterable

import numpy as np
import pandas as pd

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.abspath(os.path.join(_SCRIPT_DIR, ".."))
if _SCRIPT_DIR not in sys.path:
    sys.path.insert(0, _SCRIPT_DIR)

import owt_chinchilla_e as oce  # noqa: E402

TOKENS_PER_STEP = 4_194_304
LATE_STEP_MIN = 50_000
N_PSEUDO_EPOCHS = 6

OLMO_LADDER = {
    "1b": dict(n_params=1_000_000_000, label="OLMo-1B", csv="WB-OLMo-1B.csv"),
    "7b": dict(n_params=7_000_000_000, label="OLMo-7B", csv="WB-OLMo-7B.csv"),
    "13b": dict(n_params=13_000_000_000, label="OLMo2-13B", csv="WB-OLMo2-13B.csv"),
}
DEFAULT_TRIPLET = ("1b", "7b", "13b")

_WB_PROJECTS = {
    "1b": "ai2-llm/OLMo-1B",
    "7b": "ai2-llm/OLMo-7B",
    "13b": "ai2-llm/OLMo-2-1124-13B",
}
LOSS_KEY = "train/CrossEntropyLoss"

_FETCHED = os.path.join(_REPO, "data", "public_logs", "olmo")
_FALLBACK_FETCHED = os.path.abspath(
    os.path.join(_REPO, "..", "chinchilla-slope-one-repro", "data", "external", "fetched")
)
_OLMO_CSV = os.path.join(_FETCHED, "olmo.csv")
_FALLBACK_OLMO_CSV = os.path.abspath(
    os.path.join(_REPO, "..", "chinchilla-slope-one-repro", "data", "extra", "olmo.csv")
)
LOG_DIR = os.environ.get("OLMO_LOG_DIR") or (
    _FETCHED if os.path.isdir(_FETCHED) else _FALLBACK_FETCHED
)


def _olmo_csv_path() -> str:
    if os.path.isfile(_OLMO_CSV):
        return _OLMO_CSV
    return _FALLBACK_OLMO_CSV
OUT_DIR = os.environ.get(
    "OLMO_CHINCHILLA_DIR",
    os.path.join(_REPO, "results", "olmo_chinchilla_from_logs"),
)
OUT_DIR = os.path.abspath(OUT_DIR)


def _pseudo_epoch_bounds(step_min: int, step_max: int, n_epochs: int) -> list[tuple[int, int]]:
    edges = np.linspace(step_min, step_max, n_epochs + 1, dtype=int)
    bounds = []
    for i in range(n_epochs):
        lo, hi = int(edges[i]), int(edges[i + 1])
        if hi > lo:
            bounds.append((lo, hi))
    return bounds


def _load_olmo1b_long() -> pd.DataFrame | None:
    path = _olmo_csv_path()
    if not os.path.isfile(path):
        return None
    raw = pd.read_csv(path)
    sub = raw[raw["model_name"].str.strip() == "OLMo-1B"]
    if sub.empty or sub["pile/CrossEntropyLoss"].notna().sum() < 100:
        return None
    df = sub[["Step", "pile/CrossEntropyLoss"]].rename(
        columns={"Step": "step", "pile/CrossEntropyLoss": "loss"}
    ).dropna()
    return df.sort_values("step").drop_duplicates("step", keep="last")


def _load_wb_csv(size_key: str) -> pd.DataFrame:
    fn = OLMO_LADDER[size_key]["csv"]
    path = os.path.join(LOG_DIR, fn)
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Missing {path}; set OLMO_LOG_DIR or use --source wandb")
    df = pd.read_csv(path).rename(columns={"ce": "loss"})
    return df[["step", "loss"]].sort_values("step").drop_duplicates("step", keep="last")


def _load_olmo7b_long() -> pd.DataFrame | None:
    path = _olmo_csv_path()
    if not os.path.isfile(path):
        return None
    raw = pd.read_csv(path)
    sub = raw[raw["model_name"].str.strip() == "OLMo-7B"]
    col = "pile-validation/CrossEntropyLoss"
    if sub.empty or sub[col].notna().sum() < 100:
        return None
    df = sub[["Step", col]].rename(columns={"Step": "step", col: "loss"}).dropna()
    return df.sort_values("step").drop_duplicates("step", keep="last")


def load_curve(size_key: str, source: str) -> pd.DataFrame:
    if size_key == "1b":
        long_df = _load_olmo1b_long()
        if long_df is not None:
            return long_df
    if size_key == "7b":
        long_df = _load_olmo7b_long()
        if long_df is not None:
            return long_df
    if source == "wandb":
        return _fetch_wandb(size_key)
    return _load_wb_csv(size_key)


def _fetch_wandb(size_key: str) -> pd.DataFrame:
    if not os.environ.get("WANDB_API_KEY"):
        raise RuntimeError("Set WANDB_API_KEY to refresh OLMo logs from W&B")
    import wandb

    project = _WB_PROJECTS[size_key]
    api = wandb.Api()
    best = None
    for run in api.runs(project, per_page=15):
        rows = list(run.scan_history(keys=[LOSS_KEY, "_step"], page_size=5000))
        pts = [
            (float(r["_step"]), float(r[LOSS_KEY]))
            for r in rows
            if LOSS_KEY in r and r.get(LOSS_KEY) is not None
        ]
        if len(pts) >= 50 and (best is None or len(pts) > len(best[1])):
            best = (run, pts)
    if best is None:
        raise RuntimeError(f"No W&B run with {LOSS_KEY} in {project}")
    run, pts = best
    print(f"  [wandb] {size_key}: {len(pts)} pts from {run.name}")
    return pd.DataFrame({"step": [p[0] for p in pts], "loss": [p[1] for p in pts]})


def build_result_from_log(name: str, size_key: str, df: pd.DataFrame) -> dict:
    meta = OLMO_LADDER[size_key]
    step_max = int(df["step"].max())
    step_min = max(min(LATE_STEP_MIN, int(step_max * 0.35)), int(df["step"].min()))
    bounds = _pseudo_epoch_bounds(step_min, step_max, N_PSEUDO_EPOCHS)

    epoch_results = []
    for ep, (lo, hi) in enumerate(bounds, start=1):
        band = df[(df["step"] >= lo) & (df["step"] <= hi)]
        if len(band) < 5:
            continue
        steps = band["step"].to_numpy(dtype=np.float64)
        losses = band["loss"].to_numpy(dtype=np.float64)
        end_step = int(hi)
        t_tokens = end_step * TOKENS_PER_STEP
        fit = oce.h8_fit(steps, losses, t_min=max(lo, step_min))
        epoch_results.append(
            dict(
                epoch=ep,
                T_tokens=t_tokens,
                val_ce=float(band["loss"].iloc[-1]),
                step_end=end_step,
                h8=fit,
            )
        )

    if len(epoch_results) < 3:
        raise ValueError(f"{name}: only {len(epoch_results)} pseudo-epochs; need >= 3")

    return dict(
        name=name,
        size_key=size_key,
        label=meta["label"],
        n_params=meta["n_params"],
        corpus="Dolma / OLMo pretraining mix",
        tokens_per_step=TOKENS_PER_STEP,
        total_steps=int(df["step"].max()),
        final_train_loss=float(df["loss"].iloc[-1]),
        epochs=epoch_results,
    )


def triangulate_ladder(results: dict[str, dict]) -> dict:
    keys = sorted(results.keys(), key=lambda k: results[k]["n_params"])
    small, mid, large = keys[0], keys[len(keys) // 2], keys[-1]
    r_s, r_m, r_l = results[small], results[mid], results[large]
    tri2 = oce.chinchilla_triangulate(r_s, r_l, alpha=0.34)
    tri3_free = oce.chinchilla_triangulate_three(r_s, r_m, r_l, alpha=None)
    tri3_034 = oce.chinchilla_triangulate_three(r_s, r_m, r_l, alpha=0.34)
    rows = []
    for k in keys:
        r = results[k]
        t, c = oce._collect_T_C(r)
        e_app, beta, _ = oce._fit3_T_C(t, c)
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
    )


def write_report(tri: dict) -> str:
    t3 = tri["three_point_free"]
    t34 = tri["three_point_fixed_034"]
    lines = [
        "=" * 72,
        "OLMO CHINCHILLA-E FROM PUBLIC LOGS (ZERO TRAINING)",
        "=" * 72,
        "",
        "Corpus: Dolma / OLMo (Allen AI public logs; OLMo-1B train CE, OLMo-7B pile-val CE)",
        f"Metric: train CrossEntropyLoss, steps >= {LATE_STEP_MIN:,}",
        f"Tokens/step: {TOKENS_PER_STEP:,}",
        "",
        "LADDER:",
    ]
    for row in tri["per_model"]:
        lines.append(
            f"  {row['label']:14s}  {row['n_params']/1e9:5.1f}B params  "
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
            "NOTE: 7B/13B curves are short W&B samples — prefer Meta Step-2 or Pythia for holdout gates.",
            "",
            "=" * 72,
        ]
    )
    return "\n".join(lines)


def main(argv: Iterable[str] | None = None):
    if hasattr(sys.stdout, "reconfigure"):
        try:
            sys.stdout.reconfigure(encoding="utf-8")
        except Exception:
            pass

    ap = argparse.ArgumentParser(description="OLMo log-only Chinchilla-E triangulation")
    ap.add_argument("--sizes", default=",".join(DEFAULT_TRIPLET))
    ap.add_argument("--source", choices=("local", "wandb"), default="local")
    args = ap.parse_args(list(argv) if argv is not None else None)

    size_keys = [x.strip().lower() for x in args.sizes.split(",") if x.strip()]
    os.makedirs(OUT_DIR, exist_ok=True)

    print(f"OUT_DIR: {OUT_DIR}")
    print(f"LOG_DIR: {LOG_DIR}")
    print(f"Ladder: {size_keys}")
    print()

    results = {}
    for sk in size_keys:
        df = load_curve(sk, args.source)
        name = f"olmo_{sk}"
        results[sk] = build_result_from_log(name, sk, df)
        ep = results[sk]["epochs"][-1]
        print(
            f"[{OLMO_LADDER[sk]['label']}] steps={results[sk]['total_steps']:,}  "
            f"final_loss={results[sk]['final_train_loss']:.4f}  "
            f"last C*={ep['h8']['Cstar']:.4f}"
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
                corpus="Dolma/OLMo",
                source="Allen AI W&B CSV exports",
                tokens_per_step=TOKENS_PER_STEP,
                late_step_min=LATE_STEP_MIN,
            ),
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
