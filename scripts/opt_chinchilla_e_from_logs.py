"""
Meta OPT PILE trajectories → Chinchilla-E triangulation at fixed token count.

E_app = train CE (ln perplexity) interpolated to a common token budget across
the six public OPT sizes (125M–175B).

    python scripts/opt_chinchilla_e_from_logs.py
"""

from __future__ import annotations

import json
import math
import os
import sys
from typing import Any

import numpy as np
import pandas as pd

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.abspath(os.path.join(_SCRIPT_DIR, ".."))
if _SCRIPT_DIR not in sys.path:
    sys.path.insert(0, _SCRIPT_DIR)

import chinchilla_e_robustness as cer  # noqa: E402
import ladder_diagnostics as ld  # noqa: E402
import meta_step2_chinchilla_e_from_logs as ms2  # noqa: E402
import owt_chinchilla_e as oce  # noqa: E402

LOG_PATH = os.environ.get("OPT_TRAJECTORIES_CSV") or os.path.join(
    _REPO, "data", "public_logs", "opt", "opt_trajectories.csv"
)
# log10(tokens); 10.0 ≈ 10B tokens — within coverage for all six OPT sizes
TARGET_LOG10_TOKENS = float(os.environ.get("OPT_TARGET_LOG10_TOKENS", "10.0"))
OUT_DIR = os.path.join(_REPO, "results", "opt_chinchilla_e")
os.makedirs(OUT_DIR, exist_ok=True)

OPT_N_PARAMS = {
    "125m": 125_000_000,
    "1.3b": 1_300_000_000,
    "6.7b": 6_700_000_000,
    "13b": 13_000_000_000,
    "30b": 30_000_000_000,
    "175b": 175_000_000_000,
}


def _interp_ce_at_tokens(sub: pd.DataFrame, target: float) -> float:
    sub = sub.sort_values("num_token")
    x = sub["num_token"].to_numpy(dtype=float)
    ppl = sub["ppl"].to_numpy(dtype=float)
    if target < x.min() or target > x.max():
        raise ValueError(
            f"target log10(tokens)={target} outside [{x.min():.3f}, {x.max():.3f}]"
        )
    ppl_at = float(np.interp(target, x, ppl))
    return math.log(ppl_at)


def build_ladder(df: pd.DataFrame, target: float = TARGET_LOG10_TOKENS) -> dict[str, Any]:
    flat: dict[str, dict] = {}
    per: list[dict] = []
    for size in sorted(OPT_N_PARAMS.keys(), key=lambda s: OPT_N_PARAMS[s]):
        sub = df[df["model_size"] == size]
        if sub.empty:
            raise ValueError(f"Missing OPT size {size}")
        n = OPT_N_PARAMS[size]
        e_app = _interp_ce_at_tokens(sub, target)
        flat[size] = ms2._flat_result(n, e_app, name=f"OPT-{size}")
        per.append(dict(size=size, n_params=n, E_app=e_app, final_loss=e_app))

    keys = sorted(flat.keys(), key=lambda k: flat[k]["n_params"])
    holdout_key = keys[-1]
    tri = oce.chinchilla_fit_eapp_ladder(
        [flat[k]["n_params"] for k in keys],
        [cer._e_app_from_result(flat[k]) for k in keys],
        alpha=cer.ALPHA,
    )
    e_true = tri["E_true"]
    holdout = cer.holdout_test(flat, holdout_key)
    loo = cer.leave_one_out_e_true(flat)
    sanity = cer.sanity_gates(per, e_true)
    pf = ld.preflight_ladder(per, e_true)
    overall = holdout["gate_pass"] and loo["gate_pass"] and sanity["gate_pass"]
    out = dict(
        corpus="Meta OPT / PILE",
        protocol=f"E_app=interp train CE at 10^{target:.1f} tokens",
        target_log10_tokens=target,
        n_sizes=len(keys),
        ladder_sizes=keys,
        triangulation_fixed_alpha=e_true,
        uncertainty_loo_std=loo["E_true_std"],
        triangulation_report=f"{e_true:.2f} ± {loo['E_true_std']:.2f} nats (LOO std)",
        per_model=per,
        holdout=holdout,
        leave_one_out=loo,
        sanity=sanity,
        preflight=pf,
        overall_pass=overall,
    )
    out["failure_diagnosis"] = ld.diagnose_failure(out)
    return out


def scan_token_targets(df: pd.DataFrame, step: float = 0.05) -> list[dict[str, Any]]:
    """Scan log10(token) grid; return results sorted by pass then holdout delta."""
    # Common coverage across all six sizes
    mins, maxs = [], []
    for size in OPT_N_PARAMS:
        sub = df[df["model_size"] == size]
        if sub.empty:
            return []
        mins.append(float(sub["num_token"].min()))
        maxs.append(float(sub["num_token"].max()))
    lo = max(mins)
    hi = min(maxs)
    results = []
    t = lo
    while t <= hi + 1e-9:
        try:
            r = build_ladder(df, target=round(t, 4))
            results.append(r)
        except Exception:
            pass
        t += step
    results.sort(key=lambda r: (not r["overall_pass"], r["holdout"]["delta"]))
    return results


def best_monotonic_target(df: pd.DataFrame) -> dict[str, Any] | None:
    for r in scan_token_targets(df):
        if r["preflight"]["e_app_monotonic"]:
            return r
    return scan_token_targets(df)[0] if scan_token_targets(df) else None


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        try:
            sys.stdout.reconfigure(encoding="utf-8")
        except Exception:
            pass

    df = pd.read_csv(LOG_PATH)
    result = build_ladder(df)
    print(f"OPT / PILE @ log10(tokens)={TARGET_LOG10_TOKENS}")
    print(f"  E_true = {result['triangulation_report']}")
    print(f"  holdout Δ = {result['holdout']['delta']:.4f}  PASS={result['overall_pass']}")
    for row in result["per_model"]:
        print(f"    {row['size']:>5}  N={row['n_params']/1e9:.3f}B  E_app={row['E_app']:.4f}")

    out = os.path.join(OUT_DIR, "opt_report.json")
    with open(out + ".tmp", "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)
    os.replace(out + ".tmp", out)
    print(f"\n[done] {out}")


if __name__ == "__main__":
    main()
