"""
Kempner Institute OLMo iso-flop ladders → Chinchilla-E triangulation.

Each ladder fixes total compute (iso_flop) and sweeps model width; E_app is
final training CE at the end of that iso-flop run (not matched token budget).

    python scripts/kempner_chinchilla_e_from_logs.py
"""

from __future__ import annotations

import json
import math
import os
import sys
from typing import Any

import pandas as pd

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.abspath(os.path.join(_SCRIPT_DIR, ".."))
if _SCRIPT_DIR not in sys.path:
    sys.path.insert(0, _SCRIPT_DIR)

import chinchilla_e_robustness as cer  # noqa: E402
import ladder_diagnostics as ld  # noqa: E402
import meta_step2_chinchilla_e_from_logs as ms2  # noqa: E402
import owt_chinchilla_e as oce  # noqa: E402

LOG_PATH = os.environ.get("KEMPNER_SWEEP_CSV") or os.path.join(
    _REPO, "data", "public_logs", "kempner", "kempner_sweep.csv"
)
TRAIN_COL = "train/CrossEntropyLoss"
MIN_SIZES = 6
OUT_DIR = os.path.join(_REPO, "results", "kempner_chinchilla_e")
os.makedirs(OUT_DIR, exist_ok=True)


def _best_iso_flop(sub: pd.DataFrame) -> float | None:
    best_iso = None
    best_n = 0
    for iso in sub["iso_flop"].astype(float).unique():
        n = sub[sub["iso_flop"].astype(float) == iso]["params"].nunique()
        if n > best_n:
            best_n = n
            best_iso = float(iso)
    return best_iso if best_n >= MIN_SIZES else None


def _triangulate_flat(flat: dict[str, dict], per: list[dict], label: str, protocol: str) -> dict[str, Any]:
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
        corpus=label,
        protocol=protocol,
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


def build_ladder(corpus: str, df: pd.DataFrame, monotonic_prefix_only: bool = False) -> dict[str, Any]:
    sub = df[(df["data"] == corpus) & (df["state"] == "finished")].copy()
    iso = _best_iso_flop(sub)
    if iso is None:
        raise ValueError(f"Need >={MIN_SIZES} sizes at one iso_flop for {corpus}")
    rows = (
        sub[sub["iso_flop"].astype(float) == iso]
        .drop_duplicates("params")
        .sort_values("params")
    )
    flat: dict[str, dict] = {}
    per: list[dict] = []
    for _, r in rows.iterrows():
        n = int(r["params"])
        e_app = float(r[TRAIN_COL])
        key = f"{n // 1_000_000}M"
        flat[key] = ms2._flat_result(n, e_app, name=f"{corpus}-{key}")
        per.append(
            dict(
                size=key,
                n_params=n,
                E_app=e_app,
                final_loss=e_app,
                tokens_trained=float(r["tokens"]),
            )
        )
    keys = sorted(flat.keys(), key=lambda k: flat[k]["n_params"])
    if monotonic_prefix_only:
        e_apps = [cer._e_app_from_result(flat[k]) for k in keys]
        plen = ld.monotonic_prefix_length(e_apps)
        if plen < 3:
            raise ValueError(f"Monotonic prefix too short ({plen}) for {corpus}")
        keys = keys[:plen]
        flat = {k: flat[k] for k in keys}
        per = [p for p in per if p["size"] in keys]
    label = f"Kempner OLMo / {corpus} (iso-flop)"
    protocol = f"iso_flop={iso:.2e}; E_app=final train CE"
    if monotonic_prefix_only:
        label += " [monotonic prefix]"
        protocol += f"; prefix_n={len(keys)}"
    return _triangulate_flat(flat, per, label, protocol)


def discover_corpora(df: pd.DataFrame) -> list[str]:
    out = []
    for corpus in sorted(df["data"].unique()):
        sub = df[(df["data"] == corpus) & (df["state"] == "finished")]
        if _best_iso_flop(sub) is not None:
            out.append(corpus)
    return out


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        try:
            sys.stdout.reconfigure(encoding="utf-8")
        except Exception:
            pass

    df = pd.read_csv(LOG_PATH)
    corpora = discover_corpora(df)
    print(f"Kempner sweep: {LOG_PATH}")
    print(f"Corpora with >={MIN_SIZES}-size iso-flop ladders: {len(corpora)}")
    print()

    sections: dict[str, Any] = {}
    for corpus in corpora:
        try:
            sections[corpus] = build_ladder(corpus, df)
            d = sections[corpus]
            ok = "PASS" if d["overall_pass"] else "fail"
            print(
                f"  {corpus:<24} n={d['n_sizes']}  "
                f"E={d['triangulation_fixed_alpha']:.2f} ± {d['uncertainty_loo_std']:.2f}  {ok}"
            )
        except Exception as exc:
            sections[corpus] = dict(corpus=corpus, skipped=True, error=str(exc))
            print(f"  {corpus:<24} skipped ({exc})")

    out = os.path.join(OUT_DIR, "kempner_report.json")
    with open(out + ".tmp", "w", encoding="utf-8") as f:
        json.dump(dict(log_path=LOG_PATH, sections=sections), f, indent=2)
    os.replace(out + ".tmp", out)
    print(f"\n[done] {out}")


if __name__ == "__main__":
    main()
