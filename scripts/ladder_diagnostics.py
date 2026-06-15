"""Shared ladder preflight and failure diagnostics for Chinchilla-E triangulation."""

from __future__ import annotations

from typing import Any

import chinchilla_e_robustness as cer


def e_apps_monotonic_decreasing(e_apps: list[float]) -> bool:
    return all(e_apps[i] > e_apps[i + 1] for i in range(len(e_apps) - 1))


def monotonic_prefix_length(e_apps: list[float]) -> int:
    """Longest prefix (>=3 if possible) with strictly decreasing E_app."""
    if len(e_apps) < 3:
        return len(e_apps)
    best = 2
    for i in range(1, len(e_apps)):
        if e_apps[i - 1] > e_apps[i]:
            best = i + 1
        else:
            break
    return best


def preflight_ladder(per_model: list[dict], e_true: float | None = None) -> dict[str, Any]:
    e_apps = [float(r["E_app"]) for r in per_model]
    mono = e_apps_monotonic_decreasing(e_apps)
    prefix_len = monotonic_prefix_length(e_apps)
    below = e_true is not None and e_true < min(e_apps)
    reasons: list[str] = []
    if not mono:
        for i in range(len(e_apps) - 1):
            if e_apps[i] <= e_apps[i + 1]:
                reasons.append(
                    f"non_monotonic_at_{per_model[i]['size']}->{per_model[i+1]['size']}"
                )
                break
    if e_true is not None and not below:
        reasons.append("E_true_not_below_min_E_app")
    return dict(
        e_app_monotonic=mono,
        monotonic_prefix_len=prefix_len,
        e_true_below_min_eapp=below if e_true is not None else None,
        preflight_pass=mono and (below if e_true is not None else True),
        failure_reasons=reasons,
    )


def diagnose_failure(data: dict[str, Any]) -> str:
    if data.get("skipped"):
        return f"skipped:{data.get('error', 'unknown')[:80]}"
    if data.get("overall_pass"):
        return "pass"
    reasons: list[str] = []
    pf = data.get("preflight") or {}
    reasons.extend(pf.get("failure_reasons") or [])
    h = data.get("holdout") or data.get("holdout_410m") or {}
    if h and not h.get("gate_pass"):
        reasons.append(f"holdout_delta={h.get('delta', float('nan')):.4f}")
    lo = data.get("leave_one_out") or {}
    if lo and not lo.get("gate_pass"):
        reasons.append(f"loo_std={lo.get('E_true_std', float('nan')):.4f}")
    s = data.get("sanity") or {}
    if s and not s.get("gate_pass"):
        if not s.get("E_app_monotonic"):
            reasons.append("sanity:non_monotonic")
        if not s.get("E_true_below_min_E_app"):
            reasons.append("sanity:E_true_above_min_E_app")
    if data.get("note"):
        reasons.append(str(data["note"])[:60])
    return "; ".join(reasons) if reasons else "fail:unspecified"
