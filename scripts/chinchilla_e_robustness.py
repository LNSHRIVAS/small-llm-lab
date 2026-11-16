"""
Robustness battery for log-only / triangulation E_true estimation.

Runs CPU-only checks on:
  - Pythia/The Pile (public W&B TSVs)
  - OWT (published E_app from act4_scaling_laws.json)
  - Synthetic (if results/synthetic_chinchilla/triangulation.json exists)

Chinchilla (DeepMind) original 400-run grid: code + MassiveText are proprietary;
we compare Pythia E_true to the paper's published irreducible E ~ 1.69 nats on
MassiveText (different corpus — reference only, not a pass/fail gate).

    python scripts/chinchilla_e_robustness.py
"""

from __future__ import annotations

import json
import math
import os
import sys
from typing import Any

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.abspath(os.path.join(_SCRIPT_DIR, ".."))
if _SCRIPT_DIR not in sys.path:
    sys.path.insert(0, _SCRIPT_DIR)

import owt_chinchilla_e as oce  # noqa: E402
import meta_step2_chinchilla_e_from_logs as ms2  # noqa: E402
import olmo_chinchilla_e_from_logs as ocl  # noqa: E402
import pythia_chinchilla_e_from_logs as pcl  # noqa: E402

ALPHA = 0.34
GATE_HOLDOUT_EAPP = 0.15
GATE_LOO_STD = 0.15
GATE_SYNTHETIC = 0.15

CHINCHILLA_PAPER_E_MASSIVETEXT = 1.69  # Hoffmann et al. fitted L(N,D) intercept (nats)

OUT_DIR = os.path.join(_REPO, "results", "robustness_chinchilla_e")
os.makedirs(OUT_DIR, exist_ok=True)


def _predict_e_app(e_true: float, a_amp: float, n_params: int, alpha: float = ALPHA) -> float:
    return oce.chinchilla_predict_e_app(e_true, a_amp, n_params, alpha=alpha)


def _fit_ladder_from_results(results: dict[str, dict], keys: list[str], alpha: float = ALPHA) -> dict:
    keys = sorted(keys, key=lambda k: results[k]["n_params"])
    n_list = [results[k]["n_params"] for k in keys]
    e_list = [_e_app_from_result(results[k]) for k in keys]
    return oce.chinchilla_fit_eapp_ladder(n_list, e_list, alpha=alpha)


def _two_point_from_results(r_a: dict, r_b: dict, alpha: float = ALPHA) -> dict:
    tri = oce.chinchilla_triangulate(r_a, r_b, alpha=alpha)
    return tri


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


def holdout_test(results: dict[str, dict], holdout_key: str, alpha: float = ALPHA) -> dict:
    """Fit E_true on all sizes except holdout via N-point OLS; predict holdout E_app."""
    train = {k: v for k, v in results.items() if k != holdout_key}
    keys = sorted(train.keys(), key=lambda k: train[k]["n_params"])
    if len(keys) < 2:
        raise ValueError("Need at least 2 train sizes for holdout")
    tri = _fit_ladder_from_results(results, keys, alpha=alpha)
    e_true, a_amp = tri["E_true"], tri["A_amp"]
    n_hold = results[holdout_key]["n_params"]
    e_app_pred = _predict_e_app(e_true, a_amp, n_hold, alpha)
    e_app_actual = _e_app_from_result(results[holdout_key])
    delta = abs(e_app_pred - e_app_actual)
    return dict(
        holdout=holdout_key,
        train_sizes=keys,
        n_train=len(keys),
        E_true=e_true,
        A_amp=a_amp,
        E_app_pred=e_app_pred,
        E_app_actual=e_app_actual,
        delta=delta,
        gate_pass=delta < GATE_HOLDOUT_EAPP,
    )


def leave_one_out_e_true(results: dict[str, dict], alpha: float = ALPHA) -> dict:
    rows = []
    for drop in sorted(results.keys()):
        train_keys = [k for k in results if k != drop]
        tri = _fit_ladder_from_results(results, train_keys, alpha=alpha)
        rows.append(dict(dropped=drop, E_true=tri["E_true"]))
    e_vals = [r["E_true"] for r in rows]
    spread = float(max(e_vals) - min(e_vals))
    std = float(np_std(e_vals))
    return dict(
        runs=rows,
        E_true_min=min(e_vals),
        E_true_max=max(e_vals),
        E_true_std=std,
        E_true_spread=spread,
        gate_pass=std < GATE_LOO_STD,
    )


def np_std(xs: list[float]) -> float:
    m = sum(xs) / len(xs)
    return math.sqrt(sum((x - m) ** 2 for x in xs) / max(1, len(xs) - 1))


def sanity_gates(per_model: list[dict], e_true: float) -> dict:
    e_apps = [r["E_app"] for r in per_model]
    monotonic = all(e_apps[i] > e_apps[i + 1] for i in range(len(e_apps) - 1))
    below_eapp = e_true < min(e_apps)
    below_final = e_true < min(r["final_loss"] for r in per_model)
    return dict(
        E_app_monotonic=monotonic,
        E_true_below_min_E_app=below_eapp,
        E_true_below_min_final_loss=below_final,
        gate_pass=below_eapp and below_final,
    )


def run_pythia() -> dict[str, Any]:
    cache = os.path.join(pcl.OUT_DIR, "cache")
    size_keys = list(pcl.DEFAULT_LADDER)
    results = {}
    for sk in size_keys:
        df = pcl.load_loss_tsv(sk, cache)
        results[sk] = pcl.build_result_from_log(f"pythia_{sk}", sk, df)

    tri = pcl.triangulate_ladder(results)
    e_true = tri["n_point_fixed_034"]["E_true"]
    loo_std = leave_one_out_e_true(results)["E_true_std"]
    per = tri["per_model"]
    for row in per:
        row["final_loss"] = results[row["size"]]["final_train_loss"]

    holdout = holdout_test(results, "6.9b")
    loo = leave_one_out_e_true(results)
    gates = sanity_gates(per, e_true)

    return dict(
        corpus="The Pile (Pythia)",
        method="N-point OLS + holdout/LOO (not corpus Shannon entropy)",
        n_sizes=len(size_keys),
        ladder_sizes=size_keys,
        triangulation_fixed_alpha=e_true,
        uncertainty_loo_std=loo_std,
        triangulation_report=f"{e_true:.2f} ± {loo_std:.2f} nats (LOO std, α=0.34)",
        n_point_fit=tri["n_point_fixed_034"],
        per_model=per,
        holdout=holdout,
        leave_one_out=loo,
        sanity=gates,
        vs_chinchilla_paper=dict(
            note="MassiveText != Pile; reference only",
            chinchilla_E=CHINCHILLA_PAPER_E_MASSIVETEXT,
            delta_nats=e_true - CHINCHILLA_PAPER_E_MASSIVETEXT,
        ),
    )


def _owt_pseudo_results() -> dict[str, dict]:
    """Minimal result dicts from published E_app (no epoch curves)."""
    path = os.path.join(_REPO, "results", "summaries", "act4_scaling_laws.json")
    with open(path, encoding="utf-8") as f:
        act4 = json.load(f)
    branch = act4["branches"]["chinchilla_E_owt"]
    mapping = {
        "A_10M": ("A_10M", 10_165_681),
        "C_25M": ("C_25M", 25_147_153),
        "B_51M": ("B_51M", 51_016_529),
    }
    results = {}
    for key, (name, n) in mapping.items():
        e_app = branch["models"][name]["E_app"]
        results[key] = dict(
            name=key,
            n_params=n,
            epochs=[
                dict(T_tokens=(i + 1) * 500_000_000, h8=dict(Cstar=e_app), val_ce=e_app + 0.1)
                for i in range(6)
            ],
        )
    return results


def run_owt_summary() -> dict[str, Any]:
    results = _owt_pseudo_results()
    keys = sorted(results.keys(), key=lambda k: results[k]["n_params"])
    r_s, r_m, r_l = results[keys[0]], results[keys[1]], results[keys[2]]
    tri3 = oce.chinchilla_triangulate_three(r_s, r_m, r_l, alpha=ALPHA)
    e_true = tri3["E_true"]

    per = []
    for k in keys:
        per.append(dict(size=k, n_params=results[k]["n_params"], E_app=_e_app_from_result(results[k])))

    holdout = holdout_test(results, "B_51M")
    loo = leave_one_out_e_true(results)
    gates = sanity_gates(
        [dict(E_app=p["E_app"], final_loss=p["E_app"] + 0.05) for p in per],
        e_true,
    )

    return dict(
        corpus="OpenWebText (published E_app only)",
        note="Holdout/LOO on flat E_app - optimistic; full test needs epoch JSON curves",
        triangulation_fixed_alpha=e_true,
        per_model=per,
        holdout_B_51M=holdout,
        leave_one_out=loo,
        sanity=gates,
    )


def _run_ladder_corpus(
    label: str,
    size_keys: tuple[str, ...],
    holdout_key: str,
    load_curve,
    build_result,
    triangulate,
) -> dict[str, Any] | None:
    try:
        results = {}
        for sk in size_keys:
            df = load_curve(sk)
            results[sk] = build_result(f"{label}_{sk}", sk, df)
        tri = triangulate(results)
        e_true = tri["three_point_fixed_034"]["E_true"]
        per = tri["per_model"]
        for row in per:
            row["final_loss"] = results[row["size"]]["final_train_loss"]
        return dict(
            corpus=label,
            n_sizes=len(size_keys),
            triangulation_fixed_alpha=e_true,
            per_model=per,
            holdout=holdout_test(results, holdout_key),
            leave_one_out=leave_one_out_e_true(results),
            sanity=sanity_gates(per, e_true),
        )
    except Exception as exc:
        return dict(corpus=label, skipped=True, error=str(exc))


def run_meta_step2() -> dict[str, Any] | None:
    keys = tuple(ms2.DEFAULT_TRIPLET)

    def load(sk):
        df, _ = ms2.load_curve(sk)
        return df

    def build(name, sk, df):
        _, path = ms2.load_curve(sk)
        return ms2.build_result_from_log(name, sk, df, path)

    def adapt(raw):
        return ms2._flat_result(raw["n_params"], ms2._e_app_last_cstar(raw))

    out = _run_ladder_corpus(
        "Meta FAIR Step-2 (public CSV)",
        keys,
        "h1280",
        load,
        build,
        ms2.triangulate_ladder,
    )
    if out and not out.get("skipped"):
        raw = {}
        for sk in keys:
            df = load(sk)
            raw[sk] = build(f"meta_{sk}", sk, df)
        flat = {k: adapt(v) for k, v in raw.items()}
        e_true = out["triangulation_fixed_alpha"]
        per = out["per_model"]
        out["holdout"] = holdout_test(flat, "h1280")
        out["leave_one_out"] = leave_one_out_e_true(flat)
        out["sanity"] = sanity_gates(per, e_true)
        out["note"] = "E_app = late C*; holdout on flat asymptotes (256B-token runs)"
    return out


def run_olmo() -> dict[str, Any] | None:
    keys = tuple(ocl.DEFAULT_TRIPLET)
    return _run_ladder_corpus(
        "Dolma / OLMo (Allen AI W&B exports)",
        keys,
        "13b",
        lambda sk: ocl.load_curve(sk, "local"),
        ocl.build_result_from_log,
        ocl.triangulate_ladder,
    )


def run_synthetic() -> dict[str, Any] | None:
    path = os.path.join(_REPO, "results", "synthetic_chinchilla", "triangulation.json")
    if not os.path.isfile(path):
        return None
    with open(path, encoding="utf-8") as f:
        tri = json.load(f)
    v = tri.get("validation", {})
    h = v.get("H_true_nats", tri.get("meta", {}).get("H_true_nats"))
    e = v.get("E_true_free_alpha")
    if h is None or e is None:
        return None
    delta = abs(e - h)
    return dict(
        corpus="Synthetic Zipf (known H)",
        H_true=h,
        E_true_free=e,
        delta=delta,
        primary_pass=v.get("primary_pass"),
        gate_pass=delta < GATE_SYNTHETIC,
    )


def write_report(sections: dict[str, Any]) -> str:
    lines = [
        "=" * 72,
        "CHINCHILLA-E ROBUSTNESS BATTERY (CPU / public logs)",
        "=" * 72,
        "",
        "Chinchilla original runs: NOT public (MassiveText + code proprietary).",
        f"Paper reference E on MassiveText ~ {CHINCHILLA_PAPER_E_MASSIVETEXT} nats (compare only).",
        "",
    ]

    def block(title: str, data: dict):
        lines.append(f"--- {title} ---")
        if data is None:
            lines.append("  (skipped - no results on disk)")
            lines.append("")
            return
        if "triangulation_fixed_alpha" in data:
            lines.append(f"  E_true (fixed alpha={ALPHA}): {data['triangulation_fixed_alpha']:.4f} nats")
        if data.get("triangulation_report"):
            lines.append(f"  Report: {data['triangulation_report']}")
        if data.get("method"):
            lines.append(f"  Method: {data['method']}")
        if data.get("skipped"):
            lines.append(f"  (skipped - {data.get('error', 'unavailable')})")
            lines.append("")
            return
        if "holdout" in data:
            h = data["holdout"]
            lines.append(
                f"  Holdout {h['holdout']}: pred E_app={h['E_app_pred']:.4f}  "
                f"actual={h['E_app_actual']:.4f}  delta={h['delta']:.4f}  "
                f"PASS={h['gate_pass']} (gate < {GATE_HOLDOUT_EAPP})"
            )
        if "holdout_410m" in data:
            h = data["holdout_410m"]
            lines.append(
                f"  Holdout {h['holdout']}: pred E_app={h['E_app_pred']:.4f}  "
                f"actual={h['E_app_actual']:.4f}  delta={h['delta']:.4f}  "
                f"PASS={h['gate_pass']} (gate < {GATE_HOLDOUT_EAPP})"
            )
        if "holdout_B_51M" in data:
            h = data["holdout_B_51M"]
            lines.append(
                f"  Holdout {h['holdout']}: pred E_app={h['E_app_pred']:.4f}  "
                f"actual={h['E_app_actual']:.4f}  delta={h['delta']:.4f}  "
                f"PASS={h['gate_pass']}"
            )
        if "leave_one_out" in data:
            lo = data["leave_one_out"]
            lines.append(
                f"  LOO E_true: [{lo['E_true_min']:.4f}, {lo['E_true_max']:.4f}]  "
                f"std={lo['E_true_std']:.4f}  PASS={lo['gate_pass']} (gate < {GATE_LOO_STD})"
            )
        if "sanity" in data:
            s = data["sanity"]
            lines.append(
                f"  Sanity: E_true < min E_app={s['E_true_below_min_E_app']}  "
                f"monotonic E_app={s['E_app_monotonic']}  PASS={s['gate_pass']}"
            )
        if "H_true" in data:
            lines.append(
                f"  |E_true - H_true| = {data['delta']:.4f}  PASS={data['gate_pass']} (gate < {GATE_SYNTHETIC})"
            )
        if "vs_chinchilla_paper" in data:
            v = data["vs_chinchilla_paper"]
            lines.append(
                f"  vs Chinchilla paper E: delta={v['delta_nats']:+.2f} nats ({v['note']})"
            )
        if data.get("note"):
            lines.append(f"  Note: {data['note']}")
        lines.append("")

    block("Synthetic validation", sections.get("synthetic"))
    block("Pythia / The Pile (public logs)", sections.get("pythia"))
    block("Meta Step-2 scaling (public CSV)", sections.get("meta_step2"))
    block("OLMo / Dolma (Allen AI logs)", sections.get("olmo"))
    block("OWT (published E_app summary)", sections.get("owt"))

    def _corpus_pass(data: dict | None) -> bool:
        if not data or data.get("skipped"):
            return False
        h = data.get("holdout") or data.get("holdout_410m") or {}
        return bool(
            h.get("gate_pass")
            and data.get("leave_one_out", {}).get("gate_pass")
            and data.get("sanity", {}).get("gate_pass")
        )

    pythia = sections.get("pythia") or {}
    synth = sections.get("synthetic") or {}
    meta = sections.get("meta_step2") or {}
    p_pass = (
        (synth.get("gate_pass") if synth else True)
        and _corpus_pass(pythia)
        and _corpus_pass(meta)
    )
    pythia_report = pythia.get("triangulation_report") or (
        f"~{pythia.get('triangulation_fixed_alpha', 0):.2f} nats"
    )
    meta_e = meta.get("triangulation_fixed_alpha", 0)
    meta_loo = (meta.get("leave_one_out") or {}).get("E_true_std")
    meta_report = (
        f"~{meta_e:.2f} ± {meta_loo:.2f} nats (LOO std)"
        if meta_loo is not None
        else f"~{meta_e:.2f} nats"
    )
    lines.extend(
        [
            "OVERALL (Pythia + Meta Step-2 holdout gates; synthetic if present):",
            f"  {'PASS' if p_pass else 'FAIL'} - method supported for cheap floor estimation",
            f"  Pythia floor estimate: {pythia_report}  |  Step-2: {meta_report}",
            "=" * 72,
        ]
    )
    return "\n".join(lines)


def main():
    if hasattr(sys.stdout, "reconfigure"):
        try:
            sys.stdout.reconfigure(encoding="utf-8")
        except Exception:
            pass

    print("Running robustness battery (CPU, no GPU)...")
    print()

    sections = dict(
        synthetic=run_synthetic(),
        pythia=run_pythia(),
        meta_step2=run_meta_step2(),
        olmo=run_olmo(),
        owt=run_owt_summary(),
    )

    report = write_report(sections)
    print(report)

    out_json = os.path.join(OUT_DIR, "robustness_report.json")
    with open(out_json + ".tmp", "w", encoding="utf-8") as f:
        json.dump(sections, f, indent=2)
    os.replace(out_json + ".tmp", out_json)

    out_txt = os.path.join(OUT_DIR, "robustness_report.txt")
    with open(out_txt + ".tmp", "w", encoding="utf-8") as f:
        f.write(report)
    os.replace(out_txt + ".tmp", out_txt)
    print(f"[done] {out_txt}")
    print(f"[done] {out_json}")


if __name__ == "__main__":
    main()
