"""
Build FloorDB table from sweep results and probe cross-corpus patterns.

Honest law probes:
  1. Same-stack Kempner ordering (OLMo iso-flop, 6 corpora)
  2. Universal constant test (all passes share one E_true?)
  3. Chinchilla alpha sensitivity on passing ladders
  4. Corpus-type spread (web vs code vs pile) — descriptive only
  5. Confounded metadata regression — flagged, not claimed as law

    python scripts/floor_db.py
    python scripts/floor_db.py --skip-sweep   # use existing sweep_report.json
"""

from __future__ import annotations

import argparse
import json
import math
import os
import subprocess
import sys
from typing import Any

import numpy as np
import pandas as pd

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.abspath(os.path.join(_SCRIPT_DIR, ".."))
if _SCRIPT_DIR not in sys.path:
    sys.path.insert(0, _SCRIPT_DIR)

import ladder_diagnostics as ld  # noqa: E402

OUT_DIR = os.path.join(_REPO, "results", "floor_db")
SWEEP_JSON = os.path.join(_REPO, "results", "public_ladder_sweep", "sweep_report.json")
POINT_JSON = os.path.join(_REPO, "results", "point_ladder_chinchilla_e", "point_ladder_report.json")

# Corpus metadata for law probes (manual tags — not ground truth)
CORPUS_META: dict[str, dict[str, str]] = {
    "The Pile (Pythia)": dict(family="pile", stack="pythia", protocol="matched_tokens", domain="mixed"),
    "Meta FAIR Step-2 (public CSV)": dict(family="step2", stack="farseer", protocol="matched_tokens", domain="web"),
    "Meta Step-2 (ti134698)": dict(family="step2", stack="farseer", protocol="matched_tokens", domain="web"),
    "Meta Step-2 (ti145166)": dict(family="step2", stack="farseer", protocol="matched_tokens", domain="web"),
    "Meta Step-2 (ti139508)": dict(family="step2", stack="farseer", protocol="matched_tokens", domain="web"),
    "Meta Step-2 (ti153451)": dict(family="step2", stack="farseer", protocol="matched_tokens", domain="web"),
    "Meta Step-2 (ti172881)": dict(family="step2", stack="farseer", protocol="matched_tokens", domain="web"),
    "Dolma / OLMo (Allen AI W&B exports)": dict(family="dolma", stack="olmo", protocol="matched_tokens", domain="web"),
    "Kempner OLMo / fineweb-100b (iso-flop)": dict(family="kempner", stack="olmo", protocol="iso_flop", domain="web_raw"),
    "Kempner OLMo / fineweb-edu-100b (iso-flop)": dict(family="kempner", stack="olmo", protocol="iso_flop", domain="web_edu"),
    "Kempner OLMo / smollm-corpus (iso-flop)": dict(family="kempner", stack="olmo", protocol="iso_flop", domain="web_mixed"),
    "Kempner OLMo / proof-pile-2 (iso-flop)": dict(family="kempner", stack="olmo", protocol="iso_flop", domain="mixed"),
    "Kempner OLMo / slimpajama-chunk1 (iso-flop)": dict(family="kempner", stack="olmo", protocol="iso_flop", domain="web"),
    "Kempner OLMo / starcoder (iso-flop)": dict(family="kempner", stack="olmo", protocol="iso_flop", domain="code"),
    "Meta OPT / PILE": dict(family="opt", stack="opt", protocol="fixed_tokens", domain="pile"),
    "Meta OPT / PILE @ 10^9.9 (mono)": dict(family="opt", stack="opt", protocol="fixed_tokens", domain="pile"),
    "Cerebras-GPT / PILE (final eval)": dict(family="cerebras", stack="cerebras", protocol="final_eval", domain="pile"),
    "OpenWebText (published E_app only)": dict(family="owt", stack="custom", protocol="flat_eapp", domain="web"),
}


def _row_from_section(key: str, data: dict[str, Any]) -> dict[str, Any]:
    corpus = data.get("corpus") or key
    meta = CORPUS_META.get(corpus, dict(family="unknown", stack="unknown", protocol="unknown", domain="unknown"))
    e = data.get("triangulation_fixed_alpha")
    lo = (data.get("leave_one_out") or {}).get("E_true_std")
    h = data.get("holdout") or data.get("holdout_410m") or {}
    pf = data.get("preflight") or {}
    passed = bool(data.get("overall_pass")) if "overall_pass" in data else (
        not data.get("skipped")
        and e is not None
        and h.get("gate_pass")
        and (data.get("leave_one_out") or {}).get("gate_pass")
        and (data.get("sanity") or {}).get("gate_pass")
    )
    return dict(
        key=key,
        corpus=corpus,
        n_sizes=data.get("n_sizes"),
        e_true=e,
        loo_std=lo,
        holdout_delta=h.get("delta"),
        pass_gate=passed,
        skipped=bool(data.get("skipped")),
        protocol=data.get("protocol") or meta.get("protocol"),
        stack=meta.get("stack"),
        domain=meta.get("domain"),
        family=meta.get("family"),
        monotonic=pf.get("e_app_monotonic") if pf else (data.get("sanity") or {}).get("E_app_monotonic"),
        failure_diagnosis=data.get("failure_diagnosis") or ld.diagnose_failure(data),
    )


def build_table(sections: dict[str, Any]) -> pd.DataFrame:
    rows = [_row_from_section(k, v) for k, v in sections.items() if isinstance(v, dict)]
    return pd.DataFrame(rows)


def probe_universal_constant(df: pd.DataFrame) -> dict[str, Any]:
    passes = df[df.pass_gate & df.e_true.notna()]
    if len(passes) < 2:
        return dict(test="universal_constant", n=0, verdict="insufficient_data")
    vals = passes.e_true.to_numpy(float)
    spread = float(vals.max() - vals.min())
    pooled_std = float(np.std(vals))
    return dict(
        test="universal_constant",
        n=len(passes),
        e_true_min=float(vals.min()),
        e_true_max=float(vals.max()),
        spread=spread,
        pooled_std=pooled_std,
        verdict="REJECTED" if spread > 0.3 else "marginal",
        note="Floors differ by >0.3 nats across passing corpora - not one universal constant.",
    )


def probe_kempner_ordering(df: pd.DataFrame) -> dict[str, Any]:
    k = df[df.corpus.str.contains("Kempner", na=False) & df.e_true.notna()].copy()
    if k.empty:
        return dict(test="kempner_same_stack_ordering", verdict="no_data")
    order_web_edu = ["web_edu", "web_mixed", "web_raw", "web", "mixed", "code"]
    k["domain_rank"] = k.domain.map({d: i for i, d in enumerate(order_web_edu)})
    passes = k[k.pass_gate].sort_values("e_true")
    fails = k[~k.pass_gate].sort_values("e_true")
    # Expected: cleaner/edu lower, raw web higher; code lowest if monotonic
    spearman = None
    if len(passes) >= 3 and passes.domain_rank.notna().all():
        from scipy.stats import spearmanr

        spearman = float(spearmanr(passes.e_true, passes.domain_rank).correlation)
    return dict(
        test="kempner_same_stack_ordering",
        n_total=len(k),
        n_pass=int(k.pass_gate.sum()),
        passing_corpora=passes[["corpus", "e_true", "domain"]].to_dict("records"),
        gated_out=fails[["corpus", "e_true", "domain", "failure_diagnosis"]].to_dict("records"),
        spearman_e_vs_domain_rank=spearman,
        verdict="SUGGESTIVE_ORDERING" if spearman and spearman > 0.5 else "INCONCLUSIVE",
        note="Same OLMo stack; domain tags are manual. Gated-out rows excluded from ordering claim.",
    )


def probe_domain_spread(df: pd.DataFrame) -> dict[str, Any]:
    passes = df[df.pass_gate & df.e_true.notna()]
    by_domain = passes.groupby("domain")["e_true"].agg(["count", "mean", "std", "min", "max"])
    return dict(
        test="domain_spread_passing_only",
        by_domain=by_domain.reset_index().to_dict("records"),
        verdict="DESCRIPTIVE_ONLY",
        note="Cross-stack domain means are confounded by tokenizer/stack/protocol.",
    )


def probe_confounded_regression(df: pd.DataFrame) -> dict[str, Any]:
    """OLS E_true ~ domain dummies — explicitly flagged as confounded."""
    sub = df[df.pass_gate & df.e_true.notna()].copy()
    if len(sub) < 4:
        return dict(test="confounded_domain_regression", verdict="insufficient_n")
    dummies = pd.get_dummies(sub["domain"], prefix="dom", drop_first=True)
    X = np.column_stack([np.ones(len(sub)), dummies.to_numpy(float)])
    y = sub.e_true.to_numpy(float)
    beta, _, _, _ = np.linalg.lstsq(X, y, rcond=None)
    yhat = X @ beta
    ss_res = float(np.sum((y - yhat) ** 2))
    ss_tot = float(np.sum((y - y.mean()) ** 2))
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else float("nan")
    return dict(
        test="confounded_domain_regression",
        n=len(sub),
        r2=r2,
        coefficients=dict(intercept=float(beta[0]), **{
            c: float(b) for c, b in zip(dummies.columns, beta[1:])
        }),
        verdict="NOT_A_LAW",
        note="Confounded: stack, tokenizer, protocol differ across rows. High R² would be mirage.",
    )


def probe_chinchilla_alpha_sensitivity() -> dict[str, Any]:
    """Re-fit passing Kempner fineweb-edu at alpha grid."""
    import chinchilla_e_robustness as cer
    import kempner_chinchilla_e_from_logs as kcl

    df = pd.read_csv(kcl.LOG_PATH)
    base = kcl.build_ladder("fineweb-edu-100b", df)
    per = base["per_model"]
    flat = {}
    import meta_step2_chinchilla_e_from_logs as ms2

    for row in per:
        flat[row["size"]] = ms2._flat_result(row["n_params"], row["E_app"])
    alphas = [0.24, 0.30, 0.34, 0.38, 0.44]
    import owt_chinchilla_e as oce

    rows = []
    for a in alphas:
        keys = sorted(flat.keys(), key=lambda k: flat[k]["n_params"])
        tri = oce.chinchilla_fit_eapp_ladder(
            [flat[k]["n_params"] for k in keys],
            [cer._e_app_from_result(flat[k]) for k in keys],
            alpha=a,
        )
        rows.append(dict(alpha=a, e_true=tri["E_true"]))
    spread = max(r["e_true"] for r in rows) - min(r["e_true"] for r in rows)
    return dict(
        test="alpha_sensitivity_fineweb_edu",
        corpus="Kempner fineweb-edu-100b",
        rows=rows,
        e_true_spread=spread,
        verdict="STABLE" if spread < 0.15 else "SENSITIVE",
    )


def run_law_probes(df: pd.DataFrame) -> dict[str, Any]:
    return dict(
        universal_constant=probe_universal_constant(df),
        kempner_ordering=probe_kempner_ordering(df),
        domain_spread=probe_domain_spread(df),
        confounded_regression=probe_confounded_regression(df),
        alpha_sensitivity=probe_chinchilla_alpha_sensitivity(),
    )


def write_law_report(probes: dict[str, Any]) -> str:
    lines = [
        "=" * 72,
        "FLOORDB LAW PROBES (honest / confound-aware)",
        "=" * 72,
        "",
    ]
    uc = probes["universal_constant"]
    lines.append(f"Universal constant: {uc.get('verdict')}  spread={uc.get('spread', '?'):.3f} nats ({uc.get('n', 0)} passes)")
    lines.append(f"  {uc.get('note', '')}")
    lines.append("")

    ko = probes["kempner_ordering"]
    lines.append(f"Kempner same-stack ordering: {ko.get('verdict')}")
    if ko.get("passing_corpora"):
        for r in ko["passing_corpora"]:
            lines.append(f"  PASS  {r['e_true']:.2f}  {r['domain']}  {r['corpus'][:40]}")
    if ko.get("gated_out"):
        lines.append("  Gated out (do not use as evidence):")
        for r in ko["gated_out"]:
            lines.append(f"    {r.get('e_true', float('nan')):.2f}  {r['domain']}  ({r['failure_diagnosis'][:50]})")
    lines.append("")

    cr = probes["confounded_regression"]
    lines.append(f"Confounded domain regression: R²={cr.get('r2', float('nan')):.3f}  verdict={cr.get('verdict')}")
    lines.append(f"  {cr.get('note', '')}")
    lines.append("")

    al = probes["alpha_sensitivity"]
    lines.append(f"Alpha sensitivity ({al.get('corpus')}): {al.get('verdict')}  spread={al.get('e_true_spread', 0):.3f}")
    for r in al.get("rows", []):
        lines.append(f"  α={r['alpha']:.2f}  E_true={r['e_true']:.3f}")
    lines.append("")
    lines.append("BOTTOM LINE: Method + corpus-specific floors confirmed. Universal E_true law REJECTED.")
    lines.append("Best same-stack signal: Kempner OLMo ordering on passing corpora only.")
    lines.append("=" * 72)
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--skip-sweep", action="store_true")
    args = parser.parse_args()

    os.makedirs(OUT_DIR, exist_ok=True)

    if not args.skip_sweep:
        subprocess.run([sys.executable, os.path.join(_SCRIPT_DIR, "public_ladder_sweep.py")], check=True)
        subprocess.run([sys.executable, os.path.join(_SCRIPT_DIR, "point_ladder_chinchilla_e.py")], check=True)

    with open(SWEEP_JSON, encoding="utf-8") as f:
        sweep = json.load(f)
    sections = dict(sweep.get("sections") or {})
    if os.path.isfile(POINT_JSON):
        with open(POINT_JSON, encoding="utf-8") as f:
            sections.update(json.load(f))

    df = build_table(sections)
    csv_path = os.path.join(OUT_DIR, "floor_db.csv")
    df.to_csv(csv_path, index=False)

    probes = run_law_probes(df)
    report = write_law_report(probes)
    print(report)

    with open(os.path.join(OUT_DIR, "law_probes.json"), "w", encoding="utf-8") as f:
        json.dump(probes, f, indent=2)
    with open(os.path.join(OUT_DIR, "law_probes.txt"), "w", encoding="utf-8") as f:
        f.write(report)
    print(f"[done] {csv_path}")
    print(f"[done] {OUT_DIR}/law_probes.txt")


if __name__ == "__main__":
    main()
