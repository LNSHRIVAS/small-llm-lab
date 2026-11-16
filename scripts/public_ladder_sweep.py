"""
Batch test log-only Chinchilla-E triangulation across public model ladders.

Scans Meta FAIR Step-2 Farseer CSVs (by matched token budget) and re-runs
Pythia / Meta-3 / OLMo / Kempner / OPT / Cerebras batteries. CPU only.

    python scripts/public_ladder_sweep.py
    META_STEP2_LOG_DIR=/path/to/fetched python scripts/public_ladder_sweep.py
"""

from __future__ import annotations

import glob
import json
import os
import re
import sys
from typing import Any

import pandas as pd

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.abspath(os.path.join(_SCRIPT_DIR, ".."))
if _SCRIPT_DIR not in sys.path:
    sys.path.insert(0, _SCRIPT_DIR)

import chinchilla_e_robustness as cer  # noqa: E402
import kempner_chinchilla_e_from_logs as kcl  # noqa: E402
import ladder_diagnostics as ld  # noqa: E402
import meta_step2_chinchilla_e_from_logs as ms2  # noqa: E402
import opt_chinchilla_e_from_logs as optl  # noqa: E402
import point_ladder_chinchilla_e as plc  # noqa: E402
import pythia_chinchilla_e_from_logs as pcl  # noqa: E402

OUT_DIR = os.path.join(_REPO, "results", "public_ladder_sweep")
os.makedirs(OUT_DIR, exist_ok=True)

_FS_GLOB = "FS-step2v2_*_sc_h*_ti*.csv"


def _discover_step2_ladders(log_dir: str) -> dict[str, list[str]]:
    """Return {token_budget_tag: [h832, h1024, ...]} for tags with >=3 sizes."""
    by_tag: dict[str, set[int]] = {}
    for path in glob.glob(os.path.join(log_dir, _FS_GLOB)):
        m = re.search(r"_sc_h(\d+)_", os.path.basename(path))
        t = re.search(r"_ti(\d+)\.csv$", os.path.basename(path))
        if not m or not t:
            continue
        by_tag.setdefault(t.group(1), set()).add(int(m.group(1)))
    out = {}
    for tag, hs in sorted(by_tag.items()):
        if len(hs) >= 3:
            out[tag] = [f"h{h}" for h in sorted(hs)]
    return out


def _finalize_section(data: dict[str, Any]) -> dict[str, Any]:
    if data.get("skipped") or "triangulation_fixed_alpha" not in data:
        return data
    per = data.get("per_model") or []
    e = data.get("triangulation_fixed_alpha")
    if per and "preflight" not in data:
        data["preflight"] = ld.preflight_ladder(per, e)
    if "failure_diagnosis" not in data:
        data["failure_diagnosis"] = ld.diagnose_failure(data)
    if "overall_pass" not in data:
        data["overall_pass"] = _corpus_pass(data)
    return data


def _run_step2_ladder(log_dir: str, token_tag: str, size_keys: list[str]) -> dict[str, Any]:
    old_dir = ms2.LOG_DIR
    old_ladder = dict(ms2.LADDER)
    old_tag = ms2.TOKEN_BUDGET_TAG
    old_triplet = ms2.DEFAULT_TRIPLET
    try:
        ms2.LOG_DIR = log_dir
        ms2.TOKEN_BUDGET_TAG = token_tag
        ms2.DEFAULT_TRIPLET = tuple(size_keys[:3])
        ms2.LADDER = {}
        for sk in size_keys:
            h = sk.lstrip("h")
            ms2.LADDER[sk] = dict(
                glob=f"FS-step2v2_*_sc_h{h}_*_ti{token_tag}.csv",
                label=f"Step2-h{h}",
            )
        results = {}
        for sk in size_keys:
            df, path = ms2.load_curve(sk)
            results[sk] = ms2.build_result_from_log(f"step2_{sk}", sk, df, path)
        tri = ms2.triangulate_ladder(results)
        e_true = tri["three_point_fixed_034"]["E_true"]
        per = tri["per_model"]
        for row in per:
            row["final_loss"] = results[row["size"]]["final_train_loss"]
        flat = {
            k: ms2._flat_result(v["n_params"], ms2._e_app_last_cstar(v), v["name"])
            for k, v in results.items()
        }
        holdout_key = size_keys[-1]
        holdout = cer.holdout_test(flat, holdout_key)
        loo = cer.leave_one_out_e_true(flat)
        sanity = cer.sanity_gates(per, e_true)
        pf = ld.preflight_ladder(per, e_true)
        out = dict(
            corpus=f"Meta Step-2 (ti{token_tag})",
            protocol="matched token budget; E_app=late C*",
            n_sizes=len(size_keys),
            ladder_sizes=size_keys,
            triangulation_fixed_alpha=e_true,
            uncertainty_loo_std=loo["E_true_std"],
            triangulation_report=f"{e_true:.2f} ± {loo['E_true_std']:.2f} nats (LOO std)",
            holdout=holdout,
            leave_one_out=loo,
            sanity=sanity,
            preflight=pf,
            per_model=per,
            overall_pass=holdout["gate_pass"] and loo["gate_pass"] and sanity["gate_pass"],
        )
        out["failure_diagnosis"] = ld.diagnose_failure(out)
        return out
    except Exception as exc:
        return dict(corpus=f"Meta Step-2 (ti{token_tag})", skipped=True, error=str(exc))
    finally:
        ms2.LOG_DIR = old_dir
        ms2.LADDER = old_ladder
        ms2.TOKEN_BUDGET_TAG = old_tag
        ms2.DEFAULT_TRIPLET = old_triplet


def _run_pythia_extended() -> dict[str, Any]:
    """7-size ladder adding 1B if CSV present."""
    keys = list(pcl.DEFAULT_LADDER)
    if os.path.isfile(os.path.join(pcl._PUBLIC_PYTHIA, "Pythia-1b.csv")):
        keys = ("14m", "70m", "160m", "410m", "1b", "1.4b", "6.9b")
    cache = os.path.join(pcl.OUT_DIR, "cache")
    results = {}
    for sk in keys:
        df = pcl.load_loss_tsv(sk, cache)
        results[sk] = pcl.build_result_from_log(f"pythia_{sk}", sk, df)
    tri = pcl.triangulate_ladder(results)
    e_true = tri["n_point_fixed_034"]["E_true"]
    per = tri["per_model"]
    for row in per:
        row["final_loss"] = results[row["size"]]["final_train_loss"]
    holdout = cer.holdout_test(results, "6.9b")
    loo = cer.leave_one_out_e_true(results)
    sanity = cer.sanity_gates(per, e_true)
    pf = ld.preflight_ladder(per, e_true)
    out = dict(
        corpus=f"The Pile (Pythia {len(keys)}-size)",
        protocol="matched Pile ladder; E_app=late C*",
        n_sizes=len(keys),
        ladder_sizes=list(keys),
        triangulation_fixed_alpha=e_true,
        uncertainty_loo_std=loo["E_true_std"],
        triangulation_report=f"{e_true:.2f} ± {loo['E_true_std']:.2f} nats (LOO std)",
        per_model=per,
        holdout=holdout,
        leave_one_out=loo,
        sanity=sanity,
        preflight=pf,
        overall_pass=holdout["gate_pass"] and loo["gate_pass"] and sanity["gate_pass"],
    )
    out["failure_diagnosis"] = ld.diagnose_failure(out)
    return out


def _corpus_pass(data: dict | None) -> bool:
    if not data or data.get("skipped"):
        return False
    if "overall_pass" in data:
        return bool(data["overall_pass"])
    h = data.get("holdout") or data.get("holdout_410m") or {}
    return bool(
        h.get("gate_pass")
        and data.get("leave_one_out", {}).get("gate_pass")
        and data.get("sanity", {}).get("gate_pass")
    )


def main():
    if hasattr(sys.stdout, "reconfigure"):
        try:
            sys.stdout.reconfigure(encoding="utf-8")
        except Exception:
            pass

    log_dir = os.environ.get("META_STEP2_LOG_DIR") or ms2.LOG_DIR
    print(f"Scanning Step-2 ladders in: {log_dir}")
    print()

    sections: dict[str, Any] = {}

    sections["pythia_pile_6"] = _finalize_section(cer.run_pythia())
    sections["pythia_pile_7"] = _finalize_section(_run_pythia_extended())
    sections["meta_step2_ti134698_3pt"] = _finalize_section(cer.run_meta_step2())

    ladders = _discover_step2_ladders(log_dir)
    for tag, sizes in sorted(ladders.items()):
        key = f"meta_step2_ti{tag}_{len(sizes)}sz"
        sections[key] = _finalize_section(_run_step2_ladder(log_dir, tag, sizes))

    sections["olmo_dolma"] = _finalize_section(cer.run_olmo())

    try:
        kdf = pd.read_csv(kcl.LOG_PATH)
        for corpus in kcl.discover_corpora(kdf):
            key = f"kempner_{corpus.replace('-', '_')}"
            try:
                sections[key] = _finalize_section(kcl.build_ladder(corpus, kdf))
                if not sections[key].get("overall_pass"):
                    pkey = f"{key}_mono_prefix"
                    try:
                        sections[pkey] = _finalize_section(
                            kcl.build_ladder(corpus, kdf, monotonic_prefix_only=True)
                        )
                    except Exception:
                        pass
            except Exception as exc:
                sections[key] = dict(corpus=corpus, skipped=True, error=str(exc))
    except Exception as exc:
        sections["kempner"] = dict(corpus="Kempner OLMo", skipped=True, error=str(exc))

    try:
        odf = pd.read_csv(optl.LOG_PATH)
        sections["opt_pile_10B"] = _finalize_section(optl.build_ladder(odf))
        best = optl.best_monotonic_target(odf)
        if best:
            best = dict(best)
            best["corpus"] = f"Meta OPT / PILE @ 10^{best.get('target_log10_tokens', '?'):.1f} (mono)"
            sections["opt_pile_best_mono"] = _finalize_section(best)
    except Exception as exc:
        sections["opt_pile"] = dict(corpus="Meta OPT / PILE", skipped=True, error=str(exc))

    sections["cerebras_pile"] = _finalize_section(plc.run_cerebras_pile())
    sections["owt_trained"] = _finalize_section(cer.run_owt_summary())

    # Summary table
    lines = [
        "=" * 100,
        "PUBLIC LADDER SWEEP - log-only Chinchilla-E holdout/LOO",
        "=" * 100,
        f"{'Corpus':<46} {'n':>2}  {'E_true ± LOO':>18}  {'holdout Δ':>10}  {'PASS':>5}",
        "-" * 100,
    ]
    pass_count = 0
    test_count = 0
    fail_lines: list[str] = []

    for key, data in sections.items():
        if data.get("skipped"):
            lines.append(f"{data.get('corpus', key):<46}  - skipped ({data.get('error', '?')[:40]})")
            continue
        if "triangulation_fixed_alpha" not in data:
            continue
        test_count += 1
        e = data["triangulation_fixed_alpha"]
        lo = data.get("leave_one_out", {})
        loo_std = lo.get("E_true_std", float("nan"))
        h = data.get("holdout") or data.get("holdout_410m") or {}
        delta = h.get("delta", float("nan"))
        ok = _corpus_pass(data)
        if ok:
            pass_count += 1
        else:
            fail_lines.append(f"  FAIL  {data.get('corpus', key)}: {data.get('failure_diagnosis', '?')}")
        label = data.get("corpus") or key
        lines.append(
            f"{label:<46} {data.get('n_sizes', '?'):>2}  "
            f"{e:.2f} ± {loo_std:.2f} nats{' ':>4} "
            f"{delta:>10.4f}  {'YES' if ok else 'no':>5}"
        )

    lines.extend(
        [
            "-" * 100,
            f"Corpora tested: {test_count}  |  Holdout+LOO+sanity PASS: {pass_count}",
            "",
            "FAILURES (named causes - report prominently):",
            *fail_lines,
            "",
            "Notes:",
            "  - Method = N-point OLS (α=0.34) + holdout on largest size + LOO std < 0.15",
            "  - Kempner *_mono_prefix = sensitivity retry on longest monotonic prefix",
            "  - Run floor_db.py for FloorDB CSV + law probes",
            "=" * 100,
        ]
    )
    report = "\n".join(lines)
    print(report)

    out_json = os.path.join(OUT_DIR, "sweep_report.json")
    with open(out_json + ".tmp", "w", encoding="utf-8") as f:
        json.dump(
            dict(
                log_dir=log_dir,
                step2_ladders=ladders,
                sections=sections,
                pass_count=pass_count,
                test_count=test_count,
            ),
            f,
            indent=2,
        )
    os.replace(out_json + ".tmp", out_json)

    out_txt = os.path.join(OUT_DIR, "sweep_report.txt")
    with open(out_txt + ".tmp", "w", encoding="utf-8") as f:
        f.write(report)
    os.replace(out_txt + ".tmp", out_txt)
    print(f"[done] {out_txt}")


if __name__ == "__main__":
    main()
