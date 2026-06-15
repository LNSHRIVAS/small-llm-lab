"""
Final-checkpoint point ladders → Chinchilla-E triangulation.

For sources that publish only (N, final_loss) without step curves — e.g.
Cerebras-GPT on PILE test eval. Protocol is weaker than log-only C* but
still tests the separable ansatz on monotonic final losses.

    python scripts/point_ladder_chinchilla_e.py
"""

from __future__ import annotations

import json
import os
import sys
from typing import Any

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.abspath(os.path.join(_SCRIPT_DIR, ".."))
if _SCRIPT_DIR not in sys.path:
    sys.path.insert(0, _SCRIPT_DIR)

import chinchilla_e_robustness as cer  # noqa: E402
import ladder_diagnostics as ld  # noqa: E402
import meta_step2_chinchilla_e_from_logs as ms2  # noqa: E402
import owt_chinchilla_e as oce  # noqa: E402

OUT_DIR = os.path.join(_REPO, "results", "point_ladder_chinchilla_e")
os.makedirs(OUT_DIR, exist_ok=True)

# Hardcoded from Cerebras HF card (PILE test cross-entropy, final ckpt)
CEREBRAS_PILE = [
    ("111M", 111_000_000, 2.566),
    ("256M", 256_000_000, 2.299),
    ("590M", 590_000_000, 2.184),
    ("1.3B", 1_300_000_000, 1.996),
    ("2.7B", 2_700_000_000, 1.834),
    ("6.7B", 6_700_000_000, 1.704),
    ("13B", 13_000_000_000, 1.575),
]


def build_point_ladder(
    label: str,
    points: list[tuple[str, int, float]],
    protocol: str,
) -> dict[str, Any]:
    flat: dict[str, dict] = {}
    per: list[dict] = []
    for key, n, loss in points:
        flat[key] = ms2._flat_result(n, loss, name=f"{label}-{key}")
        per.append(dict(size=key, n_params=n, E_app=loss, final_loss=loss))

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
    return dict(
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
        failure_diagnosis=ld.diagnose_failure(
            dict(overall_pass=overall, holdout=holdout, leave_one_out=loo, sanity=sanity, preflight=pf)
        ),
    )


def run_cerebras_pile() -> dict[str, Any]:
    return build_point_ladder(
        "Cerebras-GPT / PILE (final eval)",
        CEREBRAS_PILE,
        "E_app=PILE test CE at final checkpoint (varying train FLOPs per size)",
    )


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        try:
            sys.stdout.reconfigure(encoding="utf-8")
        except Exception:
            pass

    sections = {"cerebras_pile": run_cerebras_pile()}
    for key, data in sections.items():
        ok = "PASS" if data["overall_pass"] else "fail"
        print(f"{data['corpus']}: E={data['triangulation_report']}  {ok}")
        print(f"  diagnosis: {data['failure_diagnosis']}")

    out = os.path.join(OUT_DIR, "point_ladder_report.json")
    with open(out + ".tmp", "w", encoding="utf-8") as f:
        json.dump(sections, f, indent=2)
    os.replace(out + ".tmp", out)
    print(f"\n[done] {out}")


if __name__ == "__main__":
    main()
