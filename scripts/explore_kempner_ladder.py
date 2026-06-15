"""Quick probe: Kempner OLMo iso-flop ladders for triangulation."""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
KEMP = ROOT / "data" / "public_logs" / "kempner" / "kempner_sweep.csv"


def main() -> None:
    k = pd.read_csv(KEMP)
    k = k[k.state == "finished"].copy()
    k["iso"] = k["iso_flop"].astype(float)
    train_col = "train/CrossEntropyLoss"

    for data in sorted(k.data.unique()):
        sub = k[k.data == data]
        best_iso = None
        best_n = 0
        for iso in sub["iso"].unique():
            s = sub[sub.iso == iso]
            n = s["params"].nunique()
            if n > best_n:
                best_n = n
                best_iso = iso
        if best_n < 3:
            continue
        s = sub[sub.iso == best_iso].drop_duplicates("params").sort_values("params")
        print(f"\n=== {data} iso_flop={best_iso:.2e} n_sizes={len(s)} ===")
        for _, r in s.iterrows():
            print(
                f"  N={r.params/1e6:.0f}M  tokens={r.tokens/1e9:.2f}B  "
                f"train={r[train_col]:.4f}  smollm_val={r['eval/smollm_val/CrossEntropyLoss']:.4f}"
            )


if __name__ == "__main__":
    main()
