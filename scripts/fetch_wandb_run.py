"""Fetch one W&B run by project + run name; save step/ce CSV."""
import argparse
import os
from pathlib import Path
import pandas as pd

_REPO = Path(__file__).resolve().parents[1]
_KEY = _REPO / ".wandb_api_key"
if _KEY.exists():
    os.environ.setdefault("WANDB_API_KEY", _KEY.read_text(encoding="utf-8").strip())
import wandb


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--project", required=True)
    p.add_argument("--run-name", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--loss-key", default="train/CrossEntropyLoss")
    p.add_argument("--max-rows", type=int, default=200000)
    args = p.parse_args()

    api = wandb.Api()
    run = None
    for r in api.runs(args.project, per_page=200):
        if r.name == args.run_name:
            run = r
            break
    if run is None:
        raise SystemExit(f"Run not found: {args.run_name}")

    print(f"Fetching {run.name} id={run.id}")
    rows = []
    for row in run.scan_history(keys=[args.loss_key, "_step"], page_size=5000):
        if args.loss_key in row and row.get(args.loss_key) is not None and row.get("_step") is not None:
            rows.append((float(row["_step"]), float(row[args.loss_key])))
        if len(rows) >= args.max_rows:
            break
    rows.sort(key=lambda x: x[0])
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"step": [r[0] for r in rows], "ce": [r[1] for r in rows]}).to_csv(out, index=False)
    print(f"saved {out} n={len(rows)} steps {rows[0][0]:.0f}-{rows[-1][0]:.0f}")


if __name__ == "__main__":
    main()
