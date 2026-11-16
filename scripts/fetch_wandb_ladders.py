#!/usr/bin/env python3
"""Fetch public W&B training curves into data/public_logs/ for IV-E triangulation.

    python scripts/fetch_wandb_ladders.py
    python scripts/fetch_wandb_ladders.py --probe   # list runs only

Reads WANDB_API_KEY from env or gitignored .wandb_api_key (never committed).
"""

from __future__ import annotations

import argparse
import os
import re
import sys
from pathlib import Path

import pandas as pd

_SCRIPT_DIR = Path(__file__).resolve().parent
_REPO = _SCRIPT_DIR.parent
_KEY_FILE = _REPO / ".wandb_api_key"
_PUBLIC = _REPO / "data" / "public_logs"

LOSS_KEYS = [
    "train/CrossEntropyLoss",
    "train/loss",
    "train/lm_loss",
    "loss",
]
STEP_KEYS = ["_step", "trainer/global_step", "step"]

# Base pretraining sizes (HF model cards)
SMOLLM2_PROJECTS = [
    "HuggingFaceTB/smolLM2",
    "HuggingFaceTB/SmolLM2",
    "huggingface/SmolLM3-training-logs",  # fallback: public multi-size logs
]

SMOLLM2_SIZES = {
    "135m": (135_000_000, re.compile(r"135|135m|135M", re.I)),
    "360m": (360_000_000, re.compile(r"360|360m|360M", re.I)),
    "1.7b": (1_700_000_000, re.compile(r"1\.7|1_7|1-7|1\.7b|1_7b|1-7b|1700|3b|3\.0", re.I)),
    "3b": (3_000_000_000, re.compile(r"3b|3\.0b|3000m", re.I)),
}

OLMO_TARGETS = [
    ("WB-OLMo-1B", "ai2-llm/OLMo-1B", "olmo", "train/CrossEntropyLoss"),
    ("WB-OLMo-7B", "ai2-llm/OLMo-7B", "olmo", "train/CrossEntropyLoss"),
    ("WB-OLMo2-7B", "ai2-llm/OLMo-2-1124-7B", "olmo", "train/CrossEntropyLoss"),
    ("WB-OLMo2-13B", "ai2-llm/OLMo-2-1124-13B", "olmo", "train/CrossEntropyLoss"),
]

PYTHIA_GROUPS = [
    ("Pythia-2.8b", re.compile(r"2-7B|2\.7B New")),
    ("Pythia-12b", re.compile(r"12B")),
    ("Pythia-1b", re.compile(r"800M|1B")),
]


def _load_key() -> bool:
    if os.environ.get("WANDB_API_KEY"):
        return True
    if _KEY_FILE.exists():
        key = _KEY_FILE.read_text(encoding="utf-8").strip()
        if key:
            os.environ["WANDB_API_KEY"] = key
            return True
    return False


def _curve_from_run(run, loss_key: str | None = None, min_pts: int = 500):
    """Fast path: sampled history first (avoids million-step scan_history hangs)."""
    keys = STEP_KEYS + (LOSS_KEYS if loss_key is None else [loss_key])
    lk = loss_key
    rows = []
    try:
        df = run.history(samples=500000, keys=keys)
        if df is not None and len(df) > 0:
            lk = lk or next((k for k in LOSS_KEYS if k in df.columns and df[k].notna().sum() > 50), None)
            step_col = next((k for k in STEP_KEYS if k in df.columns), None)
            if lk and step_col:
                for s, v in zip(df[step_col], df[lk]):
                    if pd.notna(s) and pd.notna(v):
                        rows.append((float(s), float(v)))
    except Exception:
        rows = []
    if len(rows) < min_pts:
        try:
            for row in run.scan_history(keys=keys, page_size=5000):
                step = next((float(row[k]) for k in STEP_KEYS if k in row and row[k] is not None), None)
                if step is None:
                    continue
                if lk is None:
                    for candidate in LOSS_KEYS:
                        val = row.get(candidate)
                        if val is not None and not (isinstance(val, float) and pd.isna(val)):
                            lk = candidate
                            rows.append((step, float(val)))
                            break
                else:
                    val = row.get(lk)
                    if val is not None and not (isinstance(val, float) and pd.isna(val)):
                        rows.append((step, float(val)))
                if len(rows) >= 200000:
                    break
        except Exception:
            return None
    if len(rows) < min_pts:
        return None
    rows.sort(key=lambda x: x[0])
    return rows, lk


def _save_csv(subdir: str, name: str, rows: list[tuple[float, float]]) -> Path:
    out_dir = _PUBLIC / subdir
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{name}.csv"
    pd.DataFrame({"step": [r[0] for r in rows], "ce": [r[1] for r in rows]}).to_csv(path, index=False)
    return path


def _match_smollm2(run) -> str | None:
    blob = " ".join(
        str(x) for x in [run.name, run.group, run.tags, (run.config or {}).get("model_name_or_path", "")]
    )
    for size_key, (_, pat) in SMOLLM2_SIZES.items():
        if pat.search(blob):
            return size_key
    return None


def probe_smollm2(api, limit: int = 20) -> None:
    for project in SMOLLM2_PROJECTS:
        print(f"Probing {project} (first {limit} runs)...")
        try:
            for i, run in enumerate(api.runs(project, per_page=limit)):
                sk = _match_smollm2(run)
                print(f"  [{i}] {run.name!r} group={run.group!r} size={sk} state={run.state}")
        except Exception as e:
            print(f"  ERR: {e}")


def fetch_smollm2(api) -> list[dict]:
    best: dict[str, tuple] = {}
    used_project = None
    for project in SMOLLM2_PROJECTS:
        try:
            runs = list(api.runs(project, per_page=200))
        except Exception as e:
            print(f"  {project}: skip ({e})")
            continue
        if not runs:
            continue
        used_project = project
        print(f"  Scanning {project} ({len(runs)} runs on first page)...")
        for run in runs:
            if run.state != "finished":
                continue
            sk = _match_smollm2(run)
            if sk is None:
                continue
            extracted = _curve_from_run(run, min_pts=1000)
            if extracted is None:
                continue
            rows, lk = extracted
            if sk not in best or len(rows) > len(best[sk][0]):
                best[sk] = (rows, lk, run)
        if len(best) >= 3:
            break
    out = []
    for sk, (rows, lk, run) in sorted(best.items(), key=lambda x: SMOLLM2_SIZES.get(x[0], (0,))[0]):
        fname = f"SmolLM2-{sk.replace('.', '_')}"
        path = _save_csv("smollm2", fname, rows)
        proj = used_project or "unknown"
        out.append(dict(size=sk, file=str(path.name), n_points=len(rows), wandb_run=f"{proj}/{run.id}", loss_key=lk))
        print(f"  {fname}: {len(rows)} pts -> {path.name}")
    if not out:
        print("  No SmolLM curves fetched (projects missing or no matching runs)")
    return out


def fetch_olmo(api) -> list[dict]:
    out = []
    for label, project, subdir, lk in OLMO_TARGETS:
        best = None
        for run in api.runs(project, per_page=8):
            extracted = _curve_from_run(run, loss_key=lk, min_pts=100)
            if extracted is None:
                continue
            rows, key = extracted
            if best is None or len(rows) > len(best[0]):
                best = (rows, key, run)
            if best and len(best[0]) >= 50000:
                break
        if best is None:
            print(f"  {label}: SKIP")
            continue
        rows, key, run = best
        path = _save_csv(subdir, label, rows)
        out.append(dict(label=label, file=str(path.name), n_points=len(rows), wandb_run=f"{project}/{run.id}"))
        print(f"  {label}: {len(rows)} pts -> {path}")
    return out


def fetch_pythia(api) -> list[dict]:
    out = []
    for label, pat in PYTHIA_GROUPS:
        best = None
        n_checked = 0
        for run in api.runs("eleutherai/pythia", per_page=80):
            n_checked += 1
            if not run.group or not pat.search(run.group):
                continue
            extracted = _curve_from_run(run, loss_key="train/lm_loss", min_pts=500)
            if extracted is None:
                continue
            rows, lk = extracted
            if best is None or len(rows) > len(best[0]):
                best = (rows, lk, run)
            if best and len(best[0]) >= 80000:
                break
        if best is None:
            print(f"  {label}: SKIP")
            continue
        rows, lk, run = best
        path = _save_csv("pythia", label, rows)
        out.append(dict(label=label, file=str(path.name), n_points=len(rows), wandb_run=f"eleutherai/pythia/{run.id}"))
        print(f"  {label}: {len(rows)} pts -> {path}")
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--probe", action="store_true")
    parser.add_argument("--smollm2-only", action="store_true")
    parser.add_argument("--olmo-only", action="store_true")
    parser.add_argument("--pythia-only", action="store_true")
    args = parser.parse_args()

    if not _load_key():
        print("Set WANDB_API_KEY or create .wandb_api_key", file=sys.stderr)
        sys.exit(1)

    import wandb

    api = wandb.Api()

    if args.probe:
        probe_smollm2(api)
        return

    manifest = {}
    do_olmo = args.olmo_only or (not args.smollm2_only and not args.pythia_only)
    do_pythia = args.pythia_only or (not args.smollm2_only and not args.olmo_only)
    do_smollm = args.smollm2_only or (not args.olmo_only and not args.pythia_only)

    if do_olmo:
        print("Fetching OLMo...")
        manifest["olmo"] = fetch_olmo(api)
    if do_pythia:
        print("Fetching Pythia...")
        manifest["pythia"] = fetch_pythia(api)
    if do_smollm:
        print("Fetching SmolLM...")
        manifest["smollm2"] = fetch_smollm2(api)

    out = _PUBLIC / "wandb_fetch_manifest.json"
    import json

    out.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"\n[done] {out}")


if __name__ == "__main__":
    main()
