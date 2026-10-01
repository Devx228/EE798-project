"""Run the cartesian product of a sweep file and collect a summary CSV.

    python scripts/sweep.py configs/sweeps/recall_capacity.yaml
    python scripts/sweep.py configs/sweeps/state_tracking.yaml --only ssm_neg --steps 500

Finished runs (a ``metrics.json`` exists) are skipped, so an interrupted sweep can be resumed by
re-running the same command.
"""

from __future__ import annotations

import argparse
import csv
import itertools
import json
import sys
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from fadingmem.config import apply_overrides, load_config  # noqa: E402
from fadingmem.train import train  # noqa: E402


def expand(sweep_path: Path):
    spec = yaml.safe_load(sweep_path.read_text())
    base = apply_overrides(load_config(sweep_path.parent / spec["base"]), spec.get("overrides", {}))
    axes = spec["axes"]
    for combo in itertools.product(*axes.values()):
        label = "__".join(c["label"] for c in combo)
        cfg = base
        for choice in combo:
            cfg = apply_overrides(cfg, choice.get("set", {}))
        yield label, dict(zip(axes.keys(), (c["label"] for c in combo))), cfg


def summarise(metrics: dict, label: str, axis_labels: dict) -> dict:
    row = {"run": label, **axis_labels}
    row.update(
        params=metrics["params"],
        state_size=metrics.get("state_size"),
        steps=metrics["steps_run"],
        steps_to_90=metrics.get("steps_to_90"),
        train_time_s=round(metrics["train_time_s"], 1),
        peak_mem_mb=None if metrics["peak_mem_mb"] is None else round(metrics["peak_mem_mb"], 1),
    )
    for name, res in metrics["eval"].items():
        row[f"acc_{name}"] = round(res["acc"], 4)
        row[f"seq_acc_{name}"] = round(res["seq_acc"], 4)
    return row


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("sweep", type=Path)
    ap.add_argument("--out", type=Path, default=Path("results"))
    ap.add_argument("--only", nargs="*", default=None, help="run only labels containing all of these substrings")
    ap.add_argument("--steps", type=int, default=None, help="override train.steps (quick trial runs)")
    ap.add_argument("--seeds", type=int, nargs="*", default=[0])
    ap.add_argument("--device", default=None)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    name = yaml.safe_load(args.sweep.read_text())["name"]
    root = args.out / name
    rows = []
    for label, axis_labels, cfg in expand(args.sweep):
        if args.only and not all(s in label for s in args.only):
            continue
        for seed in args.seeds:
            run_cfg = apply_overrides(cfg, {"train.seed": seed})
            if args.steps:
                run_cfg = apply_overrides(run_cfg, {"train.steps": args.steps})
            run_dir = root / "raw" / f"{label}__seed{seed}"
            if args.dry_run:
                print(run_dir)
                continue
            metrics_file = run_dir / "metrics.json"
            if metrics_file.exists():
                print(f"[skip] {run_dir.name}")
                metrics = json.loads(metrics_file.read_text())
            else:
                print(f"\n=== {run_dir.name} ===", flush=True)
                metrics = train(run_cfg, run_dir, args.device)
            rows.append(summarise(metrics, label, {**axis_labels, "seed": seed}))

    if rows:
        root.mkdir(parents=True, exist_ok=True)
        fields = list(dict.fromkeys(k for r in rows for k in r))
        with open(root / "summary.csv", "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fields)
            writer.writeheader()
            writer.writerows(rows)
        print(f"\nwrote {root / 'summary.csv'} ({len(rows)} runs)")


if __name__ == "__main__":
    main()
