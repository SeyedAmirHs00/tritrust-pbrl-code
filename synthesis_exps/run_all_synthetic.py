"""Run all synthetic experiment data scripts (optional --plot after each)."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def _run(script: str, extra: list[str]) -> None:
    cmd = [sys.executable, str(ROOT / script), *extra]
    print(f"\n>>> {' '.join(cmd)}", flush=True)
    subprocess.check_call(cmd, cwd=str(ROOT.parent))


def main() -> None:
    p = argparse.ArgumentParser(description="Run synthetic experiment data (and plots via entry scripts)")
    p.add_argument("--seeds", type=int, default=200)
    p.add_argument("--overwrite", action="store_true")
    p.add_argument("--plot-only", action="store_true", help="Replot from existing CSVs only")
    args = p.parse_args()

    common = ["--overwrite"] if args.overwrite else []
    if not args.overwrite and not args.plot_only:
        print("Note: pass --overwrite to replace existing result dirs.", flush=True)

    if args.plot_only:
        replot_scripts = [
            ("branch_symmetry.py", ["--replot"]),
            ("init_scale_sweep.py", ["--replot"]),
            ("expert_ratio_K10.py", ["--replot"]),
            ("overlap_sweep.py", ["--replot"]),
            ("partial_adversary.py", ["--replot"]),
            ("partial_adversary_alpha_curve.py", ["--replot"]),
            ("T_sweep_lr_scaled.py", ["--replot"]),
        ]
        for script, flags in replot_scripts:
            _run(script, flags)
        return

    runs = [
        # ("branch_symmetry.py", ["--seeds", str(args.seeds), *common]),
        # ("init_scale_sweep.py", ["--seeds", str(args.seeds), *common]),
        ("expert_ratio_K10.py", ["--seeds", str(min(args.seeds, 100)), *common]),
        ("overlap_sweep.py", ["--seeds", str(min(args.seeds, 120)), *common]),
        ("partial_adversary.py", ["--seeds", str(args.seeds), *common]),
        ("partial_adversary_alpha_curve.py", ["--seeds", str(args.seeds), *common]),
        ("T_sweep_lr_scaled.py", ["--seeds", str(min(args.seeds, 120)), *common]),
        ("enhancement_ablation.py", ["--seeds", str(args.seeds), *common]),
    ]
    for script, flags in runs:
        _run(script, flags)


if __name__ == "__main__":
    main()
