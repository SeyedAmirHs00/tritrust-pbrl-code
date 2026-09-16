#!/usr/bin/env python3
"""Analyze seed directories to find empty or anomalously small eval.csv files.

Groups seeds by exp/environment/test and detects:
  1. Empty files (0 bytes)
  2. Files with size drastically lower than peers in the same group

Anomaly detection uses a threshold-based approach:
  - Files are flagged if size < (median_size * threshold)
  - Default threshold is 0.5 (flag if < 50% of group median)

Usage:
    python analyze_outliers.py [ROOT] [--threshold 0.5] [--log-file LOG]

ROOT defaults to the current directory.
"""

from pathlib import Path
import argparse
import re
import sys
from collections import defaultdict


def find_seed_dirs(root: Path):
    """Find all seed directories matching exp_*/*/*/seed\\d+."""
    seed_re = re.compile(r"^seed\d+$")
    found = []
    for p in root.glob("exp_*/*/*/seed*"):
        if p.is_dir() and seed_re.match(p.name):
            found.append(p)
    return sorted(found)


def get_file_size(seed_dir: Path):
    """Get size of eval.csv, or None if missing/error."""
    eval_path = seed_dir / "test" / "eval.csv"
    if not eval_path.exists():
        return None
    try:
        return eval_path.stat().st_size
    except OSError:
        return None


def analyze_group(group_seeds, threshold):
    """Analyze a single group (same exp/environment/test).

    Returns:
        dict with keys:
            'empty': list of (seed_name, seed_path) with size 0
            'missing': list of (seed_name, seed_path) with missing file
            'outliers': list of (seed_name, seed_path, size, median) where size is anomalously low
            'normal': list of (seed_name, seed_path, size)
            'median': median size of normal (non-zero, non-missing) files in group
    """
    # Separate by status
    normal_sizes = {}  # seed_name -> size
    empty = []
    missing = []

    for seed_name, seed_path in group_seeds:
        size = get_file_size(seed_path)
        if size is None:
            missing.append((seed_name, seed_path))
        elif size == 0:
            empty.append((seed_name, seed_path))
        else:
            normal_sizes[seed_name] = size

    # Compute median of normal sizes
    if normal_sizes:
        sizes_list = sorted(normal_sizes.values())
        n = len(sizes_list)
        if n % 2 == 0:
            median = (sizes_list[n // 2 - 1] + sizes_list[n // 2]) / 2
        else:
            median = sizes_list[n // 2]
    else:
        median = None

    # Identify outliers (sizes below threshold * median)
    outliers = []
    normal = []
    if median is not None:
        cutoff = median * threshold
        for seed_name, size in normal_sizes.items():
            if size < cutoff:
                outliers.append((seed_name, Path("dummy"), size, median))
            else:
                normal.append((seed_name, Path("dummy"), size))

    return {
        "empty": empty,
        "missing": missing,
        "outliers": outliers,
        "normal": normal,
        "median": median,
    }


def main(root_dir: str, threshold: float, log_file: str = None):
    """Main analysis routine."""
    root = Path(root_dir)
    seeds = find_seed_dirs(root)

    if not seeds:
        print("No seed directories found matching pattern exp_*/*/*/seed\\d*")
        return 1

    # Build tree: exp -> environment -> test -> list of (seed_name, seed_path)
    tree = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    for sd in seeds:
        try:
            rel = sd.relative_to(root).parts
        except Exception:
            rel = sd.parts[-4:]

        if len(rel) >= 4:
            exp_name, env_name, test_name, seed_name = (
                rel[-4],
                rel[-3],
                rel[-2],
                rel[-1],
            )
        else:
            exp_name = rel[0] if rel else "<unknown>"
            env_name = rel[1] if len(rel) > 1 else "<unknown>"
            test_name = rel[2] if len(rel) > 2 else "<unknown>"
            seed_name = rel[-1] if rel else sd.name

        tree[exp_name][env_name][test_name].append((seed_name, sd))

    # Analyze each group
    out_lines = []

    def emit(line: str = ""):
        out_lines.append(line)
        print(line)

    emit("=" * 80)
    emit("SEED OUTLIER ANALYSIS")
    emit(f"Threshold: {threshold:.1%} of group median")
    emit("=" * 80)

    total_empty = 0
    total_missing = 0
    total_outliers = 0

    for exp_name in sorted(tree.keys()):
        emit(f"\n'{exp_name}'")
        for env_name in sorted(tree[exp_name].keys()):
            emit(f"    '{env_name}'")
            for test_name in sorted(tree[exp_name][env_name].keys()):
                group_seeds = tree[exp_name][env_name][test_name]
                analysis = analyze_group(group_seeds, threshold)

                # Only print if there are issues
                has_issues = (
                    analysis["empty"] or analysis["missing"] or analysis["outliers"]
                )
                if not has_issues:
                    continue

                emit(f"         '{test_name}'")
                median_str = (
                    f"{analysis['median']:.0f}"
                    if analysis["median"] is not None
                    else "N/A"
                )
                emit(f"              [group median: {median_str} bytes]")

                # Empty files
                for seed_name, seed_path in analysis["empty"]:
                    emit(f"              '{seed_name}'    EMPTY (0 bytes)")
                    total_empty += 1

                # Missing files
                for seed_name, seed_path in analysis["missing"]:
                    emit(f"              '{seed_name}'    MISSING")
                    total_missing += 1

                # Outliers (anomalously low size)
                for seed_name, seed_path, size, median in analysis["outliers"]:
                    pct = (size / median * 100) if median else 0
                    emit(
                        f"              '{seed_name}'    OUTLIER ({size} bytes, {pct:.1f}% of median)"
                    )
                    total_outliers += 1

    emit("\n" + "=" * 80)
    emit("SUMMARY")
    emit("=" * 80)
    emit(f"Total empty files:              {total_empty}")
    emit(f"Total missing files:            {total_missing}")
    emit(f"Total outlier-sized files:      {total_outliers}")
    emit(
        f"Total issues:                   {total_empty + total_missing + total_outliers}"
    )

    # Write log file if requested
    if log_file:
        try:
            lp = (
                (Path(root) / log_file)
                if not Path(log_file).is_absolute()
                else Path(log_file)
            )
            lp.parent.mkdir(parents=True, exist_ok=True)
            lp.write_text("\n".join(out_lines) + "\n", encoding="utf-8")
            print(f"\nWrote analysis to: {lp}")
        except Exception as e:
            print(f"Failed to write log file '{log_file}': {e}")

    # Return 0 if no issues, else return count of issues
    issues = total_empty + total_missing + total_outliers
    return 0 if issues == 0 else 2


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Detect empty and anomalously small eval.csv files in seed directories"
    )
    parser.add_argument(
        "root", nargs="?", default=".", help="project root (default '.')"
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.8,
        help="flag files with size < (median * threshold). Default 0.5 (50%% of median)",
    )
    parser.add_argument(
        "--log-file",
        "-l",
        default=None,
        help="path to write log file (relative to root unless absolute)",
    )
    args = parser.parse_args()

    sys.exit(main(args.root, args.threshold, args.log_file))
