#!/usr/bin/env python3
"""Check seed directories for non-empty test/eval.csv files.

Searches for directories matching the pattern exp_*/*/*/seed\d+ and for each
seed directory reports whether `test/eval.csv` exists and is non-empty.

Usage:
        python check.py [ROOT]

ROOT defaults to the current directory.
"""

from pathlib import Path
import argparse
import re
import sys


def find_seed_dirs(root: Path):
    # Glob seed* directories under exp_*/*/* then filter to seed\d+
    seed_re = re.compile(r"^seed\d+$")
    found = []
    for p in root.glob("exp_*/*/*/seed*"):
        if p.is_dir() and seed_re.match(p.name):
            found.append(p)
    return sorted(found)


def check_eval(seed_dir: Path):
    eval_path = seed_dir / "test" / "eval.csv"
    if not eval_path.exists():
        return "missing", None
    try:
        size = eval_path.stat().st_size
    except OSError:
        return "error", None
    if size > 0:
        return "non-empty", size
    return "empty", 0


def main(root_dir: str):
    root = Path(root_dir)
    seeds = find_seed_dirs(root)
    if not seeds:
        print("No seed directories found matching pattern exp_*/*/*/seed\\d*")
        return 1

    counts = {"total": len(seeds), "non-empty": 0, "empty": 0, "missing": 0, "error": 0}

    # Build a nested structure: exp -> environment -> testdir -> list of (seed, status, val)
    tree = {}
    for sd in seeds:
        status, val = check_eval(sd)
        # update counts
        if status in counts:
            counts[status] += 1
        else:
            counts["error"] += 1

        try:
            rel = sd.relative_to(root).parts
        except Exception:
            # fallback to name components
            rel = sd.parts[-4:]

        # Expect rel to be [exp_folder, environment, testdir, seed]
        if len(rel) >= 4:
            exp_name, env_name, test_name, seed_name = (
                rel[-4],
                rel[-3],
                rel[-2],
                rel[-1],
            )
        else:
            # If structure is unexpected, put under a generic bucket
            exp_name = rel[0] if rel else "<unknown>"
            env_name = rel[1] if len(rel) > 1 else "<unknown>"
            test_name = rel[2] if len(rel) > 2 else "<unknown>"
            seed_name = rel[-1] if rel else sd.name

        tree.setdefault(exp_name, {}).setdefault(env_name, {}).setdefault(
            test_name, []
        ).append((seed_name, status, val))

    # Collect hierarchical output lines so we can both print and write them to a log file
    out_lines = []

    def emit(line: str = ""):
        out_lines.append(line)
        print(line)

    # Print hierarchical output as requested:
    # 'exp'
    #     'environment'
    #          'test'
    #               'seed*'   status
    for exp_name in sorted(tree.keys()):
        emit(f"'{exp_name}'")
        for env_name in sorted(tree[exp_name].keys()):
            emit(f"    '{env_name}'")
            for test_name in sorted(tree[exp_name][env_name].keys()):
                emit(f"         '{test_name}'")
                for seed_name, status, val in sorted(
                    tree[exp_name][env_name][test_name]
                ):
                    if status == "non-empty":
                        status_msg = f"OK (present, {val} bytes)"
                    elif status == "empty":
                        status_msg = "EMPTY (exists but zero bytes)"
                    elif status == "missing":
                        status_msg = "MISSING"
                    else:
                        status_msg = "ERROR"
                    emit(f"              '{seed_name}'    {status_msg}")

    emit("")
    emit("Summary:")
    emit(f"  total seed dirs: {counts['total']}")
    emit(f"  non-empty eval.csv: {counts['non-empty']}")
    emit(f"  empty eval.csv: {counts['empty']}")
    emit(f"  missing eval.csv: {counts['missing']}")
    emit(f"  error checking eval.csv: {counts['error']}")

    # Write the lines to a log file in the root if requested. Default log path is
    # 'check.log' inside the provided root directory. The CLI argument parsing
    # adds the --log-file option (see parser below).
    log_path = getattr(main, "_log_path", None)
    if log_path is None:
        # no logging requested
        pass
    else:
        try:
            lp = (
                (Path(root) / log_path)
                if not Path(log_path).is_absolute()
                else Path(log_path)
            )
            lp.parent.mkdir(parents=True, exist_ok=True)
            lp.write_text("\n".join(out_lines) + "\n", encoding="utf-8")
            print(f"\nWrote log to: {lp}")
        except Exception as e:
            print(f"Failed to write log file '{log_path}': {e}")

    # Return success only when all seed dirs have non-empty eval.csv
    ok = counts["non-empty"] == counts["total"]
    return 0 if ok else 2


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Check exp_*/*/*/seed\d*/test/eval.csv non-emptiness"
    )
    parser.add_argument(
        "root", nargs="?", default=".", help="project root (default '.')"
    )
    parser.add_argument(
        "--log-file",
        "-l",
        default="check.log",
        help="path to write log file (relative to root unless absolute). Use empty string to disable",
    )
    args = parser.parse_args()
    # if the user provided an empty string for --log-file, treat as disabled
    log_file_arg = args.log_file if args.log_file != "" else None
    # attach the log path to main so it can be picked up (avoids global vars)
    setattr(main, "_log_path", log_file_arg)
    sys.exit(main(args.root))
