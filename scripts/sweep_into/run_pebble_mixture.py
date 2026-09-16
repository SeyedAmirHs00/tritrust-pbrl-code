#!/usr/bin/env python3
"""Runner for Meta-World Sweep-Into PEBBLE mixture experiments.

Convenience wrapper around ``scripts/sweep_into/run_experiments.py --method pebble_mixture``.
"""

from __future__ import annotations

import os
import sys

# Ensure runner can find run_experiments in same directory
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from run_experiments import main  # noqa: E402


if __name__ == "__main__":
    if "--method" not in sys.argv and "--methods" not in sys.argv:
        sys.argv.insert(1, "pebble_mixture")
        sys.argv.insert(1, "--method")
    sys.exit(main())
