"""TTP mixture reward model with an explicit buffer-diagnostics hook.

Subclasses the production ``mixture_reward_model_wk_sgd`` model
without modifying it. Call ``log_buffer_diagnostics(step)`` after each
preference update during training. Training itself does not auto-log diagnostics.

Writes ``buffer_diagnostics.csv`` in the run directory (not ``reward/reward.csv``)
so new columns do not clash with the reward logger's fixed CSV fieldnames.
"""

from __future__ import annotations

import os
from typing import Optional

from .diagnostics import (
    compute_reward_buffer_diagnostics,
    write_reward_buffer_diagnostics_csv,
)
from .mixture_reward_model_wk_sgd import MixtureRewardModel as _BaseMixtureRewardModel

device = "cuda"


class MixtureRewardModel(_BaseMixtureRewardModel):
    def log_buffer_diagnostics(
        self, step: int, phase: str = "post_train", out_dir: Optional[str] = None
    ):
        """Snapshot buffer diagnostics to ``buffer_diagnostics.csv``."""
        stats = compute_reward_buffer_diagnostics(
            ensemble=self.ensemble,
            reward_models=self.reward_models,
            ds=self.ds,
            da=self.da,
            device=device,
        )
        if out_dir is None:
            if self.logger is not None and hasattr(self.logger, "_log_dir"):
                out_dir = self.logger._log_dir
            else:
                out_dir = os.getcwd()
        path = write_reward_buffer_diagnostics_csv(out_dir, stats, step, phase=phase)
        print(
            f"[diagnostics @ step={step}, phase={phase}] wrote {path}\n"
            f"  rms_delta_r={stats['rms_delta_r']:.6f} "
            f"median_sq_delta_r={stats['median_sq_delta_r']:.6f} "
            f"corr_r_rstar={stats['corr_r_rstar']:.4f} "
            f"corr_segment_r_rstar={stats['corr_segment_r_rstar']:.4f} "
            f"mean_sa_var={stats['mean_sa_var']:.6f} "
            f"n_pairs={int(stats['n_pairs'])} "
            f"n_transitions={int(stats['n_transitions'])}"
        )
        return stats
