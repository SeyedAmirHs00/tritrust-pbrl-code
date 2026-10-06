"""Plot Overlap counterfactual from overlap_shared.csv (DS-Sym with q=1, s=1, theta=0)."""

from __future__ import annotations

import argparse
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from plot_utils import savefig_png_pdf


def plot_overlap_figure(table: pd.DataFrame, out_dir: str) -> None:
    colors = {"ttp": "#4c72b0", "no_alpha": "#dd8452", "ds_sym": "#55a868"}
    labels = {
        "ttp": "TTP",
        "no_alpha": r"No-$\alpha$",
        "ds_sym": r"DS-Sym ($q_0=1, \theta_0=0$)",
    }
    methods = ("ttp", "no_alpha", "ds_sym")

    q_col = "overlap_ratio_q" if "overlap_ratio_q" in table.columns else "q"
    signed_med_col = "signed_corr_median" if "signed_corr_median" in table.columns else "signed_med"
    signed_q25_col = "signed_corr_q25" if "signed_corr_q25" in table.columns else "signed_q25"
    signed_q75_col = "signed_corr_q75" if "signed_corr_q75" in table.columns else "signed_q75"

    setting_str = table["setting"].iloc[0] if "setting" in table.columns else "2R2N1A"
    qs = sorted(table[q_col].unique())
    fig, ax = plt.subplots(figsize=(6.0, 4.2))

    if len(qs) == 1:
        x = np.arange(len(methods))
        for mi, method in enumerate(methods):
            sub = table[table.method == method].iloc[0]
            val = sub[signed_med_col]
            q25 = sub[signed_q25_col]
            q75 = sub[signed_q75_col]
            ax.bar(
                mi,
                val,
                color=colors[method],
                yerr=[[val - q25], [q75 - val]],
                capsize=4,
                label=labels[method],
            )
        ax.set_xticks(x)
        ax.set_xticklabels([labels[m] for m in methods])
        ax.set_xlabel(rf"method ($q={qs[0]:g}$, {setting_str})")
    else:
        for method in methods:
            if method not in table.method.values:
                continue
            sub = table[table.method == method].sort_values(q_col)
            q_vals = sub[q_col].to_numpy(dtype=float)
            med_vals = sub[signed_med_col].to_numpy(dtype=float)
            q25_vals = sub[signed_q25_col].to_numpy(dtype=float)
            q75_vals = sub[signed_q75_col].to_numpy(dtype=float)
            ax.fill_between(
                q_vals, q25_vals, q75_vals, color=colors[method], alpha=0.15
            )
            ax.plot(q_vals, med_vals, "o-", color=colors[method], label=labels[method])
        ax.set_xlabel("shared-pair fraction $q$")
        ax.legend(fontsize=8, loc="lower right")

    ax.axhline(0.0, color="gray", ls=":", lw=0.9)
    ax.set_ylabel(r"median signed $\mathrm{corr}(\hat R,R^*)$")
    ax.set_ylim(-1.05, 1.05)
    ax.grid(True, ls=":", alpha=0.4)
    ax.set_title(f"Overlap Sweep ({setting_str})", fontsize=10)

    fig.tight_layout()
    savefig_png_pdf(
        fig,
        os.path.join(out_dir, f"{setting_str}_overlap_global_vs_local.png"),
        dpi=200,
        bbox_inches="tight",
    )
    savefig_png_pdf(
        fig,
        os.path.join(out_dir, "overlap_ds_init_global_vs_local.png"),
        dpi=200,
        bbox_inches="tight",
    )
    plt.close(fig)


def main() -> None:
    p = argparse.ArgumentParser(description="Plot overlap sweep figure (DS-Sym init q=1, s=1, theta=0)")
    p.add_argument("--out_dir", default="results/synthetic_overlap_ds_init_sweep")
    args = p.parse_args()
    csv_path = os.path.join(args.out_dir, "overlap_shared.csv")
    if not os.path.exists(csv_path):
        alt_path = os.path.join(args.out_dir, "overlap_ds_init_shared.csv")
        if os.path.exists(alt_path):
            csv_path = alt_path
    table = pd.read_csv(csv_path)
    plot_overlap_figure(table, args.out_dir)
    print(f"OUT overlap plot: {args.out_dir}")


if __name__ == "__main__":
    main()
