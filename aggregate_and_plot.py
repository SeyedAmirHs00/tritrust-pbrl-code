import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import glob
import os
import re

file_dir = os.path.dirname(__file__)
log_folder = "exp_pebble_mixture_raw"
env_name = "metaworld_door-open-v2"
test = None
results_dir = "results"
check_all_tests = True


def check_dir(
    log_folder=log_folder, env_name=env_name, test=test, results_dir=results_dir
):
    global base_dir, output_dir
    target_test = os.path.join(log_folder, env_name, test)
    base_dir = os.path.join(file_dir, target_test)
    output_dir = os.path.join(file_dir, results_dir, target_test)
    os.makedirs(output_dir, exist_ok=True)


def get_csv_files(base_dir, csv_type):
    base_dir_escaped = glob.escape(base_dir)
    csv_files_path = os.path.join(base_dir_escaped, "**", f"{csv_type}.csv")
    csv_files = sorted(glob.glob(csv_files_path, recursive=True))
    return csv_files


# Define a function to process each CSV type
def process_csv_files(csv_type):
    # Find all relevant CSV files (train, eval, reward)
    csv_files = get_csv_files(base_dir, csv_type)
    dfs = [pd.read_csv(f) for f in csv_files if os.path.getsize(f) > 0]

    if not dfs:
        print(f"No {csv_type}.csv files found or all are empty.")
        return None

    # Align by step (or episode)
    key = "step" if "step" in dfs[0].columns else "episode"
    metrics = [
        col
        for col in dfs[0].columns
        if col not in [key, "episode", "labeled_feedback", "total_feedback"]
    ]

    # Concatenate all DataFrames
    all_data = pd.concat(dfs, ignore_index=True)

    # Group by step or episode and calculate mean and std
    if "step" in all_data.columns:
        grouped = all_data.groupby("step")[metrics].agg(["mean", "std"]).reset_index()
        x_values = "step"
    elif "episode" in all_data.columns:
        grouped = (
            all_data.groupby("episode")[metrics].agg(["mean", "std"]).reset_index()
        )
        x_values = "episode"
    else:
        print(f"No step or episode column found in {csv_type}.csv files.")
        return None

    return grouped, metrics, x_values


def plot_metrics(
    grouped, metrics, x_values, csv_type, dfs=None, csv_files=None, title=test
):
    for metric in metrics:
        plt.figure(figsize=(10, 6))

        # --- Plot individual runs if provided ---
        if dfs is not None and csv_files is not None:
            for df, fname in zip(dfs, csv_files):
                if x_values in df.columns and metric in df.columns:
                    match = re.search(r"seed\d+", fname)
                    label = (
                        match.group(0)
                        if match
                        else os.path.basename(os.path.dirname(fname))
                    )
                    plt.plot(df[x_values], df[metric], alpha=0.3, label=label)

        # --- Plot aggregated mean/std ---
        metric_mean = grouped[(metric, "mean")]
        metric_std = grouped[(metric, "std")]
        plt.plot(
            grouped[x_values],
            metric_mean,
            color="black",
            linewidth=2,
            label=f"Mean {metric}",
        )
        plt.fill_between(
            grouped[x_values],
            metric_mean - metric_std,
            metric_mean + metric_std,
            alpha=0.2,
            color="gray",
            label="Std",
        )

        plt.xlabel(x_values)
        plt.ylabel(metric)
        plt.title(f"{title} - {metric} over {x_values} ({csv_type})")
        plt.legend()
        plt.tight_layout()

        # Save
        plt.savefig(os.path.join(output_dir, f"{csv_type}_{metric}_with_runs.png"))
        plt.close()
        print(f"Plot saved: {csv_type}_{metric}_with_runs.png")


# Process and plot for train.csv, reward.csv, and eval.csv
# If log_folder is None, process all exp* folders
# Build lists for log_folder / env_name / test to reduce nested ifs and duplication
log_folders = (
    sorted([os.path.basename(d) for d in glob.glob(os.path.join(file_dir, "exp*")) if os.path.isdir(d)])
    if log_folder is None
    else [log_folder]
)

if not log_folders:
    print("No log folders to process.")
else:
    for lf in log_folders:
        exp_folder_path = os.path.join(file_dir, lf)
        if not os.path.exists(exp_folder_path):
            print(f"Folder not found: {exp_folder_path}. Skipping.")
            continue

        envs = (
            sorted([os.path.basename(d) for d in glob.glob(os.path.join(exp_folder_path, "*")) if os.path.isdir(d)])
            if env_name is None
            else [env_name]
        )

        if not envs:
            print(f"No environment folders in {exp_folder_path}. Skipping.")
            continue

        for env in envs:
            env_path = os.path.join(file_dir, lf, env)
            if not os.path.exists(env_path):
                print(f"Path not found: {env_path}. Skipping.")
                continue

            tests = (
                sorted([os.path.basename(d) for d in glob.glob(os.path.join(env_path, "*")) if os.path.isdir(d)])
                if test is None
                else [test]
            )

            if not tests:
                print(f"No test folders in {env_path}. Skipping.")
                continue

            for t in tests:
                print(f"\n=== Processing: log_folder={lf}, env={env}, test={t} ===")
                check_dir(log_folder=lf, env_name=env, test=t, results_dir=results_dir)

                for csv_type in ["train", "reward", "eval"]:
                    print(f"Processing {csv_type}.csv...")
                    csv_files = get_csv_files(base_dir, csv_type)
                    print(csv_files)

                    dfs = [pd.read_csv(f) for f in csv_files if os.path.getsize(f) > 0]
                    if not dfs:
                        print(f"No non-empty {csv_type}.csv files found for {lf}/{env}/{t}.")
                        continue

                    grouped_results = process_csv_files(csv_type)
                    if grouped_results is None:
                        continue

                    grouped, metrics, x_values = grouped_results
                    plot_metrics(
                        grouped,
                        metrics,
                        x_values,
                        csv_type,
                        dfs=dfs,
                        csv_files=csv_files,
                        title=t,
                    )

    print("All specified folders processed.")
