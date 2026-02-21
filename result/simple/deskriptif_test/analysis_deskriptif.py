#!/usr/bin/env python3

import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

SCRIPT_DIR = (
    os.path.dirname(os.path.abspath(__file__)) if "__file__" in globals() else os.getcwd()
)
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
CHARTS_DIR = os.path.join(SCRIPT_DIR, "charts")
TABLES_DIR = os.path.join(SCRIPT_DIR, "tables")
NUM_RUNS = 10

METRICS = [
    ("pod_10", "response_time", "Response Time (ms)", 10),
    ("pod_10", "replica", "Replica Count", 10),
    ("pod_10", "cpu", "CPU Usage", 10),
    ("pod_10", "memory", "Memory Usage", 10),
    ("pod_20", "response_time", "Response Time (ms)", 20),
    ("pod_20", "replica", "Replica Count", 20),
    ("pod_20", "cpu", "CPU Usage", 20),
    ("pod_20", "memory", "Memory Usage", 20),
]

PLOT_CONFIG = {
    "cpu": {"metric_name": "CPU Usage", "ylabel": "CPU (%)", "ylim": 100, "threshold": None},
    "memory": {
        "metric_name": "Memory Usage",
        "ylabel": "Memory (%)",
        "ylim": 100,
        "threshold": None,
    },
    "response_time": {
        "metric_name": "Response Time",
        "ylabel": "Response Time (ms)",
        "ylim": 1300,
        "threshold": 1000,
    },
    "replica": {
        "metric_name": "Desired Replicas",
        "ylabel": "Replicas",
        "ylim": 20,
        "threshold": None,
    },
}


def load_influx_csv(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, comment="#")
    df = df.drop(columns=[c for c in df.columns if "Unnamed" in str(c)], errors="ignore")
    df["_time"] = pd.to_datetime(df["_time"], errors="coerce")
    df["_value"] = pd.to_numeric(df["_value"], errors="coerce")
    return df.dropna(subset=["_time", "_value"])


def load_all_runs(dataset: str, metric_folder: str, num_runs: int) -> dict[int, pd.DataFrame]:
    runs = {}
    for run_id in range(1, num_runs + 1):
        path = os.path.join(DATA_DIR, dataset, metric_folder, f"{run_id}.csv")
        if os.path.exists(path):
            runs[run_id] = load_influx_csv(path)
    return runs


def calculate_run_statistics(runs: dict[int, pd.DataFrame], metric_name: str) -> pd.DataFrame:
    stats = []
    for run_id, df in runs.items():
        if "deployment" not in df.columns or "_value" not in df.columns:
            continue

        hpa_data = df[df["deployment"] == "hpa-flask-app"]["_value"]
        rl_data = df[df["deployment"] == "test-flask-app"]["_value"]
        if len(hpa_data) == 0 or len(rl_data) == 0:
            continue

        hpa_mean = hpa_data.mean()
        rl_mean = rl_data.mean()
        hpa_std = hpa_data.std()
        rl_std = rl_data.std()
        hpa_max = hpa_data.max()
        rl_max = rl_data.max()

        improvement = ((hpa_mean - rl_mean) / hpa_mean * 100) if hpa_mean != 0 else 0
        stability_score = 1 / (hpa_std + rl_std + 1)

        stats.append(
            {
                "Run": run_id,
                "Metric": metric_name,
                "HPA Mean": hpa_mean,
                "RL Mean": rl_mean,
                "HPA Std": hpa_std,
                "RL Std": rl_std,
                "HPA Max": hpa_max,
                "RL Max": rl_max,
                "Improvement (%)": improvement,
                "Data Points": len(df),
                "Stability": stability_score,
            }
        )

    return pd.DataFrame(stats)


def plot_single_run_high_quality(
    df: pd.DataFrame,
    run_id: int,
    metric_name: str,
    ylabel: str,
    ylim: int,
    pod_label: str,
    threshold: int | None = None,
    save_path: str | None = None,
) -> None:
    deploy_a = "hpa-flask-app"
    deploy_b = "test-flask-app"
    colors = {deploy_a: "#0072B2", deploy_b: "#D55E00"}

    plot_df = df.copy()
    min_time = plot_df["_time"].min()
    plot_df["_elapsed_min"] = (plot_df["_time"] - min_time).dt.total_seconds() / 60

    fig, ax = plt.subplots(1, 1, figsize=(12, 6))

    for deploy in [deploy_a, deploy_b]:
        subset = plot_df[plot_df["deployment"] == deploy].sort_values("_elapsed_min")
        label = "HPA" if deploy == deploy_a else "RL Agent"
        ax.plot(
            subset["_elapsed_min"],
            subset["_value"],
            label=label,
            color=colors[deploy],
            linewidth=2,
            alpha=0.9,
        )

    ax.set_ylabel(ylabel, fontsize=13)
    ax.set_xlabel("Elapsed Time (minutes)", fontsize=13)
    ax.set_title(f"{metric_name} - Run #{run_id} ({pod_label})", fontsize=15, fontweight="bold")
    ax.set_ylim(0, ylim)

    if "Replicas" in metric_name:
        ax.yaxis.set_major_locator(plt.MaxNLocator(integer=True))

    if threshold:
        ax.axhline(
            y=threshold,
            color="red",
            linestyle="--",
            linewidth=2,
            alpha=0.7,
            label=f"Threshold ({threshold}ms)",
        )

    ax.legend(loc="upper right", fontsize=11)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"  Saved: {save_path}")

    plt.close()


def generate_deskriptif_summary(stats_by_key: dict, out_dir: str) -> str | None:
    rows = []
    for key, df in stats_by_key.items():
        pod, metric_dir = key
        if df is None or df.empty:
            continue
        for _, row in df.iterrows():
            rows.append(
                {
                    "pod": f"pod_{pod}".replace("_", "\\_"),
                    "metric_dir": metric_dir,
                    "metric_label": row.get("Metric", metric_dir),
                    "hpa_mean": row.get("HPA Mean", np.nan),
                    "rl_mean": row.get("RL Mean", np.nan),
                }
            )

    if not rows:
        print("No descriptive stats available to summarize.")
        return None

    df_all = pd.DataFrame(rows)
    summary = (
        df_all.groupby(["pod", "metric_label"])[["hpa_mean", "rl_mean"]]
        .mean()
        .reset_index()
        .sort_values(["pod", "metric_label"])
    )
    overall = (
        df_all.groupby(["metric_label"])[["hpa_mean", "rl_mean"]]
        .mean()
        .reset_index()
        .sort_values("metric_label")
    )

    os.makedirs(out_dir, exist_ok=True)

    overall_path = os.path.join(out_dir, "analisis_deskriptif_rangkuman.tex")
    lines = []
    lines.append("\\begin{table}[ht]")
    lines.append("  \\centering")
    lines.append(
        "  \\caption{Rangkuman Rata-rata per Metrik (seluruh skenario)}\\label{tab:deskriptif-summary-per-metric}"
    )
    lines.append("  \\begin{tabular}{lcc}")
    lines.append("    \\toprule")
    lines.append(r"    Metrik & Rata-rata HPA & Rata-rata RL \\")
    lines.append("    \\midrule")

    for _, row in overall.iterrows():
        metric = row["metric_label"]
        hpa_str = f"{row['hpa_mean']:.3f}" if not pd.isna(row["hpa_mean"]) else "-"
        rl_str = f"{row['rl_mean']:.3f}" if not pd.isna(row["rl_mean"]) else "-"
        lines.append(f"    {metric} & {hpa_str} & {rl_str} " + r"\\")

    lines.append("    \\bottomrule")
    lines.append("  \\end{tabular}")
    lines.append("\\end{table}")

    with open(overall_path, "w", encoding="utf-8") as handle:
        handle.write("\n".join(lines))
    print(f"Saved overall per-metric LaTeX: {overall_path}")

    for metric_label, group_df in summary.groupby("metric_label"):
        safe = metric_label.replace(" ", "_").replace("(", "").replace(")", "").replace("/", "_")
        path = os.path.join(out_dir, f"analisis_deskriptif_{safe}.tex")
        lines = []
        lines.append("\\begin{table}[ht]")
        lines.append("  \\centering")
        lines.append(
            f"  \\caption{{Rata-rata {metric_label} per Skenario}}\\label{{tab:deskriptif-{safe}}}"
        )
        lines.append("  \\begin{tabular}{lccc}")
        lines.append("    \\toprule")
        lines.append(r"    Pod & Metrik & Rata-rata HPA & Rata-rata RL \\")
        lines.append("    \\midrule")

        prev_pod = None
        for _, row in group_df.iterrows():
            pod = row["pod"]
            metric = row["metric_label"]
            hpa_str = f"{row['hpa_mean']:.3f}" if not pd.isna(row["hpa_mean"]) else "-"
            rl_str = f"{row['rl_mean']:.3f}" if not pd.isna(row["rl_mean"]) else "-"
            if prev_pod is not None and pod != prev_pod:
                lines.append("    \\midrule")
            prev_pod = pod
            lines.append(f"    {pod} & {metric} & {hpa_str} & {rl_str} " + r"\\")

        lines.append("    \\bottomrule")
        lines.append("  \\end{tabular}")
        lines.append("\\end{table}")

        with open(path, "w", encoding="utf-8") as handle:
            handle.write("\n".join(lines))
        print(f"Saved per-metric LaTeX: {path}")

    return overall_path


def main() -> None:
    runs_by_key = {}
    stats_by_key = {}

    print("Loading test runs from all steps...")
    for dataset, metric_dir, metric_label, pod in METRICS:
        key = (pod, metric_dir)
        runs = load_all_runs(dataset, metric_dir, NUM_RUNS)
        stats_df = calculate_run_statistics(runs, metric_label)

        runs_by_key[key] = runs
        stats_by_key[key] = stats_df
        print(f"  {dataset}/{metric_dir}: {len(runs)} runs")

    print("\nCreating charts for all runs...")
    for dataset, metric_dir, _, pod in METRICS:
        key = (pod, metric_dir)
        runs = runs_by_key.get(key, {})

        if not runs:
            continue

        output_dir = os.path.join(CHARTS_DIR, f"pod_{pod}", metric_dir)
        os.makedirs(output_dir, exist_ok=True)
        config = PLOT_CONFIG[metric_dir]

        for run_id in sorted(runs.keys()):
            save_path = os.path.join(output_dir, f"deskriptif_{metric_dir}_run_{run_id}.png")
            plot_single_run_high_quality(
                runs[run_id],
                run_id,
                config["metric_name"],
                config["ylabel"],
                config["ylim"],
                f"Pod {pod}",
                config["threshold"],
                save_path,
            )

    print("\nGenerating deskriptif summary tables...")
    generate_deskriptif_summary(stats_by_key, TABLES_DIR)
    print("\nDeskriptif chart generation completed.")


if __name__ == "__main__":
    main()
