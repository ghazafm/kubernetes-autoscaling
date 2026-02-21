#!/usr/bin/env python3

import os

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

matplotlib.use("Agg")

SCRIPT_DIR = (
    os.path.dirname(os.path.abspath(__file__)) if "__file__" in globals() else os.getcwd()
)
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
CHARTS_DIR = os.path.join(SCRIPT_DIR, "charts")
TABLES_DIR = os.path.join(SCRIPT_DIR, "tables")
MAX_RUNS = 20

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

DEPLOYMENTS = {"HPA": "hpa-flask-app", "RL": "test-flask-app"}


def get_run_means(base_path: str) -> dict[int, dict[str, float]]:
    runs: dict[int, dict[str, float]] = {}
    for run_id in range(1, MAX_RUNS + 1):
        path = os.path.join(base_path, f"{run_id}.csv")
        if not os.path.exists(path):
            continue

        df = pd.read_csv(path, comment="#")
        df = df.loc[:, ~df.columns.str.contains("^Unnamed")]

        if "_value" in df.columns:
            value_col = "_value"
        elif "value" in df.columns:
            value_col = "value"
        else:
            print(f"No value column in {path}, skipping")
            continue

        if "deployment" not in df.columns:
            print(f"No deployment column in {path}, skipping")
            continue

        means = df.groupby("deployment")[value_col].mean()
        runs[run_id] = {
            "hpa": means.get(DEPLOYMENTS["HPA"], np.nan),
            "rl": means.get(DEPLOYMENTS["RL"], np.nan),
        }
        print(f"Loaded {path}: {len(df)} rows (hpa={runs[run_id]['hpa']}, rl={runs[run_id]['rl']})")

    return runs


def build_paired_from_runs(runs_dict: dict[int, dict[str, float]]) -> tuple[list[float], list[float]]:
    hpa_vals: list[float] = []
    rl_vals: list[float] = []
    for run_id in sorted(runs_dict.keys()):
        values = runs_dict[run_id]
        h = values.get("hpa", np.nan)
        r = values.get("rl", np.nan)
        if not (pd.isna(h) or pd.isna(r)):
            hpa_vals.append(float(h))
            rl_vals.append(float(r))
    return hpa_vals, rl_vals


def analyze_stats(hpa_vals: list[float], rl_vals: list[float]) -> dict | None:
    n = min(len(hpa_vals), len(rl_vals))
    if n < 3:
        return None

    x = hpa_vals[:n]
    y = rl_vals[:n]

    shapiro_hpa = stats.shapiro(x)
    shapiro_rl = stats.shapiro(y)
    is_normal = (shapiro_hpa.pvalue > 0.05) and (shapiro_rl.pvalue > 0.05)

    diff = np.array(x) - np.array(y)
    if np.allclose(diff, 0):
        return {
            "N": n,
            "HPA Mean": np.mean(x),
            "RL Mean": np.mean(y),
            "Diff (%)": 0.0,
            "Normality P(HPA)": shapiro_hpa.pvalue,
            "Normality P(RL)": shapiro_rl.pvalue,
            "P-Value": 1.0,
            "Significant": False,
            "Test": "N/A (identical values)",
            "Effect Size": 0.0,
        }

    if is_normal:
        _, p_val = stats.ttest_rel(x, y)
        test_name = "Paired t-test"
    else:
        try:
            _, p_val = stats.wilcoxon(x, y)
            test_name = "Wilcoxon"
        except ValueError as err:
            print(f"  Wilcoxon failed ({err}), fallback to Paired t-test")
            _, p_val = stats.ttest_rel(x, y)
            test_name = "Paired t-test (fallback)"

    std_diff = np.std(diff, ddof=1)
    cohens_d = np.mean(diff) / std_diff if std_diff != 0 else 0

    return {
        "N": n,
        "HPA Mean": np.mean(x),
        "RL Mean": np.mean(y),
        "Diff (%)": (np.mean(y) - np.mean(x)) / np.mean(x) * 100 if np.mean(x) != 0 else np.nan,
        "Normality P(HPA)": shapiro_hpa.pvalue,
        "Normality P(RL)": shapiro_rl.pvalue,
        "P-Value": p_val,
        "Significant": p_val < 0.05,
        "Test": test_name,
        "Effect Size": cohens_d,
    }


def save_normality_plots(hpa_vals: list[float], rl_vals: list[float], pod: int, metric_dir: str, label: str) -> None:
    output_dir = os.path.join(CHARTS_DIR, f"pod_{pod}", metric_dir)
    os.makedirs(output_dir, exist_ok=True)
    print(f"Saving plots to: {output_dir}")

    plt.figure(figsize=(6, 4))
    plt.hist(hpa_vals, bins="auto", alpha=0.7, color="blue", edgecolor="black", rwidth=0.85)
    plt.title(f"HPA Histogram - {label}")
    plt.xlabel("Value")
    plt.ylabel("Frequency")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"inferensial_histogram_{metric_dir}_hpa_{pod}_pod.png"))
    plt.close()

    plt.figure(figsize=(6, 4))
    stats.probplot(hpa_vals, dist="norm", plot=plt)
    plt.title(f"HPA Q-Q Plot - {label}")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"inferensial_qq_plot_{metric_dir}_hpa_{pod}_pod.png"))
    plt.close()

    plt.figure(figsize=(6, 4))
    plt.hist(rl_vals, bins="auto", alpha=0.7, color="green", edgecolor="black", rwidth=0.85)
    plt.title(f"RL Histogram - {label}")
    plt.xlabel("Value")
    plt.ylabel("Frequency")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"inferensial_histogram_{metric_dir}_rl_{pod}_pod.png"))
    plt.close()

    plt.figure(figsize=(6, 4))
    stats.probplot(rl_vals, dist="norm", plot=plt)
    plt.title(f"RL Q-Q Plot - {label}")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"inferensial_qq_plot_{metric_dir}_rl_{pod}_pod.png"))
    plt.close()


def format_scientific(val) -> str:
    if val is None or pd.isna(val):
        return "-"
    try:
        valf = float(val)
    except Exception:
        return "-"

    if abs(valf) < 0.001 and valf != 0:
        mantissa, exponent = f"{valf:.3e}".split("e")
        return f"${mantissa} \\times 10^{{{int(exponent)}}}$"
    return f"{valf:.3f}"


def format_effect_size(val: float, test_name: str) -> str:
    if "Wilcoxon" in test_name:
        return f"{abs(val):.3f}"
    return f"{val:.3f}"


def generate_latex_table(results_df: pd.DataFrame, metric_name: str, table_label: str, caption: str) -> str:
    metric_data = results_df[results_df["Metric"] == metric_name]
    pod_10 = (
        metric_data[metric_data["Scenario"] == "pod_10"].iloc[0]
        if len(metric_data[metric_data["Scenario"] == "pod_10"]) > 0
        else None
    )
    pod_20 = (
        metric_data[metric_data["Scenario"] == "pod_20"].iloc[0]
        if len(metric_data[metric_data["Scenario"] == "pod_20"]) > 0
        else None
    )

    lines = []
    lines.append("\\begin{table}[H]")
    lines.append("  \\centering")
    lines.append(f"  \\caption{{{caption}}}\\label{{{table_label}}}")
    lines.append("  \\begin{tabular}{lcc}")
    lines.append("    \\toprule")
    lines.append("    \\textbf{Statistik} & \\textbf{10 Pod} & \\textbf{20 Pod} \\\\")
    lines.append("    \\midrule")

    n_10 = int(pod_10["N"]) if pod_10 is not None else "-"
    n_20 = int(pod_20["N"]) if pod_20 is not None else "-"
    lines.append(f"    Jumlah sampel ($N$) & {n_10} & {n_20} \\\\")

    if "Response Time" in metric_name:
        unit = " (ms)"
    elif "Replica" in metric_name:
        unit = ""
    else:
        unit = " (\\%)"

    hpa_10 = f"{pod_10['HPA Mean']:.3f}" if pod_10 is not None else "-"
    hpa_20 = f"{pod_20['HPA Mean']:.3f}" if pod_20 is not None else "-"
    lines.append(f"    Rata-rata HPA{unit} & {hpa_10} & {hpa_20} \\\\")

    rl_10 = f"{pod_10['RL Mean']:.3f}" if pod_10 is not None else "-"
    rl_20 = f"{pod_20['RL Mean']:.3f}" if pod_20 is not None else "-"
    lines.append(f"    Rata-rata RL{unit} & {rl_10} & {rl_20} \\\\")

    diff_10 = f"${pod_10['Diff (%)']:.3f}$" if pod_10 is not None else "-"
    diff_20 = f"${pod_20['Diff (%)']:.3f}$" if pod_20 is not None else "-"
    lines.append(f"    Selisih (\\%) & {diff_10} & {diff_20} \\\\")
    lines.append("    \\midrule")

    norm_hpa_10 = f"{pod_10['Normality P(HPA)']:.3f}" if pod_10 is not None else "-"
    norm_hpa_20 = f"{pod_20['Normality P(HPA)']:.3f}" if pod_20 is not None else "-"
    lines.append(f"    \\emph{{P}}-value normalitas (HPA) & {norm_hpa_10} & {norm_hpa_20} \\\\")

    norm_rl_10 = f"{pod_10['Normality P(RL)']:.3f}" if pod_10 is not None else "-"
    norm_rl_20 = f"{pod_20['Normality P(RL)']:.3f}" if pod_20 is not None else "-"
    lines.append(f"    \\emph{{P}}-value normalitas (RL) & {norm_rl_10} & {norm_rl_20} \\\\")

    test_10 = f"\\emph{{{pod_10['Test']}}}" if pod_10 is not None else "-"
    test_20 = f"\\emph{{{pod_20['Test']}}}" if pod_20 is not None else "-"
    lines.append(f"    Uji yang digunakan & {test_10} & {test_20} \\\\")

    pval_10 = format_scientific(pod_10["P-Value"]) if pod_10 is not None else "-"
    pval_20 = format_scientific(pod_20["P-Value"]) if pod_20 is not None else "-"
    lines.append(f"    \\emph{{P}}-value & {pval_10} & {pval_20} \\\\")

    sig_10 = "Ya" if pod_10 is not None and pod_10["Significant"] else "Tidak"
    sig_20 = "Ya" if pod_20 is not None and pod_20["Significant"] else "Tidak"
    lines.append(f"    Signifikan ($\\alpha = 0{{,}}05$) & {sig_10} & {sig_20} \\\\")

    effect_label = "$r$" if pod_10 is not None and "Wilcoxon" in pod_10["Test"] else "Cohen's $d$"
    effect_10 = format_effect_size(pod_10["Effect Size"], pod_10["Test"]) if pod_10 is not None else "-"
    effect_20 = format_effect_size(pod_20["Effect Size"], pod_20["Test"]) if pod_20 is not None else "-"
    lines.append(f"    \\emph{{Effect size}} ({effect_label}) & {effect_10} & {effect_20} \\\\")

    lines.append("    \\bottomrule")
    lines.append("  \\end{tabular}")
    lines.append("\\end{table}")

    return "\n".join(lines)


def main() -> None:
    results = []

    for folder, metric_dir, label, pod in METRICS:
        base_path = os.path.join(DATA_DIR, folder, metric_dir)
        if not os.path.exists(base_path):
            print(f"Warning: data path not found, skipping: {base_path}")
            continue

        runs = get_run_means(base_path)
        hpa_vals, rl_vals = build_paired_from_runs(runs)

        if len(hpa_vals) >= 3 and len(rl_vals) >= 3:
            save_normality_plots(hpa_vals, rl_vals, pod, metric_dir, label)

        stats_res = analyze_stats(hpa_vals, rl_vals)
        if stats_res:
            stats_res.update({"Scenario": f"pod_{pod}", "Metric": label})
            results.append(stats_res)

        print("")

    if results:
        columns = [
            "Scenario",
            "Metric",
            "N",
            "HPA Mean",
            "RL Mean",
            "Diff (%)",
            "Normality P(HPA)",
            "Normality P(RL)",
            "P-Value",
            "Significant",
            "Test",
            "Effect Size",
        ]
        print(pd.DataFrame(results)[columns].to_string())
    else:
        print("No valid data found for analysis.")

    os.makedirs(TABLES_DIR, exist_ok=True)
    results_df = pd.DataFrame(results)
    if results_df.empty:
        print("No results available, skipping LaTeX table generation.")
        return

    tables = [
        (
            "Response Time (ms)",
            "tab:inferensial-waktu-respon",
            "Hasil Uji Inferensial Waktu Respons",
            "inferensial_waktu_respon.tex",
        ),
        (
            "Replica Count",
            "tab:inferensial-jumlah-replika",
            "Hasil Uji Inferensial Jumlah Replika",
            "inferensial_jumlah_replika.tex",
        ),
        (
            "CPU Usage",
            "tab:inferensial-penggunaan-cpu",
            "Hasil Uji Inferensial Penggunaan CPU",
            "inferensial_penggunaan_cpu.tex",
        ),
        (
            "Memory Usage",
            "tab:inferensial-penggunaan-memori",
            "Hasil Uji Inferensial Penggunaan Memori",
            "inferensial_penggunaan_memori.tex",
        ),
    ]

    for metric_name, table_label, caption, filename in tables:
        if "Metric" not in results_df.columns:
            print(f"Skipping table {filename}: Metric column missing")
            continue

        latex_table = generate_latex_table(results_df, metric_name, table_label, caption)
        out_path = os.path.join(TABLES_DIR, filename)
        with open(out_path, "w", encoding="utf-8") as handle:
            handle.write(latex_table)

        print(f"Saved: {out_path}")
        print(latex_table)
        print("\n" + "=" * 80 + "\n")


if __name__ == "__main__":
    main()
