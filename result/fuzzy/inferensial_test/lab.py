#!/usr/bin/env python3

from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
DATA_DIR = PROJECT_ROOT / "data"
TABLES_DIR = SCRIPT_DIR / "tables"

ALPHA = 0.05
BOOTSTRAP_ITER = 5000
RNG_SEED = 42

METRIC_LABELS = {
    "response_time": "Response Time (ms)",
    "replica": "Replica Count",
    "cpu": "CPU Usage",
    "memory": "Memory Usage",
}

METRIC_ORDER = ["response_time", "replica", "cpu", "memory"]


def load_value_series(path: Path) -> pd.Series:
    df = pd.read_csv(path)
    if "_value" not in df.columns and "value" not in df.columns:
        df = pd.read_csv(path, skiprows=3, comment="#")
    df = df.loc[:, ~df.columns.str.contains("^Unnamed")]
    value_col = "_value" if "_value" in df.columns else "value"
    values = pd.to_numeric(df[value_col], errors="coerce").dropna()
    return values


def load_run_means(test_type: str, pod: str, metric: str) -> pd.DataFrame:
    metric_dir = DATA_DIR / test_type / pod / metric
    rows = []
    for csv_file in sorted(metric_dir.glob("*.csv"), key=lambda p: int(p.stem)):
        series = load_value_series(csv_file)
        if not series.empty:
            rows.append({"run": int(csv_file.stem), "mean": float(series.mean())})
    return pd.DataFrame(rows)


def discover_scenarios():
    rl_root = DATA_DIR / "rl"
    hpa_root = DATA_DIR / "hpa"

    rl_pods = {p.name for p in rl_root.glob("*") if p.is_dir()}
    hpa_pods = {p.name for p in hpa_root.glob("*") if p.is_dir()}

    for pod in sorted(rl_pods & hpa_pods, key=lambda p: int(p.split("_")[-1])):
        rl_metrics = {p.name for p in (rl_root / pod).glob("*") if p.is_dir()}
        hpa_metrics = {p.name for p in (hpa_root / pod).glob("*") if p.is_dir()}

        metric_candidates = list(rl_metrics & hpa_metrics)
        metric_candidates.sort(key=lambda m: METRIC_ORDER.index(m) if m in METRIC_ORDER else 999)

        for metric in metric_candidates:
            yield pod, metric


def shapiro_pvalue(values: np.ndarray) -> float:
    # SciPy Shapiro is valid for n >= 3
    if len(values) < 3:
        return np.nan
    return float(stats.shapiro(values).pvalue)


def hedges_g(x: np.ndarray, y: np.ndarray) -> float:
    # Effect size for independent samples (x=RL, y=HPA)
    n1, n2 = len(x), len(y)
    if n1 < 2 or n2 < 2:
        return np.nan

    var1 = np.var(x, ddof=1)
    var2 = np.var(y, ddof=1)
    pooled_var = ((n1 - 1) * var1 + (n2 - 1) * var2) / (n1 + n2 - 2)
    if pooled_var <= 0:
        return np.nan

    d = (np.mean(x) - np.mean(y)) / np.sqrt(pooled_var)
    correction = 1 - (3 / (4 * (n1 + n2) - 9))
    return float(d * correction)


def rank_biserial_from_u(u_stat: float, n1: int, n2: int) -> float:
    # x=RL group. Positive value means RL tends to be larger than HPA.
    return float((2.0 * u_stat / (n1 * n2)) - 1.0)


def bootstrap_ci_mean_diff(x: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    rng = np.random.default_rng(RNG_SEED)
    n1, n2 = len(x), len(y)
    diffs = np.empty(BOOTSTRAP_ITER, dtype=float)

    for i in range(BOOTSTRAP_ITER):
        xb = rng.choice(x, size=n1, replace=True)
        yb = rng.choice(y, size=n2, replace=True)
        diffs[i] = np.mean(xb) - np.mean(yb)

    lo, hi = np.percentile(diffs, [2.5, 97.5])
    return float(lo), float(hi)


def run_independent_inference(rl_runs: pd.DataFrame, hpa_runs: pd.DataFrame) -> dict:
    x = rl_runs["mean"].to_numpy(dtype=float)
    y = hpa_runs["mean"].to_numpy(dtype=float)

    rl_mean = float(np.mean(x))
    hpa_mean = float(np.mean(y))
    diff = rl_mean - hpa_mean
    diff_pct = np.nan if hpa_mean == 0 else (diff / hpa_mean) * 100.0

    shapiro_rl = shapiro_pvalue(x)
    shapiro_hpa = shapiro_pvalue(y)
    looks_normal = (shapiro_rl > 0.05) and (shapiro_hpa > 0.05)

    if looks_normal:
        test = stats.ttest_ind(x, y, equal_var=False, alternative="two-sided")
        test_name = "Welch t-test (independent)"
        stat = float(test.statistic)
        p_value = float(test.pvalue)
        effect = hedges_g(x, y)
        effect_name = "Hedges g"
    else:
        test = stats.mannwhitneyu(x, y, alternative="two-sided", method="auto")
        test_name = "Mann-Whitney U (independent)"
        stat = float(test.statistic)
        p_value = float(test.pvalue)
        effect = rank_biserial_from_u(stat, len(x), len(y))
        effect_name = "Rank-biserial r"

    ci_low, ci_high = bootstrap_ci_mean_diff(x, y)

    return {
        "n_hpa": len(y),
        "n_rl": len(x),
        "hpa_mean": hpa_mean,
        "rl_mean": rl_mean,
        "diff": diff,
        "diff_pct": diff_pct,
        "ci95_low": ci_low,
        "ci95_high": ci_high,
        "normality_p_hpa": shapiro_hpa,
        "normality_p_rl": shapiro_rl,
        "test": test_name,
        "stat": stat,
        "p_value": p_value,
        "significant": p_value < ALPHA,
        "effect_name": effect_name,
        "effect_value": effect,
    }


def fmt_num(value: float, ndigits: int = 3) -> str:
    if pd.isna(value):
        return "-"
    return f"{float(value):.{ndigits}f}"


def fmt_signed(value: float, ndigits: int = 3) -> str:
    if pd.isna(value):
        return "-"
    v = float(value)
    if v > 0:
        return f"+{v:.{ndigits}f}"
    return f"{v:.{ndigits}f}"


def fmt_pvalue(value: float) -> str:
    if pd.isna(value):
        return "-"
    if value < 0.001:
        return f"{value:.2e}"
    return f"{value:.4f}"


def fmt_diff(value: float, pct: float) -> str:
    if pd.isna(value):
        return "-"
    if pd.isna(pct):
        return f"{fmt_signed(value)} (-)"
    return f"{fmt_signed(value)} ({fmt_signed(pct, ndigits=2)}\\%)"


def latex_escape(text: str) -> str:
    return str(text).replace("_", "\\_")


def write_latex_table(path: Path, caption: str, label: str, columns: list[str], rows: list[list[str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    align = "l" * len(columns)

    lines = [
        "\\begin{table}[ht]",
        "  \\centering",
        f"  \\caption{{{caption}}}\\label{{{label}}}",
        f"  \\begin{{tabular}}{{{align}}}",
        "    \\toprule",
        "    " + " & ".join(columns) + " \\\\",
        "    \\midrule",
    ]
    for row in rows:
        lines.append("    " + " & ".join(row) + " \\\\")
    lines.extend(
        [
            "    \\bottomrule",
            "  \\end{tabular}",
            "\\end{table}",
        ]
    )

    path.write_text("\n".join(lines), encoding="utf-8")


def generate_latex_tables(results_df: pd.DataFrame) -> None:
    TABLES_DIR.mkdir(parents=True, exist_ok=True)

    all_rows = []
    metric_order_lookup = {m: i for i, m in enumerate(METRIC_ORDER)}

    for metric in sorted(results_df["metric"].unique(), key=lambda m: metric_order_lookup.get(m, 999)):
        metric_df = results_df[results_df["metric"] == metric].sort_values("pod")
        rows = []

        for _, row in metric_df.iterrows():
            rows.append(
                [
                    latex_escape(row["pod"]),
                    str(int(row["n_hpa"])),
                    str(int(row["n_rl"])),
                    fmt_num(row["hpa_mean"]),
                    fmt_num(row["rl_mean"]),
                    fmt_diff(row["diff"], row["diff_pct"]),
                    fmt_pvalue(row["p_value"]),
                    "Yes" if bool(row["significant"]) else "No",
                    f"{latex_escape(row['effect_name'])}={fmt_num(row['effect_value'])}",
                    latex_escape(row["test"]),
                ]
            )

            all_rows.append(
                [
                    latex_escape(row["pod"]),
                    latex_escape(METRIC_LABELS.get(row["metric"], row["metric"])),
                    str(int(row["n_hpa"])),
                    str(int(row["n_rl"])),
                    fmt_num(row["hpa_mean"]),
                    fmt_num(row["rl_mean"]),
                    fmt_diff(row["diff"], row["diff_pct"]),
                    fmt_pvalue(row["p_value"]),
                    "Yes" if bool(row["significant"]) else "No",
                    f"{latex_escape(row['effect_name'])}={fmt_num(row['effect_value'])}",
                    latex_escape(row["test"]),
                ]
            )

        metric_safe = metric.replace("/", "_").replace(" ", "_")
        write_latex_table(
            path=TABLES_DIR / f"inferensial_{metric_safe}.tex",
            caption=f"Inferential test (independent samples) for {METRIC_LABELS.get(metric, metric)}",
            label=f"tab:inferensial-{metric_safe}",
            columns=[
                "Pod",
                "$N_{HPA}$",
                "$N_{RL}$",
                "HPA Mean",
                "RL Mean",
                "Difference (RL-HPA)",
                "$p$-value",
                "Sig.",
                "Effect",
                "Test",
            ],
            rows=rows,
        )

    write_latex_table(
        path=TABLES_DIR / "inferensial_all_metrics.tex",
        caption="Inferential test (independent samples) for all metrics",
        label="tab:inferensial-all-metrics",
        columns=[
            "Pod",
            "Metric",
            "$N_{HPA}$",
            "$N_{RL}$",
            "HPA Mean",
            "RL Mean",
            "Difference (RL-HPA)",
            "$p$-value",
            "Sig.",
            "Effect",
            "Test",
        ],
        rows=all_rows,
    )


def main() -> None:
    rows = []
    for pod, metric in discover_scenarios():
        rl_runs = load_run_means("rl", pod, metric)
        hpa_runs = load_run_means("hpa", pod, metric)

        if rl_runs.empty or hpa_runs.empty:
            continue

        res = run_independent_inference(rl_runs, hpa_runs)
        rows.append({"pod": pod, "metric": metric, **res})

    results_df = pd.DataFrame(rows)
    if results_df.empty:
        return

    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    results_df.to_csv(TABLES_DIR / "inferensial_results.csv", index=False)
    generate_latex_tables(results_df)


if __name__ == "__main__":
    main()
