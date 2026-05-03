#!/usr/bin/env python3
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
DATA_DIR = PROJECT_ROOT / "data"
CHARTS_DIR = SCRIPT_DIR / "charts"
TABLES_DIR = SCRIPT_DIR / "tables"

TYPE_LABELS = {"rl": "RL", "hpa": "HPA"}

METRIC_YLABEL = {
    "response_time": "Response Time (ms)",
    "replica": "Replicas",
    "cpu": "CPU (%)",
    "memory": "Memory (%)",
}


# ---------------------------------------------------------------------------
# Plot style configuration
# ---------------------------------------------------------------------------

@dataclass
class TypeStyle:
    color: str
    linestyle: str
    marker: str
    linewidth: float
    markersize: float


@dataclass
class PlotStyle:
    name: str          # "paper" or "presentation"
    dpi: int
    figsize: tuple
    grid_alpha: float
    threshold_color: str
    types: dict        # {"rl": TypeStyle, "hpa": TypeStyle}
    font_size: Optional[int] = None   # None = matplotlib default


PAPER_STYLE = PlotStyle(
    name="paper",
    dpi=300,
    figsize=(12, 6),
    grid_alpha=0.3,
    threshold_color="red",
    types={
        "rl":  TypeStyle(color="black", linestyle="-",  marker="o", linewidth=1.5, markersize=5),
        "hpa": TypeStyle(color="black", linestyle="--", marker="s", linewidth=1.5, markersize=5),
    },
)

PRESENTATION_STYLE = PlotStyle(
    name="presentation",
    dpi=150,
    figsize=(14, 7),
    grid_alpha=0.2,
    threshold_color="#EF4444",   # vivid red
    font_size=14,
    types={
        # Blue for RL, Orange for HPA — colorblind-friendly pair
        "rl":  TypeStyle(color="#2196F3", linestyle="-",  marker="o", linewidth=2.5, markersize=7),
        "hpa": TypeStyle(color="#9E9E9E", linestyle="--", marker="s", linewidth=2.5, markersize=7),
    },
)

ALL_STYLES = [PAPER_STYLE, PRESENTATION_STYLE]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def latex_escape(value: str) -> str:
    return str(value).replace("_", "\\_")


def format_num(value: float) -> str:
    if pd.isna(value):
        return "-"
    return f"{float(value):.1f}"


def format_difference(value: float, pct: float) -> str:
    if pd.isna(value):
        return "-"
    if pd.isna(pct):
        return f"{float(value):.1f} (-)"
    value_num = float(value)
    value_str = f"+{value_num:.1f}" if value_num > 0 else f"{value_num:.1f}"
    pct_value = float(pct)
    pct_str = f"+{pct_value:.1f}" if pct_value > 0 else f"{pct_value:.1f}"
    return f"{value_str} ({pct_str}\\%)"


def write_latex_table(path: Path, caption: str, label: str, columns, rows) -> None:
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


def load_series(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    df = df.drop(columns=[c for c in df.columns if str(c).startswith("Unnamed")], errors="ignore")
    df["_time"] = pd.to_datetime(df["_time"])
    df["_value"] = pd.to_numeric(df["_value"])
    df = df.dropna(subset=["_time", "_value"]).sort_values("_time")
    df["elapsed_min"] = (df["_time"] - df["_time"].min()).dt.total_seconds() / 60.0
    return df


def pod_max_replica(pod: str) -> int:
    return int(pod.split("_")[-1])


def apply_metric_axis(ax, pod: str, metric: str, style: PlotStyle, *dfs: pd.DataFrame) -> None:
    if metric in {"cpu", "memory"}:
        ax.set_ylim(0, 100)
    elif metric == "response_time":
        ax.set_ylim(0, 1500)
        ax.axhline(
            y=1000,
            color=style.threshold_color,
            linestyle="--",
            linewidth=2,
            label="Threshold 1000 ms",
        )
    elif metric == "replica":
        try:
            ax.set_ylim(0, pod_max_replica(pod))
        except Exception:
            max_replica = 1.0
            for df in dfs:
                if not df.empty:
                    max_replica = max(max_replica, float(df["_value"].max()))
            ax.set_ylim(0, int(max_replica) + 1)


def _apply_font(style: PlotStyle) -> None:
    """Set global font size for presentation readability, or reset for paper."""
    if style.font_size:
        plt.rcParams.update({"font.size": style.font_size})
    else:
        plt.rcParams.update({"font.size": plt.rcParamsDefault["font.size"]})


# ---------------------------------------------------------------------------
# Core plot functions — style-aware
# ---------------------------------------------------------------------------

def plot_independent(
    test_type: str, pod: str, metric: str, run: str, csv_path: Path, style: PlotStyle
) -> None:
    df = load_series(csv_path)
    ts = style.types[test_type]
    markevery = max(1, len(df) // 10)

    _apply_font(style)
    fig, ax = plt.subplots(figsize=style.figsize)
    ax.plot(
        df["elapsed_min"],
        df["_value"],
        color=ts.color,
        linestyle=ts.linestyle,
        marker=ts.marker,
        markevery=markevery,
        markersize=ts.markersize,
        linewidth=ts.linewidth,
        label=TYPE_LABELS[test_type],
    )
    ax.set_title(f"{TYPE_LABELS[test_type]} | {pod} | {metric} | run {run}")
    ax.set_xlabel("Elapsed Time (minutes)")
    ax.set_ylabel(METRIC_YLABEL.get(metric, metric))
    apply_metric_axis(ax, pod, metric, style, df)
    ax.grid(alpha=style.grid_alpha)
    ax.legend()
    fig.tight_layout()

    out_path = CHARTS_DIR / style.name / test_type / pod / metric / f"{run}.png"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=style.dpi)
    plt.close(fig)


def plot_compare(
    pod: str, metric: str, run: str, rl_csv: Path, hpa_csv: Path, style: PlotStyle
) -> None:
    rl_df = load_series(rl_csv)
    hpa_df = load_series(hpa_csv)
    rl_ts = style.types["rl"]
    hpa_ts = style.types["hpa"]

    rl_markevery = max(1, len(rl_df) // 10)
    hpa_markevery = max(1, len(hpa_df) // 10)

    _apply_font(style)
    fig, ax = plt.subplots(figsize=style.figsize)
    ax.plot(
        rl_df["elapsed_min"],
        rl_df["_value"],
        color=rl_ts.color,
        linestyle=rl_ts.linestyle,
        marker=rl_ts.marker,
        markevery=rl_markevery,
        markersize=rl_ts.markersize,
        linewidth=rl_ts.linewidth,
        label=TYPE_LABELS["rl"],
    )
    ax.plot(
        hpa_df["elapsed_min"],
        hpa_df["_value"],
        color=hpa_ts.color,
        linestyle=hpa_ts.linestyle,
        marker=hpa_ts.marker,
        markevery=hpa_markevery,
        markersize=hpa_ts.markersize,
        linewidth=hpa_ts.linewidth,
        label=TYPE_LABELS["hpa"],
    )
    ax.set_title(f"Compare RL vs HPA | {pod} | {metric} | run {run}")
    ax.set_xlabel("Elapsed Time (minutes)")
    ax.set_ylabel(METRIC_YLABEL.get(metric, metric))
    apply_metric_axis(ax, pod, metric, style, rl_df, hpa_df)
    ax.grid(alpha=style.grid_alpha)
    ax.legend()
    fig.tight_layout()

    out_path = CHARTS_DIR / style.name / "compare" / pod / metric / f"{run}.png"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=style.dpi)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Generation orchestrators
# ---------------------------------------------------------------------------

def generate_independent_charts() -> None:
    for type_dir in sorted(DATA_DIR.glob("*")):
        if not type_dir.is_dir():
            continue
        test_type = type_dir.name
        if test_type not in TYPE_LABELS:
            continue

        for pod_dir in sorted(type_dir.glob("*")):
            if not pod_dir.is_dir():
                continue
            pod = pod_dir.name

            for metric_dir in sorted(pod_dir.glob("*")):
                if not metric_dir.is_dir():
                    continue
                metric = metric_dir.name

                for csv_file in sorted(metric_dir.glob("*.csv")):
                    run = csv_file.stem
                    for style in ALL_STYLES:
                        plot_independent(test_type, pod, metric, run, csv_file, style)


def generate_compare_charts() -> None:
    rl_root = DATA_DIR / "rl"
    hpa_root = DATA_DIR / "hpa"

    for rl_pod_dir in sorted(rl_root.glob("*")):
        if not rl_pod_dir.is_dir():
            continue

        pod = rl_pod_dir.name
        hpa_pod_dir = hpa_root / pod
        if not hpa_pod_dir.is_dir():
            continue

        rl_metrics = {p.name for p in rl_pod_dir.glob("*") if p.is_dir()}
        hpa_metrics = {p.name for p in hpa_pod_dir.glob("*") if p.is_dir()}

        for metric in sorted(rl_metrics & hpa_metrics):
            rl_metric_dir = rl_pod_dir / metric
            hpa_metric_dir = hpa_pod_dir / metric

            rl_runs = {p.stem for p in rl_metric_dir.glob("*.csv")}
            hpa_runs = {p.stem for p in hpa_metric_dir.glob("*.csv")}

            for run in sorted(rl_runs & hpa_runs, key=lambda x: int(x)):
                for style in ALL_STYLES:
                    plot_compare(
                        pod=pod,
                        metric=metric,
                        run=run,
                        rl_csv=rl_metric_dir / f"{run}.csv",
                        hpa_csv=hpa_metric_dir / f"{run}.csv",
                        style=style,
                    )


def collect_merged_means() -> pd.DataFrame:
    rows = []
    for type_dir in sorted(DATA_DIR.glob("*")):
        if not type_dir.is_dir():
            continue
        test_type = type_dir.name
        if test_type not in TYPE_LABELS:
            continue

        for pod_dir in sorted(type_dir.glob("*")):
            if not pod_dir.is_dir():
                continue
            pod = pod_dir.name

            for metric_dir in sorted(pod_dir.glob("*")):
                if not metric_dir.is_dir():
                    continue
                metric = metric_dir.name

                merged_values = []
                for csv_file in sorted(metric_dir.glob("*.csv"), key=lambda p: int(p.stem)):
                    df = load_series(csv_file)
                    merged_values.append(df["_value"])

                if not merged_values:
                    continue

                merged = pd.concat(merged_values, ignore_index=True)
                rows.append({"type": test_type, "pod": pod, "metric": metric, "mean": merged.mean()})

    return pd.DataFrame(rows)


def generate_latex_tables() -> None:
    merged_df = collect_merged_means()
    pivot = (
        merged_df.pivot_table(index=["pod", "metric"], columns="type", values="mean", aggfunc="mean")
        .reset_index()
        .sort_values(["metric", "pod"])
    )
    pivot["difference"] = pivot["rl"] - pivot["hpa"]
    pivot["difference_pct"] = ((pivot["rl"] - pivot["hpa"]) / pivot["hpa"]) * 100
    pivot.loc[pivot["hpa"] == 0, "difference_pct"] = pd.NA

    for metric in sorted(pivot["metric"].unique()):
        metric_df = pivot[pivot["metric"] == metric].sort_values("pod")
        rows = []
        for _, r in metric_df.iterrows():
            rows.append([
                latex_escape(r["pod"]),
                format_num(r["hpa"]),
                format_num(r["rl"]),
                format_difference(r["difference"], r["difference_pct"]),
            ])

        safe_metric = metric.replace("/", "_").replace(" ", "_")
        label_metric = metric.replace("/", "-").replace(" ", "-").replace("_", "-")
        caption_map = {
            "response_time": "Waktu Respons",
            "replica": "Jumlah Replika",
            "cpu": "Penggunaan CPU",
            "memory": "Penggunaan Memori",
        }
        caption_metric = caption_map.get(metric, latex_escape(metric))

        write_latex_table(
            path=TABLES_DIR / f"deskriptif_{safe_metric}.tex",
            caption=f"Perbandingan rata-rata metrik {caption_metric}",
            label=f"tab:deskriptif-{label_metric}",
            columns=["Pod", "HPA", "RL", "Perbedaan"],
            rows=rows,
        )

    all_rows = []
    for _, r in pivot.iterrows():
        all_rows.append([
            f"{latex_escape(r['pod'])} ({latex_escape(r['metric'])})",
            format_num(r["hpa"]),
            format_num(r["rl"]),
            format_difference(r["difference"], r["difference_pct"]),
        ])

    write_latex_table(
        path=TABLES_DIR / "deskriptif_all_metrics.tex",
        caption="Perbandingan rata-rata semua metrik",
        label="tab:deskriptif-all-metrics",
        columns=["Pod", "HPA", "RL", "Perbedaan"],
        rows=all_rows,
    )


def main() -> None:
    generate_independent_charts()
    generate_compare_charts()
    generate_latex_tables()


if __name__ == "__main__":
    main()