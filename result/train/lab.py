# /// script
# requires-python = ">=3.11"
# dependencies = ["matplotlib", "pandas", "numpy"]
# ///
"""
Visualize TensorBoard training logs exported as CSV.

File naming convention:
    {model}_logs_{date}_{run}_{metric}.csv
    e.g. stable_forecast_logs_2025-12-30-02-19_DQN_0_loss.csv
         stable_simple_logs_2026-03-06-00-27_offline_dqn_1_loss.csv

Online  training: stable_forecast / DQN_0       → loss + reward
Offline training: stable_simple   / offline_dqn → loss only (dummy env)

Run:
    uv run lab.py
"""

import re
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd

# ── Config ────────────────────────────────────────────────────────────────────
DATA_DIR  = Path(__file__).parent / "data"
CHART_DIR = Path(__file__).parent / "chart"
SMOOTH_WINDOW = 5   # rolling-average window; set 1 to disable
FIG_SIZE_COMBINED = (15, 5)
FIG_SIZE_SINGLE   = (7, 4)
DPI = 130

PALETTE = {
    "stable_forecast": ["#2196F3", "#1565C0", "#42A5F5"],
    "stable_simple":   ["#FF7043", "#BF360C", "#FFAB91"],
}

PATTERN = re.compile(
    r"^(?P<model>.+?)_logs_(?P<date>[\d-]+)_(?P<run>.+?)_(?P<metric>loss|reward)\.csv$"
)

METRIC_LABELS = {
    "loss":   ("TD Loss",        "Training step", "Loss"),
    "reward": ("Episode Reward", "Episode",       "Reward"),
}


# ── Data loading ──────────────────────────────────────────────────────────────
def load_all(data_dir: Path) -> pd.DataFrame:
    records = []
    for f in sorted(data_dir.glob("*.csv")):
        m = PATTERN.match(f.name)
        if not m:
            continue
        df = pd.read_csv(f)
        df.columns = ["wall_time", "step", "value"]
        df["model"]         = m.group("model")
        df["date"]          = m.group("date")
        df["run"]           = m.group("run")
        df["metric"]        = m.group("metric")
        df["training_type"] = "offline" if "offline" in m.group("run") else "online"
        records.append(df)
    return pd.concat(records, ignore_index=True)


def smooth(series: pd.Series, window: int) -> pd.Series:
    return series.rolling(window, min_periods=1, center=True).mean()


def _palette_color(model: str, idx: int) -> str:
    palette = PALETTE.get(model, ["#9E9E9E"] * 5)
    return palette[idx % len(palette)]


# ── Shared draw helpers ───────────────────────────────────────────────────────
def _draw_series(ax: plt.Axes, grp: pd.DataFrame, color: str, label: str,
                 linestyle: str = "-"):
    grp = grp.sort_values("step")
    y = smooth(grp["value"], SMOOTH_WINDOW)
    ax.plot(grp["step"], y, linestyle=linestyle, color=color,
            linewidth=1.8, label=label, alpha=0.92)
    ax.fill_between(grp["step"], grp["value"], y, color=color, alpha=0.1)


def _style_ax(ax: plt.Axes, title: str, xlabel: str, ylabel: str):
    ax.set_title(title, fontsize=12, fontweight="bold")
    ax.set_xlabel(xlabel, fontsize=9)
    ax.set_ylabel(ylabel, fontsize=9)
    ax.legend(fontsize=8, framealpha=0.7)
    ax.grid(True, linestyle=":", alpha=0.5)


# ── Combined chart panels ─────────────────────────────────────────────────────
def plot_loss(ax: plt.Axes, data: pd.DataFrame):
    model_idx: dict = {}
    for (model, date, run), grp in data[data["metric"] == "loss"].groupby(
        ["model", "date", "run"]
    ):
        idx = model_idx.get(model, 0)
        model_idx[model] = idx + 1
        training = grp["training_type"].iloc[0]
        color = _palette_color(model, idx)
        ls = "--" if training == "offline" else "-"
        _draw_series(ax, grp, color, f"[{training}] {model} {date}", ls)
    _style_ax(ax, "TD Loss — All Runs", "Training step", "Loss")
    ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.3f"))


def plot_reward(ax: plt.Axes, data: pd.DataFrame):
    reward_data = data[data["metric"] == "reward"]
    if reward_data.empty:
        ax.text(0.5, 0.5, "No reward data\n(offline training uses dummy env)",
                ha="center", va="center", transform=ax.transAxes,
                fontsize=10, color="grey", style="italic")
        ax.set_title("Episode Reward — Online Only", fontsize=12, fontweight="bold")
        return
    model_idx: dict = {}
    for (model, date, run), grp in reward_data.groupby(["model", "date", "run"]):
        idx = model_idx.get(model, 0)
        model_idx[model] = idx + 1
        color = _palette_color(model, idx)
        _draw_series(ax, grp, color, f"[online] {model} {date}")
    _style_ax(ax, "Episode Reward — Online Only", "Episode", "Reward")


def plot_loss_comparison(ax: plt.Axes, data: pd.DataFrame):
    """Mean ± 1σ bands, x normalised to training-progress %."""
    def normalise(g: pd.DataFrame) -> pd.DataFrame:
        g = g.sort_values("step").copy()
        rng = max(g["step"].max() - g["step"].min(), 1)
        g["pct"] = (g["step"] - g["step"].min()) / rng * 100
        return g

    common_x = np.linspace(0, 100, 300)
    for training_type, color in [("online", "#1E88E5"), ("offline", "#E53935")]:
        grps = [
            normalise(g)
            for _, g in data[
                (data["metric"] == "loss") & (data["training_type"] == training_type)
            ].groupby(["model", "date", "run"])
        ]
        if not grps:
            continue
        interp = np.array([
            np.interp(common_x, g["pct"], smooth(g["value"], SMOOTH_WINDOW))
            for g in grps
        ])
        mean_y, std_y = interp.mean(axis=0), interp.std(axis=0)
        ax.plot(common_x, mean_y, color=color, linewidth=2,
                label=f"{training_type}  (mean ± 1σ,  n={len(grps)})")
        ax.fill_between(common_x, mean_y - std_y, mean_y + std_y,
                        color=color, alpha=0.15)
    _style_ax(ax,
              "Loss: Online vs Offline\n(x = % training progress)",
              "Training progress (%)", "Loss")


# ── Individual 1×1 charts ─────────────────────────────────────────────────────
def save_individual_charts(data: pd.DataFrame, chart_dir: Path):
    """One PNG per (model, date, run, metric) series + standalone comparison."""
    for (model, date, run, metric), grp in data.groupby(
        ["model", "date", "run", "metric"]
    ):
        title_base, xlabel, ylabel = METRIC_LABELS.get(
            metric, (metric.capitalize(), "Step", "Value")
        )
        training = grp["training_type"].iloc[0]
        color = _palette_color(model, 0)
        ls = "--" if training == "offline" else "-"

        fig, ax = plt.subplots(figsize=FIG_SIZE_SINGLE, constrained_layout=True)
        _draw_series(ax, grp, color, f"[{training}] {model} {date}", ls)
        if metric == "loss":
            ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.3f"))
        _style_ax(ax, f"{title_base}  ·  {model}\n{date}", xlabel, ylabel)
        fig.suptitle(f"run: {run}", fontsize=8, color="grey", y=1.0)

        slug = f"{model}_{date}_{run}_{metric}".replace("/", "-")
        out = chart_dir / f"{slug}.png"
        fig.savefig(out, dpi=DPI, bbox_inches="tight")
        plt.close(fig)
        print(f"  saved → chart/{out.name}")

    # standalone comparison chart
    fig, ax = plt.subplots(figsize=FIG_SIZE_SINGLE, constrained_layout=True)
    plot_loss_comparison(ax, data)
    out = chart_dir / "comparison_online_vs_offline_loss.png"
    fig.savefig(out, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved → chart/{out.name}")


# ── Main ───────────────────────────────────────────────────────────────────────
def main():
    CHART_DIR.mkdir(parents=True, exist_ok=True)

    data = load_all(DATA_DIR)
    if data.empty:
        print(f"No CSV files matched in {DATA_DIR}")
        return

    n_series = data.groupby(["model", "date", "run", "metric"]).ngroups
    print(f"Loaded {len(data):,} rows  |  {n_series} series")
    print(
        data.groupby(["training_type", "model", "date", "run", "metric"])
        .size().rename("rows").to_string()
    )

    plt.style.use("seaborn-v0_8-whitegrid")

    # combined 3-panel overview
    fig, axes = plt.subplots(1, 3, figsize=FIG_SIZE_COMBINED, constrained_layout=True)
    fig.suptitle("RL Autoscaler — Training Logs", fontsize=14, fontweight="bold")
    plot_loss(axes[0], data)
    plot_reward(axes[1], data)
    plot_loss_comparison(axes[2], data)
    combined_path = CHART_DIR / "combined_overview.png"
    fig.savefig(combined_path, dpi=DPI, bbox_inches="tight")
    print(f"\nSaved combined → chart/combined_overview.png")
    plt.show()
    plt.close(fig)

    # individual 1×1 charts
    print("\nSaving individual charts:")
    save_individual_charts(data, CHART_DIR)
    print(f"\nAll charts saved to: {CHART_DIR.resolve()}")


if __name__ == "__main__":
    main()
