"""Visualisations EDA (distribution des classes, corrélations, features)."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from .paths import FIG_KW

COLOR_LEGIT = "#2e7d32"
COLOR_FRAUD = "#c62828"
COLOR_NEUTRAL = "#1565c0"


def save_fig(path: Path, **kw) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(p, **{**FIG_KW, **kw})
    plt.close()
    print(f"Saved: {p.resolve()}")


def missing_value_report(df: pd.DataFrame) -> pd.DataFrame:
    m = df.isnull().sum()
    pct = (m / len(df) * 100).round(4)
    return pd.DataFrame({"count": m, "pct": pct})


def plot_class_distribution(df: pd.DataFrame, target: str, out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(7.5, 4.8))
    vc = df[target].value_counts().sort_index()
    xlabs = ["Legitimate (0)", "Fraud (1)"]
    cols = [COLOR_LEGIT, COLOR_FRAUD]
    ax.bar(xlabs[: len(vc)], vc.values, color=cols[: len(vc)], edgecolor="white", linewidth=1.1)
    ax.set_title("Class imbalance: transaction counts", fontweight="semibold")
    ax.set_xlabel("Label")
    ax.set_ylabel("Number of transactions")
    n = len(df)
    for i, (xi, vi) in enumerate(zip(range(len(vc)), vc.values)):
        ax.text(xi, vi + n * 0.002, f"{vi:,}\n({100 * vi / n:.2f}%)", ha="center", fontsize=9)
    plt.tight_layout()
    save_fig(out_path)


def plot_correlation_heatmap(df: pd.DataFrame, out_path: Path, exclude=None):
    exclude = exclude or []
    num = df.select_dtypes(include=[np.number]).drop(columns=exclude, errors="ignore")
    fig, ax = plt.subplots(figsize=(14, 12))
    sns.heatmap(
        num.corr(),
        cmap="RdBu_r",
        center=0,
        ax=ax,
        square=False,
        cbar_kws={"shrink": 0.55, "label": "Pearson r"},
    )
    ax.set_title("Linear associations between numeric features", fontweight="semibold", pad=12)
    plt.tight_layout()
    save_fig(out_path)


def plot_feature_distributions(df: pd.DataFrame, cols: list, target: str, out_path: Path):
    cols = [c for c in cols if c in df.columns and c != target]
    n = len(cols)
    fig, axes = plt.subplots(n, 2, figsize=(11, 3.3 * max(n, 1)))
    if n == 1:
        axes = np.asarray([axes])
    for i, col in enumerate(cols):
        sns.histplot(df[col], bins=50, kde=True, ax=axes[i, 0], color=COLOR_NEUTRAL, alpha=0.85)
        axes[i, 0].set_title(f"Distribution — {col}", fontweight="medium")
        axes[i, 0].set_xlabel(col)
        axes[i, 0].set_ylabel("Count")
        sns.boxplot(
            data=df,
            x=target,
            y=col,
            hue=target,
            ax=axes[i, 1],
            palette=[COLOR_LEGIT, COLOR_FRAUD],
            dodge=False,
            linewidth=1,
            legend=False,
        )
        axes[i, 1].set_title(f"{col} by transaction type", fontweight="medium")
        axes[i, 1].set_xlabel("Class (0 = legitimate, 1 = fraud)")
        axes[i, 1].set_ylabel(col)
    fig.suptitle("Exploratory feature distributions", fontsize=14, fontweight="semibold", y=1.01)
    plt.tight_layout()
    save_fig(out_path)


def plot_smote_effect(y_before, y_after, out_path: Path) -> None:
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.5))
    xtick_labs = ["Legitimate (0)", "Fraud (1)"]
    for ax, yseries, title in zip(
        axes,
        (y_before, y_after),
        (
            "Before SMOTE (original train counts)",
            "After SMOTE (balanced train — fit-time only)",
        ),
    ):
        s = pd.Series(yseries).value_counts().sort_index()
        ax.bar(
            xtick_labs[: len(s)],
            s.values,
            color=[COLOR_LEGIT, COLOR_FRAUD][: len(s)],
            edgecolor="white",
        )
        ax.set_title(title, fontweight="medium")
        ax.set_xlabel("Class")
        ax.set_ylabel("Number of samples")
    fig.suptitle("SMOTE effect on training labels (no test data)", fontweight="semibold", y=1.02)
    plt.tight_layout()
    save_fig(out_path)


def setup_plot_style():
    sns.set_theme(style="whitegrid", context="notebook", palette="deep")
    plt.rcParams.update(
        {
            "figure.figsize": (10, 5.5),
            "figure.titlesize": 14,
            "axes.titlesize": 13,
            "axes.labelsize": 11,
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
            "legend.fontsize": 10,
            "axes.grid": True,
            "grid.alpha": 0.3,
        }
    )
