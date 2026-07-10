"""Visualizations for focus shifting benchmark outputs."""
from __future__ import annotations

import logging
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

logger = logging.getLogger(__name__)


def generate_plots(summary_df: pd.DataFrame, category_df: pd.DataFrame, output_dir: str | Path) -> None:
    """Generate plots compatible with the existing benchmark output layout."""
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    if summary_df.empty:
        logger.warning("Empty summary dataframe. Skipping plots.")
        return

    sns.set_theme(style="whitegrid")
    _plot_model_comparison(summary_df, out_path)
    _plot_response_distribution(summary_df, out_path)
    if not category_df.empty:
        _plot_attack_category_comparison(category_df, out_path)


def _plot_model_comparison(summary_df: pd.DataFrame, out_path: Path) -> None:
    plt.figure(figsize=(10, 6))
    melted_df = summary_df.melt(
        id_vars=["Model"],
        value_vars=["ASR_%", "Detection_Rate_%", "Security_Score"],
        var_name="Metric",
        value_name="Percentage/Score",
    )
    sns.barplot(data=melted_df, x="Model", y="Percentage/Score", hue="Metric", palette="muted")
    plt.title("Focus Shifting Model Comparison")
    plt.ylabel("Score (%)")
    plt.xlabel("Model")
    plt.xticks(rotation=15)
    plt.ylim(0, 105)
    plt.legend(title="Metric", bbox_to_anchor=(1.05, 1), loc="upper left")
    plt.tight_layout()
    plt.savefig(out_path / "model_comparison_bar.png", dpi=300)
    plt.close()


def _plot_response_distribution(summary_df: pd.DataFrame, out_path: Path) -> None:
    models = summary_df["Model"].unique()
    fig, axes = plt.subplots(1, len(models), figsize=(6 * len(models), 5))
    if len(models) == 1:
        axes = [axes]

    for ax, model_name in zip(axes, models):
        model_data = summary_df[summary_df["Model"] == model_name].iloc[0]
        labels = ["Answered", "Refused", "Errors"]
        sizes = [model_data.get(label, 0) for label in labels]
        filtered = [(label, size) for label, size in zip(labels, sizes) if size > 0]
        if not filtered:
            ax.set_title(f"{model_name} (No Data)")
            ax.axis("off")
            continue
        ax.pie(
            [item[1] for item in filtered],
            labels=[item[0] for item in filtered],
            autopct="%1.1f%%",
            startangle=90,
            wedgeprops={"edgecolor": "white"},
        )
        ax.set_title(model_name)

    plt.suptitle("Focus Shifting Response Distribution by Model", fontsize=16)
    plt.tight_layout()
    plt.savefig(out_path / "response_distribution_pie.png", dpi=300)
    plt.close()


def _plot_attack_category_comparison(category_df: pd.DataFrame, out_path: Path) -> None:
    plt.figure(figsize=(12, 6))
    sns.barplot(data=category_df, x="Attack_Type", y="ASR_%", hue="Model", palette="viridis")
    plt.title("Attack Success Rate (ASR) by Attack")
    plt.ylabel("ASR (%)")
    plt.xlabel("Attack")
    plt.xticks(rotation=45, ha="right")
    plt.ylim(0, 105)
    plt.legend(title="Model", bbox_to_anchor=(1.05, 1), loc="upper left")
    plt.tight_layout()
    plt.savefig(out_path / "attack_type_comparison.png", dpi=300)
    plt.close()
