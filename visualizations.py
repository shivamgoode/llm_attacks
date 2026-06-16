import logging
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

logger = logging.getLogger("visualizations")

def generate_plots(summary_df: pd.DataFrame, category_df: pd.DataFrame, output_dir: str | Path) -> None:
    """
    Generates bar graphs, pie charts, and other visualizations comparing model performance.
    """
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    
    sns.set_theme(style="whitegrid")
    
    if summary_df.empty or category_df.empty:
        logger.warning("Empty dataframes provided for visualizations. Skipping plots.")
        return

    _plot_model_comparison(summary_df, out_path)
    _plot_response_distribution(summary_df, out_path)
    _plot_attack_category_comparison(category_df, out_path)
    
    logger.info("Visualizations generated and saved to %s", out_path)

def _plot_model_comparison(summary_df: pd.DataFrame, out_path: Path) -> None:
    """Bar graph comparing overall ASR, Detection Rate, and Security Score."""
    plt.figure(figsize=(10, 6))
    
    # Melt dataframe for grouped bar plotting
    melted_df = summary_df.melt(
        id_vars=["Model"],
        value_vars=["ASR_%", "Detection_Rate_%", "Security_Score"],
        var_name="Metric",
        value_name="Percentage/Score"
    )
    
    sns.barplot(
        data=melted_df, 
        x="Model", 
        y="Percentage/Score", 
        hue="Metric",
        palette="muted"
    )
    
    plt.title("Overall Model Comparison: ASR, Detection Rate, and Security Score")
    plt.ylabel("Score (%)")
    plt.xlabel("Model")
    plt.xticks(rotation=15)
    plt.ylim(0, 105) # ensure percentages fit well
    plt.legend(title="Metric", bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig(out_path / "model_comparison_bar.png", dpi=300)
    plt.close()

def _plot_response_distribution(summary_df: pd.DataFrame, out_path: Path) -> None:
    """Pie charts showing the distribution of Answered, Refused, Errors per model."""
    models = summary_df["Model"].unique()
    num_models = len(models)
    
    fig, axes = plt.subplots(1, num_models, figsize=(6 * num_models, 5))
    if num_models == 1:
        axes = [axes]
    
    for ax, model_name in zip(axes, models):
        model_data = summary_df[summary_df["Model"] == model_name].iloc[0]
        
        labels = ["Answered", "Refused", "Errors"]
        sizes = [model_data["Answered"], model_data["Refused"], model_data["Errors"]]
        
        # Filter out slices with size 0
        filtered = [(l, s) for l, s in zip(labels, sizes) if s > 0]
        if not filtered:
            ax.set_title(f"{model_name} (No Data)")
            ax.axis('off')
            continue
            
        filtered_labels = [x[0] for x in filtered]
        filtered_sizes = [x[1] for x in filtered]
        
        # Custom colors for distinction
        color_map = {
            "Answered": "#ff9999", # Light red/pink
            "Refused": "#66b3ff",  # Light blue
            "Errors": "#99ff99"    # Light green
        }
        pie_colors = [color_map[l] for l in filtered_labels]
        
        ax.pie(
            filtered_sizes, 
            labels=filtered_labels, 
            autopct='%1.1f%%', 
            startangle=90, 
            colors=pie_colors,
            wedgeprops={'edgecolor': 'white'}
        )
        ax.set_title(f"{model_name}")
        
    plt.suptitle("Response Distribution by Model", fontsize=16)
    plt.tight_layout()
    plt.savefig(out_path / "response_distribution_pie.png", dpi=300)
    plt.close()

def _plot_attack_category_comparison(category_df: pd.DataFrame, out_path: Path) -> None:
    """Grouped bar chart comparing ASR (%) by Attack Type for each model."""
    plt.figure(figsize=(12, 6))
    
    sns.barplot(
        data=category_df, 
        x="Attack_Type", 
        y="ASR_%", 
        hue="Model",
        palette="viridis"
    )
    
    plt.title("Attack Success Rate (ASR) by Attack Category")
    plt.ylabel("ASR (%)")
    plt.xlabel("Attack Category")
    plt.xticks(rotation=45, ha="right")
    plt.ylim(0, 105) # ensure percentages fit well
    plt.legend(title="Model", bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig(out_path / "attack_type_comparison.png", dpi=300)
    plt.close()
