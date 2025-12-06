import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import logging

logger = logging.getLogger(__name__)

def plot_mse_comparison(results_df, output_dir):
    """
    Generates and saves a bar chart comparing the MSE of different models across datasets.
    """
    plt.figure(figsize=(12, 7))
    sns.barplot(data=results_df, x='dataset', y='mse_mean', hue='model')

    plt.title('Comparison of Mean Squared Error (MSE) Across Datasets', fontsize=16)
    plt.xlabel('Dataset', fontsize=12)
    plt.ylabel('Mean Squared Error (Lower is Better)', fontsize=12)
    plt.xticks(rotation=45, ha='right')
    plt.legend(title='Model')
    plt.tight_layout()

    plot_path = os.path.join(output_dir, 'mse_comparison_by_dataset.png')
    plt.savefig(plot_path)
    logger.info(f"Saved MSE comparison plot to {plot_path}")
    plt.close()

def plot_sensitivity_heatmap(results_df, output_dir):
    """
    Generates and saves heatmaps for the sensitivity of MSE to alpha and n_runs.
    """
    datasets = results_df['dataset'].unique()
    for dataset in datasets:
        dataset_df = results_df[results_df['dataset'] == dataset]
        
        # Pivot for heatmap
        pivot_df = dataset_df.pivot(index='alpha', columns='n_runs', values='soft_mse')

        plt.figure(figsize=(10, 8))
        sns.heatmap(pivot_df, annot=True, fmt=".2f", cmap="viridis_r") # _r reverses the colormap
        
        plt.title(f'Sensitivity of MSE to Alpha and N_Runs on {dataset.title()}', fontsize=16)
        plt.xlabel('Number of Runs (n_runs)', fontsize=12)
        plt.ylabel('Alpha', fontsize=12)
        
        plot_path = os.path.join(output_dir, f'sensitivity_heatmap_{dataset}.png')
        plt.savefig(plot_path)
        logger.info(f"Saved sensitivity heatmap for {dataset} to {plot_path}")
        plt.close()

def generate_visualizations(cross_val_summary, sensitivity_summary):
    """
    Main function to generate all visualizations for the regression experiments.
    """
    output_dir = './regression/results/visualizations'
    os.makedirs(output_dir, exist_ok=True)

    logger.info("--- Generating Visualizations for Regression ---")

    # 1. Plot MSE comparison from cross-validation
    if not cross_val_summary.empty:
        plot_mse_comparison(cross_val_summary, output_dir)
    else:
        logger.warning("Cross-validation summary is empty. Skipping MSE comparison plot.")

    # 2. Plot sensitivity heatmaps
    if not sensitivity_summary.empty:
        plot_sensitivity_heatmap(sensitivity_summary, output_dir)
    else:
        logger.warning("Sensitivity analysis summary is empty. Skipping heatmaps.")

    logger.info("--- Visualization Generation Complete ---")
