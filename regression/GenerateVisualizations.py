import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import logging
from mpl_toolkits.mplot3d import Axes3D

logger = logging.getLogger(__name__)

def plot_mse_comparison_per_dataset(results_df, output_dir):
    """
    Generates and saves a bar chart comparing the MSE of different models for each dataset individually.
    """
    datasets = results_df['dataset'].unique()
    for dataset in datasets:
        plt.figure(figsize=(8, 6))
        dataset_df = results_df[results_df['dataset'] == dataset]
        
        ax = sns.barplot(data=dataset_df, x='model', y='mse_mean', hue='model', dodge=False)

        plt.title(f'Model MSE Comparison for {dataset.title()}', fontsize=16)
        plt.xlabel('Model', fontsize=12)
        plt.ylabel('Mean Squared Error (Lower is Better)', fontsize=12)
        plt.legend().set_visible(False) # Hide legend as x-axis is clear
        
        # Add value labels on top of bars
        for container in ax.containers:
            ax.bar_label(container, fmt='%.2f', fontsize=10)

        plt.tight_layout()
        plot_path = os.path.join(output_dir, f'mse_comparison_{dataset}.png')
        plt.savefig(plot_path)
        logger.info(f"Saved MSE comparison plot for {dataset} to {plot_path}")
        plt.close()

def plot_sensitivity_3d(results_df, output_dir):
    """
    Generates and saves 3D surface plots for the sensitivity of MSE to alpha and n_runs.
    """
    datasets = results_df['dataset'].unique()
    for dataset in datasets:
        dataset_df = results_df[results_df['dataset'] == dataset]

        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')

        # Create the 3D surface plot
        surf = ax.plot_trisurf(
            dataset_df['alpha'], 
            dataset_df['n_runs'], 
            dataset_df['soft_mse'], 
            cmap='viridis_r',  # Use a reversed colormap where lower is "better" (darker)
            edgecolor='none'
        )

        ax.set_title(f'Sensitivity of MSE on {dataset.title()}', fontsize=16)
        ax.set_xlabel('Alpha', fontsize=12)
        ax.set_ylabel('Number of Runs (n_runs)', fontsize=12)
        ax.set_zlabel('Mean Squared Error (MSE)', fontsize=12)

        # Add a color bar
        fig.colorbar(surf, shrink=0.5, aspect=5, label='MSE')

        plot_path = os.path.join(output_dir, f'sensitivity_3d_{dataset}.png')
        plt.savefig(plot_path)
        logger.info(f"Saved 3D sensitivity plot for {dataset} to {plot_path}")
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
        plot_mse_comparison_per_dataset(cross_val_summary, output_dir)
    else:
        logger.warning("Cross-validation summary is empty. Skipping MSE comparison plot.")

    # 2. Plot sensitivity 3D plots
    if not sensitivity_summary.empty:
        plot_sensitivity_3d(sensitivity_summary, output_dir)
    else:
        logger.warning("Sensitivity analysis summary is empty. Skipping 3D plots.")

    logger.info("--- Visualization Generation Complete ---")
