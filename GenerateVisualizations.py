"""
Visualization Generator for Sensitivity Analysis
=================================================

Generates comprehensive visualizations from sensitivity analysis results.
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import numpy as np
import logging
from mpl_toolkits.mplot3d import Axes3D

logger = logging.getLogger(__name__)
logging.basicConfig(
        level=logging.INFO,
        format="%(module)s:: %(message)s"
    )

def generate_alpha_grid_run(output_dir, df):
    unique_datasets = df['dataset'].unique()
    for ds_name in unique_datasets:
        ds_data = df[df['dataset'] == ds_name].sort_values('alpha')

        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        # Plot 1: AUC vs Alpha (Left Subplot)
        axes[0].plot(ds_data['alpha'], ds_data['soft_auc'], marker='o', color='b')
        axes[0].set_title(f'Soft Path AUC vs Alpha - {ds_name}')
        axes[0].set_xlabel('Alpha')
        axes[0].set_ylabel('Soft Path AUC')
        axes[0].grid(True)
        
        # Plot 2: Accuracy vs Alpha (Right Subplot)
        axes[1].plot(ds_data['alpha'], ds_data['soft_accuracy'], marker='s', color='g')
        axes[1].set_title(f'Soft Path Accuracy vs Alpha - {ds_name}')
        axes[1].set_xlabel('Alpha')
        axes[1].set_ylabel('Soft Path Accuracy')
        axes[1].grid(True)
        
        # Adjust layout and save
        plt.tight_layout()
        plt.savefig(f'{output_dir}/{ds_name}_metrics_side_by_side.png')
        plt.close()



def generate_alpha_n_grid_run(output_dir, should_use_improved_version):

    df = pd.read_csv('./results/sensitivity/all_datasets_combined_summary.csv')
    if should_use_improved_version:
        return generate_alpha_grid_run(output_dir, df)
    
    for dataset in df["dataset"].unique().tolist():
        df2 = df[df["dataset"] == dataset]
        fig = plt.figure(figsize=(12, 6))

        ax1 = fig.add_subplot(1, 2, 1, projection='3d')
        surf = ax1.plot_trisurf(df2['alpha'], df2['n_runs'], df2['soft_accuracy'], cmap='plasma', linewidth=0.2)
        ax1.set_xlabel('alpha')
        ax1.set_ylabel('n')
        ax1.set_zlabel('Accuracy')
        ax1.set_title('Accuracy Results')
        fig.colorbar(surf, ax=ax1, shrink=0.5, aspect=5)

        ax2 = fig.add_subplot(1, 2, 2, projection='3d')
        surf = ax2.plot_trisurf(df2['alpha'], df2['n_runs'], df2['soft_auc'], cmap='plasma', linewidth=0.2)
        ax2.set_xlabel('alpha')
        ax2.set_ylabel('n')
        ax2.set_zlabel('AUC')
        ax2.set_title('AUC Results')
        fig.colorbar(surf, ax=ax2, shrink=0.5, aspect=5)

        # 3. Show or Save the plot
        plt.tight_layout()
        plt.savefig(f'{output_dir}/{dataset}.png')


def generate_metrics_bar_graphs(output_dir):
    dataset_col = 'Dataset'
    method_col = 'Method'
    df = pd.read_csv('results/cross_validation_summary.csv')
    metric_cols = [c for c in df.columns if c not in [dataset_col, method_col]]
    
    for metric in metric_cols:
        plot_data = df.pivot(index=dataset_col, columns=method_col, values=metric)
        ax = plot_data.plot(kind='bar', width=0.7, figsize=(10, 6), rot=0)

        for container in ax.containers:
            ax.bar_label(container, fmt='%.2f', padding=3, fontsize=10)

        ax.set_ylim(0, ax.get_ylim()[1] * 1.1)

        ax.set_title(f'Mean {metric} Comparison', loc='left', fontsize=16, pad=20)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_visible(False)
        ax.yaxis.grid(True, color='#EEEEEE') # Light horizontal grid
        ax.set_axisbelow(True)

        ax.legend(title=method_col, loc='upper center', bbox_to_anchor=(0.5, 1.1), 
                  frameon=False, ncol=len(plot_data.columns))
        
        plt.tight_layout()
        filename = f"{output_dir}/{metric}_plot.png"
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        plt.close()



def generate_all_visualizations(should_use_improved_version, sensitivity_dir='./results/sensitivity',
                                output_dir='./results/visualizations'):
    """
    Generate all visualizations from sensitivity analysis results.
    
    Parameters
    ----------
    sensitivity_dir : str
        Directory containing sensitivity analysis CSV files
    output_dir : str
        Directory to save visualization plots
    """
    os.makedirs(output_dir, exist_ok=True)
    
    logger.info("="*80)
    logger.info("GENERATING VISUALIZATIONS")
    logger.info("="*80)

    generate_metrics_bar_graphs(output_dir)
    generate_alpha_n_grid_run(output_dir, should_use_improved_version)


    
    
