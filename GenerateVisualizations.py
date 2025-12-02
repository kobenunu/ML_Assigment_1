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


def generate_alpha_n_grid_run():
    df = pd.read_csv('./results/sensitivity/all_datasets_combined_summary.csv')

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
        plt.savefig(f'./results/sensitivity/{dataset}.png')

def generate_all_visualizations(sensitivity_dir='./results/sensitivity',
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

    generate_alpha_n_grid_run()


    
    
