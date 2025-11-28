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

logger = logging.getLogger(__name__)
logging.basicConfig(
        level=logging.INFO,
        format="%(module)s:: %(message)s"
    )

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 10


def plot_alpha_sensitivity(results_df, dataset_name, save_path=None):
    """
    Plot alpha sensitivity analysis results.
    
    Creates two subplots:
    1. Performance vs Alpha (Accuracy and AUC)
    2. Improvement vs Alpha
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Plot 1: Performance vs Alpha
    ax1.plot(results_df['alpha'], results_df['standard_accuracy'], 
             'o-', label='Standard DT (Accuracy)', linewidth=2, markersize=8)
    ax1.plot(results_df['alpha'], results_df['soft_accuracy'], 
             's-', label='Soft Split DT (Accuracy)', linewidth=2, markersize=8)
    ax1.plot(results_df['alpha'], results_df['standard_auc'], 
             '^--', label='Standard DT (AUC)', linewidth=2, markersize=8)
    ax1.plot(results_df['alpha'], results_df['soft_auc'], 
             'v--', label='Soft Split DT (AUC)', linewidth=2, markersize=8)
    
    ax1.set_xlabel('Alpha', fontsize=12)
    ax1.set_ylabel('Performance', fontsize=12)
    ax1.set_title(f'Alpha Sensitivity - {dataset_name}', fontsize=14, fontweight='bold')
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Improvement vs Alpha
    ax2.plot(results_df['alpha'], results_df['accuracy_improvement'], 
             'o-', label='Accuracy Improvement', linewidth=2, markersize=8, color='green')
    ax2.plot(results_df['alpha'], results_df['auc_improvement'], 
             's-', label='AUC Improvement', linewidth=2, markersize=8, color='blue')
    ax2.axhline(y=0, color='red', linestyle='--', alpha=0.5, label='No Improvement')
    
    ax2.set_xlabel('Alpha', fontsize=12)
    ax2.set_ylabel('Improvement', fontsize=12)
    ax2.set_title(f'Performance Improvement vs Alpha - {dataset_name}', 
                  fontsize=14, fontweight='bold')
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Saved alpha sensitivity plot to {save_path}")
    
    plt.close()


def plot_n_runs_sensitivity(results_df, dataset_name, save_path=None):
    """
    Plot n_runs sensitivity analysis results.
    
    Creates two subplots:
    1. Performance vs n_runs (Accuracy and AUC)
    2. Improvement vs n_runs
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Plot 1: Performance vs n_runs
    ax1.plot(results_df['n_runs'], results_df['standard_accuracy'], 
             'o-', label='Standard DT (Accuracy)', linewidth=2, markersize=8)
    ax1.plot(results_df['n_runs'], results_df['soft_accuracy'], 
             's-', label='Soft Split DT (Accuracy)', linewidth=2, markersize=8)
    ax1.plot(results_df['n_runs'], results_df['standard_auc'], 
             '^--', label='Standard DT (AUC)', linewidth=2, markersize=8)
    ax1.plot(results_df['n_runs'], results_df['soft_auc'], 
             'v--', label='Soft Split DT (AUC)', linewidth=2, markersize=8)
    
    ax1.set_xlabel('n_runs', fontsize=12)
    ax1.set_ylabel('Performance', fontsize=12)
    ax1.set_title(f'n_runs Sensitivity - {dataset_name}', fontsize=14, fontweight='bold')
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Improvement vs n_runs
    ax2.plot(results_df['n_runs'], results_df['accuracy_improvement'], 
             'o-', label='Accuracy Improvement', linewidth=2, markersize=8, color='green')
    ax2.plot(results_df['n_runs'], results_df['auc_improvement'], 
             's-', label='AUC Improvement', linewidth=2, markersize=8, color='blue')
    ax2.axhline(y=0, color='red', linestyle='--', alpha=0.5, label='No Improvement')
    
    ax2.set_xlabel('n_runs', fontsize=12)
    ax2.set_ylabel('Improvement', fontsize=12)
    ax2.set_title(f'Performance Improvement vs n_runs - {dataset_name}', 
                  fontsize=14, fontweight='bold')
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Saved n_runs sensitivity plot to {save_path}")
    
    plt.close()


def plot_combined_heatmap(results_df, dataset_name, metric='auc_improvement', save_path=None):
    """
    Create heatmap showing performance across alpha and n_runs combinations.
    
    Parameters
    ----------
    results_df : pandas.DataFrame
        Combined sensitivity results
    dataset_name : str
        Name of dataset
    metric : str
        Metric to display ('auc_improvement', 'accuracy_improvement', 'soft_auc', etc.)
    save_path : str
        Path to save figure
    """
    # Pivot data for heatmap
    pivot_data = results_df.pivot(index='alpha', columns='n_runs', values=metric)
    
    plt.figure(figsize=(10, 8))
    
    # Create heatmap
    sns.heatmap(pivot_data, annot=True, fmt='.4f', cmap='RdYlGn', 
                center=0 if 'improvement' in metric else None,
                cbar_kws={'label': metric.replace('_', ' ').title()})
    
    plt.title(f'{metric.replace("_", " ").title()} - {dataset_name}', 
              fontsize=14, fontweight='bold')
    plt.xlabel('n_runs', fontsize=12)
    plt.ylabel('Alpha', fontsize=12)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Saved heatmap to {save_path}")
    
    plt.close()


def plot_cross_dataset_comparison(summary_df, metric='auc_improvement', 
                                  parameter='alpha', save_path=None):
    """
    Compare performance across all datasets for a given parameter.
    
    Parameters
    ----------
    summary_df : pandas.DataFrame
        Summary data across all datasets
    metric : str
        Metric to plot
    parameter : str
        Parameter to vary ('alpha' or 'n_runs')
    save_path : str
        Path to save figure
    """
    plt.figure(figsize=(14, 8))
    
    datasets = summary_df['dataset'].unique()
    
    for dataset in datasets:
        data = summary_df[summary_df['dataset'] == dataset]
        plt.plot(data[parameter], data[metric], 'o-', label=dataset, 
                linewidth=2, markersize=8)
    
    plt.xlabel(parameter.replace('_', ' ').title(), fontsize=12)
    plt.ylabel(metric.replace('_', ' ').title(), fontsize=12)
    plt.title(f'{metric.replace("_", " ").title()} vs {parameter.upper()} - All Datasets', 
              fontsize=14, fontweight='bold')
    plt.legend(fontsize=10, loc='best')
    plt.grid(True, alpha=0.3)
    
    if 'improvement' in metric:
        plt.axhline(y=0, color='red', linestyle='--', alpha=0.5)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Saved cross-dataset comparison to {save_path}")
    
    plt.close()


def plot_best_parameters_summary(summary_df, save_path=None):
    """
    Create bar chart showing best alpha value for each dataset based on AUC improvement.
    """
    # Find best alpha for each dataset
    best_params = summary_df.loc[summary_df.groupby('dataset')['auc_improvement'].idxmax()]
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Plot 1: Best Alpha per Dataset
    ax1.bar(range(len(best_params)), best_params['alpha'], color='steelblue')
    ax1.set_xticks(range(len(best_params)))
    ax1.set_xticklabels(best_params['dataset'], rotation=45, ha='right')
    ax1.set_ylabel('Best Alpha', fontsize=12)
    ax1.set_title('Best Alpha Value per Dataset', fontsize=14, fontweight='bold')
    ax1.grid(True, alpha=0.3, axis='y')
    
    # Plot 2: AUC Improvement at Best Alpha
    ax2.bar(range(len(best_params)), best_params['auc_improvement'], color='green')
    ax2.set_xticks(range(len(best_params)))
    ax2.set_xticklabels(best_params['dataset'], rotation=45, ha='right')
    ax2.set_ylabel('AUC Improvement', fontsize=12)
    ax2.set_title('AUC Improvement at Best Alpha', fontsize=14, fontweight='bold')
    ax2.axhline(y=0, color='red', linestyle='--', alpha=0.5)
    ax2.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Saved best parameters summary to {save_path}")
    
    plt.close()


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
    
    # Get all datasets
    alpha_files = [f for f in os.listdir(sensitivity_dir) 
                   if f.endswith('_alpha_sensitivity.csv')]
    
    # Generate per-dataset plots
    for alpha_file in alpha_files:
        dataset_name = alpha_file.replace('_alpha_sensitivity.csv', '')
        logger.info(f"\nGenerating plots for {dataset_name}...")
        
        # Alpha sensitivity plots
        alpha_path = os.path.join(sensitivity_dir, alpha_file)
        alpha_df = pd.read_csv(alpha_path)
        
        plot_alpha_sensitivity(
            alpha_df, 
            dataset_name,
            save_path=os.path.join(output_dir, f'{dataset_name}_alpha_sensitivity.png')
        )
        
        # n_runs sensitivity plots
        nruns_file = f'{dataset_name}_nruns_sensitivity.csv'
        nruns_path = os.path.join(sensitivity_dir, nruns_file)
        
        if os.path.exists(nruns_path):
            nruns_df = pd.read_csv(nruns_path)
            plot_n_runs_sensitivity(
                nruns_df,
                dataset_name,
                save_path=os.path.join(output_dir, f'{dataset_name}_nruns_sensitivity.png')
            )
        
        # Combined heatmaps
        combined_file = f'{dataset_name}_combined_sensitivity.csv'
        combined_path = os.path.join(sensitivity_dir, combined_file)
        
        if os.path.exists(combined_path):
            combined_df = pd.read_csv(combined_path)
            
            # AUC improvement heatmap
            plot_combined_heatmap(
                combined_df,
                dataset_name,
                metric='auc_improvement',
                save_path=os.path.join(output_dir, f'{dataset_name}_auc_heatmap.png')
            )
            
            # Accuracy improvement heatmap
            plot_combined_heatmap(
                combined_df,
                dataset_name,
                metric='accuracy_improvement',
                save_path=os.path.join(output_dir, f'{dataset_name}_accuracy_heatmap.png')
            )
    
    # Generate cross-dataset comparison plots
    logger.info("\nGenerating cross-dataset comparison plots...")
    
    # Alpha comparison
    alpha_summary_path = os.path.join(sensitivity_dir, 'all_datasets_alpha_summary.csv')
    if os.path.exists(alpha_summary_path):
        alpha_summary = pd.read_csv(alpha_summary_path)
        
        plot_cross_dataset_comparison(
            alpha_summary,
            metric='auc_improvement',
            parameter='alpha',
            save_path=os.path.join(output_dir, 'all_datasets_alpha_auc_comparison.png')
        )
        
        plot_cross_dataset_comparison(
            alpha_summary,
            metric='accuracy_improvement',
            parameter='alpha',
            save_path=os.path.join(output_dir, 'all_datasets_alpha_accuracy_comparison.png')
        )
        
        plot_best_parameters_summary(
            alpha_summary,
            save_path=os.path.join(output_dir, 'best_alpha_summary.png')
        )
    
    # n_runs comparison
    nruns_summary_path = os.path.join(sensitivity_dir, 'all_datasets_nruns_summary.csv')
    if os.path.exists(nruns_summary_path):
        nruns_summary = pd.read_csv(nruns_summary_path)
        
        plot_cross_dataset_comparison(
            nruns_summary,
            metric='auc_improvement',
            parameter='n_runs',
            save_path=os.path.join(output_dir, 'all_datasets_nruns_auc_comparison.png')
        )
        
        plot_cross_dataset_comparison(
            nruns_summary,
            metric='accuracy_improvement',
            parameter='n_runs',
            save_path=os.path.join(output_dir, 'all_datasets_nruns_accuracy_comparison.png')
        )
    
    logger.info("\n" + "="*80)
    logger.info("VISUALIZATION GENERATION COMPLETE")
    logger.info("="*80)
    logger.info(f"\nAll plots saved to: {output_dir}/")
