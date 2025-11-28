"""
Sensitivity Analysis Runner
============================

Runs sensitivity analysis on alpha and n_runs parameters
for all datasets and saves results for visualization.
"""

import logging
from rich.logging import RichHandler
import pandas as pd
import os
from SensitivityAnalysis import (
    run_alpha_sensitivity,
    run_n_runs_sensitivity,
    run_combined_sensitivity
)

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(module)s.%(funcName)s:: %(message)s",
    handlers=[RichHandler()]
)
logger = logging.getLogger(__name__)


def get_dataset_name(file_path):
    """Extract dataset name from file path"""
    basename = os.path.basename(file_path)
    return basename.replace('_processed.csv', '')


def run_sensitivity_on_all_datasets(
    alpha_values=[0.01, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3],
    n_runs_values=[10, 25, 50, 75, 100, 150, 200],
    combined_alpha=[0.05, 0.1, 0.15, 0.2],
    combined_n_runs=[50, 100, 150],
    test_size=0.2,
    random_state=42
):
    """
    Run sensitivity analysis on all datasets.
    
    Parameters
    ----------
    alpha_values : list
        Alpha values to test in alpha sensitivity
    n_runs_values : list
        n_runs values to test in n_runs sensitivity
    combined_alpha : list
        Alpha values for combined analysis
    combined_n_runs : list
        n_runs values for combined analysis
    test_size : float
        Test set proportion
    random_state : int
        Random seed
    
    Returns
    -------
    dict
        All sensitivity analysis results
    """
    processed_dir = './datasets/processed'
    results_dir = './results/sensitivity'
    os.makedirs(results_dir, exist_ok=True)
    
    # Get all processed datasets
    files = [f for f in os.listdir(processed_dir) if f.endswith('_processed.csv')]
    
    if len(files) == 0:
        logger.error(f"No processed datasets found in {processed_dir}")
        return None
    
    logger.info(f"Found {len(files)} datasets for sensitivity analysis:")
    for f in files:
        logger.info(f"  - {f}")
    
    all_results = {
        'alpha_sensitivity': {},
        'n_runs_sensitivity': {},
        'combined_sensitivity': {}
    }
    
    for file in files:
        file_path = os.path.join(processed_dir, file)
        dataset_name = get_dataset_name(file_path)
        
        logger.info("\n" + "="*80)
        logger.info(f"SENSITIVITY ANALYSIS: {dataset_name.upper()}")
        logger.info("="*80)
        
        try:
            df = pd.read_csv(file_path)
            logger.info(f"Loaded {len(df)} samples from {file}")
            
            # Alpha sensitivity
            logger.info(f"\nRunning alpha sensitivity analysis...")
            alpha_results = run_alpha_sensitivity(
                df=df,
                target_column='target',
                alpha_values=alpha_values,
                n_runs=100,  # Fixed for alpha analysis
                test_size=test_size,
                random_state=random_state
            )
            all_results['alpha_sensitivity'][dataset_name] = alpha_results
            
            # Save alpha results
            alpha_path = os.path.join(results_dir, f'{dataset_name}_alpha_sensitivity.csv')
            alpha_results.to_csv(alpha_path, index=False)
            logger.info(f"Saved alpha sensitivity to {alpha_path}")
            
            # n_runs sensitivity
            logger.info(f"\nRunning n_runs sensitivity analysis...")
            n_runs_results = run_n_runs_sensitivity(
                df=df,
                target_column='target',
                n_runs_values=n_runs_values,
                alpha=0.1,  # Fixed for n_runs analysis
                test_size=test_size,
                random_state=random_state
            )
            all_results['n_runs_sensitivity'][dataset_name] = n_runs_results
            
            # Save n_runs results
            n_runs_path = os.path.join(results_dir, f'{dataset_name}_nruns_sensitivity.csv')
            n_runs_results.to_csv(n_runs_path, index=False)
            logger.info(f"Saved n_runs sensitivity to {n_runs_path}")
            
            # Combined sensitivity (for heatmaps)
            logger.info(f"\nRunning combined sensitivity analysis...")
            combined_results = run_combined_sensitivity(
                df=df,
                target_column='target',
                alpha_values=combined_alpha,
                n_runs_values=combined_n_runs,
                test_size=test_size,
                random_state=random_state
            )
            all_results['combined_sensitivity'][dataset_name] = combined_results
            
            # Save combined results
            combined_path = os.path.join(results_dir, f'{dataset_name}_combined_sensitivity.csv')
            combined_results.to_csv(combined_path, index=False)
            logger.info(f"Saved combined sensitivity to {combined_path}")
            
            logger.info(f"\n✓ Completed sensitivity analysis for {dataset_name}")
            
        except Exception as e:
            logger.error(f"Error processing {dataset_name}: {str(e)}")
            import traceback
            traceback.print_exc()
            continue
    
    # Create summary across all datasets
    logger.info("\n" + "="*80)
    logger.info("CREATING CROSS-DATASET SUMMARIES")
    logger.info("="*80)
    
    # Alpha sensitivity summary
    alpha_summary = []
    for dataset_name, results in all_results['alpha_sensitivity'].items():
        for _, row in results.iterrows():
            alpha_summary.append({
                'dataset': dataset_name,
                'alpha': row['alpha'],
                'auc_improvement': row['auc_improvement'],
                'accuracy_improvement': row['accuracy_improvement'],
                'soft_auc_mean': row['soft_auc_mean'],
                'soft_accuracy_mean': row['soft_accuracy_mean']
            })
    
    alpha_summary_df = pd.DataFrame(alpha_summary)
    alpha_summary_path = os.path.join(results_dir, 'all_datasets_alpha_summary.csv')
    alpha_summary_df.to_csv(alpha_summary_path, index=False)
    logger.info(f"Saved alpha summary to {alpha_summary_path}")
    
    # n_runs sensitivity summary
    n_runs_summary = []
    for dataset_name, results in all_results['n_runs_sensitivity'].items():
        for _, row in results.iterrows():
            n_runs_summary.append({
                'dataset': dataset_name,
                'n_runs': row['n_runs'],
                'auc_improvement': row['auc_improvement'],
                'accuracy_improvement': row['accuracy_improvement'],
                'soft_auc_mean': row['soft_auc_mean'],
                'soft_accuracy_mean': row['soft_accuracy_mean']
            })
    
    n_runs_summary_df = pd.DataFrame(n_runs_summary)
    n_runs_summary_path = os.path.join(results_dir, 'all_datasets_nruns_summary.csv')
    n_runs_summary_df.to_csv(n_runs_summary_path, index=False)
    logger.info(f"Saved n_runs summary to {n_runs_summary_path}")
    
    # Combined sensitivity summary
    combined_summary = []
    for dataset_name, results in all_results['combined_sensitivity'].items():
        for _, row in results.iterrows():
            combined_summary.append({
                'dataset': dataset_name,
                'alpha': row['alpha'],
                'n_runs': row['n_runs'],
                'auc_improvement': row['auc_improvement'],
                'accuracy_improvement': row['accuracy_improvement'],
                'soft_auc_mean': row['soft_auc_mean'],
                'soft_accuracy_mean': row['soft_accuracy_mean']
            })
    
    combined_summary_df = pd.DataFrame(combined_summary)
    combined_summary_path = os.path.join(results_dir, 'all_datasets_combined_summary.csv')
    combined_summary_df.to_csv(combined_summary_path, index=False)
    logger.info(f"Saved combined summary to {combined_summary_path}")
    
    logger.info("\n" + "="*80)
    logger.info("SENSITIVITY ANALYSIS COMPLETE")
    logger.info("="*80)
    logger.info(f"\nAll results saved to: {results_dir}/")
    logger.info("\nGenerated files:")
    logger.info("  Per-dataset results:")
    logger.info("    - {dataset}_alpha_sensitivity.csv")
    logger.info("    - {dataset}_nruns_sensitivity.csv")
    logger.info("    - {dataset}_combined_sensitivity.csv")
    logger.info("  Cross-dataset summaries:")
    logger.info("    - all_datasets_alpha_summary.csv")
    logger.info("    - all_datasets_nruns_summary.csv")
    logger.info("    - all_datasets_combined_summary.csv")
    
    return all_results


if __name__ == "__main__":
    logger.info("Starting Sensitivity Analysis")
    logger.info("="*80)
    
    # Run sensitivity analysis on all datasets
    results = run_sensitivity_on_all_datasets(
        alpha_values=[0.01, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3],
        n_runs_values=[10, 25, 50, 75, 100, 150, 200],
        combined_alpha=[0.05, 0.1, 0.15, 0.2],
        combined_n_runs=[50, 100, 150],
        test_size=0.2,
        random_state=42
    )
    
    if results:
        logger.info("\n✓ Sensitivity analysis completed successfully!")
    else:
        logger.error("Sensitivity analysis failed!")