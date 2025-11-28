"""
Cross-Validation Runner for Decision Tree Models
=================================================

This script runs repeated K-fold cross-validation on all preprocessed datasets
to evaluate both standard and soft split decision tree classifiers.

Requirements from instructions.md:
- Use repeated K-fold cross-validation with at least 2 repetitions and 5 folds
- Evaluate on at least 5 datasets
- Report accuracy and AUC metrics
- Perform sensitivity analysis to alpha and n parameters
"""

import logging
from rich.logging import RichHandler
import pandas as pd
import os
from KFoldCrossValidation import repeated_kfold_cross_validation
import json

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

def run_cross_validation_on_all_datasets(n_splits=5, n_repeats=2, alpha=0.1, 
                                         n_runs=100, random_state=42):
    """
    Run cross-validation on all preprocessed datasets.
    
    Parameters
    ----------
    n_splits : int
        Number of folds for K-fold cross-validation (default: 5)
    n_repeats : int
        Number of times to repeat the cross-validation (default: 2)
    alpha : float
        Soft split alpha parameter (default: 0.1)
    n_runs : int
        Number of iterations for soft split prediction (default: 100)
    random_state : int
        Random state for reproducibility (default: 42)
    
    Returns
    -------
    dict
        Dictionary containing results for all datasets
    """
    processed_dir = './datasets/processed'
    
    # Get all processed datasets
    files = [f for f in os.listdir(processed_dir) if f.endswith('_processed.csv')]
    
    if len(files) == 0:
        logger.error(f"No processed datasets found in {processed_dir}")
        return None
    
    logger.info(f"Found {len(files)} datasets to evaluate:")
    for f in files:
        logger.info(f"  - {f}")
    
    all_results = {}
    summary_data = []
    
    for file in files:
        file_path = os.path.join(processed_dir, file)
        dataset_name = get_dataset_name(file_path)
        
        logger.info("\n" + "="*80)
        logger.info(f"PROCESSING DATASET: {dataset_name.upper()}")
        logger.info("="*80)
        
        try:
            # Load dataset
            df = pd.read_csv(file_path)
            logger.info(f"Loaded {len(df)} samples from {file}")
            
            # Run cross-validation
            cv_results = repeated_kfold_cross_validation(
                df=df,
                target_column='target',
                n_splits=n_splits,
                n_repeats=n_repeats,
                alpha=alpha,
                n_runs=n_runs,
                random_state=random_state
            )
            
            # Store results
            all_results[dataset_name] = cv_results
            
            # Extract summary statistics for table
            stats = cv_results['statistics']
            
            # Add to summary for Standard DT
            summary_data.append({
                'Dataset': dataset_name,
                'Method': 'Standard DT',
                'Alpha': 'NA',
                'n_runs': 'NA',
                'Accuracy': f"{stats['standard']['accuracy_mean']:.4f}",
                'Accuracy_Std': f"{stats['standard']['accuracy_std']:.4f}",
                'AUC': f"{stats['standard']['auc_mean']:.4f}",
                'AUC_Std': f"{stats['standard']['auc_std']:.4f}"
            })
            
            # Add to summary for Soft Split DT
            summary_data.append({
                'Dataset': dataset_name,
                'Method': 'Soft Split DT',
                'Alpha': alpha,
                'n_runs': n_runs,
                'Accuracy': f"{stats['soft']['accuracy_mean']:.4f}",
                'Accuracy_Std': f"{stats['soft']['accuracy_std']:.4f}",
                'AUC': f"{stats['soft']['auc_mean']:.4f}",
                'AUC_Std': f"{stats['soft']['auc_std']:.4f}"
            })
            
            logger.info(f"\n✓ Completed cross-validation for {dataset_name}")
            
        except Exception as e:
            logger.error(f"Error processing {dataset_name}: {str(e)}")
            import traceback
            traceback.print_exc()
            continue
    
    # Create final summary DataFrame
    summary_df = pd.DataFrame(summary_data)
    
    logger.info("\n" + "="*80)
    logger.info("FINAL SUMMARY - ALL DATASETS")
    logger.info("="*80)
    logger.info("\n" + summary_df.to_string(index=False))
    
    # Save results
    results_dir = './results'
    os.makedirs(results_dir, exist_ok=True)
    
    # Save summary table
    summary_path = os.path.join(results_dir, 'cross_validation_summary.csv')
    summary_df.to_csv(summary_path, index=False)
    logger.info(f"\n✓ Saved summary to {summary_path}")
    
    # Save detailed results
    for dataset_name, results in all_results.items():
        # Save fold details
        fold_details_path = os.path.join(results_dir, f'{dataset_name}_fold_details.csv')
        results['fold_details'].to_csv(fold_details_path, index=False)
        logger.info(f"✓ Saved fold details for {dataset_name} to {fold_details_path}")
    
    return {
        'summary': summary_df,
        'all_results': all_results,
        'config': {
            'n_splits': n_splits,
            'n_repeats': n_repeats,
            'alpha': alpha,
            'n_runs': n_runs,
            'random_state': random_state
        }
    }

def run_sensitivity_analysis(dataset_path, target_column='target', 
                             alphas=[0.05, 0.1, 0.15, 0.2],
                             n_runs_list=[50, 100, 150, 200],
                             n_splits=5, n_repeats=2, random_state=42):
    """
    Perform sensitivity analysis on alpha and n_runs parameters.
    
    Parameters
    ----------
    dataset_path : str
        Path to the dataset CSV file
    target_column : str
        Name of the target column
    alphas : list
        List of alpha values to test
    n_runs_list : list
        List of n_runs values to test
    n_splits : int
        Number of folds
    n_repeats : int
        Number of repeats
    random_state : int
        Random state for reproducibility
    
    Returns
    -------
    dict
        Dictionary containing sensitivity analysis results
    """
    logger.info("\n" + "="*80)
    logger.info("SENSITIVITY ANALYSIS")
    logger.info("="*80)
    
    df = pd.read_csv(dataset_path)
    dataset_name = get_dataset_name(dataset_path)
    
    results = {
        'alpha_sensitivity': [],
        'n_runs_sensitivity': []
    }
    
    # Alpha sensitivity (fix n_runs=100)
    logger.info(f"\n--- Testing Alpha Sensitivity (n_runs=100) ---")
    for alpha in alphas:
        logger.info(f"\nTesting alpha={alpha}")
        cv_results = repeated_kfold_cross_validation(
            df=df,
            target_column=target_column,
            n_splits=n_splits,
            n_repeats=n_repeats,
            alpha=alpha,
            n_runs=100,
            random_state=random_state
        )
        
        stats = cv_results['statistics']
        results['alpha_sensitivity'].append({
            'alpha': alpha,
            'n_runs': 100,
            'accuracy_mean': stats['soft']['accuracy_mean'],
            'accuracy_std': stats['soft']['accuracy_std'],
            'auc_mean': stats['soft']['auc_mean'],
            'auc_std': stats['soft']['auc_std']
        })
    
    # n_runs sensitivity (fix alpha=0.1)
    logger.info(f"\n--- Testing n_runs Sensitivity (alpha=0.1) ---")
    for n_runs in n_runs_list:
        logger.info(f"\nTesting n_runs={n_runs}")
        cv_results = repeated_kfold_cross_validation(
            df=df,
            target_column=target_column,
            n_splits=n_splits,
            n_repeats=n_repeats,
            alpha=0.1,
            n_runs=n_runs,
            random_state=random_state
        )
        
        stats = cv_results['statistics']
        results['n_runs_sensitivity'].append({
            'alpha': 0.1,
            'n_runs': n_runs,
            'accuracy_mean': stats['soft']['accuracy_mean'],
            'accuracy_std': stats['soft']['accuracy_std'],
            'auc_mean': stats['soft']['auc_mean'],
            'auc_std': stats['soft']['auc_std']
        })
    
    # Create DataFrames
    alpha_df = pd.DataFrame(results['alpha_sensitivity'])
    n_runs_df = pd.DataFrame(results['n_runs_sensitivity'])
    
    logger.info("\n" + "="*80)
    logger.info("ALPHA SENSITIVITY RESULTS")
    logger.info("="*80)
    logger.info("\n" + alpha_df.to_string(index=False))
    
    logger.info("\n" + "="*80)
    logger.info("N_RUNS SENSITIVITY RESULTS")
    logger.info("="*80)
    logger.info("\n" + n_runs_df.to_string(index=False))
    
    # Save results
    results_dir = './results'
    os.makedirs(results_dir, exist_ok=True)
    
    alpha_path = os.path.join(results_dir, f'{dataset_name}_alpha_sensitivity.csv')
    alpha_df.to_csv(alpha_path, index=False)
    logger.info(f"\n✓ Saved alpha sensitivity results to {alpha_path}")
    
    n_runs_path = os.path.join(results_dir, f'{dataset_name}_nruns_sensitivity.csv')
    n_runs_df.to_csv(n_runs_path, index=False)
    logger.info(f"✓ Saved n_runs sensitivity results to {n_runs_path}")
    
    return {
        'alpha_sensitivity': alpha_df,
        'n_runs_sensitivity': n_runs_df,
        'dataset': dataset_name
    }

if __name__ == "__main__":
    logger.info("Starting Cross-Validation Analysis")
    logger.info("="*80)
    
    # Run cross-validation on all datasets
    results = run_cross_validation_on_all_datasets(
        n_splits=5,      # At least 5 folds (requirement)
        n_repeats=2,     # At least 2 repetitions (requirement)
        alpha=0.1,       # Default soft split parameter
        n_runs=100,      # Default number of runs
        random_state=42
    )
    
    if results:
        logger.info("\n✓ Cross-validation completed successfully!")
        logger.info(f"Results saved to ./results/")
        
        # Run sensitivity analysis on first dataset as example
        processed_dir = './datasets/processed'
        files = [f for f in os.listdir(processed_dir) if f.endswith('_processed.csv')]
        
        if len(files) > 0:
            first_dataset = os.path.join(processed_dir, files[0])
            logger.info(f"\nRunning sensitivity analysis on {files[0]}...")
            
            sensitivity_results = run_sensitivity_analysis(
                dataset_path=first_dataset,
                target_column='target',
                alphas=[0.05, 0.1, 0.15, 0.2],
                n_runs_list=[50, 100, 150],
                n_splits=5,
                n_repeats=2,
                random_state=42
            )
            
            logger.info("\n✓ Sensitivity analysis completed!")
    else:
        logger.error("Cross-validation failed!")