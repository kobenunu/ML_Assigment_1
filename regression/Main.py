import logging
from rich.logging import RichHandler
import pandas as pd
import os
from regression.KFoldCrossValidation import repeated_kfold_cross_validation_regression
from regression.SensitivityAnalysis import run_combined_sensitivity_regression

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(module)s.%(funcName)s:: %(message)s",
    handlers=[RichHandler()]
)
logger = logging.getLogger(__name__)
logging.getLogger("sklearn").setLevel(logging.WARNING)
logging.getLogger("pandas").setLevel(logging.WARNING)

def load_regression_data(dataset_path):
    """Loads a regression dataset."""
    logger.info(f"Loading regression dataset from: {dataset_path}")
    if not os.path.exists(dataset_path):
        logger.error(f"Dataset not found at {dataset_path}")
        return None
    return pd.read_csv(dataset_path)

def run_regression_experiments(dataset_path, target_column):
    """
    Runs cross-validation and sensitivity analysis for a given regression dataset.
    """
    df = load_regression_data(dataset_path)
    if df is None:
        return

    # --- Repeated K-Fold Cross-Validation ---
    repeated_kfold_cross_validation_regression(
        df=df,
        target_column=target_column,
        n_splits=5,
        n_repeats=2,
        alpha=0.1,
        n_runs=100,
        random_state=42
    )

    # --- Sensitivity Analysis ---
    run_combined_sensitivity_regression(
        df=df,
        target_column=target_column,
        alpha_values=[0.05, 0.1, 0.15, 0.2, 0.25],
        n_runs_values=[20, 50, 100, 150],
        test_size=0.2,
        random_state=42
    )

if __name__ == "__main__":
    # This is a placeholder. When datasets are available, this will be updated.
    # Example: run_regression_experiments('path/to/your/dataset.csv', 'your_target_column')
    logger.info("Regression analysis script is ready.")
    logger.info("To run experiments, update the main block with your dataset path and target column.")
    # Example usage (commented out):
    # dataset_files = {
    #     'dataset1': {'path': './regression/datasets/dataset1.csv', 'target': 'target1'},
    #     'dataset2': {'path': './regression/datasets/dataset2.csv', 'target': 'target2'}
    # }
    # for name, info in dataset_files.items():
    #     logger.info(f"--- Running experiments for {name} ---")
    #     run_regression_experiments(info['path'], info['target'])
