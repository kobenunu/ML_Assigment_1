import logging
from rich.logging import RichHandler
import pandas as pd
import os
from regression.KFoldCrossValidation import repeated_kfold_cross_validation_regression
from regression.SensitivityAnalysis import run_combined_sensitivity_regression
from regression.preprocess import main as preprocess_main

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
    # 1. Preprocess all datasets
    # preprocess_main()

    # 2. Define processed datasets and their target columns
    processed_datasets = {
        'fiat': 'another-fiat-500-dataset-1538-rows_processed',
        'food_delivery': 'food-delivery-time-prediction_processed',
        'laptop_price': 'laptop-price-prediction-dataset_processed',
        'second_hand_car': 'second-hand-used-cars-data-set-linear-regression_processed'
    }

    # 3. Run experiments on each processed dataset
    for name, filename in processed_datasets.items():
        logger.info(f"\n{'='*20} Running Experiments for: {name.upper()} {'='*20}")
        dataset_path = f"./regression/datasets/processed/{filename}.csv"
        run_regression_experiments(dataset_path, target_column='target')
