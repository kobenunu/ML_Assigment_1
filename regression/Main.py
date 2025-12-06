import logging
from rich.logging import RichHandler
import pandas as pd
import os
from .KFoldCrossValidation import repeated_kfold_cross_validation_regression
from .SensitivityAnalysis import run_combined_sensitivity_regression
from .preprocess import main as preprocess_main
from .GenerateVisualizations import generate_visualizations

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
        return None, None

    # --- Repeated K-Fold Cross-Validation ---
    cv_results = repeated_kfold_cross_validation_regression(
        df=df,
        target_column=target_column,
        n_splits=5,
        n_repeats=2,
        alpha=0.1,
        n_runs=100,
        random_state=42
    )

    # --- Sensitivity Analysis ---
    sensitivity_results_df = run_combined_sensitivity_regression(
        df=df,
        target_column=target_column,
        alpha_values=[0.05, 0.1, 0.15, 0.2, 0.25],
        n_runs_values=[20, 50, 100, 150],
        test_size=0.2,
        random_state=42
    )
    
    return cv_results, sensitivity_results_df

if __name__ == "__main__":
    # 1. Preprocess all datasets
    # preprocess_main()

    # 2. Define processed datasets and their target columns
    processed_datasets = {
        'fiat': 'another-fiat-500-dataset-1538-rows_processed',
        'food_delivery': 'food-delivery-time-prediction_processed',
        'laptop_price': 'laptop-price-prediction-dataset_processed',
        'second_hand_car': 'second-hand-used-cars-data-set-linear-regression_processed',
        'bank_churn': 'bank_churn_processed'
    }

    # 3. Run experiments and collect results
    all_cv_results = []
    all_sensitivity_results = []

    for name, filename in processed_datasets.items():
        logger.info(f"\n{'='*20} Running Experiments for: {name.upper()} {'='*20}")
        dataset_path = f"./regression/datasets/processed/{filename}.csv"
        
        cv_res, sens_res = run_regression_experiments(dataset_path, target_column='target')

        if cv_res:
            standard_mse = cv_res['statistics']['standard']['mse_mean']
            soft_mse = cv_res['statistics']['soft']['mse_mean']
            all_cv_results.append({'dataset': name, 'model': 'Standard DTR', 'mse_mean': standard_mse})
            all_cv_results.append({'dataset': name, 'model': 'Soft Split DTR', 'mse_mean': soft_mse})

        if sens_res is not None:
            sens_res['dataset'] = name
            all_sensitivity_results.append(sens_res)

    # 4. Combine results into DataFrames
    cv_summary_df = pd.DataFrame(all_cv_results)
    sensitivity_summary_df = pd.concat(all_sensitivity_results, ignore_index=True) if all_sensitivity_results else pd.DataFrame()

    # 5. Generate visualizations
    generate_visualizations(cv_summary_df, sensitivity_summary_df)
