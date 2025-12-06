import numpy as np
import pandas as pd
import logging
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import LabelEncoder
from .SoftSplitTreeRegressor import SoftSplitTreeRegressor

logger = logging.getLogger(__name__)

def prepare_data_regression(df, target_column='target', test_size=0.2, random_state=42):
    """Prepare data for regression sensitivity analysis."""
    df_processed = df.copy()

    # Encode categorical features
    for col in df_processed.drop(columns=[target_column]).columns:
        if df_processed[col].dtype == 'object' or not np.issubdtype(df_processed[col].dtype, np.number):
            le = LabelEncoder()
            df_processed[col] = le.fit_transform(df_processed[col].astype(str))

    X = df_processed.drop(columns=[target_column]).values
    y = df_processed[target_column].values

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )

    return X_train, X_test, y_train, y_test

def run_combined_sensitivity_regression(df, target_column='target',
                                        alpha_values=[0.05, 0.1, 0.15, 0.2],
                                        n_runs_values=[50, 100, 150],
                                        test_size=0.2, random_state=42):
    """
    Test all combinations of alpha and n_runs for regression.
    """
    logger.info("="*80)
    logger.info("COMBINED SENSITIVITY ANALYSIS FOR REGRESSION")
    logger.info("="*80)
    logger.info(f"Testing alpha values: {alpha_values}")
    logger.info(f"Testing n_runs values: {n_runs_values}")
    logger.info(f"Total combinations: {len(alpha_values) * len(n_runs_values)}")

    X_train, X_test, y_train, y_test = prepare_data_regression(
        df, target_column, test_size, random_state
    )

    logger.info(f"Train samples: {len(X_train)}, Test samples: {len(X_test)}")

    # Train standard model once
    logger.info("\nTraining Standard Decision Tree Regressor...")
    standard_dtr = DecisionTreeRegressor(random_state=random_state)
    standard_dtr.fit(X_train, y_train)
    y_pred_standard = standard_dtr.predict(X_test)
    standard_mse = mean_squared_error(y_test, y_pred_standard)
    logger.info(f"Standard DTR - MSE: {standard_mse:.4f}")

    # Test all combinations
    results = []
    total = len(alpha_values) * len(n_runs_values)
    current = 0

    for alpha in alpha_values:
        for n_runs in n_runs_values:
            current += 1
            logger.info(f"\n[{current}/{total}] Testing alpha={alpha}, n_runs={n_runs}")

            soft_dtr = SoftSplitTreeRegressor(
                alpha=alpha,
                n_runs=n_runs,
                random_state=random_state
            )
            soft_dtr.fit(X_train, y_train)
            y_pred_soft = soft_dtr.predict(X_test)
            soft_mse = mean_squared_error(y_test, y_pred_soft)
            logger.info(f"Soft Split DTR - MSE: {soft_mse:.4f}")

            results.append({
                'alpha': alpha,
                'n_runs': n_runs,
                'standard_mse': standard_mse,
                'soft_mse': soft_mse,
                'mse_change': soft_mse - standard_mse
            })

    results_df = pd.DataFrame(results)

    logger.info("\n" + "="*80)
    logger.info("COMBINED SENSITIVITY RESULTS FOR REGRESSION")
    logger.info("="*80)
    logger.info("\n" + results_df.to_string(index=False))

    return results_df