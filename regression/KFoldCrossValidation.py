import numpy as np
import pandas as pd
from sklearn.model_selection import RepeatedKFold
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import LabelEncoder
from scipy import stats
import logging

from regression.SoftSplitTreeRegressor import SoftSplitTreeRegressor

logger = logging.getLogger(__name__)

def repeated_kfold_cross_validation_regression(df, target_column, n_splits=5, n_repeats=2,
                                            alpha=0.1, n_runs=100, random_state=42):
    """
    Perform repeated K-fold cross-validation for regression tasks.
    """
    logger.info("="*70)
    logger.info("REPEATED K-FOLD CROSS-VALIDATION FOR REGRESSION")
    logger.info("="*70)
    logger.info(f"\nParameters:")
    logger.info(f"  Number of folds (K): {n_splits}")
    logger.info(f"  Number of repeats: {n_repeats}")
    logger.info(f"  Total iterations: {n_splits * n_repeats}")
    logger.info(f"  Soft split alpha: {alpha}")
    logger.info(f"  Soft split n_runs: {n_runs}")
    logger.info(f"  Random state: {random_state}")

    if target_column not in df.columns:
        raise ValueError(f"Target column '{target_column}' not found in dataframe")

    df_processed = df.copy()

    # Encode categorical features
    for col in df_processed.drop(columns=[target_column]).columns:
        if df_processed[col].dtype == 'object' or not np.issubdtype(df_processed[col].dtype, np.number):
            le = LabelEncoder()
            df_processed[col] = le.fit_transform(df_processed[col].astype(str))

    X = df_processed.drop(columns=[target_column]).values
    y = df_processed[target_column].values

    logger.info(f"\nDataset Information:")
    logger.info(f"  Total samples: {len(df)}")
    logger.info(f"  Number of features: {X.shape[1]}")

    rkf = RepeatedKFold(n_splits=n_splits, n_repeats=n_repeats, random_state=random_state)

    standard_scores = {'mse': []}
    soft_scores = {'mse': []}
    fold_details = []

    logger.info("\n" + "="*70)
    logger.info("CROSS-VALIDATION ITERATIONS")
    logger.info("="*70)

    iteration = 0
    for train_index, test_index in rkf.split(X):
        iteration += 1
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]

        logger.info(f"\n--- Iteration {iteration}/{n_splits * n_repeats} ---")
        logger.info(f"Train size: {len(X_train)}, Test size: {len(X_test)}")

        # Standard Decision Tree Regressor
        standard_dtr = DecisionTreeRegressor(random_state=random_state)
        standard_dtr.fit(X_train, y_train)
        y_pred_standard = standard_dtr.predict(X_test)
        mse_standard = mean_squared_error(y_test, y_pred_standard)
        standard_scores['mse'].append(mse_standard)
        logger.info(f"Standard DTR - MSE: {mse_standard:.4f}")

        # Soft Split Decision Tree Regressor
        soft_dtr = SoftSplitTreeRegressor(alpha=alpha, n_runs=n_runs, random_state=random_state)
        soft_dtr.fit(X_train, y_train)
        y_pred_soft = soft_dtr.predict(X_test)
        mse_soft = mean_squared_error(y_test, y_pred_soft)
        soft_scores['mse'].append(mse_soft)
        logger.info(f"Soft Split DTR - MSE: {mse_soft:.4f}")

        fold_details.append({
            'iteration': iteration,
            'standard_mse': mse_standard,
            'soft_mse': mse_soft,
            'mse_diff': mse_soft - mse_standard
        })

    logger.info("\n" + "="*70)
    logger.info("CROSS-VALIDATION RESULTS SUMMARY")
    logger.info("="*70)

    standard_mse_mean = np.mean(standard_scores['mse'])
    standard_mse_std = np.std(standard_scores['mse'])
    soft_mse_mean = np.mean(soft_scores['mse'])
    soft_mse_std = np.std(soft_scores['mse'])

    summary_df = pd.DataFrame({
        'Model': ['Standard DTR', 'Soft Split DTR'],
        'MSE (Mean ± Std)': [
            f"{standard_mse_mean:.4f} ± {standard_mse_std:.4f}",
            f"{soft_mse_mean:.4f} ± {soft_mse_std:.4f}"
        ]
    })
    logger.info("\n" + summary_df.to_string(index=False))

    mse_improvement = soft_mse_mean - standard_mse_mean
    logger.info(f"\n" + "="*70)
    logger.info("COMPARISON")
    logger.info("="*70)
    logger.info(f"MSE change: {mse_improvement:+.4f} (lower is better)")

    t_stat_mse, p_value_mse = stats.ttest_rel(soft_scores['mse'], standard_scores['mse'])
    logger.info(f"\nPaired t-test for MSE:")
    logger.info(f"  t-statistic: {t_stat_mse:.4f}, p-value: {p_value_mse:.4f}")
    if p_value_mse < 0.05:
        logger.info(f"  ✓ MSE difference is statistically significant (p < 0.05)")
    else:
        logger.info(f"  ✗ MSE difference is NOT statistically significant (p ≥ 0.05)")

    fold_details_df = pd.DataFrame(fold_details)
    logger.info(f"\n" + "="*70)
    logger.info("DETAILED FOLD RESULTS")
    logger.info("="*70)
    logger.info("\n" + fold_details_df.to_string(index=False))

    return {
        'summary': summary_df,
        'fold_details': fold_details_df,
        'standard_scores': standard_scores,
        'soft_scores': soft_scores,
        'statistics': {
            'standard': {'mse_mean': standard_mse_mean, 'mse_std': standard_mse_std},
            'soft': {'mse_mean': soft_mse_mean, 'mse_std': soft_mse_std},
            'comparison': {
                'mse_improvement': mse_improvement,
                't_test_mse': {'t_stat': t_stat_mse, 'p_value': p_value_mse}
            }
        },
        'config': {
            'n_splits': n_splits, 'n_repeats': n_repeats, 'alpha': alpha,
            'n_runs': n_runs, 'random_state': random_state
        }
    }