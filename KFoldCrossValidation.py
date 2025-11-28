import numpy as np
import pandas as pd
from sklearn.model_selection import RepeatedKFold
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.preprocessing import LabelEncoder, label_binarize
from scipy import stats
import logging

from SoftSplitTreeModel import SoftSplitDecisionTreeClassifier

logger = logging.getLogger(__name__)

def repeated_kfold_cross_validation(df, target_column, n_splits=5, n_repeats=2,
                                    alpha=0.1, n_runs=100, random_state=42):
    """
    Perform repeated K-fold cross-validation to evaluate both standard and soft split decision trees.
    
    Parameters
    ----------
    df : pandas.DataFrame
        The dataset containing features and target
    target_column : str
        Name of the target column in the dataframe
    n_splits : int, default=5
        Number of folds for K-fold cross-validation
    n_repeats : int, default=2
        Number of times to repeat the K-fold cross-validation
    alpha : float, default=0.1
        Soft split alpha parameter (probability of opposite routing)
    n_runs : int, default=100
        Number of iterations for soft split prediction
    random_state : int, default=42
        Random state for reproducibility
        
    Returns
    -------
    dict
        Dictionary containing all cross-validation results including:
        - Mean and std of accuracy and AUC for both models
        - Detailed results for each fold
        - Statistical comparison between models
    """
    logger.info("="*70)
    logger.info("REPEATED K-FOLD CROSS-VALIDATION")
    logger.info("="*70)
    logger.info(f"\nParameters:")
    logger.info(f"  Number of folds (K): {n_splits}")
    logger.info(f"  Number of repeats: {n_repeats}")
    logger.info(f"  Total iterations: {n_splits * n_repeats}")
    logger.info(f"  Soft split alpha: {alpha}")
    logger.info(f"  Soft split n_runs: {n_runs}")
    logger.info(f"  Random state: {random_state}")
    
    # Prepare data
    if target_column not in df.columns:
        raise ValueError(f"Target column '{target_column}' not found in dataframe")
    
    df_processed = df.copy()
    
    # Encode categorical features
    feature_encoders = {}
    X_df = df_processed.drop(columns=[target_column])
    
    for col in X_df.columns:
        if X_df[col].dtype == 'object' or not np.issubdtype(X_df[col].dtype, np.number):
            le = LabelEncoder()
            df_processed[col] = le.fit_transform(X_df[col].astype(str))
            feature_encoders[col] = le
    
    # Extract features and target
    X = df_processed.drop(columns=[target_column]).values
    y = df_processed[target_column].values
    
    # Encode target if categorical
    label_encoder = None
    if df[target_column].dtype == 'object' or not np.issubdtype(df[target_column].dtype, np.number):
        label_encoder = LabelEncoder()
        y = label_encoder.fit_transform(df[target_column].astype(str))
        logger.info(f"\nTarget classes: {label_encoder.classes_}")
    
    n_classes = len(np.unique(y))
    logger.info(f"\nDataset Information:")
    logger.info(f"  Total samples: {len(df)}")
    logger.info(f"  Number of features: {X.shape[1]}")
    logger.info(f"  Number of classes: {n_classes}")
    logger.info(f"  Class distribution: {dict(zip(*np.unique(y, return_counts=True)))}")
    
    # Initialize repeated K-fold
    rkf = RepeatedKFold(n_splits=n_splits, n_repeats=n_repeats, random_state=random_state)
    
    # Storage for results
    standard_scores = {
        'accuracy': [],
        'auc': []
    }
    soft_scores = {
        'accuracy': [],
        'auc': []
    }
    
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
        
        # ========================================================================
        # Standard Decision Tree
        # ========================================================================
        standard_dt = DecisionTreeClassifier(random_state=random_state)
        standard_dt.fit(X_train, y_train)
        
        y_pred_standard = standard_dt.predict(X_test)
        y_proba_standard = standard_dt.predict_proba(X_test)
        
        acc_standard = accuracy_score(y_test, y_pred_standard)
        
        # Calculate AUC
        if n_classes == 2:
            auc_standard = roc_auc_score(y_test, y_proba_standard[:, 1])
        else:
            y_test_bin = label_binarize(y_test, classes=np.unique(y))
            auc_standard = roc_auc_score(y_test_bin, y_proba_standard,
                                        multi_class='ovr', average='weighted')
        
        standard_scores['accuracy'].append(acc_standard)
        standard_scores['auc'].append(auc_standard)
        
        logger.info(f"Standard DT - Accuracy: {acc_standard:.4f}, AUC: {auc_standard:.4f}")
        
        # ========================================================================
        # Soft Split Decision Tree
        # ========================================================================
        soft_dt = SoftSplitDecisionTreeClassifier(
            alpha=alpha,
            n_runs=n_runs,
            random_state=random_state
        )
        soft_dt.fit(X_train, y_train)
        
        y_proba_soft = soft_dt.predict_proba(X_test)
        y_pred_soft = soft_dt.predict(X_test)
        
        acc_soft = accuracy_score(y_test, y_pred_soft)
        
        # Calculate AUC
        if n_classes == 2:
            auc_soft = roc_auc_score(y_test, y_proba_soft[:, 1])
        else:
            auc_soft = roc_auc_score(y_test_bin, y_proba_soft,
                                    multi_class='ovr', average='weighted')
        
        soft_scores['accuracy'].append(acc_soft)
        soft_scores['auc'].append(auc_soft)
        
        logger.info(f"Soft Split DT - Accuracy: {acc_soft:.4f}, AUC: {auc_soft:.4f}")
        
        # Store fold details
        fold_details.append({
            'iteration': iteration,
            'standard_accuracy': acc_standard,
            'standard_auc': auc_standard,
            'soft_accuracy': acc_soft,
            'soft_auc': auc_soft,
            'accuracy_diff': acc_soft - acc_standard,
            'auc_diff': auc_soft - auc_standard
        })
    
    # ========================================================================
    # Aggregate Results
    # ========================================================================
    logger.info("\n" + "="*70)
    logger.info("CROSS-VALIDATION RESULTS SUMMARY")
    logger.info("="*70)
    
    # Calculate statistics
    standard_acc_mean = np.mean(standard_scores['accuracy'])
    standard_acc_std = np.std(standard_scores['accuracy'])
    standard_auc_mean = np.mean(standard_scores['auc'])
    standard_auc_std = np.std(standard_scores['auc'])
    
    soft_acc_mean = np.mean(soft_scores['accuracy'])
    soft_acc_std = np.std(soft_scores['accuracy'])
    soft_auc_mean = np.mean(soft_scores['auc'])
    soft_auc_std = np.std(soft_scores['auc'])
    
    # Create summary DataFrame
    summary_df = pd.DataFrame({
        'Model': ['Standard DT', 'Soft Split DT'],
        'Accuracy (Mean ± Std)': [
            f"{standard_acc_mean:.4f} ± {standard_acc_std:.4f}",
            f"{soft_acc_mean:.4f} ± {soft_acc_std:.4f}"
        ],
        'AUC (Mean ± Std)': [
            f"{standard_auc_mean:.4f} ± {standard_auc_std:.4f}",
            f"{soft_auc_mean:.4f} ± {soft_auc_std:.4f}"
        ]
    })
    
    logger.info("\n" + summary_df.to_string(index=False))
    
    # Statistical comparison
    acc_improvement = soft_acc_mean - standard_acc_mean
    auc_improvement = soft_auc_mean - standard_auc_mean
    
    logger.info(f"\n" + "="*70)
    logger.info("COMPARISON")
    logger.info("="*70)
    logger.info(f"Accuracy improvement: {acc_improvement:+.4f}")
    logger.info(f"AUC improvement: {auc_improvement:+.4f}")
    
    # Perform paired t-test
    t_stat_acc, p_value_acc = stats.ttest_rel(soft_scores['accuracy'], 
                                               standard_scores['accuracy'])
    t_stat_auc, p_value_auc = stats.ttest_rel(soft_scores['auc'],
                                               standard_scores['auc'])
    
    logger.info(f"\nPaired t-test results:")
    logger.info(f"  Accuracy - t-statistic: {t_stat_acc:.4f}, p-value: {p_value_acc:.4f}")
    logger.info(f"  AUC - t-statistic: {t_stat_auc:.4f}, p-value: {p_value_auc:.4f}")
    
    if p_value_acc < 0.05:
        logger.info(f"  ✓ Accuracy difference is statistically significant (p < 0.05)")
    else:
        logger.info(f"  ✗ Accuracy difference is NOT statistically significant (p ≥ 0.05)")
    
    if p_value_auc < 0.05:
        logger.info(f"  ✓ AUC difference is statistically significant (p < 0.05)")
    else:
        logger.info(f"  ✗ AUC difference is NOT statistically significant (p ≥ 0.05)")
    
    # Create detailed results DataFrame
    fold_details_df = pd.DataFrame(fold_details)
    
    logger.info(f"\n" + "="*70)
    logger.info("DETAILED FOLD RESULTS")
    logger.info("="*70)
    logger.info("\n" + fold_details_df.to_string(index=False))
    
    # Return comprehensive results
    return {
        'summary': summary_df,
        'fold_details': fold_details_df,
        'standard_scores': standard_scores,
        'soft_scores': soft_scores,
        'statistics': {
            'standard': {
                'accuracy_mean': standard_acc_mean,
                'accuracy_std': standard_acc_std,
                'auc_mean': standard_auc_mean,
                'auc_std': standard_auc_std
            },
            'soft': {
                'accuracy_mean': soft_acc_mean,
                'accuracy_std': soft_acc_std,
                'auc_mean': soft_auc_mean,
                'auc_std': soft_auc_std
            },
            'comparison': {
                'accuracy_improvement': acc_improvement,
                'auc_improvement': auc_improvement,
                't_test_accuracy': {'t_stat': t_stat_acc, 'p_value': p_value_acc},
                't_test_auc': {'t_stat': t_stat_auc, 'p_value': p_value_auc}
            }
        },
        'config': {
            'n_splits': n_splits,
            'n_repeats': n_repeats,
            'alpha': alpha,
            'n_runs': n_runs,
            'random_state': random_state
        }
    }