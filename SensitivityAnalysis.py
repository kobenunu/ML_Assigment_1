"""
Sensitivity Analysis for Soft Split Decision Trees
==================================================

Performs sensitivity analysis on alpha and n_runs parameters
by training models once and testing different parameter values.
"""

import numpy as np
import pandas as pd
import logging
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.preprocessing import LabelEncoder, label_binarize
from SoftSplitTreeModel import SoftSplitDecisionTreeClassifier

logger = logging.getLogger(__name__)


def prepare_data(df, target_column='target', test_size=0.2, random_state=42):
    """Prepare data for sensitivity analysis"""
    df_processed = df.copy()
    
    # Encode categorical features
    feature_encoders = {}
    X_df = df_processed.drop(columns=[target_column])
    
    for col in X_df.columns:
        if X_df[col].dtype == 'object' or not np.issubdtype(X_df[col].dtype, np.number):
            le = LabelEncoder()
            df_processed[col] = le.fit_transform(X_df[col].astype(str))
            feature_encoders[col] = le
    
    X = df_processed.drop(columns=[target_column]).values
    y = df_processed[target_column].values
    
    # Encode target if categorical
    label_encoder = None
    if df[target_column].dtype == 'object' or not np.issubdtype(df[target_column].dtype, np.number):
        label_encoder = LabelEncoder()
        y = label_encoder.fit_transform(df[target_column].astype(str))
    
    n_classes = len(np.unique(y))
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    
    return X_train, X_test, y_train, y_test, n_classes


def run_alpha_sensitivity(df, target_column='target',
                         alpha_values=[0.01, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3],
                         n_runs=100, test_size=0.2, random_state=42):
    """
    Test different alpha values with fixed n_runs.
    
    Parameters
    ----------
    df : pandas.DataFrame
        The dataset
    target_column : str
        Name of target column
    alpha_values : list
        List of alpha values to test
    n_runs : int
        Fixed number of runs for predictions
    test_size : float
        Test set proportion
    random_state : int
        Random seed
        
    Returns
    -------
    pandas.DataFrame
        Results for each alpha value
    """
    logger.info("="*80)
    logger.info("ALPHA SENSITIVITY ANALYSIS")
    logger.info("="*80)
    logger.info(f"Testing alpha values: {alpha_values}")
    logger.info(f"Fixed n_runs: {n_runs}")
    
    X_train, X_test, y_train, y_test, n_classes = prepare_data(
        df, target_column, test_size, random_state
    )
    
    logger.info(f"Train samples: {len(X_train)}, Test samples: {len(X_test)}")
    
    # Train standard model once
    logger.info("\nTraining Standard Decision Tree...")
    standard_dt = DecisionTreeClassifier(random_state=random_state)
    standard_dt.fit(X_train, y_train)
    
    y_pred_standard = standard_dt.predict(X_test)
    y_proba_standard = standard_dt.predict_proba(X_test)
    
    standard_accuracy = accuracy_score(y_test, y_pred_standard)
    
    if n_classes == 2:
        standard_auc = roc_auc_score(y_test, y_proba_standard[:, 1])
    else:
        y_test_bin = label_binarize(y_test, classes=np.unique(y_train))
        standard_auc = roc_auc_score(y_test_bin, y_proba_standard,
                                    multi_class='ovr', average='weighted')
    
    logger.info(f"Standard DT - Accuracy: {standard_accuracy:.4f}, AUC: {standard_auc:.4f}")
    
    # Test different alpha values
    results = []
    
    for alpha in alpha_values:
        logger.info(f"\nTesting alpha = {alpha}")
        
        soft_dt = SoftSplitDecisionTreeClassifier(
            alpha=alpha,
            n_runs=n_runs,
            random_state=random_state
        )
        soft_dt.fit(X_train, y_train)
        
        y_proba_soft = soft_dt.predict_proba(X_test)
        y_pred_soft = np.argmax(y_proba_soft, axis=1)
        
        soft_accuracy = accuracy_score(y_test, y_pred_soft)
        
        if n_classes == 2:
            soft_auc = roc_auc_score(y_test, y_proba_soft[:, 1])
        else:
            soft_auc = roc_auc_score(y_test_bin, y_proba_soft,
                                    multi_class='ovr', average='weighted')
        
        logger.info(f"Soft Split DT - Accuracy: {soft_accuracy:.4f}, AUC: {soft_auc:.4f}")
        
        results.append({
            'alpha': alpha,
            'n_runs': n_runs,
            'standard_accuracy': standard_accuracy,
            'standard_auc': standard_auc,
            'soft_accuracy': soft_accuracy,
            'soft_auc': soft_auc,
            'accuracy_improvement': soft_accuracy - standard_accuracy,
            'auc_improvement': soft_auc - standard_auc
        })
    
    results_df = pd.DataFrame(results)
    
    logger.info("\n" + "="*80)
    logger.info("ALPHA SENSITIVITY RESULTS")
    logger.info("="*80)
    logger.info("\n" + results_df.to_string(index=False))
    
    return results_df


def run_n_runs_sensitivity(df, target_column='target',
                          n_runs_values=[10, 25, 50, 75, 100, 150, 200],
                          alpha=0.1, test_size=0.2, random_state=42):
    """
    Test different n_runs values with fixed alpha.
    
    Parameters
    ----------
    df : pandas.DataFrame
        The dataset
    target_column : str
        Name of target column
    n_runs_values : list
        List of n_runs values to test
    alpha : float
        Fixed alpha value
    test_size : float
        Test set proportion
    random_state : int
        Random seed
        
    Returns
    -------
    pandas.DataFrame
        Results for each n_runs value
    """
    logger.info("="*80)
    logger.info("N_RUNS SENSITIVITY ANALYSIS")
    logger.info("="*80)
    logger.info(f"Testing n_runs values: {n_runs_values}")
    logger.info(f"Fixed alpha: {alpha}")
    
    X_train, X_test, y_train, y_test, n_classes = prepare_data(
        df, target_column, test_size, random_state
    )
    
    logger.info(f"Train samples: {len(X_train)}, Test samples: {len(X_test)}")
    
    # Train standard model once
    logger.info("\nTraining Standard Decision Tree...")
    standard_dt = DecisionTreeClassifier(random_state=random_state)
    standard_dt.fit(X_train, y_train)
    
    y_pred_standard = standard_dt.predict(X_test)
    y_proba_standard = standard_dt.predict_proba(X_test)
    
    standard_accuracy = accuracy_score(y_test, y_pred_standard)
    
    if n_classes == 2:
        standard_auc = roc_auc_score(y_test, y_proba_standard[:, 1])
    else:
        y_test_bin = label_binarize(y_test, classes=np.unique(y_train))
        standard_auc = roc_auc_score(y_test_bin, y_proba_standard,
                                    multi_class='ovr', average='weighted')
    
    logger.info(f"Standard DT - Accuracy: {standard_accuracy:.4f}, AUC: {standard_auc:.4f}")
    
    # Test different n_runs values
    results = []
    
    for n_runs in n_runs_values:
        logger.info(f"\nTesting n_runs = {n_runs}")
        
        soft_dt = SoftSplitDecisionTreeClassifier(
            alpha=alpha,
            n_runs=n_runs,
            random_state=random_state
        )
        soft_dt.fit(X_train, y_train)
        
        y_proba_soft = soft_dt.predict_proba(X_test)
        y_pred_soft = soft_dt.predict(X_test)
        
        soft_accuracy = accuracy_score(y_test, y_pred_soft)
        
        if n_classes == 2:
            soft_auc = roc_auc_score(y_test, y_proba_soft[:, 1])
        else:
            soft_auc = roc_auc_score(y_test_bin, y_proba_soft,
                                    multi_class='ovr', average='weighted')
        
        logger.info(f"Soft Split DT - Accuracy: {soft_accuracy:.4f}, AUC: {soft_auc:.4f}")
        
        results.append({
            'alpha': alpha,
            'n_runs': n_runs,
            'standard_accuracy': standard_accuracy,
            'standard_auc': standard_auc,
            'soft_accuracy': soft_accuracy,
            'soft_auc': soft_auc,
            'accuracy_improvement': soft_accuracy - standard_accuracy,
            'auc_improvement': soft_auc - standard_auc
        })
    
    results_df = pd.DataFrame(results)
    
    logger.info("\n" + "="*80)
    logger.info("N_RUNS SENSITIVITY RESULTS")
    logger.info("="*80)
    logger.info("\n" + results_df.to_string(index=False))
    
    return results_df


def run_combined_sensitivity(df, target_column='target',
                             alpha_values=[0.05, 0.1, 0.15, 0.2],
                             n_runs_values=[50, 100, 150],
                             test_size=0.2, random_state=42):
    """
    Test all combinations of alpha and n_runs (for heatmap visualization).
    
    Parameters
    ----------
    df : pandas.DataFrame
        The dataset
    target_column : str
        Name of target column
    alpha_values : list
        List of alpha values to test
    n_runs_values : list
        List of n_runs values to test
    test_size : float
        Test set proportion
    random_state : int
        Random seed
        
    Returns
    -------
    pandas.DataFrame
        Results for all combinations
    """
    logger.info("="*80)
    logger.info("COMBINED SENSITIVITY ANALYSIS")
    logger.info("="*80)
    logger.info(f"Testing alpha values: {alpha_values}")
    logger.info(f"Testing n_runs values: {n_runs_values}")
    logger.info(f"Total combinations: {len(alpha_values) * len(n_runs_values)}")
    
    X_train, X_test, y_train, y_test, n_classes = prepare_data(
        df, target_column, test_size, random_state
    )
    
    logger.info(f"Train samples: {len(X_train)}, Test samples: {len(X_test)}")
    
    # Train standard model once
    logger.info("\nTraining Standard Decision Tree...")
    standard_dt = DecisionTreeClassifier(random_state=random_state)
    standard_dt.fit(X_train, y_train)
    
    y_pred_standard = standard_dt.predict(X_test)
    y_proba_standard = standard_dt.predict_proba(X_test)
    
    standard_accuracy = accuracy_score(y_test, y_pred_standard)
    
    if n_classes == 2:
        standard_auc = roc_auc_score(y_test, y_proba_standard[:, 1])
    else:
        y_test_bin = label_binarize(y_test, classes=np.unique(y_train))
        standard_auc = roc_auc_score(y_test_bin, y_proba_standard,
                                    multi_class='ovr', average='weighted')
    
    logger.info(f"Standard DT - Accuracy: {standard_accuracy:.4f}, AUC: {standard_auc:.4f}")
    
    # Test all combinations
    results = []
    total = len(alpha_values) * len(n_runs_values)
    current = 0
    
    for alpha in alpha_values:
        for n_runs in n_runs_values:
            current += 1
            logger.info(f"\n[{current}/{total}] Testing alpha={alpha}, n_runs={n_runs}")
            
            soft_dt = SoftSplitDecisionTreeClassifier(
                alpha=alpha,
                n_runs=n_runs,
                random_state=random_state
            )
            soft_dt.fit(X_train, y_train)
            
            y_proba_soft = soft_dt.predict_proba(X_test)
            y_pred_soft = soft_dt.predict(X_test)
            
            soft_accuracy = accuracy_score(y_test, y_pred_soft)
            
            if n_classes == 2:
                soft_auc = roc_auc_score(y_test, y_proba_soft[:, 1])
            else:
                soft_auc = roc_auc_score(y_test_bin, y_proba_soft,
                                        multi_class='ovr', average='weighted')
            
            logger.info(f"Soft Split DT - Accuracy: {soft_accuracy:.4f}, AUC: {soft_auc:.4f}")
            
            results.append({
                'alpha': alpha,
                'n_runs': n_runs,
                'standard_accuracy': standard_accuracy,
                'standard_auc': standard_auc,
                'soft_accuracy': soft_accuracy,
                'soft_auc': soft_auc,
                'accuracy_improvement': soft_accuracy - standard_accuracy,
                'auc_improvement': soft_auc - standard_auc
            })
    
    results_df = pd.DataFrame(results)
    
    logger.info("\n" + "="*80)
    logger.info("COMBINED SENSITIVITY RESULTS")
    logger.info("="*80)
    logger.info("\n" + results_df.to_string(index=False))
    
    return results_df