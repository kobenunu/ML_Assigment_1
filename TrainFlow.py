import pickle
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score, classification_report
from sklearn.preprocessing import LabelEncoder
import numpy as np
from SoftSplitTreeModel import SoftSplitDecisionTreeClassifier
from sklearn.tree import DecisionTreeClassifier
import pandas as pd
import logging

logger = logging.getLogger(__name__)

def train_and_evaluate_models(df, target_column, test_size=0.2, alpha=0.1,
                              n_runs=100, random_state=42, save_models=True,
                              model_prefix='model'):
    """
    Train and evaluate both standard and soft split decision tree classifiers.

    Parameters
    ----------
    df : pandas.DataFrame
        The dataset containing features and target
    target_column : str
        Name of the target column in the dataframe
    test_size : float, default=0.2
        Proportion of dataset to use as test set (0.0 to 1.0)
    alpha : float, default=0.1
        Soft split alpha parameter (probability of opposite routing)
    n_runs : int, default=100
        Number of iterations for soft split prediction
    random_state : int, default=42
        Random state for reproducibility
    save_models : bool, default=True
        Whether to save the trained models to disk
    model_prefix : str, default='model'
        Prefix for saved model filenames

    Returns
    -------
    dict
        Dictionary containing models, predictions, and evaluation metrics
    """

    logger.info("="*70)
    logger.info("MODEL TRAINING AND EVALUATION")
    logger.info("="*70)

    # Separate features and target
    if target_column not in df.columns:
        raise ValueError(f"Target column '{target_column}' not found in dataframe")

    # Create a copy to avoid modifying original dataframe
    df_processed = df.copy()

    # Encode categorical features in X
    feature_encoders = {}
    X_df = df_processed.drop(columns=[target_column])

    for col in X_df.columns:
        if X_df[col].dtype == 'object' or not np.issubdtype(X_df[col].dtype, np.number):
            le = LabelEncoder()
            df_processed[col] = le.fit_transform(X_df[col].astype(str))
            feature_encoders[col] = le
            logger.info(f"Encoded feature '{col}': {len(le.classes_)} unique values")

    # Now extract X and y as numeric arrays
    X = df_processed.drop(columns=[target_column]).values
    y = df_processed[target_column].values

    # Store original column names
    feature_names = df_processed.drop(columns=[target_column]).columns.tolist()

    # Encode target if categorical
    label_encoder = None
    if df[target_column].dtype == 'object' or not np.issubdtype(df[target_column].dtype, np.number):
        label_encoder = LabelEncoder()
        y = label_encoder.fit_transform(df[target_column].astype(str))
        logger.info(f"\nTarget classes encoded: {label_encoder.classes_}")

    # Split the data
    train_size = 1.0 - test_size
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    # Dataset information
    logger.info(f"\nDataset Information:")
    logger.info(f"  Total samples: {len(df)}")
    logger.info(f"  Number of features: {X.shape[1]}")
    logger.info(f"  Number of classes: {len(np.unique(y))}")
    logger.info(f"  Train/Test split: {train_size:.0%}/{test_size:.0%}")
    logger.info(f"  Training samples: {X_train.shape[0]}")
    logger.info(f"  Test samples: {X_test.shape[0]}")
    logger.info(f"  Class distribution (train): {dict(zip(*np.unique(y_train, return_counts=True)))}")
    logger.info(f"  Class distribution (test): {dict(zip(*np.unique(y_test, return_counts=True)))}")

    # ========================================================================
    # Train Standard Decision Tree
    # ========================================================================
    logger.info("\n" + "="*70)
    logger.info("1. TRAINING STANDARD DECISION TREE")
    logger.info("="*70)

    standard_dt = DecisionTreeClassifier(random_state=random_state)
    standard_dt.fit(X_train, y_train)

    y_pred_standard = standard_dt.predict(X_test)
    y_proba_standard = standard_dt.predict_proba(X_test)

    accuracy_standard = accuracy_score(y_test, y_pred_standard)
    logger.info(f"\nAccuracy: {accuracy_standard:.4f}")

    # Calculate AUC
    n_classes = len(np.unique(y))
    if n_classes == 2:
        auc_standard = roc_auc_score(y_test, y_proba_standard[:, 1])
        logger.info(f"AUC: {auc_standard:.4f}")
    else:
        from sklearn.preprocessing import label_binarize
        y_test_bin = label_binarize(y_test, classes=np.unique(y))
        auc_standard = roc_auc_score(y_test_bin, y_proba_standard,
                                     multi_class='ovr', average='weighted')
        logger.info(f"AUC (weighted): {auc_standard:.4f}")

    # ========================================================================
    # Train Soft Split Decision Tree
    # ========================================================================
    logger.info("\n" + "="*70)
    logger.info("2. TRAINING SOFT SPLIT DECISION TREE")
    logger.info("="*70)
    logger.info(f"\nParameters: alpha={alpha}, n_runs={n_runs}")

    soft_dt = SoftSplitDecisionTreeClassifier(
        alpha=alpha,
        n_runs=n_runs,
        random_state=random_state
    )
    soft_dt.fit(X_train, y_train)

    logger.info("Running soft split predictions (may take a moment)...")
    y_proba_soft = soft_dt.predict_proba(X_test)
    y_pred_soft = soft_dt.predict(X_test)

    accuracy_soft = accuracy_score(y_test, y_pred_soft)
    logger.info(f"\nAccuracy: {accuracy_soft:.4f}")

    # Calculate AUC
    if n_classes == 2:
        auc_soft = roc_auc_score(y_test, y_proba_soft[:, 1])
        logger.info(f"AUC: {auc_soft:.4f}")
    else:
        auc_soft = roc_auc_score(y_test_bin, y_proba_soft,
                                multi_class='ovr', average='weighted')
        logger.info(f"AUC (weighted): {auc_soft:.4f}")

    # ========================================================================
    # Comparison
    # ========================================================================
    logger.info("\n" + "="*70)
    logger.info("3. COMPARISON SUMMARY")
    logger.info("="*70)

    results_df = pd.DataFrame({
        'Method': ['Standard DT', 'Soft Split DT'],
        'Alpha': ['NA', alpha],
        'n_runs': ['NA', n_runs],
        'Accuracy': [f"{accuracy_standard:.4f}", f"{accuracy_soft:.4f}"],
        'AUC': [f"{auc_standard:.4f}", f"{auc_soft:.4f}"]
    })

    logger.info("\n", results_df.to_string(index=False))

    acc_diff = accuracy_soft - accuracy_standard
    auc_diff = auc_soft - auc_standard

    logger.info(f"\nAccuracy difference: {acc_diff:+.4f}")
    logger.info(f"AUC difference: {auc_diff:+.4f}")

    # ========================================================================
    # Save Models
    # ========================================================================
    if save_models:
        logger.info("\n" + "="*70)
        logger.info("4. SAVING MODELS")
        logger.info("="*70)

        # Save standard model
        standard_filename = f"{model_prefix}_standard_dt.pkl"
        with open(standard_filename, 'wb') as f:
            pickle.dump(standard_dt, f)
        logger.info(f"\nSaved standard model to: {standard_filename}")

        # Save soft split model
        soft_filename = f"{model_prefix}_soft_split_dt.pkl"
        with open(soft_filename, 'wb') as f:
            pickle.dump(soft_dt, f)
        logger.info(f"Saved soft split model to: {soft_filename}")

        # Save label encoder if used
        if label_encoder is not None:
            encoder_filename = f"{model_prefix}_label_encoder.pkl"
            with open(encoder_filename, 'wb') as f:
                pickle.dump(label_encoder, f)
            logger.info(f"Saved label encoder to: {encoder_filename}")

        # Save feature encoders
        if feature_encoders:
            feature_enc_filename = f"{model_prefix}_feature_encoders.pkl"
            with open(feature_enc_filename, 'wb') as f:
                pickle.dump(feature_encoders, f)
            logger.info(f"Saved feature encoders to: {feature_enc_filename}")

        # Save feature names
        features_filename = f"{model_prefix}_feature_names.pkl"
        with open(features_filename, 'wb') as f:
            pickle.dump(feature_names, f)
        logger.info(f"Saved feature names to: {features_filename}")

    # ========================================================================
    # Return results
    # ========================================================================
    return {
        'standard_model': standard_dt,
        'soft_model': soft_dt,
        'label_encoder': label_encoder,
        'feature_encoders': feature_encoders,
        'feature_names': feature_names,
        'X_train': X_train,
        'X_test': X_test,
        'y_train': y_train,
        'y_test': y_test,
        'y_pred_standard': y_pred_standard,
        'y_pred_soft': y_pred_soft,
        'y_proba_standard': y_proba_standard,
        'y_proba_soft': y_proba_soft,
        'accuracy_standard': accuracy_standard,
        'accuracy_soft': accuracy_soft,
        'auc_standard': auc_standard,
        'auc_soft': auc_soft,
        'results_df': results_df
    }