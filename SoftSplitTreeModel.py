from sklearn.tree import DecisionTreeClassifier
from traitlets import Float, Int
from joblib import Parallel, delayed
from typing import Any, Literal, Mapping, Sequence
from numpy import ndarray
from numpy.random import RandomState
from pandas import DataFrame
import numpy as np
import pandas as pd
import logging
logger = logging.getLogger(__name__)

class SoftSplitTreeModel(DecisionTreeClassifier):
    def __init__(
        self,
        alpha: Float = 0.0,
        n_samples_predictions: Int = 1,
        *,
        criterion: Literal["gini", "entropy", "log_loss", "gini"] = "gini",
        splitter: Literal["best", "random", "best"] = "best",

        max_depth: None | Int = None,
        min_samples_split: float | int = 2,
        min_samples_leaf: float | int = 1,
        min_weight_fraction_leaf: Float = 0.0,
        max_features: float | None | Literal["auto", "sqrt", "log2"] | int = None,

        random_state: RandomState | None | int = None,
        max_leaf_nodes: None | Int = None,
        min_impurity_decrease: Float = 0.0,
        class_weight: None | Mapping | str | Sequence[Mapping] = None,
        ccp_alpha: float = 0.0,
    ) -> None:
        super(SoftSplitTreeModel, self).__init__(
            criterion=criterion,
            splitter=splitter,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            min_weight_fraction_leaf=min_weight_fraction_leaf,
            max_features=max_features,
            random_state=random_state,
            max_leaf_nodes=max_leaf_nodes,
            min_impurity_decrease=min_impurity_decrease,
            class_weight=class_weight,
            ccp_alpha=ccp_alpha,
        )
        self.alpha = alpha
        self.n_samples_predictions = n_samples_predictions
    def _modified_predict_single(self, x: pd.Series) -> ndarray:  # Helper function for parallelization
        probas = []
        for _ in range(self.n_samples_predictions):
            left = self.tree_.children_left
            right = self.tree_.children_right
            node_idx = 0
            feature_idx = self.tree_.feature[node_idx]
            threshold = self.tree_.threshold[node_idx]
            while feature_idx != -2:
                random_val = np.random.uniform(0, 1)
                if random_val > self.alpha:
                    if x.iloc[feature_idx] <= threshold:
                        node_idx = left[node_idx]
                    else:
                        node_idx = right[node_idx]
                else:
                    if x.iloc[feature_idx] > threshold:
                        node_idx = left[node_idx]
                    else:
                        node_idx = right[node_idx]
                feature_idx = self.tree_.feature[node_idx]
                threshold = self.tree_.threshold[node_idx]

            val = self.tree_.value[node_idx][0]
            probas.append(val)
        temps = np.vstack(probas)
        temps = np.mean(temps, axis=0)
        return temps

    def predict_proba(self, X: Any | DataFrame, check_input: bool = True) -> ndarray | list[ndarray]:
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X) # Convert to DataFrame if it's not already

        n_jobs = -1 # Use all available cores
        classes = Parallel(n_jobs=n_jobs)(
            delayed(self._modified_predict_single)(row) for _, row in X.iterrows()
        )
        return np.vstack(classes)