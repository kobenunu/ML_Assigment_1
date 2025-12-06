from sklearn.tree import DecisionTreeRegressor
import numpy as np
import logging

logger = logging.getLogger(__name__)

class SoftSplitTreeRegressor(DecisionTreeRegressor):
    """
    A Decision Tree Regressor that employs 'soft splits' during inference.

    During prediction, at each split node, a sample is routed to the opposite
    direction of the split condition with probability 'alpha', and according
    to the condition with probability '1 - alpha'.
    The final prediction is the average prediction over 'n_runs' runs.
    """

    def __init__(self, alpha=0.1, n_runs=100, **kwargs):
        """
        Initializes the SoftSplitTreeRegressor.

        Args:
            alpha (float): Probability of routing a sample in the opposite
                           direction of the split condition (0.0 <= alpha <= 1.0).
            n_runs (int): Number of times to run the soft-split prediction for
                          each sample.
            **kwargs: Arguments passed to the base DecisionTreeRegressor.
        """
        if not 0.0 <= alpha <= 1.0:
            raise ValueError("alpha must be between 0.0 and 1.0.")
        if not isinstance(n_runs, int) or n_runs <= 0:
            raise ValueError("n_runs must be a positive integer.")

        super().__init__(**kwargs)
        self.alpha = alpha
        self.n_runs = n_runs

    def _get_leaf_prediction(self, node_id):
        """
        Retrieves the prediction value at a specific leaf node.
        """
        # For a regressor, the 'value' is the predicted value for the leaf.
        # It has shape (1, 1, 1), so we extract the float value.
        return self.tree_.value[node_id][0, 0]

    def _soft_split_predict_single_run(self, X):
        """
        Performs a single run of the soft-split prediction for all samples in X.
        """
        n_samples = X.shape[0]
        predictions = np.zeros(n_samples)

        tree = self.tree_
        feature = tree.feature
        threshold = tree.threshold
        children_left = tree.children_left
        children_right = tree.children_right

        for i in range(n_samples):
            current_node = 0
            sample = X[i, :]

            while feature[current_node] != -2:  # -2 indicates a leaf node
                feat = feature[current_node]
                thresh = threshold[current_node]

                is_left_path = sample[feat] <= thresh
                random_draw = np.random.rand()
                take_opposite_path = random_draw < self.alpha

                if (is_left_path and not take_opposite_path) or \
                   (not is_left_path and take_opposite_path):
                    current_node = children_left[current_node]
                else:
                    current_node = children_right[current_node]

            predictions[i] = self._get_leaf_prediction(current_node)

        return predictions

    def predict(self, X, check_input=True):
        """
        Predicts regression values using soft splits averaged over n_runs.

        Args:
            X (array-like of shape (n_samples, n_features)): The input samples.
            check_input (bool): Unused, for compatibility with sklearn API.

        Returns:
            array-like of shape (n_samples,): The average predicted values.
        """
        # Input validation is handled by the base class fit method.
        # We can call the internal predict method of the base class to ensure
        # the tree is fitted and X is valid.
        super().predict(X, check_input=True)

        total_predictions = np.zeros(X.shape[0])

        for _ in range(self.n_runs):
            total_predictions += self._soft_split_predict_single_run(X)

        final_predictions = total_predictions / self.n_runs

        return final_predictions