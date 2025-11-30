from sklearn.tree import DecisionTreeClassifier
import numpy as np
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
        self.n_runs = n_runs

    def _get_leaf_prediction(self, node_id):
        """
        Retrieves the probability distribution at a specific leaf node.

        This method accesses the internal structure of the trained tree.
        """
        # The 'value' array contains the counts of each class for the samples
        # that reached this node. The shape is (1, 1, n_classes).
        counts = self.tree_.value[node_id][0]
        # Normalize the counts to get probabilities
        total_samples = counts.sum()
        if total_samples == 0:
            return np.full(counts.shape, 1.0 / counts.size) # Handle empty node case
        return counts / total_samples

    def _soft_split_predict_proba_single_run(self, X):
        """
        Performs a single run of the soft-split prediction for all samples in X.
        """
        n_samples = X.shape[0]
        n_classes = self.n_classes_

        # Initialize the probability matrix for this run
        probas = np.zeros((n_samples, n_classes))

        # Access internal tree structure
        tree = self.tree_
        feature = tree.feature
        threshold = tree.threshold
        children_left = tree.children_left
        children_right = tree.children_right

        for i in range(n_samples):
            current_node = 0  # Start at the root node
            sample = X[i, :]

            # Route the sample down the tree until a leaf is reached
            while children_left[current_node] != children_right[current_node]:
                # 1. Determine the 'correct' route based on the condition
                feat = feature[current_node]
                thresh = threshold[current_node]

                # Check the actual split condition: X[i, feat] <= threshold
                is_left_path = sample[feat] <= thresh

                # 2. Randomly decide on the routing direction
                # Generate a uniform random number between 0 and 1
                random_draw = np.random.rand()

                # Should the sample follow the opposite path?
                # Opposite path probability: alpha
                # Correct path probability: 1 - alpha
                take_opposite_path = random_draw < self.alpha

                # 3. Choose the next node
                if (is_left_path and not take_opposite_path) or \
                   (not is_left_path and take_opposite_path):
                    # Route to the left child
                    current_node = children_left[current_node]
                else:
                    # Route to the right child
                    current_node = children_right[current_node]

            # Reached a leaf node, get the prediction
            probas[i, :] = self._get_leaf_prediction(current_node)

        return probas

    def predict_proba(self, X):
        """
        Predicts class probabilities using soft splits averaged over n_runs.

        Args:
            X (array-like of shape (n_samples, n_features)): The input samples.

        Returns:
            array-like of shape (n_samples, n_classes): The average
            predicted class probabilities.
        """
        # Initialize the array to accumulate probabilities
        total_probas = np.zeros((X.shape[0], self.n_classes_))

        # Run the soft-split prediction n_runs times
        for _ in range(self.n_runs):
            total_probas += self._soft_split_predict_proba_single_run(X)

        # Average the results
        final_probas = total_probas / self.n_runs

        return final_probas