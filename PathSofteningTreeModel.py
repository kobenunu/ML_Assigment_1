from sklearn.tree import DecisionTreeClassifier
import numpy as np
import logging
logger = logging.getLogger(__name__)
class PathSofteningTreeModel(DecisionTreeClassifier):
    def __init__(self, alpha=0.1, **kwargs):
        if not 0.0 < alpha < 1.0:
            raise ValueError("alpha must be between 0.0 and 1.0.")
        super().__init__(**kwargs)
        self.alpha = alpha

    def predict_proba(self, X: np.ndarray):
        """
        Predict class probabilities using path softening.
        The prediction is a weighted average of all leaves, where weights
        are the probability of the sample reaching that leaf given alpha.
        """        
        tree = self.tree_
        
        n_samples = X.shape[0]
        n_classes = self.n_classes_
        
        # This will accumulate the weighted class probabilities for all samples
        final_probs = np.zeros((n_samples, n_classes))
        
        # We start at the root (node 0) with a probability (weight) of 1.0 for all samples
        initial_weights = np.ones(n_samples)
        
        # Stack for traversal: (node_id, current_sample_weights)
        # We use a stack to simulate recursion without hitting recursion limits
        stack = [(0, initial_weights)]
        
        while stack:
            node_id, weights = stack.pop()
            
            # If weights are all effectively zero, stop processing this branch
            if np.sum(weights) < 1e-9:
                continue

            feature_idx = tree.feature[node_id]
            is_leaf = feature_idx == -2
            
            if is_leaf:
                # Get class counts probabilities for this leaf
                node_counts = tree.value[node_id][0]
                leaf_class_probs = node_counts / np.sum(node_counts)
                
                # Add this leaf's contribution to the final total
                # Contribution = Weight of reaching leaf * Class Probs at leaf
                final_probs += weights[:, np.newaxis] * leaf_class_probs
                
            else:
                # It's a split node 
                #             
                threshold = tree.threshold[node_id]
                
                # Determine which way the normal split would go
                goes_left = X[:, feature_idx] <= threshold
                
                # Calculate weights for the next step
                # If goes_left is True:
                #   Left child gets weight * (1 - alpha)
                #   Right child gets weight * alpha
                # If goes_left is False:
                #   Left child gets weight * alpha
                #   Right child gets weight * (1 - alpha)
                
                left_path_probs = np.where(goes_left, 1 - self.alpha, self.alpha)
                right_path_probs = 1.0 - left_path_probs # Probabilities sum to 1
                
                # Update weights for children
                weights_left = weights * left_path_probs
                weights_right = weights * right_path_probs
                
                # Push children to stack
                stack.append((tree.children_right[node_id], weights_right))
                stack.append((tree.children_left[node_id], weights_left))
        
        # Normalize just in case (though mathematically they should sum to 1)
        row_sums = final_probs.sum(axis=1, keepdims=True)
        # Avoid division by zero
        row_sums[row_sums == 0] = 1.0 
        return final_probs / row_sums

    def predict(self, X):
        """
        Predict class labels based on the soft probabilities.
        """
        proba = self.predict_proba(X)
        return self.classes_[np.argmax(proba, axis=1)]
