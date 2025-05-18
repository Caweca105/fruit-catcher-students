import numpy as np

class DecisionTree:
    """
    Decision tree classifier for categorical data using information gain.
    """
    def __init__(self, X, y, threshold: float = 0.0, max_depth: int = None, depth: int = 0):
        # Store hyperparameters
        self.threshold = threshold
        self.max_depth = max_depth
        self.depth = depth

        # Convert to numpy arrays
        X = np.array(X, dtype=object)
        y = np.array(y, dtype=int)

        # Compute majority class for this node
        labels, counts = np.unique(y, return_counts=True)
        self.prediction = labels[np.argmax(counts)]

        # Stopping criteria: pure node or reached max depth
        if len(labels) == 1 or (self.max_depth is not None and self.depth >= self.max_depth):
            self.is_leaf = True
            return

        # Compute entropy at current node
        current_entropy = self._entropy(y)

        # Search for best split
        best_gain = 0.0
        best_feature = None
        best_splits = None
        n_samples, n_features = X.shape

        for feature_idx in range(n_features):
            values = X[:, feature_idx]
            unique_vals = np.unique(values)
            weighted_entropy = 0.0
            splits = {}

            for val in unique_vals:
                mask = (values == val)
                y_subset = y[mask]
                X_subset = X[mask]
                prob = len(y_subset) / n_samples
                weighted_entropy += prob * self._entropy(y_subset)
                splits[val] = (X_subset, y_subset)

            info_gain = current_entropy - weighted_entropy
            if info_gain > best_gain:
                best_gain = info_gain
                best_feature = feature_idx
                best_splits = splits

        # If no beneficial split, make leaf
        if best_gain <= self.threshold or best_splits is None:
            self.is_leaf = True
            return

        # Otherwise, create an internal node
        self.is_leaf = False
        self.feature = best_feature
        self.children = {
            val: DecisionTree(
                X_subset,
                y_subset,
                threshold=self.threshold,
                max_depth=self.max_depth,
                depth=self.depth + 1
            )
            for val, (X_subset, y_subset) in best_splits.items()
        }

    def _entropy(self, y: np.ndarray) -> float:
        labels, counts = np.unique(y, return_counts=True)
        probs = counts / counts.sum()
        return -np.sum([p * np.log2(p) for p in probs if p > 0])

    def predict(self, x) -> int:
        """
        Predict the label for a single sample.

        Parameters:
        - x: array-like of shape (n_features,), feature values in the same order as training.

        Returns:
        - Predicted class label: int
        """
        if self.is_leaf:
            return self.prediction

        val = x[self.feature]
        # Recursively follow the branch
        if val in self.children:
            return self.children[val].predict(x)
        # Unseen feature value: fallback to majority class at this node
        return self.prediction


def train_decision_tree(X, y, threshold: float = 0.0, max_depth: int = None) -> DecisionTree:
    """
    Factory function to train a DecisionTree classifier.

    Parameters:
    - X: array-like of shape (n_samples, n_features)
    - y: array-like of shape (n_samples,)
    - threshold: minimum information gain to split
    - max_depth: maximum tree depth

    Returns:
    - Trained DecisionTree instance
    """
    return DecisionTree(X, y, threshold=threshold, max_depth=max_depth)
