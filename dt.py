import numpy as np

class DecisionTree:
    def __init__(self, X, y, threshold=0.0, max_depth=None, depth=0):
        """
        Decision tree classifier for categorical data using information gain.

        Parameters:
        - X: array-like of shape (n_samples, n_features), feature matrix
        - y: list or array-like of shape (n_samples,), target labels
        - threshold: float, minimum information gain required to split
        - max_depth: int or None, maximum depth of the tree
        - depth: int, current depth (used internally)
        """
        # Store parameters
        self.threshold = threshold
        self.max_depth = max_depth
        self.depth = depth

        # Ensure X,y are numpy arrays
        X = np.array(X)
        y = np.array(y)

        # Compute majority class prediction at this node
        labels, counts = np.unique(y, return_counts=True)
        self.prediction = labels[np.argmax(counts)]

        # Stopping: pure node or depth limit
        if len(labels) == 1 or (self.max_depth is not None and self.depth >= self.max_depth):
            self.is_leaf = True
            return

        # Compute entropy of current node
        current_entropy = self._entropy(y)

        # Find best split by information gain
        best_gain = 0.0
        best_feature = None
        best_splits = None
        n_samples, n_features = X.shape

        for feature in range(n_features):
            values = X[:, feature]
            unique_vals = np.unique(values)
            weighted_entropy = 0.0
            splits = {}

            for val in unique_vals:
                mask = (values == val)
                y_subset = y[mask]
                X_subset = X[mask]
                weight = len(y_subset) / n_samples
                weighted_entropy += weight * self._entropy(y_subset)
                splits[val] = (X_subset, y_subset)

            info_gain = current_entropy - weighted_entropy
            if info_gain > best_gain:
                best_gain = info_gain
                best_feature = feature
                best_splits = splits

        # If no split yields sufficient gain, make leaf
        if best_gain <= self.threshold or best_splits is None:
            self.is_leaf = True
            return

        # Otherwise, create internal node
        self.is_leaf = False
        self.feature = best_feature
        self.children = {}

        for val, (X_subset, y_subset) in best_splits.items():
            self.children[val] = DecisionTree(
                X_subset, y_subset,
                threshold=self.threshold,
                max_depth=self.max_depth,
                depth=self.depth + 1
            )

    def _entropy(self, y):
        """
        Compute entropy of label array y.
        """
        labels, counts = np.unique(y, return_counts=True)
        probabilities = counts / counts.sum()
        return -np.sum([p * np.log2(p) for p in probabilities if p > 0])

    def predict(self, x):
        """
        Predict the label for a single sample x.

        Parameters:
        - x: array-like of shape (n_features,), feature values

        Returns:
        - prediction: predicted label for x
        """
        if self.is_leaf:
            return self.prediction

        val = x[self.feature]
        # Traverse child if value seen
        if val in self.children:
            return self.children[val].predict(x)
        else:
            # Unseen category: fallback to majority at this node
            return self.prediction


def train_decision_tree(X, y, threshold=0.01, max_depth=5):
    """
    Train and return a DecisionTree classifier using given hyperparameters.

    Parameters:
    - X: array-like, feature matrix
    - y: array-like, target labels
    - threshold: float, minimum information gain to split (default=0.01)
    - max_depth: int or None, maximum tree depth (default=5)

    Returns:
    - DecisionTree: trained decision tree model
    """
    return DecisionTree(X, y, threshold=threshold, max_depth=max_depth)
