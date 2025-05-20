import numpy as np

class ID3DecisionTree:
    def __init__(self, max_depth=None):
        self.max_depth = max_depth
        self.tree = None
        self.n_classes = None  # Track the total number of classes

    def _entropy(self, y):
        """Calculate entropy of a dataset."""
        values, counts = np.unique(y, return_counts=True)
        probabilities = counts / len(y)
        return -np.sum(probabilities * np.log2(probabilities))

    def _information_gain(self, X, y, feature_index):
        """Calculate information gain for a feature."""
        total_entropy = self._entropy(y)
        values, counts = np.unique(X[:, feature_index], return_counts=True)
        weighted_entropy = np.sum([
            (counts[i] / len(y)) * self._entropy(y[X[:, feature_index] == values[i]])
            for i in range(len(values))
        ])
        return total_entropy - weighted_entropy

    def _best_feature(self, X, y):
        """Find the best feature to split on."""
        gains = [self._information_gain(X, y, i) for i in range(X.shape[1])]
        return np.argmax(gains)

    def _build_tree(self, X, y, depth):
        """Recursively build the decision tree."""
        if len(np.unique(y)) == 1:
            return y[0]
        if X.shape[1] == 0 or (self.max_depth is not None and depth >= self.max_depth):
            return np.bincount(y).argmax()

        best_feature_index = self._best_feature(X, y)
        tree = {best_feature_index: {}}

        for value in np.unique(X[:, best_feature_index]):
            subset_X = X[X[:, best_feature_index] == value]
            subset_y = y[X[:, best_feature_index] == value]
            subtree = self._build_tree(
                np.delete(subset_X, best_feature_index, axis=1),
                subset_y,
                depth + 1
            )
            tree[best_feature_index][value] = subtree

        return tree

    def fit(self, X, y):
        """Train the decision tree."""
        self.tree = self._build_tree(X, y, depth=0)
        self.n_classes = len(np.unique(y))  # Determine the total number of classes

    def _predict_sample(self, sample, tree):
        """Predict a single sample."""
        if not isinstance(tree, dict):
            return tree
        feature = list(tree.keys())[0]
        value = sample[feature]
        # Handle unseen values or missing branches
        if value not in tree[feature]:
            # Get majority class if possible, otherwise return a default value
            majority_classes = [v for v in tree[feature].values() if isinstance(v, int)]
            if majority_classes:
                return np.bincount(majority_classes).argmax()
            else:
                return 0  # Default class (e.g., 0) if no majority class is found
        return self._predict_sample(sample, tree[feature][value])

    def predict(self, X):
        """Predict for multiple samples."""
        return np.array([int(self._predict_sample(sample, self.tree)) for sample in X])

    def predict_proba(self, X):
        """Predict class probabilities for multiple samples."""
        predictions = self.predict(X)
        probabilities = np.zeros((len(X), self.n_classes))  # Use the total number of classes
        for i, pred in enumerate(predictions):
            probabilities[i, pred] = 1
        return probabilities


class CustomRandomForest:
    def __init__(self, n_estimators=100, max_features="sqrt", max_depth=None, random_state=None):
        self.n_estimators = n_estimators
        self.max_features = max_features
        self.max_depth = max_depth
        self.random_state = random_state
        self.trees = []
        self.feature_indices = []
        self.n_classes = None  # Track the total number of classes

    def _bootstrap_sample(self, X, y):
        """Generate a bootstrap sample from the dataset."""
        n_samples = X.shape[0]
        indices = np.random.choice(n_samples, size=n_samples, replace=True)
        return X[indices], y[indices]

    def _select_features(self, X):
        """Randomly select features for each tree."""
        n_features = X.shape[1]
        if self.max_features == "sqrt":
            n_selected_features = int(np.sqrt(n_features))
        elif self.max_features == "log2":
            n_selected_features = int(np.log2(n_features))
        else:
            n_selected_features = n_features

        selected_indices = np.random.choice(n_features, size=n_selected_features, replace=False)
        return selected_indices

    def fit(self, X, y):
        """Train the Random Forest."""
        np.random.seed(self.random_state)
        self.n_classes = len(np.unique(y))  # Determine the total number of classes
        for _ in range(self.n_estimators):
            # Bootstrap sampling
            X_sample, y_sample = self._bootstrap_sample(X, y)
            
            # Feature selection
            selected_features = self._select_features(X_sample)
            self.feature_indices.append(selected_features)
            
            # Train an ID3 decision tree
            tree = ID3DecisionTree(max_depth=self.max_depth)
            tree.fit(X_sample[:, selected_features], y_sample)
            self.trees.append(tree)

    def predict(self, X):
        """Predict using the Random Forest."""
        tree_predictions = np.array([
            tree.predict(X[:, features]) for tree, features in zip(self.trees, self.feature_indices)
        ])
        # Majority voting
        return np.apply_along_axis(lambda x: np.bincount(x).argmax(), axis=0, arr=tree_predictions)

    def predict_proba(self, X):
        """Predict class probabilities using the Random Forest."""
        # Collect probabilities from each tree
        tree_probabilities = np.array([
            tree.predict_proba(X[:, features]) for tree, features in zip(self.trees, self.feature_indices)
        ])
        # Average probabilities across all trees
        return np.mean(tree_probabilities, axis=0)

