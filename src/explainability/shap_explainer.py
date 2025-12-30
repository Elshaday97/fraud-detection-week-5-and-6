import shap
import numpy as np
import pandas as pd


class TreeSHAPExplainer:
    """
    Efficient SHAP explainer for tree-based binary classifiers.
    """

    def __init__(
        self, model, X_background, max_background_samples=200, random_state=42
    ):
        self.model = model
        self.random_state = random_state

        X_bg = self._to_numpy(X_background)

        if X_bg.shape[0] > max_background_samples:
            rng = np.random.default_rng(random_state)
            idx = rng.choice(X_bg.shape[0], max_background_samples, replace=False)
            X_bg = X_bg[idx]

        self.explainer = shap.TreeExplainer(
            model, data=X_bg, feature_perturbation="interventional"
        )

    def explain_global(self, X, max_samples=300):
        X_np = self._to_numpy(X)

        if X_np.shape[0] > max_samples:
            rng = np.random.default_rng(self.random_state)
            idx = rng.choice(X_np.shape[0], max_samples, replace=False)
            X_np = X_np[idx]

        shap_values = self.explainer.shap_values(X_np)
        if isinstance(shap_values, list):
            shap_values = shap_values[1]

        return shap_values, X_np

    def explain_single(self, X_row):
        """
        Compute SHAP values for a single instance.
        """
        X_np = self._to_numpy(X_row)

        shap_values = self.explainer.shap_values(X_np)

        # Binary classification → positive class
        if isinstance(shap_values, list):
            shap_values = shap_values[1]

        base_value = self.explainer.expected_value
        if isinstance(base_value, (list, np.ndarray)):
            base_value = base_value[1]

        return shap_values[0], base_value

    @staticmethod
    def _to_numpy(X):
        if isinstance(X, pd.DataFrame):
            return X.values
        if hasattr(X, "toarray"):
            return X.toarray()
        return np.asarray(X)
