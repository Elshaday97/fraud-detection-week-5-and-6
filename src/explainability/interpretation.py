import pandas as pd
import numpy as np


def top_shap_drivers(shap_values, X, top_n=5, positive_class=1):
    """
    Identify strongest global SHAP drivers for binary classification.
    """

    shap_values = np.asarray(shap_values)

    # Case 1: (n_samples, n_features, 2)
    if shap_values.ndim == 3:
        shap_values = shap_values[:, :, positive_class]

    # Case 2: (n_features, 2)
    elif shap_values.ndim == 2 and shap_values.shape[1] == 2:
        shap_values = shap_values[:, positive_class]

    mean_abs_shap = np.abs(shap_values).mean(axis=0)

    return (
        pd.DataFrame(
            {
                "feature": X.columns,
                "mean_abs_shap": mean_abs_shap,
            }
        )
        .sort_values("mean_abs_shap", ascending=False)
        .head(top_n)
        .reset_index(drop=True)
    )
