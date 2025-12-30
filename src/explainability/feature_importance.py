import pandas as pd
import matplotlib.pyplot as plt
import os
import numpy as np
from sklearn.inspection import permutation_importance
from scipy.sparse import issparse


def plot_tree_feature_importance(model, feature_names, output_path, top_n=10):
    importances = model.feature_importances_

    features_aligned = feature_names[: len(importances)]

    fi_df = pd.DataFrame(
        {"feature": features_aligned, "importance": importances}
    ).sort_values("importance", ascending=False)

    top = fi_df.head(top_n)

    plt.figure(figsize=(8, 5))
    plt.barh(top["feature"], top["importance"])
    plt.gca().invert_yaxis()
    plt.title("Top Feature Importances")

    if output_path:
        plt.savefig(output_path, bbox_inches="tight")

    plt.show()

    return top
