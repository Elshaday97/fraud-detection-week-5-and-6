import pandas as pd
from scripts.constants import Model_Names
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    confusion_matrix,
)
from sklearn.model_selection import StratifiedKFold
import numpy as np


class TrainModel:
    def __init__(self, X_train, y_train, X_test, y_test):
        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
        self.y_test = y_test
        self.y_pred = None
        self.model = None
        self.metrics = {}

    def _get_correct_model(self, model_name, **kwargs):
        model = None

        if model_name == Model_Names.LOGISTIC_REGRESSION.value:
            allowed_kwargs = {
                k: v
                for k, v in kwargs.items()
                if k in ["C", "penalty", "solver", "max_iter", "class_weight"]
            }
            model = LogisticRegression(random_state=42, max_iter=100, **allowed_kwargs)
        elif model_name == Model_Names.RANDOM_FOREST.value:
            model = RandomForestClassifier(random_state=42, **kwargs)
        elif model_name == Model_Names.XGBoost.value:
            model = XGBClassifier(random_state=42, **kwargs)
        else:
            raise ValueError(f"Unknown Model: {model_name}")
        return model

    def train_model(self, model_name=Model_Names.LOGISTIC_REGRESSION.value, **kwargs):
        print(f"Training {model_name}...")
        self.model = self._get_correct_model(model_name, **kwargs)
        self.model.fit(self.X_train, self.y_train)
        self.y_pred = self.model.predict(self.X_test)
        print(f"Training {model_name} complete!")
        return self.model, self.y_pred

    def evaluate_model(self):
        if self.model is None or self.y_pred is None:
            raise ValueError("Model Not Trained Yet")

        # 1. Confusion Matrix
        cm = confusion_matrix(self.y_test, self.y_pred)

        # 2. F1-Score
        f1 = f1_score(self.y_test, self.y_pred)

        # 3. AUC-PR
        y_scores = self.model.predict_proba(self.X_test)[:, 1]
        auc_pr = average_precision_score(self.y_test, y_score=y_scores)

        print("Confusion Matrix\n:", cm)
        print("F1-Score", f1)
        print("AUC-PR", auc_pr)

        return {"confusion_matrix": cm, "fl_score": f1, "auc_pr": auc_pr}

    def cross_val_evaluate(
        self, model_name=Model_Names.LOGISTIC_REGRESSION.value, **kwargs
    ):
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

        f1_scores = []
        auc_pr_scores = []

        for train_idx, val_idx in skf.split(self.X_train, self.y_train):
            X_tr, X_val = self.X_train[train_idx], self.X_train[val_idx]
            y_tr, y_val = self.y_train[train_idx], self.y_train[val_idx]

            model = self._get_correct_model(model_name, **kwargs)
            model.fit(X_tr, y_tr)
            y_val_pred = model.predict(X_val)
            y_val_proba = model.predict_proba(X_val)[:, 1]

            f1_scores.append(f1_score(y_val, y_val_pred))
            auc_pr_scores.append(average_precision_score(y_val, y_val_proba))

        mean_f1, std_f1 = np.mean(f1_scores), np.std(f1_scores)
        mean_auc_pr, std_auc_pr = np.mean(auc_pr_scores), np.std(auc_pr_scores)

        print(f"{model_name} CV Results:")
        print(f"F1-Score: {mean_f1:.3f} ± {std_f1:.3f}")
        print(f"AUC-PR : {mean_auc_pr:.3f} ± {std_auc_pr:.3f}")

        return {
            "model_name": model_name,
            "f1_mean": mean_f1,
            "f1_std": std_f1,
            "auc_pr_mean": mean_auc_pr,
            "auc_pr_std": std_auc_pr,
        }
