import json
from scripts.constants import Model_Names
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    confusion_matrix,
    precision_recall_curve,
    ConfusionMatrixDisplay,
)
from sklearn.model_selection import StratifiedKFold
import numpy as np
import matplotlib.pyplot as plt
import os
from joblib import dump, load
from pathlib import Path


class TrainModel:
    def __init__(
        self, X_train, y_train, X_test, y_test, project_name="fraud_detection"
    ):
        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
        self.y_test = y_test
        self.y_pred = None
        self.model = None
        self.model_name = None
        self.project_name = project_name
        self.metrics = {}

        # Setup directories
        self.root_dir = Path.cwd().parent
        self.root_reports_dir = os.path.join(self.root_dir, "reports", project_name)
        self.root_models_dir = os.path.join(self.root_dir, "models", project_name)
        self.metrics_dir = os.path.join(self.root_reports_dir, "metrics")
        self.figures_dir = os.path.join(self.root_reports_dir, "figures")
        os.makedirs(self.root_models_dir, exist_ok=True)
        os.makedirs(self.metrics_dir, exist_ok=True)
        os.makedirs(self.figures_dir, exist_ok=True)

    def _get_correct_model(self, model_name, **kwargs):
        """
        Returns the correct model based on the model name.
        """
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
        """Train the specified model."""
        print(f"Training {model_name}...")
        self.model_name = model_name
        self.model = self._get_correct_model(model_name, **kwargs)
        self.model.fit(self.X_train, self.y_train)
        self.y_pred = self.model.predict(self.X_test)
        print(f"Training {model_name} complete!")
        return self.model, self.y_pred

    def evaluate_model(self):
        """Evaluate the trained model and return metrics."""
        if self.model is None or self.y_pred is None:
            raise ValueError("Model Not Trained Yet")

        # 1. Confusion Matrix
        cm = confusion_matrix(self.y_test, self.y_pred)

        # 2. F1-Score
        f1 = f1_score(self.y_test, self.y_pred)

        # 3. AUC-PR
        y_scores = self.model.predict_proba(self.X_test)[:, 1]
        auc_pr = average_precision_score(self.y_test, y_score=y_scores)

        self.metrics = {
            "confusion_matrix": cm.tolist(),
            "f1_score": f1,
            "auc_pr": auc_pr,
        }

        # Save metrics
        metrics_file = os.path.join(self.metrics_dir, f"{self.model_name}_metrics.json")
        with open(metrics_file, "w") as f:
            json.dump(self.metrics, f, indent=4)
        print(f"Metrics saved to {metrics_file}")

        # Save plots
        self._save_confusion_matrix(cm, self.model_name)
        self._save_pr_curve(self.y_test, y_scores, self.model_name)

        print("Confusion Matrix\n:", cm)
        print("F1-Score", f1)
        print("AUC-PR", auc_pr)

        return {"confusion_matrix": cm, "fl_score": f1, "auc_pr": auc_pr}

    def cross_val_evaluate(
        self, model_name=Model_Names.LOGISTIC_REGRESSION.value, **kwargs
    ):
        """Perform cross-validation and return mean and std of metrics."""
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

        results = {
            "model": model_name,
            "f1_mean": np.mean(f1_scores),
            "f1_std": np.std(f1_scores),
            "auc_pr_mean": np.mean(auc_pr_scores),
            "auc_pr_std": np.std(auc_pr_scores),
        }

        # Save CV metrics
        metrics_file = os.path.join(self.metrics_dir, f"{model_name}_cv_metrics.json")
        with open(metrics_file, "w") as f:
            json.dump(results, f, indent=4)
        print(f"Cross-validation metrics saved to {metrics_file}")

        return results

    # -------------------------
    # Helper Functions to Save Plots
    # -------------------------
    def _save_confusion_matrix(self, cm, model_name):
        disp = ConfusionMatrixDisplay(confusion_matrix=cm)
        disp.plot()
        fig_path = os.path.join(self.figures_dir, f"{model_name}_confusion_matrix.png")
        plt.savefig(fig_path)
        plt.close()
        print(f"Confusion matrix saved to {fig_path}")

    def _save_pr_curve(self, y_true, y_scores, model_name):
        precision, recall, _ = precision_recall_curve(y_true, y_scores)
        plt.plot(recall, precision, marker=".", label=model_name)
        plt.xlabel("Recall")
        plt.ylabel("Precision")
        plt.title(f"PR Curve - {model_name}")
        plt.legend()
        fig_path = os.path.join(self.figures_dir, f"{model_name}_pr_curve.png")
        plt.savefig(fig_path)
        plt.close()
        print(f"PR curve saved to {fig_path}")

    # -------------------------
    # Save / Load Model
    # -------------------------
    def save_model(self, filename: str):
        if self.model is None:
            raise ValueError("No trained model to save!")
        file_path = os.path.join(self.root_models_dir, filename)
        dump(self.model, file_path)
        print(f"Model saved to {file_path}")

    def load_model(self, filename: str):
        file_path = os.path.join(self.root_models_dir, filename)
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"{file_path} does not exist")
        self.model = load(file_path)
        print(f"Model loaded from {file_path}")
