import pandas as pd
from scripts.constants import Model_Names
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import (
    precision_recall_curve,
    average_precision_score,
    f1_score,
    confusion_matrix,
)


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
            model = LogisticRegression(random_state=42, max_iter=100, **kwargs)
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
