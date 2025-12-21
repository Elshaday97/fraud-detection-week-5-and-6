import pandas as pd
from scripts.decorator import handle_errors
from scripts.constants import FRAUD_DATA_NUMERIC_COLS, FRAUD_DATA_CATEGORICAL_COLS
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from imblearn.over_sampling import SMOTE


class DataTransformer:
    def __init__(self):
        self.numeric_cols = []
        self.categorical_cols = []
        self._is_fitted = False
        self.preprocessor = None

    @handle_errors
    def fit(self, X: pd.DataFrame):
        if not isinstance(X, pd.DataFrame):
            raise ValueError("Please use a valid dataframe")

        if X.empty:
            raise ValueError("Please use a non empty training data")

        self.numeric_cols = []
        self.categorical_cols = []

        for col in X.columns:
            if col in FRAUD_DATA_NUMERIC_COLS and pd.api.types.is_numeric_dtype(X[col]):
                self.numeric_cols.append(col)
            elif (
                col in FRAUD_DATA_CATEGORICAL_COLS
                and pd.api.types.is_object_dtype(X[col])
                or pd.api.types.is_categorical_dtype(X[col])
            ):
                self.categorical_cols.append(col)

        self.preprocessor = ColumnTransformer(
            transformers=[
                ("num", StandardScaler(), self.numeric_cols),
                (
                    "cat",
                    OneHotEncoder(handle_unknown="ignore"),
                    self.categorical_cols,
                ),
            ]
        )
        self.preprocessor.fit(X)
        self._is_fitted = True

        return self

    @handle_errors
    def transform(self, X: pd.DataFrame):
        if not self._is_fitted:
            raise RuntimeError("Transformer must be fitted first")
        return self.preprocessor.transform(X)

    @handle_errors
    def fit_transform(self, X: pd.DataFrame):
        self.fit(X)
        return self.transform(X)


class ImbalanceHandler:

    def __init__(self):
        self.sm = SMOTE(random_state=42, sampling_strategy=0.35)

    @handle_errors
    def fit_resample(self, X: pd.DataFrame, y: pd.Series):
        X_res, y_res = self.sm.fit_resample(X, y)
        return X_res, y_res
