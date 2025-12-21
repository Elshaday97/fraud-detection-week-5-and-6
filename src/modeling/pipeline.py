import pandas as pd
from scripts.decorator import handle_errors
from scripts.constants import FRAUD_DATA_NUMERIC_COLS, FRAUD_DATA_CATEGORICAL_COLS
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from imblearn.over_sampling import SMOTE


class DataTransformer:
    """Data transformation class for preprocessing features."""

    def __init__(
        self,
        custom_numeric_cols_checker: list = None,
        custom_categoric_cols_checker: list = None,
    ):
        self.numeric_cols = []
        self.categorical_cols = []
        self._is_fitted = False
        self.preprocessor = None
        self.custom_numeric_cols_checker = custom_numeric_cols_checker
        self.custom_categoric_cols_checker = custom_categoric_cols_checker

    @handle_errors
    def fit(
        self,
        X: pd.DataFrame,
    ):
        """
        Fit the transformer to the DataFrame
        :param X: Input DataFrame
        :return: Fitted transformer"""
        if not isinstance(X, pd.DataFrame):
            raise ValueError("Please use a valid dataframe")

        if X.empty:
            raise ValueError("Please use a non empty training data")

        self.numeric_cols = []
        self.categorical_cols = []

        numeric_cols_checker = (
            self.custom_numeric_cols_checker or FRAUD_DATA_NUMERIC_COLS
        )
        categoric_cols_checker = (
            self.custom_categoric_cols_checker or FRAUD_DATA_CATEGORICAL_COLS
        )
        # Identify numeric and categorical columns
        for col in X.columns:
            if col in numeric_cols_checker and pd.api.types.is_numeric_dtype(X[col]):
                self.numeric_cols.append(col)
            elif (
                (len(categoric_cols_checker) > 0 and col in categoric_cols_checker)
                and pd.api.types.is_object_dtype(X[col])
                or pd.api.types.is_categorical_dtype(X[col])
            ):
                self.categorical_cols.append(col)

        # Create the ColumnTransformer
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

        # Fit the preprocessor
        self.preprocessor.fit(X)
        # Mark as fitted
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
    """Class to handle imbalanced datasets using SMOTE."""

    def __init__(self):
        self.sm = SMOTE(random_state=42, sampling_strategy=0.35)

    @handle_errors
    def fit_resample(self, X: pd.DataFrame, y: pd.Series):
        X_res, y_res = self.sm.fit_resample(X, y)
        return X_res, y_res
