import pandas as pd
from scripts.decorator import handle_errors
from scripts.constants import Fraud_Data_Columns
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline


class TimeFeatureExtractor(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        working_df = X.copy()

        working_df[Fraud_Data_Columns.HOUR_OF_DAY.value] = working_df[
            Fraud_Data_Columns.PURCHASE_TIME.value
        ].dt.hour
        working_df[Fraud_Data_Columns.DAY_OF_WEEK.value] = working_df[
            Fraud_Data_Columns.PURCHASE_TIME.value
        ].dt.dayofweek
        working_df[Fraud_Data_Columns.TIME_SINCE_SIGN_UP.value] = (
            working_df[Fraud_Data_Columns.PURCHASE_TIME.value]
            - working_df[Fraud_Data_Columns.SIGN_UP_TIME.value]
        ).dt.total_seconds()

        return working_df


# I have not implemented any aggregator since data set is already one row per user


class FeatEngineer:
    def __init__(self, df: pd.DataFrame):
        self.df = df
        self.pipeline = Pipeline(
            [
                ("time_feature_extractor", TimeFeatureExtractor()),
            ]
        )

    @handle_errors
    def transform_all(self):
        if self.df.empty:
            raise ValueError("Please use a non empty data set")
        self.df.sort_values(
            [Fraud_Data_Columns.USER_ID.value, Fraud_Data_Columns.PURCHASE_TIME.value],
            inplace=True,
        )
        return self.pipeline.fit_transform(self.df)
