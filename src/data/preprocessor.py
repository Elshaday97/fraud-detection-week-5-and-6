import pandas as pd
from scripts.decorator import handle_errors
from scripts.constants import (
    Fraud_Data_Columns,
    FRAUD_DATA_NUMERIC_COLS,
    FRAUD_DATA_DATE_COLS,
)


class DataPreProcessor:
    def __init__(self, df: pd.DataFrame):
        self.raw_df = df
        self.cleaned_df = df

    @handle_errors
    def _handle_duplicates(self, df: pd.DataFrame):
        working_df = df.copy()
        duplicted_rows = working_df.duplicated().sum()
        if duplicted_rows == 0:
            print("No duplicated rows found.")
        else:
            working_df.drop_duplicates()
            print("Dropped duplicates.")
        return working_df

    @handle_errors
    def _handle_dtype(self, df: pd.DataFrame):
        working_df = df.copy()

        # Numeric Cols
        for numeric_col in FRAUD_DATA_NUMERIC_COLS:
            col_dtype = working_df[numeric_col].dtype
            if col_dtype != "int64":
                working_df[numeric_col] = working_df[numeric_col].astype(int)
                print(f"Converted {numeric_col} to numeric.")

        # Date Cols
        for date_col in FRAUD_DATA_DATE_COLS:
            col_dtype = working_df[date_col].dtype
            if col_dtype != "datetime64[ns]":
                working_df[date_col] = pd.to_datetime(
                    working_df[date_col], errors="coerce"
                )
                print(f"Converted {date_col} to datetime.")

        return working_df

    @handle_errors
    def _sanity_check(self, df: pd.DataFrame):
        working_df = df.copy()

        # Check Generic Null Values
        null_rows = working_df.isna().values.any()
        if not null_rows:
            print("No generic null values found.")

        # Check for negative purchase amounts
        neg_mask = working_df[Fraud_Data_Columns.PURCHASE_VALUE.value] <= 0
        neg_purchase_amounts = working_df[neg_mask]
        print(
            f"{neg_purchase_amounts.shape[0]} transactions with purchase value <= 0 found"
        )

        # Check transactions where Sign up time is larger than Purchase Time
        time_mask = (
            working_df[Fraud_Data_Columns.SIGN_UP_TIME.value]
            > working_df[Fraud_Data_Columns.PURCHASE_TIME.value]
        )
        invalid_transactions = working_df[time_mask]
        print(
            f"{invalid_transactions.shape[0]} transactions where sign up date > purchase date values found"
        )

        # Check Ages outside of reasonable times, zero value transactions
        age_mask = (working_df[Fraud_Data_Columns.AGE.value] < 0) | (
            working_df[Fraud_Data_Columns.AGE.value] > 100
        )
        invalid_ages = working_df[age_mask]
        print(
            f"{invalid_ages.shape[0]} transactions with age values below 0 or above 100 found"
        )

        return working_df

    def get_cleaned_data(self):
        df = self.raw_df
        df = self._handle_duplicates(df)
        df = self._handle_dtype(df)
        df = self._sanity_check(df)
        self.cleaned_df = df

        print("Data preprocessing complete!")
        return df
