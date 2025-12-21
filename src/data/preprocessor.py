import pandas as pd
from scripts.decorator import handle_errors
from scripts.constants import (
    Fraud_Data_Columns,
    FRAUD_DATA_NUMERIC_COLS,
    FRAUD_DATA_DATE_COLS,
    RAW_CREDIT_IP_TO_COUNTRY_FILE_NAME,
    IP_To_Country_Columns,
    Credit_Card_Data_Columns,
)
from .loader import DataLoader
import numpy as np


class DataPreProcessor:
    def __init__(self, df: pd.DataFrame = None, init_ip=False):
        self.raw_df = df
        self.cleaned_df = None
        if init_ip:
            self.ip_to_country_df = (
                DataLoader()
                .load_csv(file_name=RAW_CREDIT_IP_TO_COUNTRY_FILE_NAME)
                .sort_values("lower_bound_ip_address")
                .reset_index(drop=True)
            )
        else:
            self.ip_to_country_df = None

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

    @handle_errors
    def _map_ip_address(self, df: pd.DataFrame):
        if self.ip_to_country_df.empty:
            raise ValueError(
                "IP Data source is empty. Please initialize instance with init_ip set to True."
            )
        working_df = df.copy()
        working_ip_df = self.ip_to_country_df.copy()

        # Convert scaled IP Address to IPv4 integer
        working_df[Fraud_Data_Columns.IP_INT.value] = working_df["ip_address"].astype(
            "int64"
        )
        # Sort by IPv4
        working_df = working_df.sort_values(Fraud_Data_Columns.IP_INT.value)

        # Convert upper and lower ip bounds to integer
        working_ip_df[IP_To_Country_Columns.UPPER_BOUND.value] = working_ip_df[
            IP_To_Country_Columns.UPPER_BOUND.value
        ].astype("int64")

        working_ip_df[IP_To_Country_Columns.LOWER_BOUND.value] = working_ip_df[
            IP_To_Country_Columns.LOWER_BOUND.value
        ].astype("int64")

        # Sort by lower bound ip
        working_ip_df = working_ip_df.sort_values(
            IP_To_Country_Columns.LOWER_BOUND.value
        )

        # Merge dfs
        merged_df = pd.merge_asof(
            working_df,
            working_ip_df,
            left_on=Fraud_Data_Columns.IP_INT.value,
            right_on=IP_To_Country_Columns.LOWER_BOUND.value,
            direction="backward",
        )

        # Filter by Upper bound
        merged_df[Fraud_Data_Columns.COUNTRY.value] = np.where(
            (
                merged_df[Fraud_Data_Columns.IP_INT.value]
                <= merged_df[IP_To_Country_Columns.UPPER_BOUND.value]
            )
            & (~merged_df[IP_To_Country_Columns.UPPER_BOUND.value].isna()),
            merged_df[Fraud_Data_Columns.COUNTRY.value],
            "Unknown",
        )

        # Drop Bounds
        merged_df.drop(
            columns=[
                IP_To_Country_Columns.LOWER_BOUND.value,
                IP_To_Country_Columns.UPPER_BOUND.value,
            ],
            inplace=True,
        )

        print("Mapped IP Address to Country!")
        return merged_df

    def get_cleaned_data(self):
        df = self.raw_df
        df = self._handle_duplicates(df)
        df = self._handle_dtype(df)
        df = self._sanity_check(df)
        df = self._map_ip_address(df)
        self.cleaned_df = df

        print("Data preprocessing complete!")
        return df

    def scale_features(self, df: pd.DataFrame):
        working_df = df.copy()
        return working_df


class CreditCardDataProcessor:
    def __init__(self, raw_df: pd.DataFrame):
        self.raw_df = raw_df

    @handle_errors
    def _handle_missing(self, df: pd.DataFrame):
        working_df = df.copy()
        missing_count = working_df.isna().sum()
        if missing_count.sum() == 0:
            print("No missing values in data")
        # Skipping missing data handling since data has no missing values

        return working_df

    @handle_errors
    def _handle_duplicates(self, df: pd.DataFrame):
        working_df = df.copy()
        duplicated_rows = working_df.duplicated().sum()
        if duplicated_rows == 0:
            print("No duplicated rows found")
        else:
            working_df = working_df.drop_duplicates()
            print(f"Dropped {duplicated_rows} duplicated rows.")
        return working_df

    @handle_errors
    def _handle_dtypes(self, df: pd.DataFrame):
        working_df = df.copy()
        cols_list = [col.value for col in Credit_Card_Data_Columns]

        for col in cols_list:
            if col not in working_df.columns:
                raise ValueError(f"Column {col} not found in data set")
            col_dtype = working_df[col].dtype
            if col_dtype != "int64" and col_dtype != "float64":
                working_df[col] = working_df[col].astype(int)
                print(f"Converted {col} to numeric.")

        return working_df

    def get_cleaned_data(self):
        df = self.raw_df
        df = self._handle_missing(df)
        df = self._handle_duplicates(df)
        df = self._handle_dtypes(df)
        return df
