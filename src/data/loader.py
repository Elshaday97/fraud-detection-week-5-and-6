from scripts.constants import (
    CLEAN_DATA_DIR,
    RAW_DATA_DIR,
    RAW_FRAUD_DATA_FILE_NAME,
    Fraud_Data_Columns,
)
from scripts.decorator import handle_errors
from pathlib import Path
import pandas as pd


class DataLoader:
    """
    DataLoader class for loading and saving CSV files
    from specified directories.
    """

    def __init__(self):
        pass

    @handle_errors
    def load_csv(
        self,
        file_name: str = RAW_FRAUD_DATA_FILE_NAME,
        file_path: str = RAW_DATA_DIR,
        should_parse_dates=False,
    ) -> pd.DataFrame | None:
        """
        Load CSV file from given path and return as DataFrame
        :param file_name: Name of the CSV file
        :param file_path: Path to the directory containing the CSV file
        :return: DataFrame containing the CSV data
        """
        path = Path(file_path) / Path(file_name)

        if not Path(path).exists():
            raise FileNotFoundError(f"{path}: File Not Found!")

        parse_dates = (
            [
                Fraud_Data_Columns.PURCHASE_TIME.value,
                Fraud_Data_Columns.SIGN_UP_TIME.value,
            ]
            if should_parse_dates
            else None
        )
        df = pd.read_csv(
            path,
            parse_dates=parse_dates,
        )
        if "Unnamed: 0" in df.columns:
            df.drop(columns=["Unnamed: 0"], inplace=True)

        print(f"Loaded {path} to dataframe!")
        return df

    @handle_errors
    def save_csv(
        self, df: pd.DataFrame, file_name: str, file_path: str = CLEAN_DATA_DIR
    ):
        """
        Save DataFrame to CSV file at given path
        :param df: DataFrame to be saved
        :param file_name: Name of the CSV file
        :param file_path: Path to the directory where the CSV file will be saved"""
        if df.empty:
            raise ValueError("Please use a non empty data source to save")
        if not Path(file_path).exists():
            raise FileNotFoundError(f"Directory {file_path} does not exist.")
        path = Path(file_path) / Path(file_name)
        df.to_csv(path)
        print(f"Saved dataframe to {path} to successfully!")
