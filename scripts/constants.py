from enum import Enum

RAW_DATA_DIR = "../data/raw"
CLEAN_DATA_DIR = "../data/processed"

RAW_FRAUD_DATA_FILE_NAME = "fraud_data.csv"
RAW_CREDIT_CARD_DATA_FILE_NAME = "credit_card.csv"
RAW_CREDIT_IP_TO_COUNTRY_FILE_NAME = "ip_to_country.csv"

CLEAN_FRAUD_DATA_FILE_NAME = "fraud_data_clean.csv"
CLEAN_CREDIT_CARD_DATA_FILE_NAME = "credit_card_clean.csv"
CLEAN_CREDIT_IP_TO_COUNTRY_FILE_NAME = "ip_to_country_clean.csv"


class Fraud_Data_Columns(Enum):
    USER_ID = "user_id"
    SIGN_UP_TIME = "signup_time"
    PURCHASE_TIME = "purchase_time"
    PURCHASE_VALUE = "purchase_value"
    DEVICE_ID = "device_id"
    SOURCE = "source"
    BROWSER = "browser"
    SEX = "sex"
    AGE = "age"
    IP_ADDRESS = "ip_address"
    CLASS = "class"


FRAUD_DATA_NUMERIC_COLS = [
    Fraud_Data_Columns.PURCHASE_VALUE.value,
    Fraud_Data_Columns.AGE.value,
]

FRAUD_DATA_CATEGORICAL_COLS = [
    Fraud_Data_Columns.DEVICE_ID.value,
    Fraud_Data_Columns.SOURCE.value,
    Fraud_Data_Columns.BROWSER.value,
    Fraud_Data_Columns.SEX.value,
]

FRAUD_DATA_DATE_COLS = [
    Fraud_Data_Columns.PURCHASE_TIME.value,
    Fraud_Data_Columns.SIGN_UP_TIME.value,
]
