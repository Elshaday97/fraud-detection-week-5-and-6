from enum import Enum

RAW_DATA_DIR = "../data/raw"
CLEAN_DATA_DIR = "../data/processed"

RAW_FRAUD_DATA_FILE_NAME = "fraud_data.csv"
RAW_CREDIT_CARD_DATA_FILE_NAME = "credit_card.csv"
RAW_CREDIT_IP_TO_COUNTRY_FILE_NAME = "ip_to_country.csv"

CLEAN_FRAUD_DATA_FILE_NAME = "fraud_data_clean.csv"
CLEAN_CREDIT_CARD_DATA_FILE_NAME = "credit_card_clean.csv"

LR_MODEL_FILEPATH = "logistic_regression.pkl"
RF_MODEL_FILEPATH = "random_forest.pkl"
XG_BOOST_FILEPATH = "xg_boost.pkl"


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

    # Additional Cols
    IP_INT = "ip_int"
    COUNTRY = "country"
    HOUR_OF_DAY = "hour_of_day"
    DAY_OF_WEEK = "day_of_week"
    TIME_SINCE_SIGN_UP = "time_since_signup"
    # TXN Frequency
    TXN_COUNT = "txn_count"
    TXN_LAST_HOUR = "txn_last_hour"
    TXN_LAST_DAY = "txn_last_day"
    TXN_LAST_WEEK = "txn_last_week"
    # TXN Velocity
    TIME_SINCE_LAST_TXN = "time_since_last_txn"
    MIN_TIME_BETWEEN_TXNS = "min_time_between_txns"
    MEDIAN_TIME_BETWEEN_TXNS = "median_time_between_txns"


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


class IP_To_Country_Columns(Enum):
    LOWER_BOUND = "lower_bound_ip_address"
    UPPER_BOUND = "upper_bound_ip_address"
    COUNTRY = "country"


class Credit_Card_Data_Columns(Enum):
    AMOUNT = "Amount"
    CLASS = "Class"
    TIME = "Time"
    V10 = "V10"
    V11 = "V11"
    V12 = "V12"
    V13 = "V13"
    V14 = "V14"
    V15 = "V15"
    V16 = "V16"
    V17 = "V17"
    V18 = "V18"
    V19 = "V19"
    V20 = "V20"
    V21 = "V21"
    V22 = "V22"
    V23 = "V23"
    V24 = "V24"
    V25 = "V25"
    V26 = "V26"
    V27 = "V27"
    V28 = "V28"


class Model_Names(Enum):
    LOGISTIC_REGRESSION = "Logistic Regression"
    RANDOM_FOREST = "Random Forest"
    XGBoost = "XGBoost"
    LIGHT_GBM = "LightGBM"
