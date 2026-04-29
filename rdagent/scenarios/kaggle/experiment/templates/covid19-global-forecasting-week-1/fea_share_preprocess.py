import os

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder


def prepreprocess():
    # Load the data
    train = pd.read_csv("/kaggle/input/train.csv")
    test = pd.read_csv("/kaggle/input/test.csv")

    # Combine train and test for preprocessing
    all_data = pd.concat([train, test], sort=False)

    # Convert date to datetime
    all_data["Date"] = pd.to_datetime(all_data["Date"])

    # Create new features
    all_data["Day"] = all_data["Date"].dt.day
    all_data["Month"] = all_data["Date"].dt.month
    all_data["Year"] = all_data["Date"].dt.year

    # Encode categorical variables
    le = LabelEncoder()
    all_data["Country/Region"] = le.fit_transform(all_data["Country/Region"])
    all_data["Province/State"] = le.fit_transform(all_data["Province/State"].fillna("None"))

    # Split back into train and test
    train = all_data[all_data["ForecastId"].isna()]
    test = all_data[all_data["ForecastId"].notna()]

    # Prepare features and targets
    features = ["Country/Region", "Province/State", "Day", "Month", "Year"]
    X = train[features]
    y = train[["ConfirmedCases", "Fatalities"]]

    # Split into train and validation sets
    X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.2, random_state=42)

    return X_train, X_valid, y_train, y_valid, test[features], test["ForecastId"]


def preprocess_script():
    if os.path.exists("/kaggle/input/X_train.pkl"):
        X_train = pd.read_pickle("/kaggle/input/X_train.pkl")  # nosec B301
        X_valid = pd.read_pickle("/kaggle/input/X_valid.pkl")  # nosec B301
        y_train = pd.read_pickle("/kaggle/input/y_train.pkl")  # nosec B301
        y_valid = pd.read_pickle("/kaggle/input/y_valid.pkl")  # nosec B301
        X_test = pd.read_pickle("/kaggle/input/X_test.pkl")  # nosec B301
        forecast_ids = pd.read_pickle("/kaggle/input/forecast_ids.pkl")  # nosec B301
    else:
        X_train, X_valid, y_train, y_valid, X_test, forecast_ids = prepreprocess()

        # Save preprocessed data
        X_train.to_pickle("/kaggle/input/X_train.pkl")  # nosec
        X_valid.to_pickle("/kaggle/input/X_valid.pkl")  # nosec
        y_train.to_pickle("/kaggle/input/y_train.pkl")  # nosec
        y_valid.to_pickle("/kaggle/input/y_valid.pkl")  # nosec
        X_test.to_pickle("/kaggle/input/X_test.pkl")  # nosec
        forecast_ids.to_pickle("/kaggle/input/forecast_ids.pkl")  # nosec

    return X_train, X_valid, y_train, y_valid, X_test, forecast_ids
