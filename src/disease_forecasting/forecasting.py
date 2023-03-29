# src/disease_forecasting/forecasting.py

import os
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder

RANDOM_SEED = 42


def load_preprocessed_data(file_path):
    return pd.read_csv(file_path)


def prepare_data(df, target_col, test_size=0.2):
    X = df.drop(columns=[target_col])
    y = df[target_col]

    le = LabelEncoder()
    y = le.fit_transform(y)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=RANDOM_SEED)

    return X_train, X_test, y_train, y_test, le


def train_forecasting_model(X_train, y_train):
    model = xgb.XGBClassifier(use_label_encoder=False, eval_metric='mlogloss', random_state=RANDOM_SEED)
    model.fit(X_train, y_train)
    return model


def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    return accuracy


def save_model(model, output_file):
    model.save_model(output_file)


def main():
    input_file = 'data/preprocessed/preprocessed_data.csv'
    output_file = 'models/disease_forecasting/forecasting_model.pkl'

    # Load preprocessed data
    data = load_preprocessed_data(input_file)

    # Prepare data for training and testing
    X_train, X_test, y_train, y_test, label_encoder = prepare_data(data, target_col='disease')

    # Train the forecasting model
    model = train_forecasting_model(X_train, y_train)

    # Evaluate the model
    accuracy = evaluate_model(model, X_test, y_test)
    print(f"Model accuracy: {accuracy:.4f}")

    # Save the trained model
    save_model(model, output_file)


if __name__ == '__main__':
    main()
