import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder

RANDOM_SEED = 42


def load_preprocessed_data(file_path):
    """
    Load preprocessed data from the given file path.
    :param file_path: Path to the preprocessed data file
    :return: A pandas DataFrame containing the preprocessed data
    """
    return pd.read_csv(file_path)


def prepare_data(df, target_col, test_size=0.2):
    """
    Prepare data for training and testing.
    :param df: DataFrame containing the preprocessed data
    :param target_col: Column name containing the target variable (disease)
    :param test_size: Proportion of the dataset to include in the test split
    :return: X_train, X_test, y_train, y_test, and a label encoder object
    """
    X = df.drop(columns=[target_col])
    y = df[target_col]

    le = LabelEncoder()
    y = le.fit_transform(y)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=RANDOM_SEED)

    return X_train, X_test, y_train, y_test, le


def train_forecasting_model(X_train, y_train):
    """
    Train the disease forecasting model using X_train and y_train.
    :param X_train: Training data
    :param y_train: Training labels
    :return: A trained forecasting model
    """
    model = xgb.XGBClassifier(use_label_encoder=False, eval_metric='mlogloss', random_state=RANDOM_SEED)
    model.fit(X_train, y_train)
    return model


def evaluate_model(model, X_test, y_test):
    """
    Evaluate the forecasting model on the test data.
    :param model: The trained forecasting model
    :param X_test: Test data
    :param y_test: Test labels
    :return: Model accuracy
    """
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    return accuracy


def save_model(model, output_file):
    """
    Save the trained model to the specified output file.
    :param model: The trained forecasting model
    :param output_file: Path to the output file
    """
    model.save_model(output_file)


def forecast_disease(model, preprocessed_data):
    """
    Forecast the disease using the trained model and preprocessed input data.
    :param model: The trained forecasting model
    :param preprocessed_data: Preprocessed input data
    :return: Disease label
    """
    input_data = xgb.DMatrix(preprocessed_data)
    predictions = model.predict(input_data)
    disease_index = np.argmax(predictions)
    disease_label = model.classes_[disease_index]
    return disease_label
