# src/data_preprocessing/data_preprocessing.py

import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler


def load_data(file_paths):
    """
    Load data from the given file paths.
    :param file_paths: A dictionary containing the file names and their paths
    :return: A dictionary containing the file names and their data
    """
    datasets = {}
    for file_name, file_path in file_paths.items():
        datasets[file_name] = pd.read_csv(file_path)
    return datasets


def clean_data(datasets):
    """
    Clean the data.
    :param datasets: A dictionary containing the file names and their data
    :return: A dictionary containing the file names and their cleaned data
    """
    cleaned_data = {}

    # Climate data cleaning
    climate_data = datasets['climate_data'].interpolate(method='linear', limit_direction='both')
    cleaned_data['climate_data'] = climate_data

    # Plant and disease data cleaning
    for data_name in ['plant_data', 'disease_data']:
        data = datasets[data_name].dropna()
        cleaned_data[data_name] = data

    return cleaned_data


def normalize_data(climate_data, columns):
    """
    Normalize the given columns of the climate data.
    :param climate_data:    The climate data
    :param columns:       The columns to normalize
    :return:            The normalized climate data
    """
    scaler = StandardScaler()
    climate_data[columns] = scaler.fit_transform(climate_data[columns])
    return climate_data


def merge_data(cleaned_data, merge_key):
    """
    Merge the datasets.
    :param cleaned_data:  A dictionary containing the file names and their cleaned data
    :param merge_key:  The key to merge the datasets on
    :return:  The merged data
    """
    merged_data = cleaned_data['climate_data']
    for data_name in ['plant_data', 'disease_data', 'geolocation_data']:
        merged_data = pd.merge(merged_data, cleaned_data[data_name], on=merge_key)
    return merged_data


def save_preprocessed_data(data, output_file):
    """
    Save the preprocessed data to the given file.
    :param data:  The preprocessed data
    :param output_file:  The output file
    :return:  None
    """
    data.to_csv(output_file, index=False)


def main():
    input_data_paths = {
        'climate_data': 'data/raw/climate_data.csv',
        'plant_data': 'data/raw/plant_data.csv',
        'disease_data': 'data/raw/disease_data.csv',
        'geolocation_data': 'data/raw/geolocation_data.csv',
    }

    output_file = 'data/preprocessed/preprocessed_data.csv'

    # Load raw data
    datasets = load_data(input_data_paths)

    # Clean data
    cleaned_data = clean_data(datasets)

    # Normalize climate data
    numeric_features = ['temperature', 'humidity', 'precipitation', 'solar_radiation', 'wind_speed']
    cleaned_data['climate_data'] = normalize_data(cleaned_data['climate_data'], numeric_features)

    # Merge datasets
    merged_data = merge_data(cleaned_data, 'id')

    # Save preprocessed data
    save_preprocessed_data(merged_data, output_file)


if __name__ == '__main__':
    main()
