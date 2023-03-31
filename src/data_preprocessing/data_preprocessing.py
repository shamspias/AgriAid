import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler


def load_data(file_paths):
    return {file_name: pd.read_csv(file_path) for file_name, file_path in file_paths.items()}


def clean_climate_data(climate_data):
    return climate_data.interpolate(method='linear', limit_direction='both')


def clean_other_data(data):
    return data.dropna()


def normalize_data(data, columns):
    scaler = StandardScaler()
    data[columns] = scaler.fit_transform(data[columns])
    return data


def merge_data(cleaned_data, merge_key):
    merged_data = cleaned_data['climate_data']
    for data_name in ['plant_data', 'disease_data', 'geolocation_data']:
        merged_data = pd.merge(merged_data, cleaned_data[data_name], on=merge_key)
    return merged_data


def save_preprocessed_data(data, output_file):
    data.to_csv(output_file, index=False)


def main():
    input_data_paths = {
        'climate_data': 'data/raw/climate_data.csv',
        'plant_data': 'data/raw/plant_data.csv',
        'disease_data': 'data/raw/disease_data.csv',
        'geolocation_data': 'data/raw/geolocation_data.csv',
    }
    output_file = 'data/preprocessed/preprocessed_data.csv'

    datasets = load_data(input_data_paths)

    cleaned_data = {
        'climate_data': clean_climate_data(datasets['climate_data']),
        'plant_data': clean_other_data(datasets['plant_data']),
        'disease_data': clean_other_data(datasets['disease_data']),
        'geolocation_data': datasets['geolocation_data']
    }

    numeric_features = ['temperature', 'humidity', 'precipitation', 'solar_radiation', 'wind_speed']
    cleaned_data['climate_data'] = normalize_data(cleaned_data['climate_data'], numeric_features)

    merged_data = merge_data(cleaned_data, 'id')

    save_preprocessed_data(merged_data, output_file)


if __name__ == '__main__':
    main()
