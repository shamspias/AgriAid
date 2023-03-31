import pandas as pd
from sklearn.preprocessing import StandardScaler


# Load datasets from the specified file paths
def load_data(file_paths):
    datasets = {}
    for file_name, file_path in file_paths.items():
        datasets[file_name] = pd.read_csv(file_path)
    return datasets


# Clean and interpolate missing values in the datasets
def clean_data(datasets):
    cleaned_data = {}

    climate_data = datasets['climate_data'].interpolate(method='linear', limit_direction='both')
    cleaned_data['climate_data'] = climate_data

    for data_name in ['plant_data', 'disease_data']:
        data = datasets[data_name].dropna()
        cleaned_data[data_name] = data

    return cleaned_data


# Normalize specified columns in the climate data
def normalize_data(climate_data, columns):
    scaler = StandardScaler()
    climate_data[columns] = scaler.fit_transform(climate_data[columns])
    return climate_data


# Merge datasets on a specified key
def merge_data(cleaned_data, merge_key):
    merged_data = cleaned_data['climate_data']
    for data_name in ['plant_data', 'disease_data', 'geolocation_data']:
        merged_data = pd.merge(merged_data, cleaned_data[data_name], on=merge_key)
    return merged_data


# Preprocess input climate data by cleaning and normalizing it
def preprocess_input_data(climate_data, columns_to_normalize):
    cleaned_data = climate_data.interpolate(method='linear', limit_direction='both')
    normalized_data = normalize_data(cleaned_data, columns_to_normalize)
    return normalized_data


# Main function to preprocess the data
def main():
    input_data_paths = {
        'climate_data': 'data/raw/climate_data.csv',
        'plant_data': 'data/raw/plant_data.csv',
        'disease_data': 'data/raw/disease_data.csv',
        'geolocation_data': 'data/raw/geolocation_data.csv',
    }

    output_file = 'data/preprocessed/preprocessed_data.csv'

    # Load datasets
    datasets = load_data(input_data_paths)

    # Clean datasets
    cleaned_data = clean_data(datasets)

    # Normalize climate data
    numeric_features = ['temperature', 'humidity', 'precipitation', 'solar_radiation', 'wind_speed']
    cleaned_data['climate_data'] = normalize_data(cleaned_data['climate_data'], numeric_features)

    # Merge datasets on the 'id' column
    merged_data = merge_data(cleaned_data, 'id')

    # Save preprocessed data to a CSV file
    merged_data.to_csv(output_file, index=False)


if __name__ == '__main__':
    main()
