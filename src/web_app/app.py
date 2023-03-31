# app.py

from flask import Flask, request, jsonify
from src.utils import utilities
from src.disease_identification import identification
from src.disease_forecasting import forecasting
from src.data_preprocessing import data_preprocessing

app = Flask(__name__)


@app.route('/api/forecast', methods=['POST'])
def forecast_disease():
    """
    Forecast the disease based on the climate data
    :return: JSON object containing the forecasted disease
    """
    climate_data = request.json

    # Preprocess input data
    preprocessed_data = data_preprocessing.preprocess_input_data(climate_data)

    # Load the trained model for forecasting
    model_file = 'models/disease_forecasting/forecasting_model.h5'
    model = utilities.load_model(model_file)

    # Forecast the disease
    forecast_result = forecasting.forecast_disease(model, preprocessed_data)

    # Format and return the result as a JSON object
    response = {
        'disease_forecast': forecast_result
    }
    return jsonify(response)


@app.route('/api/identify', methods=['POST'])
def identify_disease():
    """
    Identify the disease based on the image
    :return: JSON object containing the identified disease
    """
    image_file = request.files['image']

    # Load the image data
    image_data = utilities.load_image_data(image_file)

    # Load the trained model for identification
    model_file = 'models/disease_identification/identification_model.h5'
    model = utilities.load_model(model_file)

    # Identify the disease based on the image
    disease_identification_result = identification.identify_disease(model, image_data)

    # Format and return the result as a JSON object
    response = {
        'disease_identification': disease_identification_result
    }
    return jsonify(response)


if __name__ == '__main__':
    app.run(debug=True)
