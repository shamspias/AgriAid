from flask import Flask, request, jsonify
from src.web_app.utils import utilities
from src.disease_identification import identification
from src.disease_forecasting import forecasting
from src.data_preprocessing import data_preprocessing
import config

app = Flask(__name__)


@app.route('/api/forecast', methods=['POST'])
def forecast_disease():
    """
    Forecast the disease based on the climate data
    :return: JSON object containing the disease forecast result
    """
    climate_data = request.json

    # Preprocess input data
    preprocessed_data = data_preprocessing.preprocess_input_data(climate_data)

    # Load the trained model for forecasting
    model = utilities.load_model(config.FORECASTING_MODEL_PATH)

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
    :return: JSON object containing the disease identification result
    """
    image_file = request.files['image']

    # Load the image data
    image_data = utilities.load_image_data(image_file)

    # Load the trained model for identification
    model = utilities.load_model(config.IDENTIFICATION_MODEL_PATH)

    # Identify the disease based on the image
    disease_identification_result = identification.identify_disease(model, image_data)

    # Format and return the result as a JSON object
    response = {
        'disease_identification': disease_identification_result
    }
    return jsonify(response)


if __name__ == '__main__':
    app.run(debug=True)
