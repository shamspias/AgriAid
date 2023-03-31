# AgriAid: AI-Powered Early Detection and Diagnosis of Plant Diseases in Bangladesh 🌾🇧🇩

## Overview
AgriAid is an AI-powered software application designed to assist farmers, agricultural extension agents, and other stakeholders in Bangladesh with early detection and diagnosis of plant diseases. Developed using advanced machine learning and deep learning techniques, AgriAid aims to increase crop yield and food security in Bangladesh by providing timely and accurate information on potential disease outbreaks.

## Features
1. **Disease Forecasting**: AgriAid forecasts the occurrence of plant diseases based on climate data, allowing users to take preventive measures.

2. **Disease Identification**: AgriAid identifies plant diseases based on symptoms or images, enabling users to effectively manage plant diseases.

3. **User-friendly Interface**: The software's intuitive user interface allows users to easily input data and receive disease forecasts and identification results.REST API for easy integration with web and mobile applications.

## Technologies
- **Programming Language**: Python, with extensive library support including TensorFlow, Keras, scikit-learn, PyTorch, Pandas, NumPy, and Matplotlib.
- **Machine Learning Techniques**: Random Forest, Support Vector Machines, and Gradient Boosting Machines for disease forecasting.
- **Deep Learning Techniques**: Convolutional Neural Networks (CNNs) and transfer learning for disease identification based on images.
- **Web Frameworks**: Flask or Django for backend development and integration of AI models.

## Future Scope
1. **Expansion to Other Countries**: Adapting the software for use in other countries and regions to address global food security challenges.
2. **Integration with IoT Devices**: Incorporating data from IoT devices, such as sensors and drones, to enhance disease detection and diagnosis.
3. **Mobile Application**: Developing a mobile app to make the platform more accessible to farmers and agricultural agents in remote areas.
4. **Support for Additional Crops**: Extending the software's capabilities to support a wider range of crops and diseases.

## Installation

1. Clone the repository:
    
    ```bash
   git clone https://github.com/shamspias/agriaid.git
   cd agriaid
    ```

2. Create a virtual environment and activate it:

    ```bash
   python3 -m venv venv
   source venv/bin/activate
    ```

3. Install the project and its dependencies:

    ```bash
   pip install -e .[dev]
    ```

4. Create a `.env` file in the project root directory with the required environment variables:

    ```bash
   API_KEY=your_api_key
   RANDOM_SEED=42
   IMAGE_SIZE=224,224
   ```

5. Run the Flask app:

    ```bash
   export FLASK_APP=app.py # On Windows, use set FLASK_APP=app.py
   flask run
    ```


## Usage

The AgriAid software exposes two REST API endpoints for forecasting and identifying plant diseases.

### Forecasting

To forecast the occurrence of a plant disease based on climate data, send a POST request to the `/api/forecast` endpoint with the climate data as JSON.

Example:

```python
import requests

climate_data = {
 "temperature": 30.0,
 "humidity": 80,
 "rainfall": 100,
}

response = requests.post("http://localhost:5000/api/forecast", json=climate_data)
print(response.json())
```
## Identification
To identify a plant disease based on an image of plant symptoms, send a POST request to the /api/identify endpoint with the image file as form data.

Example:
```python
import requests

with open("path/to/image.jpg", "rb") as image_file:
    response = requests.post("http://localhost:5000/api/identify", files={"image": image_file})
    print(response.json())

```
