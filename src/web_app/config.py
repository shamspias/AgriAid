import os
from dotenv import load_dotenv

# Load environment variables from the .env file
load_dotenv()

# Directories
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, 'data')
MODEL_DIR = os.path.join(BASE_DIR, 'models')

# Data paths
CLIMATE_DATA_PATH = os.path.join(DATA_DIR, 'climate_data.csv')
PLANT_DATA_PATH = os.path.join(DATA_DIR, 'plant_data.csv')
DISEASE_DATA_PATH = os.path.join(DATA_DIR, 'disease_data.csv')

# Model paths
FORECASTING_MODEL_PATH = os.path.join(MODEL_DIR, 'disease_forecasting', 'forecasting_model.h5')
IDENTIFICATION_MODEL_PATH = os.path.join(MODEL_DIR, 'disease_identification', 'identification_model.h5')

# API keys and credentials
API_KEY = os.getenv('API_KEY')

# Other settings
RANDOM_SEED = int(os.getenv('RANDOM_SEED', 42))
IMAGE_SIZE = tuple(map(int, os.getenv('IMAGE_SIZE', '224,224').split(',')))
