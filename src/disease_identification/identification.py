import numpy as np
from tensorflow.keras.applications.efficientnet import preprocess_input


def preprocess_image_data(image_data):
    """
    Preprocess image data for the identification model.
    :param image_data: Image data as a NumPy array
    :return: Preprocessed image data
    """
    return preprocess_input(image_data)


def identify_disease(model, image_data):
    """
    Identify the disease in the input image using the trained model.
    :param model: Trained disease identification model
    :param image_data: Preprocessed image data
    :return: Identified disease label
    """
    preprocessed_data = preprocess_image_data(image_data)
    predictions = model.predict(preprocessed_data)
    predicted_class_index = np.argmax(predictions, axis=-1)
    return predicted_class_index
