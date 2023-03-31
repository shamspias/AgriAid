import os
import numpy as np
from PIL import Image
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import random
import tensorflow as tf
import keras

RANDOM_SEED = 42


def load_images_from_directory(directory, target_size=(224, 224)):
    """
    Load images from the specified directory.
    :param directory: Directory containing the images
    :param target_size: Target size of the images
    :return: Images and their corresponding labels
    """
    images = []
    labels = []

    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.lower().endswith(('.jpg', '.jpeg', '.png')):
                file_path = os.path.join(root, file)
                label = os.path.basename(root)

                img = Image.open(file_path).resize(target_size)
                img_array = np.array(img)

                images.append(img_array)
                labels.append(label)

    return images, labels


def prepare_data(images, labels, test_size=0.2):
    """
    Prepare data for training and testing.
    :param images: List of images
    :param labels: List of labels
    :param test_size: The proportion of the dataset to include in the test split
    :return: Training and testing data along with the label encoder
    """
    X = np.array(images)
    y = np.array(labels)

    le = LabelEncoder()
    y = le.fit_transform(y)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=RANDOM_SEED)

    return X_train, X_test, y_train, y_test, le


def save_model(model, output_file):
    """
    Save the trained model to the specified file.
    :param model: The trained model
    :param output_file: The file to save the model to
    """
    model.save(output_file)


def load_model(model_file):
    """
    Load the trained model from the specified file.
    :param model_file: The file containing the trained model
    :return: The trained model
    """
    return keras.models.load_model(model_file)


def set_random_seed(seed_value):
    """
    Set the random seed for reproducibility.
    :param seed_value: The seed value
    """
    np.random.seed(seed_value)
    random.seed(seed_value)
    os.environ['PYTHONHASHSEED'] = str(seed_value)
    tf.random.set_seed(seed_value)
