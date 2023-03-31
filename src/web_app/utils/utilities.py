import os
import random
import numpy as np
from PIL import Image
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import tensorflow as tf
from tensorflow import keras

RANDOM_SEED = 42


def load_images_from_directory(directory, target_size=(224, 224)):
    """
    Load images and their labels from a directory.
    :param directory: Path to the directory containing images
    :param target_size: Tuple specifying the desired dimensions of the resized images
    :return: List of images and their corresponding labels
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
    Split the image and label data into training and testing sets.
    :param images: List of images
    :param labels: List of corresponding labels
    :param test_size: Proportion of the dataset to include in the test split (default: 0.2)
    :return: Train and test sets for image data and labels, and the label encoder
    """
    X = np.array(images)
    y = np.array(labels)

    le = LabelEncoder()
    y = le.fit_transform(y)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=RANDOM_SEED)

    return X_train, X_test, y_train, y_test, le


def load_image_data(image_file, target_size=(224, 224)):
    """
    Load image data from a file.
    :param image_file: Image file
    :param target_size: Tuple specifying the desired dimensions of the resized image
    :return: Image array
    """
    img = Image.open(image_file).resize(target_size)
    img_array = np.array(img)
    img_array = np.expand_dims(img_array, axis=0)
    return img_array


def save_model(model, output_file):
    """
    Save a trained model to a file.
    :param model: Trained model
    :param output_file: Path to the output file
    :return: None
    """
    model.save(output_file)


def load_model(model_file):
    """
    Load a trained model from a file.
    :param model_file: Path to the model file
    :return: Loaded model
    """
    return keras.models.load_model(model_file)


def set_random_seed(seed_value):
    """
    Set the random seed for reproducibility.
    :param seed_value: Seed value
    :return: None
    """
    np.random.seed(seed_value)
    random.seed(seed_value)
    os.environ['PYTHONHASHSEED'] = str(seed_value)
    tf.random.set_seed(seed_value)
