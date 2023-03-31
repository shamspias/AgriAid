import os
import numpy as np
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

RANDOM_SEED = 42
IMAGE_SIZE = (224, 224)
BATCH_SIZE = 32
EPOCHS = 10


def load_image_data(file_path, image_size):
    """
    Load image data
    :param file_path:   file path
    :param image_size:  image size
    :return:    image data
    """
    img = image.load_img(file_path, target_size=image_size)
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    return img_array


def prepare_data(images, labels, test_size=0.2):
    """
    Prepare data for training
    :param images:  images
    :param labels:  labels
    :param test_size:   test size
    :return:    training and test data, label encoder
    """
    X = np.array(images)
    y = np.array(labels)

    le = LabelEncoder()
    y = le.fit_transform(y)

    return train_test_split(X, y, test_size=test_size, random_state=RANDOM_SEED), le


def build_identification_model(num_classes):
    """
    Build the identification model
    :param num_classes: number of classes
    :return:   identification model
    """
    base_model = EfficientNetB0(weights='imagenet', include_top=False, input_shape=(IMAGE_SIZE[0], IMAGE_SIZE[1], 3))
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(1024, activation='relu')(x)
    predictions = Dense(num_classes, activation='softmax')(x)

    model = Model(inputs=base_model.input, outputs=predictions)

    for layer in base_model.layers:
        layer.trainable = False

    model.compile(optimizer=Adam(lr=0.001), loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    return model


def train_identification_model(model, X_train, y_train, batch_size, epochs):
    """
    Train the identification model
    :param model:   identification model
    :param X_train:     training data
    :param y_train:     training labels
    :param batch_size:  batch size
    :param epochs:    number of epochs
    :return:    trained identification model
    """
    data_gen = ImageDataGenerator(
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest'
    )

    data_gen.fit(X_train)

    model.fit(data_gen.flow(X_train, y_train, batch_size=batch_size), steps_per_epoch=len(X_train) // batch_size,
              epochs=epochs)

    return model


def evaluate_model(model, X_test, y_test):
    scores = model.evaluate(X_test, y_test)
    return scores[1]  # Return the accuracy score


def save_model(model, output_file):
    model.save(output_file)


def main():
    # Load and prepare image data
    # Replace the following code with the actual code to load and preprocess your image data
    images = []
    labels = []

    (X_train, X_test, y_train, y_test), label_encoder = prepare_data(images, labels)

    # Build and train the identification model
    num_classes = len(np.unique(y_train))
    model = build_identification_model(num_classes)
    model = train_identification_model(model, X_train, y_train, BATCH_SIZE, EPOCHS)

    # Evaluate the model
    accuracy = evaluate_model(model, X_test, y_test)
    print(f"Model accuracy: {accuracy:.4f}")

    # Save the trained model
    output_file = 'models/disease_identification/identification_model.h5'
    save_model(model, output_file)


if __name__ == '__main__':
    main()
