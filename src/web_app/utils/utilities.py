import os
import numpy as np
from PIL import Image
from sklearn.preprocessing import LabelEncoder

RANDOM_SEED = 42


def load_images_from_directory(directory, target_size=(224, 224)):
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
    X = np.array(images)
    y = np.array(labels)

    le = LabelEncoder()
    y = le.fit_transform(y)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=RANDOM_SEED)

    return X_train, X_test, y_train, y_test, le


def save_model(model, output_file):
    model.save(output_file)


def load_model(model_file):
    return keras.models.load_model(model_file)


def set_random_seed(seed_value):
    np.random.seed(seed_value)
    random.seed(seed_value)
    os.environ['PYTHONHASHSEED'] = str(seed_value)
    tf.random.set_seed(seed_value)
