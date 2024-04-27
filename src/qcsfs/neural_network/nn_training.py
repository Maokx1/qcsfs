# Dataset: https://www.kaggle.com/datasets/thomasdubail/screwanomalies-detection
# Partially based on: https://www.analyticsvidhya.com/blog/2020/10/create-image-classification-model-python-keras/
# Explanation of the f1-score: https://deepai.org/machine-learning-glossary-and-terms/f-score
# For better explanation check docs/neural_network.md

import csv
import logging
import os
import traceback
import yaml

import cv2
import numpy as np
from sklearn.metrics import classification_report
import tensorflow as tf


def read_yaml(file_path: str | os.PathLike) -> dict:
    """
    This function reads YAML config file containing the neural network training configuration information.
    #### Args:
        file_path (str | os.PathLike): This should be the valid path to the yaml file.
    #### Returns:
        (dict): Dictionary with values from yaml config file.
    #### Raises:
        FileNotFoundError: When the yaml file does not exist or when passing a directory.
    """
    if not os.path.isfile(file_path):
        raise FileNotFoundError
    with open(file_path, "r") as file_:
        return yaml.safe_load(file_)


def get_data(
    data_path: str | os.PathLike,
    label_list: list[str],
    label_: str,
    shape_: tuple[int, int],
    path_list: list = [],
    is_test: bool = False,
) -> list[list[np.ndarray]] | list[list[int]]:
    """
    This function loads, labels and resizes images. It loads every image from the given folder.
    If is_test is true path to the image will be added to the path list.
    #### Args:
        data_path (str | os.PathLike): This should be a valid path to the directory with images or image.
        label_list (list[str]): List of possible labels.
        label_ (str): Label, that should be given to the image.
        shape_ (tuple[int]): Target shape of the image. If the tuple is empty, the image size will be retained.
        path_list (list): List of paths to test images.
        is_test (bool): If True image path is added to path_list.
    #### Returns:
        (list): List of images and corresponding label number.
    #### Raises:
        FileNotFoundError: When no file was found  in the path.
    """

    data = []
    class_num = label_list.index(label_)
    if os.path.isfile(data_path):
        img_arr = cv2.imread(data_path)  # type: ignore
        if type(img_arr) == np.ndarray:
            if shape_:
                if len(img_arr.shape) < 3:
                    resized_arr = cv2.resize(img_arr, shape_)
                    # MobileNet-v2 requires RGB images, so repeat values for all 3 channels
                    img_arr = cv2.cvtColor(resized_arr, cv2.COLOR_GRAY2RGB)
                else:
                    img_arr = cv2.resize(img_arr, shape_)
            if is_test:
                path_list.append(data_path)
            data.append([img_arr, class_num])
    else:
        for img_path in os.listdir(data_path):
            img_arr = cv2.imread(os.path.join(data_path, img_path))
            if type(img_arr) == np.ndarray:
                if shape_:
                    if len(img_arr.shape) < 3:
                        resized_arr = cv2.resize(img_arr, shape_)
                        # MobileNet-v2 requires RGB images, so repeat values for all 3 channels
                        img_arr = cv2.cvtColor(resized_arr, cv2.COLOR_GRAY2RGB)
                    else:
                        img_arr = cv2.resize(img_arr, shape_)
                if is_test:
                    path_list.append(img_path)
                data.append([img_arr, class_num])
    if not data:
        raise FileNotFoundError
    return data


def create_model():
    """
    This function creates model based on MobileNetV2 architecture.
    """
    csv_path = "data/training_results/results.csv"
    model_path = "data/models/new_model.keras"
    logging.basicConfig(level=logging.INFO)
    # logging.basicConfig(filename='data/training_results/logs.log', filemode='w', level=logging.INFO)

    if os.path.isfile("nn_config.yaml"):
        config_values = read_yaml("nn_config.yaml")
    else:
        yaml_dir = os.path.dirname(__file__)
        config_values = read_yaml(os.path.join(yaml_dir, "nn_config.yaml"))

    try:
        labels = config_values["TRAINING"]["LABELS"].split(",")
        img_shape = tuple(
            [int(i) for i in config_values["TRAINING"]["IMG_SHAPE"].split(",")]
        )
        base_learning_rate = float(config_values["TRAINING"]["BASE_LEARNING_RATE"])
        epochs = int(config_values["TRAINING"]["EPOCHS"])
        batch_size = int(config_values["TRAINING"]["BATCH_SIZE"])
        membership_boundary = float(config_values["TRAINING"]["MEMBERSHIP_BOUNDARY"])
    except KeyError:
        traceback.print_exc()
        exit("Some default keys were changed in yaml file without updating code.")
    except ValueError:
        traceback.print_exc()
        exit(
            "The newly entered data in the yaml file does not meet the requirements stated in the comments."
        )

    path = []
    x_train = []
    y_train = []
    x_val = []
    y_val = []
    x_test = []
    y_test = []

    # creating datasets
    for img, label in get_data(
        data_path="data/screws/train/good",
        label_list=labels,
        label_=labels[0],
        shape_=img_shape,
        is_test=False,
        path_list=[],
    ) + get_data(
        data_path="data/screws/train/damaged",
        label_list=labels,
        label_=labels[1],
        shape_=img_shape,
        is_test=False,
        path_list=[],
    ):
        x_train.append(img)
        y_train.append(label)

    for img, label in get_data(
        data_path="data/screws/validation/good",
        label_list=labels,
        label_=labels[0],
        shape_=img_shape,
        is_test=False,
        path_list=[],
    ) + get_data(
        data_path="data/screws/validation/damaged",
        label_list=labels,
        label_=labels[1],
        shape_=img_shape,
        is_test=False,
        path_list=[],
    ):
        x_val.append(img)
        y_val.append(label)

    for img, label in get_data(
        data_path="data/screws/test/good",
        label_list=labels,
        label_=labels[0],
        shape_=img_shape,
        is_test=True,
        path_list=path,
    ) + get_data(
        data_path="data/screws/test/damaged",
        label_list=labels,
        label_=labels[1],
        shape_=img_shape,
        is_test=True,
        path_list=path,
    ):
        x_test.append(img)
        y_test.append(label)

    # reshaping labels to be a column vector
    x_train = np.asarray(x_train)
    y_train = np.asarray(y_train).reshape((-1, 1))

    x_val = np.asarray(x_val)
    y_val = np.asarray(y_val).reshape((-1, 1))

    x_test = np.asarray(x_test)
    y_test = np.asarray(y_test).reshape((-1, 1))

    logging.info(
        f"Init learning rate = {base_learning_rate},"
        f" Batch size = {batch_size}, # of epochs = {epochs}"
    )

    # choosing MobileNetV2 architecture (pretrained on imagenet), best results were obtained using Adam optimizer
    base_model = tf.keras.applications.MobileNetV2(
        input_shape=(img_shape[0], img_shape[1], 3),
        include_top=False,
        weights="imagenet",
    )
    base_model.trainable = True

    model = tf.keras.Sequential(
        [
            base_model,
            tf.keras.layers.GlobalAveragePooling2D(),
            tf.keras.layers.Dropout(rate=0.1),
            tf.keras.layers.Dense(1, activation="sigmoid"),
        ]
    )

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=base_learning_rate),
        loss=tf.keras.losses.BinaryCrossentropy(from_logits=False),
        metrics=["accuracy"],
    )

    callback = tf.keras.callbacks.EarlyStopping(monitor="loss", patience=5)

    history = model.fit(
        x_train,
        y_train,
        batch_size=batch_size,
        epochs=epochs,
        validation_data=(x_val, y_val),
        callbacks=[callback],
    )

    with open(csv_path, "w", encoding="UTF8") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "Training_accuracy",
                "Validation_accuracy",
                "Training_loss",
                "Validation_loss",
            ]
        )
        for t_a, v_a, t_l, v_l in zip(
            history.history["accuracy"],
            history.history["val_accuracy"],
            history.history["loss"],
            history.history["val_loss"],
        ):
            writer.writerow([t_a, v_a, t_l, v_l])
    logging.info(f"Saved training history to {csv_path}.")

    model.save(model_path)
    logging.info(f"Saved model to {model_path}.")

    # assigning objects to specific class
    predictions = model.predict(x_test)
    for img_name, membership_val in zip(path, predictions):
        if membership_val < membership_boundary:
            print(
                f"Neural network thinks {img_name} shows good screw. "
                f"Membership value: {round(float(membership_val), 4)}\n"
            )
        else:
            print(
                f"Neural network thinks {img_name} shows damaged screw. "
                f"Membership value: {round(float(membership_val), 4)}\n"
            )

    predictions[predictions < membership_boundary] = 0
    predictions[predictions >= membership_boundary] = 1

    print(
        classification_report(
            y_test,
            predictions.astype(np.uint8),
            target_names=[f"{labels[0]} screw(Class 0)", f"{labels[1]} screw(Class 1)"],
        )
    )
