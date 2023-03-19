# For better explanation check docs/neural_network.md

import os

import numpy as np
import tensorflow as tf


def load_model(model_path: str | os.PathLike):
    """
    This functions loads saved model.
    #### Args:
        model_path (str | os.PathLike): This should be a valid path to the saved model.
    #### Returns:
        Object of the loaded model.
    """
    return tf.keras.models.load_model(model_path)


def predict(model_, test_image: np.ndarray, true_label: str, labels_: list[str],
            membership_boundary_: float) -> tuple[str, str]:
    """
    This function uses existing model to predict image label.
    #### Args:
        model_ : Loaded model object.
        test_image (np.ndarray): Input image of a size that matches the input of an existing neural network model. 
        By default, it must be 224x224 pixels.
        true_label (str): Ground truth label.
        labels_ (list[str]): List of possible labels.
        membership_boundary_ (float): The boundary, which determines class membership.
    #### Returns:
        (tuple[str, str]): The tuple containing the true label and label determined by the neural network.
    """
    if model_.predict(test_image) < membership_boundary_:
        return true_label, labels_[0]
    else:
        return true_label, labels_[1]
