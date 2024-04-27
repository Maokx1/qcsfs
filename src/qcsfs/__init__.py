import os
import sys

import yaml

sys.path.append(os.path.dirname(__file__))
from gui import app, gui_utils
from measure_screws import measure
from neural_network import nn_training, nn_predicting
from spc import spc_charts


def create_folders():
    """Creates directories needed for the use of the package."""
    cwd = os.getcwd()

    os.makedirs(os.path.join(cwd, "data/measuring_results"))
    os.makedirs(os.path.join(cwd, "data/models"))

    os.makedirs(os.path.join(cwd, "data/screws/test/good"))
    os.makedirs(os.path.join(cwd, "data/screws/test/damaged"))

    os.makedirs(os.path.join(cwd, "data/screws/train/good"))
    os.makedirs(os.path.join(cwd, "data/screws/train/damaged"))

    os.makedirs(os.path.join(cwd, "data/screws/validation/good"))
    os.makedirs(os.path.join(cwd, "data/screws/validation/damaged"))

    os.makedirs(os.path.join(cwd, "data/training_results"))


def create_yaml():
    """Creates yaml file with neural network config needed for the use of the package."""
    cwd = os.getcwd()
    comments = """\
# LABELS: must be two values of type string separated by commas.
# IMG_SHAPE: Targeted image size. It must be two values(width and height), both an integer type, separated by commas.
# Because of MobileNet-v2's requirements it can only be: 96 or 128 or 160 or 192 or 224
# BASE_LEARNING_RATE: must be of type float.
# EPOCHS: must be of type integer (greater than 0).
# BATCH_SIZE: must be of type integer (greater than 0).
# MEMBERSHIP_BOUNDARY: must be of type float and be between 0 and 1.\n"""

    data = """
TRAINING:
  LABELS: Good,Damaged
  IMG_SHAPE: 224,224
  BASE_LEARNING_RATE: 0.0001
  EPOCHS: 10
  BATCH_SIZE: 7
  MEMBERSHIP_BOUNDARY: 0.5"""

    data = yaml.safe_load(data)
    with open(os.path.join(cwd, "nn_config.yaml"), "w") as f:
        f.write(comments)
        yaml.dump(data, f, default_flow_style=False)
