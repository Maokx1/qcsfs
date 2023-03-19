# QualityControlSystemForScrews #
<p align="left">
    <img alt="Python version" src="https://img.shields.io/badge/python-3.10-blue.svg">
    <img alt="Python tests" src="https://github.com/Maokx1/qcsfs/actions/workflows/tests.yml/badge.svg?event=push">
</p>

<p align="center">
    <img alt="Main application" src="https://github.com/Maokx1/qcsfs/blob/main/docs/imgs/gui.png">
</p>

A Quality Control System For Screws (qcsfs), that checks the quality of screws using a [neural network](https://github.com/Maokx1/qcsfs/blob/main/docs/neural_network.md), [classical image analysis methods](https://github.com/Maokx1/qcsfs/blob/main/docs/measure_screws.md) and [SPC (Statistical Process Control)](https://github.com/Maokx1/qcsfs/blob/main/docs/SPC.md). It is something like a simple simulation of a production line. Based on the results of the neural network, the system indicates which screws are correctly manufactured and which are defective. In a real life implementation, defective objects would be removed from the production line. To simulate this, I decided that only correctly manufactured screws would be subjected to further inspection, which meant measuring the screw length. Based on the aggregated results of the screw length measurements, control charts are displayed: mean values (X̄, X-bar) and standard deviation (S). All the results can be seen in the GUI of the application.

## Installation ##

### Requirements ###
```
Python >= 3.10
numpy ~= 1.24.2
opencv-python ~= 4.7.0.72
tensorflow ~= 2.11.0
matplotlib ~= 3.7.1
scikit-learn ~= 1.2.2
scipy ~= 1.10.1
imutils ~= 0.5.4
pandas ~= 1.5.3
PySimpleGUI ~= 4.60.4
Pillow ~= 9.4.0
PyYAML ~= 5.4.1
```
Detailed requirements can be found in file [requirements.txt](https://github.com/Maokx1/qcsfs/blob/main/requirements.txt).

Currently, the only way to install this package is to clone [this repository](https://github.com/Maokx1/qcsfs). After that, I recommend building the project using:

**Unix/macOS:**
```
cd /path/to/cloned/repo/
python3 -m pip install --upgrade pip
python3 -m pip install --upgrade build
python3 -m build
``` 

**Windows:**
```
cd /path/to/cloned/repo/
py -m pip install --upgrade pip
py -m pip install --upgrade build
py -m build
```

This allows you to freely use the package in your other projects, simply use the installation via pip locally. **Assuming you are in a folder with your project and have an active virtual environment, use:**  

```
pip install /path/to/cloned/repo
```
Ta paczka wymaga również następujących katalogów:
```
project_root_dir
┗ data
  ┣ measuring_results
  ┣ models
  ┣ screws
  ┃ ┣ test
  ┃ ┃ ┣ damaged
  ┃ ┃ ┗ good
  ┃ ┣ train
  ┃ ┃ ┣ damaged
  ┃ ┃ ┗ good
  ┃ ┗ validation
  ┃ ┃ ┣ damaged
  ┃ ┃ ┗ good
  ┗ training_results
```

Once the package has been successfully installed, a terminal command can be used to create such a directory:
```
qcsfs_create_folders
```

The package also requires a configuration file for the neural network. Create one using the command: 
```
qcsfs_create_yaml
```

It is possible that the package will appear on PyPI in the future, but before that happens, it would be necessary to:
* add the option to download a sample dataset and sample neural network model,
* add a data augmentation function,
* add control of neural network learning parameters from the GUI,
* add the option to select a neural network model from a list,
* add built-in live charts in the GUI,
* add the option to control the size of the group on which the average is calculated,
* add multithreading,
* refactored GUI code.

## Usage ##

Before using the package, populate data/screws subdirectories with images of the screws. To do this, use this [collection](https://www.kaggle.com/datasets/thomasdubail/screwanomalies-detection) and read [neural_network.md](https://github.com/Maokx1/qcsfs/blob/main/docs/neural_network.md) on how to do this correctly.
```
from qcsfs.gui import app

app.main()
```

## Sources ##

* [Image classification model in Python](https://www.analyticsvidhya.com/blog/2020/10/create-image-classification-model-python-keras/)
* [F-score](https://deepai.org/machine-learning-glossary-and-terms/f-score)
* [Canny Edge Detection](https://docs.opencv.org/3.4/da/d22/tutorial_py_canny.html)
* [Contours Hierarchy](https://docs.opencv.org/3.4/d9/d8b/tutorial_py_contours_hierarchy.html)
* [PCA](https://en.wikipedia.org/wiki/Principal_component_analysis)
* [How to Determine the Orientation of an Object Using OpenCV](https://automaticaddison.com/how-to-determine-the-orientation-of-an-object-using-opencv/)
* [Straighten rotated object](https://stackoverflow.com/questions/11627362/how-to-straighten-a-rotated-rectangle-area-of-an-image-using-opencv-in-python)
* [SPC - Western electric rules](https://en.wikipedia.org/wiki/Western_Electric_rules)
* [X-bar and S control chart](https://sixsigmastudyguide.com/x-bar-s-chart/)
* [PySimpleGUI](https://www.pysimplegui.org/en/latest/cookbook/)
