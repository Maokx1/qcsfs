# Neural network based on MobileNetV2 architecture #  

## Creating dataset ##
In this project I am using the [dataset](https://www.kaggle.com/datasets/thomasdubail/screwanomalies-detection) that was used in the paper:
```
Paul Bergmann, Michael Fauser, David Sattlegger, and Carsten Steger, "A Comprehensive Real-World Dataset for Unsupervised Anomaly Detection", IEEE Conference on Computer Vision and Pattern Recognition, 2019 
```
Example screws look like this:

<p align="center">
    <img alt="Good screw" src="https://github.com/Maokx1/qcsfs/blob/main/data/sample_screws/good/005.png">
</p>

<p align="center">
    <img alt="Damaged screw (Manipulated front)" src="https://github.com/Maokx1/qcsfs/blob/main/data/sample_screws/damaged/020_mf.png">
</p>

<p align="center">
    <img alt="Damaged screw (Scratched neck)" src="https://github.com/Maokx1/qcsfs/blob/main/data/sample_screws/damaged/020_sn.png">
</p>

<p align="center">
    <img alt="Damaged screw (Thread top)" src="https://github.com/Maokx1/qcsfs/blob/main/data/sample_screws/damaged/020_tt.png">
</p>

For my application, I resize the images to 512x512 pixels. Unfortunately, this collection only contains around 500 images. This is too few for my application. For this purpose, data augmentation must be used. For my application, I used [tensorflow's built-in options](https://www.tensorflow.org/api_docs/python/tf/keras/preprocessing/image/ImageDataGenerator) such as: 
```
tf.keras.preprocessing.image.ImageDataGenerator(zoom_range=[0.8, 1.2])

tf.keras.preprocessing.image.ImageDataGenerator(brightness_range=[0.2, 1.0])

tf.keras.preprocessing.image.ImageDataGenerator(width_shift_range=[-100, 100])

tf.keras.preprocessing.image.ImageDataGenerator(height_shift_range=0.1)

tf.keras.preprocessing.image.ImageDataGenerator(horizontal_flip=True)
```
As you can see, I used options such as zooming, changing brightness, moving the subject vertically and horizontally and flipping the image (Please note that the damage of the screws should still be visible on the image). I used these options randomly, which allowed me to increase the number of images to 1,500, which was already enough. I have mainly focused on increasing the number of images of damaged screws, as there are the fewest. In the final set, the number of images of good and damaged screws is equal to each other.
Finally, I divide the images into 3 groups: 
* in the train directory I place: 500 images of good screws and 500 damaged ones.
* in the validation directory, 100 of each type (200 in total)
* in the test directory, 150 of each type (300 in total)

During the research, this number of images and their ratio appeared to give the best results.
**This approach allowed me to obtain satisfactory results for my applications. However, I do not believe that I have succeeded in determining the optimal method for creating a dataset for this kaggle dataset, and for this reason I am not publishing either the model or dataset I have created. At this point, I leave all the steps involved in creating the dataset to the user.**

## Why MobileNetV2? ##

From my perspective, the task of dividing the screws into good and damaged seemed straightforward. Therefore, I was looking for an architecture that is relatively small and models based on it are able to achieve very high accuracy. While searching, I found the following comparison in one [article](https://culurciello.medium.com/analysis-of-deep-neural-networks-dcf398e71aae):

<p align="center">
    <img alt="Accuracy vs number of operations chart" src="https://github.com/Maokx1/qcsfs/blob/main/docs/imgs/nn_chart1.png">
</p>

<p align="center">
    <img alt="Accuracy per number of parameters" src="https://github.com/Maokx1/qcsfs/blob/main/docs/imgs/nn_chart2.png">
</p>

Both charts appeared in the [paper](https://arxiv.org/abs/1605.07678) linked by the article.

Based on the charts above, I made the decision to choose the MobileNetV2 architecture because it has a very high accuracy score per number of parameters it creates.

## Quick implementation summary ##

One of the first decisions I made during implementation was to put the learning parameters in the yaml file. The decisions are motivated by the fact that such file can be edited by anyone without knowledge of Python. Next, I implemented functions to load and resize images for MobileNetV2 input. This architecture accepts images with resolutions of 96x96, 128x128, 160x160, 192x192 or 224x224. I decided on 224x224, but this can be changed in the yaml file. For anyone wanting to better understand how to implement their own network based on MobileNetV2, I recommend the article: ["Create Your Own Image Classification Model using Python and Keras"](https://www.analyticsvidhya.com/blog/2020/10/create-image-classification-model-python-keras/). For my own purposes, I decided to change the sequential model of the neural network and add an early stop condition. Training will stop early if there is no result improvement for the loss function for 5 epochs in a row. The output of the model created with the [code](https://github.com/Maokx1/qcsfs/blob/main/src/qcsfs/neural_network/nn_training.py) is between 0 and 1. Where the value 0 means 100% certainty that the screw belongs to the good class, and 1 means that it belongs to the damaged class. For this reason, I introduced membership boundary in the yaml file. By default it is 0.5, which means that all screws that received a value less than 0.5 from the neural network will be classified as good, and the rest are classified as damaged. 

## Best results ##

To determine which model is the best I used the [f1-score](https://deepai.org/machine-learning-glossary-and-terms/f-score). I was able to get the following results:

- | Precision | Recall | F1-Score | Support
| :---:  | :---: | :---: | :---: | :---:|
Good Screw(Class 0) | 1.00 | 0.92 | 0.96 | 150
Damaged Screw(Class 1) | 0.93 | 1.00 | 0.96 | 150

Accuracy | Support
| :---:  | :---: 
0.96 | 300

## Sources ##

* [Dataset](https://www.kaggle.com/datasets/thomasdubail/screwanomalies-detection)
* [Data Augmentation](https://www.tensorflow.org/api_docs/python/tf/keras/preprocessing/image/ImageDataGenerator)
* [Comparison of deep neural networks](https://culurciello.medium.com/analysis-of-deep-neural-networks-dcf398e71aae)
* [Create Your Own Image Classification Model using Python and Keras](https://www.analyticsvidhya.com/blog/2020/10/create-image-classification-model-python-keras/)
* [F-score](https://deepai.org/machine-learning-glossary-and-terms/f-score)