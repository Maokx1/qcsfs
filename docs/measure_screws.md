# Measuring methods for screws #

## Why measure the length of a screw? ##

In the case of this package, the goal of the neural network model is to sort the screws into 2 groups: good and damaged screws. However, a screw may have no surface defects and still not meet predetermined standards. For this reason, I decided to check the length of the screw (as a example parameter). In this case I decided to measure the length using traditional image analysis methods. Since I don't have a reference point, I'm stuck with measuring the length in pixels.   

## Implementation summary ##

First of all, I decided to measure the screws when the images are 512x512 pixels. This saves me some time while I'm still getting accurate measurements. 
Before measuring, you need to get rid of the noise in the image. In my research, I've tested several approaches such as: mean filter, Gaussian filter, median filter and non-local means filter. Of all the above the non-local means filter is the best. While it is the best in de-noise an image, it is also the slowest (up to 350 times slower than the Gaussian filter). For the most accurate results, I highly recommend using NLMeans filter, but if you need a safe middle ground between speed and accuracy, go for Gaussian filter as it is the second best. After de-noising the image, I've decided the best way to measure the length of a screw is to figure out the area it occupies in the image. So I decided to use methods that determine the contour of the object. In my research, I've tested 2 approaches: [Sobel method](https://docs.opencv.org/3.4/d2/d2c/tutorial_sobel_derivatives.html) and [Canny method](https://docs.opencv.org/3.4/da/d22/tutorial_py_canny.html). To test them, I manually marked the area occupied by the screws in the image, and then compared with the areas marked by both methods. In this case, Canny's method turned out to be much better. The average relative error for the Canny method was 1.53%, and for the Sobel method 6.45%. Powyższe badania pozwoliły mi zaimplementować funkcję [get_contours](https://github.com/Maokx1/qcsfs/src/qcsfs/measure_screws/measure.py). An example result:

<p align="center">
    <img alt="Contoured screw" src="https://github.com/Maokx1/qcsfs/docs/imgs/contoured_screw.png" width="512" height="512">
</p>

### Rectangle method ###

Having a the screw contour made me think all I need now is a rectangle that can fit all these points approximating contour. Specifically, it should be a minimal area rectangle that will fit all these points. And OpenCV has exactly such a function. Having such a rectangle, you only need to measure its longest side, which should correspond to the length of the screw. Unfortunately, minimal area rectangle doesn't mean that its sides will be parallel to axes of the screw. You can see it in the image below:

<p align="center">
    <img alt="Boxed screw" src="https://github.com/Maokx1/qcsfs/docs/imgs/boxed_screw.png" width="512" height="512">
</p>

### Axis method ###

To solve the problem with the rectangle method, I've implemented axis method. In this method, the axis of the screw is determined first, so that the screw can be rotated vertically and than measured. To determine axes of the screw I used [PCA](https://en.wikipedia.org/wiki/Principal_component_analysis). This approach only works because the screw has an almost identical number of points on both sides approximating its contour. After rotating and measuring, we get the following exemplary results:

<p align="center">
    <img alt="Axis method results" src="https://github.com/Maokx1/qcsfs/docs/imgs/cut_screw.png">
</p>

## Comparisons and conclusions ##

To compare the two approaches, I decided to measure a few screws by hand and compare with the results obtained by these methods. Here are the results:

<p align="center">
    <img alt="Comparison chart" src="https://github.com/Maokx1/qcsfs/docs/imgs/method_comparison.png">
</p>

**On average, the axis method is off by 4 pixels(relative error = 0,91%) and the rectangle method is off by 2 (relative error = 0,46%). This means that both methods are very accurate, but the rectangle method is better.**

The reason why the axis method has worse results than rectangle method is that the image changes when rotating. As a result of the rotation, colors are interpolated, which has a particularly strong effect on the blurring of the screw edge, causing errors in the length measurement. You can clearly see this in the images below:

<table>
  <tr>
    <td>Tip of the screw before rotation</td>
    <td>Tip of the screw after rotation</td>
  </tr>
  <tr>
    <td><img alt="Before rotation" src="https://github.com/Maokx1/qcsfs/docs/imgs/before_rotation.png"></td>
    <td><img alt="After rotation" src="https://github.com/Maokx1/qcsfs/docs/imgs/after_rotation.png"></td>
  </tr>
</table>

## Sources ##

* [Contours Hierarchy](https://docs.opencv.org/3.4/d9/d8b/tutorial_py_contours_hierarchy.html)
* [Sobel Derivatives](https://docs.opencv.org/3.4/d2/d2c/tutorial_sobel_derivatives.html)
* [Canny Edge Detection](https://docs.opencv.org/3.4/da/d22/tutorial_py_canny.html)
* [PCA](https://en.wikipedia.org/wiki/Principal_component_analysis)
* [How to Determine the Orientation of an Object Using OpenCV](https://automaticaddison.com/how-to-determine-the-orientation-of-an-object-using-opencv/)
* [Straighten rotated object](https://stackoverflow.com/questions/11627362/how-to-straighten-a-rotated-rectangle-area-of-an-image-using-opencv-in-python)