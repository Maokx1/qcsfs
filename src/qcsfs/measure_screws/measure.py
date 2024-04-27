# Finding object's contour: https://docs.opencv.org/3.4/da/d22/tutorial_py_canny.html
# Finding the main axis of the screw: https://automaticaddison.com/how-to-determine-the-orientation-of-an-object-using-opencv/
# Explanation of PCA: https://www.youtube.com/watch?v=FgakZw6K1QQ
# Partially based on: https://stackoverflow.com/questions/11627362/how-to-straighten-a-rotated-rectangle-area-of-an-image-using-opencv-in-python
# For better explanation check docs/measure_screws.md

import csv
from math import atan2
import os
import random

import cv2
from imutils import perspective
import numpy as np
from scipy.spatial import distance


def random_img(image_dir: str | os.PathLike) -> np.ndarray:
    """
    This function picks random image (files with extensions: .png, .jpg, .jpeg)
    from the given directory and loads it.
    #### Args:
        image_dir (str | os.PathLike): A valid path to the images directory.
    #### Returns:
        (np.ndarray): The image represented as numpy ndarray.
    #### Raises:

    """
    extensions = [".png", ".jpg", ".jpeg"]
    imgs = []
    for element in os.listdir(image_dir):
        if os.path.splitext(element)[-1].lower() in extensions:
            imgs.append(element)
    if not imgs:
        raise Exception("NoImages")
    rand_img = random.choice(imgs)
    img_ = cv2.imread(os.path.join(image_dir, rand_img))
    return cv2.resize(img_, (512, 512))


def get_contours(
    image_: np.ndarray,
    threshold: list[int] = [50, 70],
    show_canny: bool = False,
    min_area: int = 1000,
) -> list[np.ndarray]:
    """
    This function determines the contours of an object.
    It copies the original image, so that it is not modified.
    #### Args:
        image (np.ndarray): Image of the screw.
        threshold (list[int]): List containing lower and upper bound of threshold.
        show_canny (bool): If true, a window will appear showing image of detected contours.
        min_area (int): Objects with smaller surface area than min_area won't be taken into the account.
    #### Returns:
        (list[np.ndarray]): List of points approximating the contour of the screw.
    """
    copy_image = image_.copy()
    # if image is BGR convert it to grayscale
    if len(copy_image.shape) == 3:
        copy_image = cv2.cvtColor(copy_image, cv2.COLOR_BGR2GRAY)
    # de-noising, non local means is better at de-noising but slower than gaussian blur
    denoised_image = cv2.fastNlMeansDenoising(copy_image).astype(np.uint8)
    # denoised_image = cv2.GaussianBlur(copy_image, (5, 5), 0)
    # using canny algorithm to find edges
    canny_image = cv2.Canny(denoised_image, threshold[0], threshold[1])
    if show_canny:
        cv2.imshow("CannyImage", canny_image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    dilated_image = cv2.dilate(canny_image, np.ones((5, 5)), iterations=3)
    eroded_image = cv2.erode(dilated_image, np.ones((5, 5)), iterations=3)
    # contouring edged image; cv2.RETR_EXTERNAL -> it returns only extreme outer flags
    # https://docs.opencv.org/3.4/d9/d8b/tutorial_py_contours_hierarchy.html
    # cv2.CHAIN_APPROX_SIMPLE -> compressing segments and leaves only endpoints (rectangle is encoded as 4 points)
    # https://docs.opencv.org/3.4/d3/dc0/group__imgproc__shape.html#gga4303f45752694956374734a03c54d5ffa5f2883048e654999209f88ba04c302f5
    contours, _ = cv2.findContours(
        eroded_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    final_contours = []
    # getting rid of contours with small surface area (potential noise)
    for con in contours:
        area = cv2.contourArea(con)
        if area > min_area:
            final_contours.append(con)
    return final_contours


def measure_with_rect(image_: np.ndarray) -> tuple[np.ndarray, float]:
    """
    This function measures the screw using the rectangle method.
    Basically it determines box with the smallest surface area, that fits the whole screw.
    It copies the original image, so that it is not modified.
    #### Args:
        image (np.ndarray): Image of the screw.
    #### Returns:
        (tuple[np.ndarray, int]): Image of boxed screw and its length.
    """

    def midpoint(p1: np.ndarray, p2: np.ndarray) -> np.ndarray:
        """
        This function calculates midpoint between two points.
        #### Args:
            p1 (np.ndarray): First point.
            p2 (np.ndarray): Second point.
        #### Returns:
            (np.ndarray): A middle point between p1 and p2.
        """
        return (p1 + p2) * 0.5

    copy_image = image_.copy()
    final_contours = get_contours(copy_image, show_canny=False)
    # Returns minimal area rectangle containing whole screw
    # in the form (center(x, y), (width, height), angle of rotation)
    box = cv2.minAreaRect(final_contours[0])
    # Finds the four vertices of a rotated rect.
    box = cv2.boxPoints(box).astype(np.int32)
    # Returns coordinates of vertices in order: upper left, upper right, lower right, and lower left.
    box = perspective.order_points(box).astype(np.int32)

    top_left, top_right, bottom_right, bottom_left = box
    # Boxed screw, contourIdx=-1 means: take all points and connect them
    cv2.polylines(
        copy_image,
        [box],
        True,
        (0, 255, 0),
        2,
    )
    # Calculating midpoints between vertices to mark potential main axis of the screw
    top_midpoint = midpoint(top_left, top_right)
    bottom_midpoint = midpoint(bottom_left, bottom_right)
    left_midpoint = midpoint(top_left, bottom_left)
    right_midpoint = midpoint(top_right, bottom_right)

    top_bottom_distance = distance.euclidean(top_midpoint, bottom_midpoint)
    left_right_distance = distance.euclidean(left_midpoint, right_midpoint)
    # Screw is always longer than wide, so check which side is greater
    if top_bottom_distance > left_right_distance:
        top_midpoint = top_midpoint.astype(np.int32)
        bottom_midpoint = bottom_midpoint.astype(np.int32)
        cv2.line(copy_image, top_midpoint, bottom_midpoint, (255, 0, 255), 2)
        cv2.putText(
            copy_image,
            f"{top_bottom_distance:.3f}",
            (105, 50),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (255, 0, 255),
            2,
        )
        cv2.circle(copy_image, top_midpoint, 4, (0, 20, 50), -1)
        cv2.circle(copy_image, bottom_midpoint, 4, (0, 20, 50), -1)

        return copy_image, top_bottom_distance
    else:
        left_midpoint = left_midpoint.astype(np.int32)
        right_midpoint = right_midpoint.astype(np.int32)
        cv2.line(copy_image, left_midpoint, right_midpoint, (255, 0, 255), 2)
        cv2.putText(
            copy_image,
            f"{left_right_distance:.3f}",
            (105, 50),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (255, 0, 255),
            2,
        )
        cv2.circle(copy_image, left_midpoint, 4, (0, 20, 50), -1)
        cv2.circle(copy_image, right_midpoint, 4, (0, 20, 50), -1)

        return copy_image, left_right_distance


def measure_with_axis(
    image_: np.ndarray, show_axis: bool = False
) -> tuple[np.ndarray, int]:
    """
    This function measures the screw using the axis method.
    Basically it rotates the screw into the vertical position and measures the screw defined by its contour.
    It copies the original image, so that it is not modified.
    #### Args:
        image (np.ndarray): Image of the screw.
        show_axis (bool): If true, a window will appear showing image of screw with marked main axis.
    #### Returns:
        (tuple[np.ndarray, int]): Cropped image of the screw and its length.
    """
    # A vector with point approximating contour of the screw
    final_contours = get_contours(image_, show_canny=False)
    data_pts = (
        final_contours[0]
        .reshape(final_contours[0].shape[0], final_contours[0].shape[2])
        .astype(np.float64)
    )
    # If screw contour is determined correctly direction of one of eigenvectors
    # is the same as direction of main axis
    mean, eigenvectors, eigenvalues = cv2.PCACompute2(data_pts, np.empty(0))
    # Store the geometric centre of the screw
    center = (int(mean[0, 0]), int(mean[0, 1]))
    tilt = atan2(eigenvectors[0, 1], eigenvectors[0, 0])  # in radians
    tilt *= 180 / np.pi  # in degrees
    tilt -= 90  # rotate screw to a vertical position
    # determining 2nd point to construct axis
    p1 = (
        int(center[0] + eigenvectors[0, 0] * eigenvalues[0, 0]),
        int(center[1] + eigenvectors[0, 1] * eigenvalues[0, 0]),
    )
    if show_axis:
        ax_img = image_.copy()
        # marking main axis with a green line
        cv2.line(ax_img, p1, center, (0, 200, 0), 1, cv2.LINE_AA)
        # marking geometric center with a red dot
        cv2.circle(ax_img, center, 3, (0, 0, 255), -1)
        cv2.imshow("Main axis of an object", ax_img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    tilted_image = image_.copy()
    # cv2.warpAffine expects shape in format (length, height)
    shape = (tilted_image.shape[1], tilted_image.shape[0])
    rotation_matrix = cv2.getRotationMatrix2D(
        center=(shape[0] / 2, shape[1] / 2), angle=tilt, scale=1
    )
    tilted_image = cv2.warpAffine(src=tilted_image, M=rotation_matrix, dsize=shape)
    final_contours = get_contours(
        tilted_image, threshold=[50, 70], show_canny=False, min_area=1000
    )

    min_y, min_x, _ = tilted_image.shape
    # computing the bounding box for the contour, and drawing it on the tilted_image
    (x, y, w, h) = cv2.boundingRect(final_contours[0])
    min_x, max_x = min(x, min_x), max(x + w, 0)
    min_y, max_y = min(y, min_y), max(y + h, 0)
    final_image = tilted_image[min_y:max_y, min_x:max_x]
    length = final_image.shape[0]
    cv2.putText(
        final_image,
        f"{length}",
        (40, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (255, 0, 255),
        2,
    )  # marking screw length in pink

    return final_image, length


def rect_method(
    image_: np.ndarray, show_contour: bool = False
) -> tuple[np.ndarray, float]:
    """
    This function measures screw using rectangle method.
    It copies the original image, so that it is not modified.
    #### Args:
        image_ (np.ndarray): Image of the screw.
        show_image (bool): If true, two windows will appear showing original image and image of detected contours.
    #### Returns:
        (tuple[np.ndarray, float]): Image with marked length and length itself.
    """
    copy_image = image_.copy()
    if show_contour:
        cv2.imshow("OriginalImage", image_)
        contoured_image = image_.copy()
        # Contouring screw
        contours = get_contours(image_, show_canny=False)
        if len(contours) != 0:
            for con in contours:
                cv2.polylines(
                    img=contoured_image,
                    pts=[con],
                    isClosed=True,
                    color=(0, 0, 0),
                    thickness=2,
                )
        # Showing contoured screw
        cv2.imshow("Contoured screw", contoured_image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    return measure_with_rect(copy_image)


def axis_method(
    image_: np.ndarray, show_contour: bool = False
) -> tuple[np.ndarray, int]:
    """
    This function measures screw using axis method.
    It copies the original image, so that it is not modified.
    #### Args:
        image_ (np.ndarray): Image of the screw.
        show_image (bool): If true, two windows will appear showing original image and image of detected contours.
    #### Returns:
        (tuple[np.ndarray, int]): Image with marked length and length itself.
    """
    copy_image = image_.copy()
    if show_contour:
        cv2.imshow("OriginalImage", image_)
        contoured_image = image_.copy()
        contours = get_contours(contoured_image, show_canny=False)
        if len(contours) != 0:
            for con in contours:
                cv2.polylines(
                    img=contoured_image,
                    pts=[con],
                    isClosed=True,
                    color=(0, 0, 0),
                    thickness=2,
                )
        cv2.imshow("Contoured screw", contoured_image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    return measure_with_axis(copy_image, show_axis=False)


def log_measurements_rect(path_: str | os.PathLike):
    """
    This function loads all the screw images from the specified path,
    measures them using the rectangle method and saves the results to a csv file.
    #### Args:
        path_ (str | os.PathLike): A valid path to the images directory.
    """
    csv_path = "data/measuring_results/measurements_rect.csv"
    extensions = [".png", ".jpg", ".jpeg"]
    imgs = []
    for element in os.listdir(path_):
        if os.path.splitext(element)[-1].lower() in extensions:
            imgs.append(element)
    if not imgs:
        raise Exception("NoImages")
    else:
        with open(csv_path, "w", encoding="UTF8", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["path", "length"])
            for image_name in os.listdir(path_):
                image_ = cv2.imread(os.path.join(path_, image_name))
                image_ = cv2.resize(image_, (512, 512))
                _, length = measure_with_rect(image_)
                writer.writerow([os.path.join(path_, image_name), round(length, 3)])


def log_measurements_axis(path_: str | os.PathLike):
    """
    This function loads all the screw images from the specified path,
    measures them using the axis method and saves the results to a csv file.
    #### Args:
        path_ (str | os.PathLike): A valid path to the images directory.
    """
    csv_path = "data/measuring_results/measurements_axis.csv"
    extensions = [".png", ".jpg", ".jpeg"]
    imgs = []
    for element in os.listdir(path_):
        if os.path.splitext(element)[-1].lower() in extensions:
            imgs.append(element)
    if not imgs:
        raise Exception("NoImages")
    else:
        with open(csv_path, "w", encoding="UTF8", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["path", "length"])
            for image_name in imgs:
                image_ = cv2.imread(os.path.join(path_, image_name))
                image_ = cv2.resize(image_, (512, 512))
                _, length = measure_with_axis(image_)
                writer.writerow([os.path.join(path_, image_name), length])
