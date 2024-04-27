import numpy as np
import pytest

from . import (
    random_img,
    get_contours,
    measure_with_rect,
    log_measurements_rect,
    measure_with_axis,
    log_measurements_axis,
)


def test_random_img_no_img():
    with pytest.raises(Exception, match="NoImages"):
        random_img("tests")


def test_random_img():
    assert type(random_img("data/sample_screws/good")) == np.ndarray


@pytest.fixture
def sample_image() -> np.ndarray:
    """Returns sample image from good sample screws."""
    return random_img("data/sample_screws/good")


def test_get_contour(sample_image):
    assert len(get_contours(sample_image)) == 1


@pytest.fixture
def rect_result(sample_image) -> tuple[np.ndarray, float]:
    """Returns results of measuring screw with rectangle method."""
    return measure_with_rect(sample_image)


def test_measure_with_rect_marked_box(rect_result):
    img, _ = rect_result
    # checks if screw got boxed by looking for green (0, 255, 0) pixel
    assert (img == (0, 255, 0)).any(axis=-1).max()


def test_measure_with_rect_length(rect_result):
    _, length = rect_result
    assert round(length, 3) in [456.947, 454.529]


@pytest.fixture
def axis_result(sample_image) -> tuple[np.ndarray, int]:
    """Returns results of measuring screw with axis method."""
    return measure_with_axis(sample_image)


def test_measure_with_axis_marked_length(axis_result):
    img, _ = axis_result
    # checks if length of the screw is in the image (marked pinkish (255, 0, 255))
    assert (img == (255, 0, 255)).any(axis=-1).max()


def test_measure_with_axis_length(axis_result):
    _, length = axis_result
    assert round(length, 3) in [456, 458]


def test_log_measurments_rect_no_imgs():
    with pytest.raises(Exception, match="NoImages"):
        log_measurements_rect("tests")


def test_log_measurments_axis_no_imgs():
    with pytest.raises(Exception, match="NoImages"):
        log_measurements_axis("tests")
