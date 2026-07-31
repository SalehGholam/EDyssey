# -*- coding: utf-8 -*-
"""Tests for py4DTomo.io_utils.contrast - the 8-bit contrast-stretching
methods (percentile/minmax/std) added this session, including the
outlier-robustness behaviour that's the whole point of defaulting to
percentile instead of minmax (see contrast.py's module docstring).
"""
import numpy as np
import pytest
import hyperspy.api as hs

from py4DTomo.io_utils import contrast


def _signal(data):
    return hs.signals.Signal2D(data.astype(np.float32))


@pytest.fixture
def clean_stack():
    rng = np.random.default_rng(0)
    return rng.normal(100, 10, size=(3, 32, 32)).astype(np.float32)


@pytest.fixture
def stack_with_hot_pixel(clean_stack):
    data = clean_stack.copy()
    data[0, 0, 0] = 5000.0  # single outlier pixel in frame 0
    return data


# ---------------------------------------------------------------------------
# _contrast_bounds
# ---------------------------------------------------------------------------

def test_contrast_bounds_minmax_matches_raw_extremes():
    data = np.array([[1.0, 2.0], [3.0, 4.0]])
    lo, hi = contrast._contrast_bounds(data, axis=None, method='minmax')
    assert lo == 1.0
    assert hi == 4.0


def test_contrast_bounds_percentile_is_between_min_and_max():
    data = np.arange(100, dtype=np.float32)
    lo, hi = contrast._contrast_bounds(data, axis=None, method='percentile', plow=1.0, phigh=99.0)
    assert data.min() <= lo <= hi <= data.max()
    assert lo > data.min()  # percentile clip is strictly inside the raw range here
    assert hi < data.max()


def test_contrast_bounds_std_uses_mean_and_std():
    data = np.array([10.0, 20.0, 30.0])
    mean, std = data.mean(), data.std()
    lo, hi = contrast._contrast_bounds(data, axis=None, method='std', n_std=2.0)
    assert lo == pytest.approx(mean - 2 * std)
    assert hi == pytest.approx(mean + 2 * std)


def test_contrast_bounds_unknown_method_raises():
    data = np.array([1.0, 2.0, 3.0])
    with pytest.raises(ValueError):
        contrast._contrast_bounds(data, axis=None, method='bogus')


# ---------------------------------------------------------------------------
# convert_to_8bit / convert_img_to_8bit - shape/dtype/range contract
# ---------------------------------------------------------------------------

@pytest.mark.parametrize('method', list(contrast.CONTRAST_METHODS))
def test_convert_to_8bit_output_contract(clean_stack, method):
    s8 = contrast.convert_to_8bit(_signal(clean_stack), method=method)
    assert s8.data.shape == clean_stack.shape
    assert s8.data.dtype == np.uint8
    assert s8.data.min() >= 0
    assert s8.data.max() <= 255


@pytest.mark.parametrize('method', list(contrast.CONTRAST_METHODS))
def test_convert_img_to_8bit_output_contract(clean_stack, method):
    img8 = contrast.convert_img_to_8bit(clean_stack[0], method=method)
    assert img8.shape == clean_stack[0].shape
    assert img8.dtype == np.uint8


def test_convert_to_8bit_does_not_mutate_input(clean_stack):
    original = clean_stack.copy()
    contrast.convert_to_8bit(_signal(clean_stack), method='percentile')
    np.testing.assert_array_equal(clean_stack, original)


# ---------------------------------------------------------------------------
# Outlier robustness: percentile should shrug off a single hot pixel far
# more than minmax does - this is the entire reason percentile is the
# default (see contrast.py's docstring).
# ---------------------------------------------------------------------------

def test_percentile_is_more_robust_to_hot_pixel_than_minmax(stack_with_hot_pixel):
    frame = stack_with_hot_pixel[0]  # contains the 5000.0 outlier
    normal_pixel_value = stack_with_hot_pixel[1, 5, 5]  # a typical ~100 value, different frame

    img_minmax = contrast.convert_img_to_8bit(frame, method='minmax')
    img_percentile = contrast.convert_img_to_8bit(frame, method='percentile', plow=1.0, phigh=99.0)

    # Under minmax, the hot pixel dominates the range and crushes everything
    # else toward 0; under percentile, normal-range pixels still spread out
    # meaningfully. Compare each method's own 5th-percentile pixel value as
    # a proxy for "how much of the normal range survived the stretch".
    assert np.percentile(img_percentile, 50) > np.percentile(img_minmax, 50)
