# -*- coding: utf-8 -*-
"""ROI Tracker's "Extract!" must only process frames inside the ROI's current Start-End range,
even when tracking already stored boxes for frames beyond a later-lowered End."""
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path[:0] = [ROOT, os.path.join(ROOT, 'workers')]

from ui_tabs.tab_tracking_cv2 import frames_to_extract  # noqa: E402


def _out_rois(n, first_tracked=0, last_tracked=None):
    """(n, 4) boxes, non-empty only for frames [first_tracked, last_tracked)."""
    last_tracked = n if last_tracked is None else last_tracked
    out = np.zeros((n, 4), dtype=np.int16)
    out[first_tracked:last_tracked] = (5, 5, 10, 10)
    return out


def test_all_tracked_frames_when_end_is_unchanged():
    assert frames_to_extract(_out_rois(10), 0, 10, 10) == list(range(10))


def test_lowering_end_after_tracking_limits_extraction():
    assert frames_to_extract(_out_rois(10), 0, 6, 10) == [0, 1, 2, 3, 4, 5]


def test_start_frame_is_respected():
    assert frames_to_extract(_out_rois(10), 3, 7, 10) == [3, 4, 5, 6]


def test_untracked_frames_are_skipped_even_when_end_is_higher():
    assert frames_to_extract(_out_rois(10, 2, 6), 0, 10, 10) == [2, 3, 4, 5]


def test_never_exceeds_the_number_of_files_or_boxes():
    assert frames_to_extract(_out_rois(10), 0, 10, 4) == [0, 1, 2, 3]
    assert frames_to_extract(_out_rois(5), 0, 10, 10) == [0, 1, 2, 3, 4]


def test_empty_range_gives_nothing():
    assert frames_to_extract(_out_rois(10), 6, 6, 10) == []
    assert frames_to_extract(_out_rois(10), 8, 3, 10) == []
