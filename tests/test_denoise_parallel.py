# -*- coding: utf-8 -*-
"""ContrastScalingBox.rescale_async's parallel per-frame denoise path
(SAM2 Tracker/ROI Tracker "Apply to All Images", driven by the ribbon's
"CPU Cores" spinbox via n_workers=) - added so a long navigation stack's
denoise step (the only per-frame Python loop in rescale_async; the
contrast stretch itself is vectorized) can use more than one core.

Threads, not processes: every EDyssey.io_utils.denoise method is
scipy.ndimage/skimage, which release the GIL in their own compiled inner
loops - see contrast_scaling.rescale_async's own n_workers docstring for
why this is a plain ThreadPoolExecutor, unlike eventem_backend's
confirmed-unsafe-to-parallelize eventem/pyeventem calls.

n_workers=1 must reproduce the exact previous sequential behavior (the
default, so every pre-existing caller is unaffected); n_workers>1 must
produce a bit-identical result (each frame is denoised independently, so
the only thing concurrency changes is completion order) plus correct
final progress.
"""
import os
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

pytest.importorskip("PyQt5")
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import PyQt5.QtWidgets as qtw  # noqa: E402
from PyQt5.QtCore import QThreadPool  # noqa: E402
import hyperspy.api as hs  # noqa: E402

from ui_tabs.contrast_scaling import ContrastScalingBox  # noqa: E402


@pytest.fixture(scope="module")
def qapp():
    return qtw.QApplication.instance() or qtw.QApplication(sys.argv)


@pytest.fixture
def stack_signal():
    rng = np.random.default_rng(0)
    data = rng.normal(100, 10, size=(6, 24, 24)).astype(np.float32)
    return hs.signals.Signal2D(data)


def _box_with_gaussian_denoise(qapp):
    box = ContrastScalingBox()
    box.box_denoise.combo_method.setCurrentIndex(
        box.box_denoise.combo_method.findText('Gaussian Blur'))
    box.box_denoise.spinbox_param.setValue(1.5)
    return box


def _run_rescale_async(qapp, box, raw_signal, n_workers, threadpool=None):
    threadpool = threadpool or QThreadPool()
    results = {}
    progress_calls = []
    box.rescale_async(
        raw_signal, threadpool,
        on_done=lambda s8: results.__setitem__('s8', s8),
        on_error=lambda tb: results.__setitem__('error', tb),
        on_progress=lambda cur, tot: progress_calls.append((cur, tot)),
        n_workers=n_workers,
    )
    threadpool.waitForDone(10000)
    # Cross-thread queued signal delivery needs the event loop pumped even
    # after the worker itself has finished (waitForDone only guarantees the
    # QRunnable ran, not that its queued signals were already dispatched).
    for _ in range(50):
        qapp.processEvents()
    return results, progress_calls


def test_n_workers_1_matches_prior_sequential_default(qapp, stack_signal):
    """Omitting n_workers (the default) must still work and produce the
    same result as explicitly passing 1."""
    box = _box_with_gaussian_denoise(qapp)
    results_default, _ = _run_rescale_async(qapp, box, stack_signal, n_workers=1)
    box2 = _box_with_gaussian_denoise(qapp)
    threadpool = QThreadPool()
    results_explicit = {}
    box2.rescale_async(stack_signal, threadpool,
                       on_done=lambda s8: results_explicit.__setitem__('s8', s8))
    threadpool.waitForDone(10000)
    for _ in range(50):
        qapp.processEvents()
    assert 'error' not in results_default
    np.testing.assert_array_equal(results_default['s8'].data, results_explicit['s8'].data)


def test_parallel_denoise_matches_sequential_result(qapp, stack_signal):
    box_seq = _box_with_gaussian_denoise(qapp)
    results_seq, progress_seq = _run_rescale_async(qapp, box_seq, stack_signal, n_workers=1)
    assert 'error' not in results_seq, results_seq.get('error')

    box_par = _box_with_gaussian_denoise(qapp)
    results_par, progress_par = _run_rescale_async(qapp, box_par, stack_signal, n_workers=4)
    assert 'error' not in results_par, results_par.get('error')

    np.testing.assert_array_equal(results_seq['s8'].data, results_par['s8'].data)

    total = stack_signal.data.shape[0]
    assert progress_seq[-1] == (total, total)
    assert progress_par[-1] == (total, total)
    # Every frame accounted for exactly once, regardless of completion order.
    assert sorted(c for c, _ in progress_par) == list(range(1, total + 1))


def test_parallel_denoise_disabled_is_unaffected_by_n_workers(qapp, stack_signal):
    """method='None' (the default) never enters the per-frame loop at all -
    n_workers must be a harmless no-op in that case."""
    box = ContrastScalingBox()  # denoise left at its default ('None')
    results, progress_calls = _run_rescale_async(qapp, box, stack_signal, n_workers=8)
    assert 'error' not in results, results.get('error')
    assert progress_calls == []  # nothing to report - no per-frame loop ran
    np.testing.assert_array_equal(
        results['s8'].data.shape, stack_signal.data.shape)
