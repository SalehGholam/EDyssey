# -*- coding: utf-8 -*-
"""Tests for EDyssey.io_utils.video's frame/clip export safety: failures are logged and raised
(never silent), clips render on a worker thread without pyplot, and a dying ffmpeg is reported."""
import os
import subprocess
import sys
import threading

import numpy as np
import pytest

from EDyssey.io_utils import video


class _FakeLogger:
    def __init__(self):
        self.infos, self.errors = [], []

    def info(self, msg, *args, **kwargs):
        self.infos.append(msg % args if args else msg)

    def error(self, msg, *args, **kwargs):
        self.errors.append(msg % args if args else msg)


def _stack(n=3, size=24):
    rng = np.random.default_rng(0)
    return rng.integers(1, 200, size=(n, size, size)).astype(np.uint16)


def _run_in_thread(func, *args, **kwargs):
    """Run on a non-main thread, like the app's QThreadPool workers; returns the raised exception."""
    box = []

    def _target():
        try:
            func(*args, **kwargs)
        except Exception as exc:
            box.append(exc)

    t = threading.Thread(target=_target)
    t.start()
    t.join()
    return box[0] if box else None


def test_create_frames_writes_all_frames_and_only_clears_old_tifs(tmp_path):
    out = tmp_path / 'frames'
    out.mkdir()
    (out / '0099.tif').write_bytes(b'old')
    (out / 'keep.txt').write_text('mine')
    log = _FakeLogger()
    video.create_frames(str(out), _stack(3), logger=log)
    assert sorted(p.name for p in out.glob('*.tif')) == ['0001.tif', '0002.tif', '0003.tif']
    assert (out / 'keep.txt').exists()
    assert not log.errors


def test_create_frames_creates_missing_parents(tmp_path):
    out = tmp_path / 'a' / 'b' / 'frames'
    video.create_frames(str(out), _stack(2))
    assert len(list(out.glob('*.tif'))) == 2


def test_create_frames_reports_failures_but_keeps_going(tmp_path, monkeypatch):
    real = video.tifffile.imwrite
    calls = []

    def flaky(path, data, *a, **k):
        calls.append(path)
        if len(calls) == 2:
            raise OSError('disk says no')
        return real(path, data, *a, **k)

    monkeypatch.setattr(video.tifffile, 'imwrite', flaky)
    log = _FakeLogger()
    with pytest.raises(RuntimeError, match=r'1 of 3 TIFF frames failed'):
        video.create_frames(str(tmp_path / 'frames'), _stack(3), logger=log)
    assert len(calls) == 3
    assert any('disk says no' in e for e in log.errors)


@pytest.mark.parametrize('make_clip', [
    lambda fn, log: video.create_clip_dp(fn, _stack(), scale=0.01, dpi=40, fps=2, logger=log),
    lambda fn, log: video.create_clip_tracking(
        fn, _stack(), rois=[(2, 2, 6, 6)] * 3, ref_rois=[(1, 1, 10, 10)] * 3, scale=1.0,
        dpi=40, fps=2, logger=log),
    lambda fn, log: video.create_clip_tracking_with_mask(
        fn, _stack(), np.ones((3, 24, 24), dtype=bool), obj_id=2, scale=1.0, dpi=40, fps=2, logger=log),
], ids=['dp', 'tracking', 'tracking_with_mask'])
def test_clips_fall_back_to_gif_on_a_worker_thread(tmp_path, monkeypatch, make_clip):
    monkeypatch.setattr(video, '_find_ffmpeg', lambda: None)
    log = _FakeLogger()
    fn = str(tmp_path / 'clip')
    assert _run_in_thread(make_clip, fn, log) is None
    assert os.path.getsize(fn + '.gif') > 0
    assert any('is created!' in m for m in log.infos)
    assert not log.errors


def test_clip_uses_ffmpeg_when_available(tmp_path):
    if not video._find_ffmpeg():
        pytest.skip('no ffmpeg available')
    fn = str(tmp_path / 'clip')
    assert _run_in_thread(video.create_clip_dp, fn, _stack(), dpi=40, fps=2) is None
    assert os.path.getsize(fn + '.mp4') > 0


@pytest.mark.skipif(sys.platform != 'win32', reason='stale std handles are a Windows failure mode')
def test_ffmpeg_starts_even_with_invalid_std_handles(tmp_path):
    """Regression: an invalid OS stdout/stderr handle made an inheriting Popen fail with
    OSError WinError 6 (a reported Save Results failure). Runs in a child so pytest is unaffected."""
    child = (
        'import ctypes, os, subprocess, sys, tempfile\n'
        'from EDyssey.io_utils import video\n'
        'k = ctypes.windll.kernel32\n'
        'k.SetStdHandle(-11, ctypes.c_void_p(0x7FFFFFF0)); k.SetStdHandle(-12, ctypes.c_void_p(0x7FFFFFF4))\n'
        'with tempfile.TemporaryFile() as err:\n'
        '    p = video._start_ffmpeg([sys.executable, "-c", "import sys; sys.stdin.read()"], err)\n'
        '    p.stdin.close()\n'
        '    sys.exit(p.wait())\n'
    )
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    result = subprocess.run([sys.executable, '-c', child], cwd=root, timeout=120,
                            stdin=subprocess.DEVNULL, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    assert result.returncode == 0, result.stderr.decode(errors='replace')[-500:]


def test_clip_reports_a_dying_ffmpeg(tmp_path, monkeypatch):
    monkeypatch.setattr(video, '_find_ffmpeg', lambda: sys.executable)  # python rejects ffmpeg's flags
    log = _FakeLogger()
    exc = _run_in_thread(video.create_clip_dp, str(tmp_path / 'clip'), _stack(), dpi=40, fps=2, logger=log)
    assert isinstance(exc, RuntimeError) and 'ffmpeg' in str(exc)
    assert any('DP clip failed' in e for e in log.errors)
