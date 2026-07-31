# -*- coding: utf-8 -*-
"""Backward-compatible re-export shim.

This module's content used to live entirely in this one file. It's now
split into cohesive siblings (loaders.py, metadata.py, nav_image.py,
contrast.py, plotting.py, video.py, progress.py, pets.py) - see each for
what it owns. This file just re-exports everything from them, unchanged, so
every existing call site (`import py4DTomo.io_utils as io; io.convert_to_8bit(...)`
etc.) keeps working without modification.

Absolute imports (not relative `from .x import *`) are used deliberately:
worker_nav_img.py imports this file directly as a bare top-level module
(`sys.path.append('.../py4DTomo/io_utils'); import io_utils_ui as io`),
without importing the `py4DTomo.io_utils` package first - in that context
this module has no parent package, so a relative import would fail with
"attempted relative import with no known parent package". Absolute imports
work either way, since the repo root ends up on `sys.path` regardless of
which of the two ways this module gets imported.
"""
from py4DTomo.io_utils.progress import *
from py4DTomo.io_utils.plotting import *
from py4DTomo.io_utils.loaders import *
from py4DTomo.io_utils.metadata import *
from py4DTomo.io_utils.nav_image import *
from py4DTomo.io_utils.contrast import *
from py4DTomo.io_utils.video import *
from py4DTomo.io_utils.pets import *
from py4DTomo.io_utils.smart_scan import *
