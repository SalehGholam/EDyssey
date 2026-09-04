import os
import sys

# workers/ (repo root, or sys._MEIPASS when frozen) holds worker_*.py,
# which several tab modules below bare-import - must be on sys.path first.
_workers_dir = os.path.join(
    getattr(sys, '_MEIPASS', os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    'workers')
if _workers_dir not in sys.path:
    sys.path.append(_workers_dir)

from .tab_create_navSignal import Tab_Create_NavSignal
from .tab_tracking_cv2 import Tab_Tracking_CV2
from .tab_roi_4d import Tab_ROI_on_4D
from .tab_sam2 import Tab_SAM2
from .tab_edit import EditSettingsDialog
from .worker_thread import WorkerThread_General
from .loading_label import LoadingSpinner
from .object_detection_widget import Object_Detector_Widget