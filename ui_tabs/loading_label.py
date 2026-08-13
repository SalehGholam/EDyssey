from PyQt5.QtWidgets import QLabel
from PyQt5.QtGui import QPixmap, QTransform
from PyQt5.QtCore import Qt, QSize, QTimer
import os

class LoadingSpinner(QLabel):
    def __init__(self, parent=None):
        """Build the overlay: a translucent background, the EDyssey logo
        spinning in place, and a "LOADING..." text label on top - hidden
        until start() is called. A rotating static image rather than a GIF
        (as this used to be) since ui_tabs/logo/ - the only asset folder
        EDyssey.spec bundles as `datas` - is the only place this can live
        and be found in a frozen build."""
        super().__init__(parent)
        self.setAlignment(Qt.AlignCenter)
        self.setStyleSheet("background-color: rgba(0, 0, 0, 100); border-radius: 10px;")

        file_path = os.path.abspath(__file__)
        directory_path = os.path.dirname(os.path.dirname(file_path))
        size = QSize(300, 300)
        self._base_pixmap = QPixmap(os.path.join(directory_path, 'ui_tabs', 'logo', 'EDyssey_logo.png')
                                     ).scaled(size, Qt.KeepAspectRatio, Qt.SmoothTransformation)
        self.setPixmap(self._base_pixmap)

        self._angle = 0
        self._spin_timer = QTimer(self)
        self._spin_timer.setInterval(30)
        self._spin_timer.timeout.connect(self._advance)

        self.text_label = QLabel("LOADING...", self)
        self.text_label.setStyleSheet("""
            color: white;
            font-size: 24px;
            font-weight: bold;
            background-color: transparent;
        """)
        self.text_label.setAlignment(Qt.AlignTop | Qt.AlignHCenter)
        self.text_label.resize(size)

        self.resize(size)
        self.hide()

    def _advance(self):
        self._angle = (self._angle + 4) % 360
        rotated = self._base_pixmap.transformed(QTransform().rotate(self._angle), Qt.SmoothTransformation)
        self.setPixmap(rotated)

    def start(self):
        self.show()
        self._spin_timer.start()

    def stop(self):
        self._spin_timer.stop()
        self._angle = 0
        self.setPixmap(self._base_pixmap)
        self.hide()
