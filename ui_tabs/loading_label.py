from PyQt5.QtWidgets import QLabel
from PyQt5.QtGui import QPixmap
from PyQt5.QtCore import Qt, QSize
import os
import sys

class LoadingSpinner(QLabel):
    def __init__(self, parent=None):
        """Build the overlay: a translucent background, the (static) EDyssey
        logo, and a "LOADING..." text label on top - hidden until start() is
        called."""
        super().__init__(parent)
        self.setAlignment(Qt.AlignCenter)
        self.setStyleSheet("background-color: rgba(0, 0, 0, 100); border-radius: 10px;")

        # __file__ isn't reliable here in a frozen build: this module is
        # compiled into the app's embedded archive, so it always reports
        # sys._MEIPASS/ui_tabs/loading_label.py regardless of where the
        # actual logo/ data folder is staged (see EDyssey.spec's datas=).
        if hasattr(sys, '_MEIPASS'):
            fn_logo = os.path.join(sys._MEIPASS, 'EDyssey', 'ui_tabs', 'logo', 'EDyssey_logo.png')
        else:
            directory_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            fn_logo = os.path.join(directory_path, 'ui_tabs', 'logo', 'EDyssey_logo.png')
        size = QSize(300, 300)
        self._base_pixmap = QPixmap(fn_logo).scaled(
            size, Qt.KeepAspectRatio, Qt.SmoothTransformation)
        self.setPixmap(self._base_pixmap)

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

    def start(self):
        self.show()

    def stop(self):
        self.hide()
