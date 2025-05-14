from PyQt5.QtWidgets import QLabel
from PyQt5.QtGui import QMovie
from PyQt5.QtCore import Qt, QSize
import os

class LoadingSpinner(QLabel):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setAlignment(Qt.AlignCenter)
        self.setStyleSheet("background-color: rgba(0, 0, 0, 100); border-radius: 10px;")
        # self.setFixedSize(500, 500)

        file_path = os.path.abspath(__file__)
        directory_path = os.path.dirname(file_path)
        self.movie = QMovie(os.path.join(directory_path, 'scream_loading_2.gif'))
        
        size = QSize(452, 573)
        self.movie.setScaledSize(size)
        self.setMovie(self.movie)

        # Remove fixed size so the QLabel matches the GIF
        # self.movie.jumpToFrame(0)  # Load first frame to get size
        # gif_size = self.movie.currentPixmap().size()
        
        
        self.resize(size)
        self.hide()

    def start(self):
        self.show()
        self.movie.start()

    def stop(self):
        self.movie.stop()
        self.hide()
