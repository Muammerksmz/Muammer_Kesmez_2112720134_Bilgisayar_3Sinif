import sys
import cv2
import numpy as np
from PyQt5.QtWidgets import QApplication, QWidget, QVBoxLayout, QPushButton, QLabel, QFileDialog
from PyQt5.QtGui import QPixmap, QImage
from PyQt5.QtCore import Qt
from datetime import datetime
import os

class PhotoEffectApp(QWidget):
    def __init__(self):
        super().__init__()

        self.image_label = QLabel(self)
        self.image_label.setAlignment(Qt.AlignCenter)

        self.load_button = QPushButton('Fotoğraf Yükle', self)
        self.load_button.clicked.connect(self.load_image)

        self.take_photo_button = QPushButton('Fotoğraf Çek', self)
        self.take_photo_button.clicked.connect(self.take_photo)

        self.adaptive_button = QPushButton('Adaptive Thresholding', self)
        self.adaptive_button.clicked.connect(self.apply_adaptive_threshold)

        self.otsu_button = QPushButton('Otsu Threshold', self)
        self.otsu_button.clicked.connect(self.apply_otsu_threshold)

        self.blur_button = QPushButton('Blur', self)
        self.blur_button.clicked.connect(self.apply_blur)

        self.gaussian_button = QPushButton('Gaussian Blur', self)
        self.gaussian_button.clicked.connect(self.apply_gaussian_blur)

        self.border_button = QPushButton('Kenarlık Ekle', self)
        self.border_button.clicked.connect(self.add_border)

        self.kernel_filter_button = QPushButton('Kernel Tabanlı Filtreleme', self)
        self.kernel_filter_button.clicked.connect(self.apply_kernel_filter)

        self.gamma_filter_button = QPushButton('Gamma Filtreleme', self)
        self.gamma_filter_button.clicked.connect(self.apply_gamma_filter)

        self.histogram_button = QPushButton('Histogram Eşitleme', self)
        self.histogram_button.clicked.connect(self.apply_histogram_equalization)

        self.edge_detection_button = QPushButton('Kenar Tespit Algoritmaları', self)
        self.edge_detection_button.clicked.connect(self.apply_edge_detection)


        layout = QVBoxLayout(self)
        layout.addWidget(self.load_button)
        layout.addWidget(self.take_photo_button)
        layout.addWidget(self.adaptive_button)
        layout.addWidget(self.otsu_button)
        layout.addWidget(self.blur_button)
        layout.addWidget(self.gaussian_button)
        layout.addWidget(self.border_button)
        layout.addWidget(self.kernel_filter_button)
        layout.addWidget(self.gamma_filter_button)
        layout.addWidget(self.histogram_button)
        layout.addWidget(self.edge_detection_button)
        layout.addWidget(self.image_label)

        self.efektsayisi = 0

        self.image = None
        self.original_image = None

        self.video_source = 0
        self.cap = cv2.VideoCapture(self.video_source)

        self.delay = 10
        self.update()

        self.captured_photo_path = None

    def take_photo(self):
        _, frame = self.cap.read()
        frame = cv2.flip(frame, +1)

        save_dir = "DATA"
        os.makedirs(save_dir, exist_ok=True)
        now = datetime.now()
        timestamp = now.strftime("%Y%m%d_%H%M%S")
        file_name = f"{save_dir}/foto_{timestamp}.png"
        cv2.imwrite(file_name, frame)
        print(f"Fotoğraf başarıyla kaydedildi: {file_name}")

        self.display_captured_photo(file_name)
        self.display_original_photo(frame)

        self.captured_photo_path = file_name

    def apply_adaptive_threshold(self):
        if self.original_image is not None:
            self.image = cv2.adaptiveThreshold(cv2.cvtColor(self.original_image, cv2.COLOR_BGR2GRAY), 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 11, 2)
            self.display_image()

    def apply_otsu_threshold(self):
        if self.original_image is not None:
            _, self.image = cv2.threshold(cv2.cvtColor(self.original_image, cv2.COLOR_BGR2GRAY), 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            self.display_image()

    def apply_blur(self):
        if self.original_image is not None:
            self.image = cv2.blur(cv2.cvtColor(self.original_image, cv2.COLOR_BGR2RGB), (5, 5))
            self.display_image()

    def apply_gaussian_blur(self):
        if self.original_image is not None:
            self.image = cv2.GaussianBlur(cv2.cvtColor(self.original_image, cv2.COLOR_BGR2RGB), (5, 5), 0)
            self.display_image()

    def add_border(self):
        if self.original_image is not None:
            border_color = (0, 255, 0)  # You can change the border color (BGR format)
            border_size = 20  # You can adjust the size of the border as needed

            self.image = cv2.copyMakeBorder(cv2.cvtColor(self.original_image, cv2.COLOR_BGR2RGB), border_size, border_size, border_size, border_size, cv2.BORDER_CONSTANT, value=border_color)
            self.display_image()

    def apply_kernel_filter(self):
        if self.original_image is not None:
            kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])  # Örnek bir filtre matrisi, istediğinizi kullanabilirsiniz
            self.image = cv2.filter2D(self.original_image, -1, kernel)
            self.display_image()

    def apply_gamma_filter(self):
        if self.original_image is not None:
            gamma = 1.5  # Gamma değerini ihtiyacınıza göre ayarlayabilirsiniz
            self.image = np.power(self.original_image / 255.0, gamma) * 255
            self.image = self.image.astype(np.uint8)
            self.display_image()

    def apply_histogram_equalization(self):
        if self.original_image is not None:
            gray_image = cv2.cvtColor(self.original_image, cv2.COLOR_BGR2GRAY)
            equalized_image = cv2.equalizeHist(gray_image)
            self.image = cv2.cvtColor(equalized_image, cv2.COLOR_GRAY2RGB)
            self.display_image()

    def apply_edge_detection(self):
        if self.original_image is not None:
            gray_image = cv2.cvtColor(self.original_image, cv2.COLOR_BGR2GRAY)
            edges = cv2.Canny(gray_image, 50, 150)
            self.image = cv2.cvtColor(edges, cv2.COLOR_GRAY2RGB)
            self.display_image()


    def load_image(self):
        options = QFileDialog.Options()
        file_name, _ = QFileDialog.getOpenFileName(self, "Fotoğraf Seç", "", "Images (*.png *.jpg *.bmp *.jpeg);;All Files (*)", options=options)

        if file_name:
            self.original_image = cv2.imread(file_name)
            self.original_image = cv2.cvtColor(self.original_image, cv2.COLOR_BGR2RGB)  # Convert to RGB
            self.image = None  # Set to None when loading a new image
            self.display_image()

    def display_captured_photo(self, image):
        if isinstance(image, str):
            image = cv2.imread(image)
        elif not isinstance(image, (np.ndarray, np.generic)):
            print("Invalid image format.")
            return

        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        aspect_ratio = image.shape[1] / image.shape[0]
        new_width = int(self.image_label.height() * aspect_ratio)
        resized_image = cv2.resize(image, (new_width, self.image_label.height()))

        q_image = QImage(resized_image.data, new_width, self.image_label.height(), resized_image.strides[0], QImage.Format_RGB888)
        pixmap = QPixmap.fromImage(q_image)

        self.image_label.setPixmap(pixmap)

    def display_original_photo(self, image):
        if not isinstance(image, (np.ndarray, np.generic)):
            print("Invalid image format.")
            return

        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        aspect_ratio = image.shape[1] / image.shape[0]
        new_width = int(self.image_label.height() * aspect_ratio)
        resized_image = cv2.resize(image, (new_width, self.image_label.height()))

        q_image = QImage(resized_image.data, new_width, self.image_label.height(), resized_image.strides[0], QImage.Format_RGB888)
        pixmap = QPixmap.fromImage(q_image)

        self.image_label.setPixmap(pixmap)

    def display_image(self):
        if self.original_image is not None:
            display_image = self.original_image.copy() if self.image is None else cv2.cvtColor(self.image, cv2.COLOR_BGR2RGB)

            h, w, ch = display_image.shape
            bytes_per_line = ch * w
            q_image = QImage(display_image.data, w, h, bytes_per_line, QImage.Format_RGB888)
            pixmap = QPixmap.fromImage(q_image)
            self.image_label.setPixmap(pixmap)

    def update(self):
        _, frame = self.cap.read()
        frame = cv2.flip(frame, +1)

        if frame is not None:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            h, w, ch = frame.shape
            bytes_per_line = ch * w
            q_image = QImage(frame.data, w, h, bytes_per_line, QImage.Format_RGB888)
            pixmap = QPixmap.fromImage(q_image)
            self.image_label.setPixmap(pixmap)

        self.image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        self.original_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        self.cap.release()
        self.cap = cv2.VideoCapture(self.video_source)

        self.original_image = None

        self.image_label.repaint()

        self.image_label.setPixmap(pixmap)
        self.original_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        self.cap.release()
        self.cap = cv2.VideoCapture(self.video_source)

        self.image_label.repaint()

        self.image_label.setPixmap(pixmap)
        self.original_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        self.cap.release()
        self.cap = cv2.VideoCapture(self.video_source)

        self.image_label.repaint()

if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = PhotoEffectApp()
    window.setGeometry(100, 100, 800, 600)
    window.show()
    sys.exit(app.exec_())
