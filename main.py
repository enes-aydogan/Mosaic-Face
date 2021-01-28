"""This code written by Muhammet Enes AYDOĞAN"""
from PyQt5.QtWidgets import *
from PyQt5.QtGui import QPixmap, QImage
import sys
import cv2
import numpy as np


class Window(QWidget):
    CASCADE_PATH = "haarcascade_frontalface_default.xml"

    def __init__(self):
        super().__init__()
        self.hBox = QHBoxLayout()
        self.List = QListWidget()

        """loading the photo with cv2"""
        self.img = cv2.imread('second.jpg')
        self.imgCpy = self.img.copy()
        self.label = QLabel("Label")

        """convert from an opencv image to QPixmap"""
        self.qtImg = self.convert_cv_qt(self.img)
        self.pixmap = QPixmap(self.qtImg)

        self.findFaces()
        self.initUI()

    def initUI(self):
        self.setWindowTitle("Image")
        self.setGeometry(0, 0, 600, 600)
        self.label.setPixmap(self.pixmap)
        self.label.resize(self.pixmap.width(),
                          self.pixmap.height())

        self.List.doubleClicked.connect(self.listClicked)

        """the process of adding photo and face list to horizontal box"""
        self.hBox.addWidget(self.label)
        self.hBox.addWidget(self.List)
        self.setLayout(self.hBox)
        self.show()

    def findFaces(self):
        """face finding process"""
        face_Cascade = cv2.CascadeClassifier(self.CASCADE_PATH)
        gray = cv2.cvtColor(self.img, cv2.COLOR_BGR2GRAY)
        self.faces = face_Cascade.detectMultiScale(gray, 1.1, 4)
        self.List.addItem("none")

        """Adding faces to the list widget with ids."""
        for i in range(len(self.faces)):
            text = "Face " + str((i + 1))
            self.List.addItem(text)


    def mosaic(self, img, nsize=10):
        rows, cols, _ = img.shape
        img_copy = img.copy()

        for y in range(0, rows, nsize):
            for x in range(0, cols, nsize):
                img_copy[y:y + nsize, x:x + nsize] = (
                np.random.randint(0, 255), np.random.randint(0, 255), np.random.randint(0, 255))

        return img_copy


    def listClicked(self):
        self.img = self.imgCpy.copy()
        if self.List.currentItem().text() != "none":

            """find the id of the selected face"""
            item = self.List.currentItem().text().split(" ")
            sub = int(item[1]) - 1

            """the process of covering the face with mosaic."""
            (x, y, w, h) = self.faces[sub]
            cv2.rectangle(self.img, (x, y), (x + w, y + h), (255, 0, 0), 2)
            singleFace = self.img[y:y + h, x:x + w]
            mosaicedSingleFace = self.mosaic(singleFace)

            """opafication process"""
            self.img[y:y + h, x:x + w] = cv2.addWeighted(mosaicedSingleFace, 0.85, singleFace, 0.15, 2.0)

        self.qtImg = self.convert_cv_qt(self.img)
        self.pixmap = QPixmap(self.qtImg)
        self.label.setPixmap(self.pixmap)
        self.label.resize(self.pixmap.width(),
                          self.pixmap.height())

    def convert_cv_qt(self, cv_img):
        """Convert from an opencv image to QPixmap"""
        rgb_image = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb_image.shape
        bytes_per_line = ch * w
        convert_to_Qt_format = QImage(rgb_image.data, w, h, bytes_per_line, QImage.Format_RGB888)
        return QPixmap.fromImage(convert_to_Qt_format)


if __name__ == '__main__':
    App = QApplication(sys.argv)
    window = Window()
    sys.exit(App.exec())
