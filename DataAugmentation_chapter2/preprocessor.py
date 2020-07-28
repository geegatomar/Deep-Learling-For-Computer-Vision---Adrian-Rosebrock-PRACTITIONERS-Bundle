import cv2

class SimplePreprocessor:
    def __init__(self, height, width, inter=cv2.INTER_AREA):
        self.height = height
        self.width = width
        self.inter = inter

    def preprocess(self, image):
        return cv2.resize(image, (self.width, self.height), interpolation=self.inter)
