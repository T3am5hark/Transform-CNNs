import cv2

class ResizePreprocessor:
    def __init__(self, width, height, interp=cv2.INTER_AREA):
        # Store the target image width, height, and interpolation
        self.width = width
        self.height = height
        self.interp = interp

    def preprocess(self, image):
        return cv2.resize(image, (self.width, self.height),
                          interpolation=self.interp;
