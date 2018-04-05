import numpy as np
import cv2
import os

class SimpleDatasetLoader:
    def __init__(self, preprocessors=None):
        self.preprocessors=preprocessors
        if self.preprocessors is None:
            self.preprocessors = []

    def preprocess(self, image):
        if self.preprocessors is None:
            return image
        for p in self.preprocessors:
            image = p.preprocess(image)
        return image

    def load(self, paths, verbose=-1):
        data = []
        labels = []
        for (i, imagepath) in enumerate(paths):
            # load image and extract class label assuming structure:
            # /path/to/dataset/{class}/{image}.jpg
            image = cv2.imread(imagePath)
            label = imagepath.split(os.path.sep)[-2]

            data.append(self.preprocess(image))
            labels.append(label)

            if verbose > 0 and i > 0 and (i+1) % verbose == 0:
                print("[INFO] processed {}/{}".format(i+1, len(paths)))

        return (np.array(data), np.array(labels))


