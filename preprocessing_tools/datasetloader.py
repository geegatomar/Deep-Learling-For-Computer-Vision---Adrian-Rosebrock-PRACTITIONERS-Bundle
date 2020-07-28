import cv2
import os
import numpy as np


class SimpleDatasetLoader:
    def __init__(self, preprocessors=None):
        self.preprocessors = preprocessors
        if preprocessors is None:
            preprocessors = []
    

    def load(self, imagePaths, verbose=-1):
        
        data = []
        labels = []

        for (i, imagePath) in enumerate(imagePaths):

            label = imagePath.split(os.path.sep)[-2]
            image = cv2.imread(imagePath)
            
            if self.preprocessors is not None:
                for p in self.preprocessors:
                    image = p.preprocess(image)
            
            data.append(image)
            labels.append(label)

            if verbose > 0 and i + 1 % verbose == 0:
                print("[INFO] processed {}/{}".format(i + 1, len(imageFiles)))

        return (np.array(data), np.array(labels))

                


