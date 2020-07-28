import cv2
import os
import numpy as np

class SimpleDatasetLoader:
    
    def __init__(self, preprocessors = []):
        self.preprocessors = preprocessors
    
    def load(self, imagePaths, verbose=-1):
        data = []
        labels = []

        for (i, imagePath) in enumerate(imagePaths):
            image = cv2.imread(imagePath)
            label = imagePath.split(os.path.sep)[-2]

            for p in self.preprocessors:
                image = p.preprocess(image)

            data.append(image)
            labels.append(label)
        
        if verbose > 0 and i > 0 and (i + 1) % verbose == 0:
            print("[INFO] loading and preprocessing {} / {}".format(i + 1, len(imagePaths)))

        return (np.array(data), np.array(labels))

