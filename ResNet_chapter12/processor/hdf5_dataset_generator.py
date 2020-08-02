from keras.utils import np_utils
import numpy as np
import h5py
import sys
import gc

class HDF5DatasetGenerator:
    def __init__(self, dbPath, batchSize, preprocessors=None, aug=None, binarize=True, classes=2):
        self.batchSize = batchSize
        self.preprocessors = preprocessors
        self.aug = aug
        self.binarize = binarize
        self.classes = classes
        
        self.db = h5py.File(dbPath, "r")
        self.numImages = self.db["labels"].shape[0]

    def generator(self, passes=np.inf):
        epochs = 0
        #images = None
        #labels = None
        while epochs < passes:
            #print("Processing epoch# ", epochs)
            for i in np.arange(0, self.numImages, self.batchSize):
                images = self.db["images"][i : i + self.batchSize]
                labels = self.db["labels"][i : i + self.batchSize]
                
                #print(hex(id(images)))
                #print("After alloc: ", sys.getrefcount(object))

                if self.binarize:
                    labels = np_utils.to_categorical(labels, self.classes)

                if self.preprocessors is not None:
                    procImages = []
                
                    for image in images:
                        for p in self.preprocessors:
                            image = p.preprocess(image)
                        procImages.append(image)

                    images = np.array(procImages)

                if self.aug is not None:
                    (images, labels) = next(self.aug.flow(images, labels, batch_size=self.batchSize))

                #print("Before yield: ", sys.getrefcount(object))
                yield (images, labels)
                #print("After yield: ", sys.getrefcount(object))
                #del images
                #del labels
                #print("Calling gc.collect()");
                #gc.collect()
            epochs += 1
   
    def close(self):
        self.db.close()



