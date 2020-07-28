from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from imagetoarraypreprocessor import ImageToArrayPreprocessor
from preprocessor import SimplePreprocessor
from datasetloader import SimpleDatasetLoader
from aspectawarepreprocessor import AspectAwarePreprocessor
from minivggnet import MiniVGGNet
from keras.optimizers import SGD
import argparse
import cv2
import imutils
from imutils import paths
import os
import numpy as np


ap = argparse.ArgumentParser()
ap.add_argument("-d", "--dataset", required=True, help="Path to input dataset")
args = vars(ap.parse_args())

print("[INFO] loading images...")
imagePaths = list(paths.list_images(args["dataset"]))
classNames = [p.split(os.path.sep)[-2] for p in imagePaths]
classNames = [str(x) for x in np.unique(classNames)]

sp = SimplePreprocessor(64, 64)
iap = ImageToArrayPreprocessor()
sdl = SimpleDatasetLoader(preprocessors=[sp, iap])
(data, labels) = sdl.load(imagePaths, verbose=500)
data = data.astype("float")/255.0

(X_train, X_test, y_train, y_test) = train_test_split(data, labels, test_size=0.25)

lb = LabelBinarizer()
y_train = lb.fit_transform(y_train)
y_test = lb.transform(y_test)

print("[INFO] building and training model...")
model = MiniVGGNet.build(64, 64, 3, len(classNames))
opt = SGD(lr=0.05)
model.compile(loss="categorical_crossentropy", optimizer=opt, metrics=["accuracy"])
H = model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=100, batch_size=32, verbose=1)

model.save('minivggnet_without_data_augmentation.hdf5')

print("[INFO] evaluating network...")
preds = model.predict(X_test, batch_size=32)
print(classification_report(y_test.argmax(axis=1), preds.argmax(axis=1), target_names=classNames))



import matplotlib.pyplot as plt
plt.style.use("ggplot")
plt.figure()
plt.plot(np.arange(0, 100), H.history["loss"], label="train_loss")
plt.plot(np.arange(0, 100), H.history["val_loss"], label="val_loss")
plt.plot(np.arange(0, 100), H.history["accuracy"], label="train_acc")
plt.plot(np.arange(0, 100), H.history["val_accuracy"], label="val_acc")
plt.title("Training Loss and Accuracy")
plt.xlabel("Epoch #")
plt.ylabel("Loss/Accuracy")
plt.legend()
plt.show()

