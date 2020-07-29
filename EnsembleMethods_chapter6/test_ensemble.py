from sklearn.preprocessing import LabelBinarizer
from sklearn.metrics import classification_report
from keras.models import load_model
from keras.datasets import cifar10
import numpy as np
import argparse
import glob
import os

ap = argparse.ArgumentParser()
ap.add_argument("-m", "--model", required=True, help="Path to models directory")
args = vars(ap.parse_args())

(X_test, y_test) = cifar10.load_data()[1]
X_test = X_test.astype("float")/255.0

lb = LabelBinarizer()
y_test = lb.fit_transform(y_test)

labelNames = ["airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"]

modelPaths = os.path.sep.join([args["model"], "*.model"])
modelPaths = list(glob.glob(modelPaths))
models = []

for (i, modelPath) in enumerate(modelPaths):
    print("[INFO] loading {} / {}".format(i + 1, len(modelPaths)))
    models.append(load_model(modelPath))


print("[INFO] evaluating ensemble...")
predictions = []

for model in models:
    predictions.append(model.predict(X_test, batch_size=64))

predictions = np.average(predictions, axis=0)
print(classification_report(y_test.argmax(axis=1), predictions.argmax(axis=1), target_names=labelNames))







