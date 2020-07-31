import matplotlib
matplotlib.use("Agg")

from sklearn.preprocessing import LabelBinarizer
from processor.training_monitor import TrainingMonitor
from minigooglenet import MiniGoogLeNet
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import LearningRateScheduler
from keras.optimizers import SGD
from keras.datasets import cifar10
import numpy as np
import argparse
import os

NUM_EPOCHS = 70
INIT_LR = 5e-3

def poly_decay(epoch):
    maxEpochs = NUM_EPOCHS
    baseLR = INIT_LR
    power = 1.0

    alpha = baseLR * (1 - (epoch/float(maxEpochs))) ** power
    
    return alpha



ap = argparse.ArgumentParser()
ap.add_argument("-m", "--model", required=True, help="Path to output model")
ap.add_argument("-o", "--output", required=True, help="Path to output directory (logs, plots, etc.)")
args = vars(ap.parse_args())

print("loading dataset cifar10...")
((X_train, y_train), (X_test, y_test)) = cifar10.load_data()
X_train = X_train.astype("float")
X_test = X_test.astype("float")

mean = np.mean(X_train, axis=0)
X_train -= mean
X_test -= mean

lb = LabelBinarizer()
y_train = lb.fit_transform(y_train)
y_test = lb.transform(y_test)

aug = ImageDataGenerator(width_shift_range=0.1, height_shift_range=0.1, horizontal_flip=True, fill_mode="nearest")


figPath = os.path.sep.join([args["output"], "{}.png".format(os.getpid())])
jsonPath = os.path.sep.join([args["output"], "{}.json".format(os.getpid())])
callbacks = [TrainingMonitor(figPath, jsonPath=jsonPath), LearningRateScheduler(poly_decay)]

print("[INFO] compiling model...")
opt = SGD(lr=INIT_LR, momentum=0.9)
model = MiniGoogLeNet.build(width=32, height=32, depth=3, classes=10)
model.compile(loss="categorical_crossentropy", optimizer=opt, metrics=["accuracy"])

print("[INFO] training model...")
model.fit_generator(aug.flow(X_train, y_train, batch_size=64), validation_data=(X_test, y_test), steps_per_epoch=len(X_train)//64, epochs=NUM_EPOCHS, callbacks=callbacks, verbose=1)

print("[INFO] serializing network...")
model.save(args["model"])


