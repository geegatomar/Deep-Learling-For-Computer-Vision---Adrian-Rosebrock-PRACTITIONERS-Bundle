import matplotlib
matplotlib.use("Agg")

from sklearn.preprocessing import LabelBinarizer
from sklearn.metrics import classification_report
from mini_vggnet import MiniVGGNet
from keras.optimizers import SGD
from keras.preprocessing.image import ImageDataGenerator
from keras.datasets import cifar10
import matplotlib.pyplot as plt
import numpy as np
import argparse
import os


ap = argparse.ArgumentParser()
ap.add_argument("-o", "--output", required=True, help="Path to output directory")
ap.add_argument("-m", "--model", required=True, help="Path to output models directory")
ap.add_argument("-n", "--num_models", type=int, default=5, help="# of models to train")
args = vars(ap.parse_args())

print("[INFO] loading dataset...")
((X_train, y_train), (X_test, y_test)) = cifar10.load_data()
X_train = X_train.astype("float")/255.0
X_test = X_test.astype("float")/255.0
lb = LabelBinarizer()
y_train = lb.fit_transform(y_train)
y_test = lb.transform(y_test)


labelNames = ["airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"]

aug = ImageDataGenerator(rotation_range=10, width_shift_range=0.1, height_shift_range=0.1, zoom_range=0.2, shear_range=0.2, horizontal_flip=True, fill_mode="nearest")



for i in np.arange(0, args["num_models"]):

    print("[INFO] training model {} / {}".format(i + 1, args["num_models"]))
    model = MiniVGGNet.build(32, 32, 3, 10)
    opt = SGD(lr=0.01, decay=0.01/40, momentum=0.9, nesterov=True)
    model.compile(loss="categorical_crossentropy", optimizer=opt, metrics=["accuracy"])
    H = model.fit_generator(aug.flow(X_train, y_train, batch_size=64), validation_data=(X_test, y_test), epochs=63, steps_per_epoch=len(X_train)//64, verbose=1)

    p = [args["model"], "model_{}.model".format(i)]
    model.save(os.path.sep.join(p))

    preds = model.predict(X_test, batch_size=64)
    report = classification_report(y_test.argmax(axis=1), preds.argmax(axis=1), target_names=labelNames)

    p = [args["output"], "model_{}.txt".format(i)]
    f = open(os.path.sep.join(p), "w")
    f.write(report)
    f.close()


    p = [args["output"], "model_{}.png".format(i)]
    plt.style.use("ggplot")
    plt.figure()
    plt.plot(np.arange(0, 40), H.history["loss"], label="train_loss")
    plt.plot(np.arange(0, 40), H.history["val_loss"], label="val_loss")
    plt.plot(np.arange(0, 40), H.history["accuracy"], label="train_acc")
    plt.plot(np.arange(0, 40), H.history["val_accuracy"], label="val_acc")
    plt.title("Training Loss and Accuracy for model {}".format(i))
    plt.xlabel("Epoch #")
    plt.ylabel("Loss/Accuracy")
    plt.legend()
    plt.savefig(os.path.sep.join(p))
    plt.close()



