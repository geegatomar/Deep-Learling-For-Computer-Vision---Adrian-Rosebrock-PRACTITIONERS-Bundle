from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from preprocessing_tools.aspectawarepreprocessor import AspectAwarePreprocessor
from preprocessing_tools.image_to_array_preprocessor import ImageToArrayPreprocessor
from preprocessing_tools.datasetloader import SimpleDatasetLoader
from fc_headnet import FCHeadNet
from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import RMSprop, SGD
from keras.applications import VGG16
from keras.layers import Input
from keras.models import Model
from imutils import paths
import numpy as np
import argparse
import os


ap = argparse.ArgumentParser()
ap.add_argument("-d", "--dataset", required=True, help="Path to inpput dataset")
ap.add_argument("-m", "--model", required=True, help="Path to output model")
args = vars(ap.parse_args())


aug = ImageDataGenerator(rotation_range=30, width_shift_range=0.1, height_shift_range=0.1, shear_range=0.2, zoom_range=0.2, horizontal_flip=True, fill_mode="nearest")


print("[INFO] loading images...")
imagePaths = list(paths.list_images(args["dataset"]))
classNames = [p.split(os.path.sep)[-2] for p in imagePaths]
classNames = [str(x) for x in np.unique(classNames)]

aap = AspectAwarePreprocessor(224, 224)
iap = ImageToArrayPreprocessor()
sdl = SimpleDatasetLoader(preprocessors=[aap, iap])
(data, labels) = sdl.load(imagePaths, verbose=500)
data = data.astype("float")/255.0

(X_train, X_test, y_train, y_test) = train_test_split(data, labels, train_size=0.25)

lb = LabelBinarizer()
y_train = lb.fit_transform(y_train)
y_test = lb.transform(y_test)


print("[INFO] performing network surgery...")

baseModel = VGG16(weights="imagenet", include_top=False, input_tensor=Input(shape=(224, 224, 3)))

headModel = FCHeadNet.build(baseModel, len(classNames), 256)   
                                            # 256 is num of nodes we want to put in dense layer
model = Model(inputs=baseModel.input, outputs=headModel)

for layer in baseModel.layers:
    layer.trainable = False


print("[INFO] training model...")
opt = RMSprop(lr=0.001)
model.compile(loss="categorical_crossentropy", optimizer=opt, metrics=["accuracy"])

print("[INFO] training head...")
model.fit_generator(aug.flow(X_train, y_train, batch_size=32), validation_data=(X_test, y_test),
epochs = 25, steps_per_epoch = len(X_train)//32, verbose=1)


print("[INFO] evaluating after initialization...")
predictions = model.predict(X_test, batch_size=32)
print(classification_report(y_test.argmax(axis=1), predictions.argmax(axis=1), target_names=classNames))


for layer in baseModel.layers[15:]:
    layer.trainable = True



print("[INFO] re-compiling model...")
opt = SGD(0.001)
model.compile(loss="categorical_crossentropy", optimizer=opt, metrics=["accuracy"])

print("[INFO] fine-tuning model...")
model.fit_generator(aug.flow(X_train, y_train, batch_size=32), validation_data=(X_train, y_train), epochs=100, steps_per_epoch = len(X_train)//32, verbose=1)

print("[INFO] evaluating after fine-tuning...")
preds = model.predict(X_test, batch_size=32)
print(classification_report(y_test.argmax(axis=1), preds.argmax(axis=1), target_names=classNames))


print("[INFO] serializing model...")
model.save(args["model"])






