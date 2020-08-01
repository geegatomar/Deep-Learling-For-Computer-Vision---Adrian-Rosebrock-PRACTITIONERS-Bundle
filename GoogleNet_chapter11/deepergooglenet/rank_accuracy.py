from config import tiny_imagenet_config as config
from processor.simplepreprocessor import SimplePreprocessor
from processor.meanpreprocessor import MeanPreprocessor
from processor.image_to_array_preprocessor import ImageToArrayPreprocessor
from processor.hdf5_dataset_generator import HDF5DatasetGenerator
from processor.ranked import rank5_accuracy
from keras.models import load_model
import json

means = json.loads(open(config.DATASET_MEAN).read())

sp = SimplePreprocessor(64, 64)
mp = MeanPreprocessor(means["R"], means["G"], means["B"])
iap = ImageToArrayPreprocessor()

testGen = HDF5DatasetGenerator(config.TEST_HDF5, 64, preprocessors=[sp, mp, iap], classes=config.NUM_CLASSES)

print("[INFO] loading model...")
model = load_model(config.MODEL_PATH)


print("[INFO] predicting on test data...")
predictions = model.predict_generator(testGen.generator(), steps=testGen.numImages // 64, max_queue_size=64 * 2)
(rank1, rank5) = rank5_accuracy(predictions, testGen.db["labels"])
print("[INFO] rank-1: {:.2f}%".format(rank1 * 100))
print("[INFO] rank-5: {:.2f}%".format(rank5 * 100))
testGen.close()




