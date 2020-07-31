import matplotlib
matplotlib.use("Agg")

from config import tiny_imagenet_config as config
from processor.simplepreprocessor import SimplePreprocessor
from processor.meanpreprocessor import MeanPreprocessor
from processor.image_to_array_preprocessor import ImageToArrayPreprocessor
from processor.hdf5_dataset_generator import HDF5DatasetGenerator
from deeper_googlenet import DeeperGoogLeNet 
from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import Adam
from keras.models import load_model
import keras.backend as K
import argparse
import json

ap = argparse.ArgumentParser()
ap.add_argument("-c", "--checkpoints", required=True, help="path to output checkpoint directory")
ap.add_argument("-m", "--model", type=str, help="path to *specific* model checkpoint to load")
ap.add_argument("-s", "--start-epoch", type=int, default=0, help="epoch to restart training at")
args = vars(ap.parse_args())




