import os
import tensorflow as tf
import numpy as np
from random import shuffle
import cv2
from tqdm import tqdm
import tflearn
from tflearn.layers.conv import conv_2d,max_pool_2d
from tflearn.layers.core import input_data,dropout,fully_connected
from tflearn.layers.estimator import regression
from model import get_model
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('-learning_rate', type=float, default=0.001, help="Learning rate")
parser.add_argument('-image_size', type=int, default=50, help="Image size")
parser.add_argument('-dropout', type=float, default=0.8, help="Dropout")
parser.add_argument('-filter_count', type=int, default=32, help="Filter count")
parser.add_argument('-brain_size', type=int, default=1024, help="Brain size")
parser.add_argument('-epochs', type=int, default=1, help="Epochs")
parser.add_argument('-steps', type=int, default=500, help="Steps")
parser.add_argument('-images_count', type=int, default=100000, help="Image count limit")
parser.add_argument('-validation_count', type=int, default=500, help="Validation count")
parser.add_argument('-name', type=str, default="dogsvscats", help="Model name")
args = parser.parse_args()

TRAIN_DIR = os.getenv('VH_INPUTS_DIR', '/work') + "/training_data"
LEARNING_RATE = args.learning_rate
MODEL_NAME = os.getenv('VH_OUTPUTS_DIR', '/work/models') + "/%s.model" % args.name
IMAGE_SIZE = args.image_size
EPOCHS = args.epochs
STEPS = args.steps
DROPOUT = args.dropout
FILTER_COUNT = args.filter_count
BRAIN_SIZE = args.brain_size
IMAGES_COUNT = args.images_count
VALIDATION_COUNT = args.validation_count

def label_image(img):
    img_name = img.split(".")[-3]
    if img_name == "cat":
        return [1,0]
    elif img_name == "dog":
        return [0,1]

def train_data_loader():
    training_data = []
    for img in tqdm(os.listdir(path=TRAIN_DIR)[:IMAGES_COUNT]):
        img_lable = label_image(img)
        path_to_img = os.path.join(TRAIN_DIR,img)
        img = cv2.resize(cv2.imread(path_to_img,cv2.IMREAD_GRAYSCALE),(IMAGE_SIZE,IMAGE_SIZE))
        training_data.append([np.array(img),np.array(img_lable)])

    shuffle(training_data)
    np.save("training_data.npy",training_data)
    return training_data

print("Loading images...")

train_data_loader()
train_data_g = np.load('training_data.npy', allow_pickle=True)

model = get_model(LEARNING_RATE, IMAGE_SIZE, DROPOUT, BRAIN_SIZE, FILTER_COUNT)

train = train_data_g[:-VALIDATION_COUNT]
test = train_data_g[-VALIDATION_COUNT:]
X = np.array([i[0] for i in train]).reshape(-1,IMAGE_SIZE,IMAGE_SIZE,1)
Y = [i[1] for i in train]
test_x = np.array([i[0] for i in test]).reshape(-1,IMAGE_SIZE,IMAGE_SIZE,1)
test_y = [i[1] for i in test]
model.fit(X, Y, n_epoch=EPOCHS, validation_set=(test_x,  test_y),
    snapshot_step=STEPS, show_metric=True, run_id=MODEL_NAME)

print("Saving model %s" % MODEL_NAME)
model.save(MODEL_NAME)
