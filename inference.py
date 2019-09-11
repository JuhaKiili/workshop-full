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
parser.add_argument('-image_size', type=int, default=50, help="Image size")
parser.add_argument('-images_count', type=int, default=100000, help="Image count limit")
parser.add_argument('-name', type=str, default="dogsvscats", help="Model name")
args = parser.parse_args()

INFERENCE_DIR = '/work/inference'
TRAIN_DIR = '/work/train'

MODEL_NAME = "/work/models/%s.model" % args.name
IMAGE_SIZE = args.image_size
IMAGES_COUNT = args.images_count

def testing_data_loader():
    test_data = []
    for img in tqdm(os.listdir(INFERENCE_DIR)[:IMAGES_COUNT]):
        img_labels = img.split(".")[0]
        path_to_img = os.path.join(INFERENCE_DIR,img)
        img = cv2.resize(cv2.imread(path_to_img,cv2.IMREAD_GRAYSCALE),(IMAGE_SIZE,IMAGE_SIZE))
        test_data.append([np.array(img),np.array(img_labels)])

    shuffle(test_data)
    np.save("test_data.npy",test_data)
    return test_data

print("Loading images...")
testing_data_loader()

tf.reset_default_graph()

convnet = input_data(shape=[None, IMAGE_SIZE, IMAGE_SIZE, 1], name='input')

if os.path.exists("{}.meta".format(MODEL_NAME)):
    model.load(MODEL_NAME)
    print("Model Loaded")
else:
    print("Error loading model from %s" % ("{}.meta".format(MODEL_NAME)))
    return

test_data = np.load("test_data.npy", allow_pickle=True)
with open("final_results.csv",mode="w") as f:
    f.write("id,pred \n")
with open("final_results.csv",mode="a") as r:
    for data in test_data:
        img_class = data[1]
        img = data[0]
        imgs = img.reshape((IMAGE_SIZE,IMAGE_SIZE,1))
        model_out = model.predict([imgs])[0]
        r.write("{},{} \n".format(img_class,model_out[1]))
        print("Image %s doggy score: %d" % (img_class, model_out[1]))
