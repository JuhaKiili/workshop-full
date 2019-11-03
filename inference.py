import os
import tensorflow as tf
from tensorflow.keras import datasets, layers, models, callbacks
import argparse
import cv2
import numpy as np

def prediction_to_text(prediction):
    if prediction[0] > prediction[1]:
        return "It's a cat! (%s%%)" % round(prediction[0] * 100, 2)
    else:
        return "It's a dog! (%s%%)" % round(prediction[1] * 100, 2)

parser = argparse.ArgumentParser()
parser.add_argument('-data_path', type=str, default="test", help="Where are the images")
parser.add_argument('-model_path', type=str, default="model.h5", help="Model file path")
parser.add_argument('-image_size', type=int, default=128, help="Image size")
args = parser.parse_args()

DATA_PATH = args.data_path
MODEL_PATH = args.model_path
IMAGE_SIZE = args.image_size

# Recreate the exact same model
print(MODEL_PATH)
new_model = models.load_model(MODEL_PATH)

for img in os.listdir(path=DATA_PATH):
    if img.endswith('jpg'):
        path_to_img = os.path.join(DATA_PATH, img)
        img = cv2.resize(cv2.imread(path_to_img,cv2.IMREAD_COLOR),(IMAGE_SIZE,IMAGE_SIZE))
        new_predictions = new_model.predict(np.array(img).reshape(-1, IMAGE_SIZE, IMAGE_SIZE, 3).astype(float))
        print(path_to_img, prediction_to_text(new_predictions[0]))

