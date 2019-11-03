import os
import json
import tensorflow as tf
import numpy as np
from random import shuffle
import cv2
from tqdm import tqdm
import tensorflow as tf
from tensorflow.keras import datasets, layers, models, callbacks
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from datetime import datetime
from model import get_model
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('-learning_rate', type=float, default=0.001, help="Learning rate")
parser.add_argument('-image_size', type=int, default=50, help="Image size")
parser.add_argument('-drop_out', type=float, default=0.8, help="Dropout")
parser.add_argument('-filter_count', type=int, default=32, help="Filter count")
parser.add_argument('-dense_size', type=int, default=1024, help="Dense size")
parser.add_argument('-epochs', type=int, default=1, help="Epochs")
parser.add_argument('-batch_size', type=int, default=500, help="Batch size")
parser.add_argument('-images_count', type=int, default=100000, help="Image count limit")
parser.add_argument('-validation_count', type=int, default=500, help="Validation count")
parser.add_argument('-name', type=str, default="dogsvscats", help="Model name")
args = parser.parse_args()

TRAIN_DIR = os.getenv('VH_REPOSITORY_DIR', '/work') + '/training_data'
CACHE_DIR = os.getenv('VH_REPOSITORY_DIR', '/work')
MODEL_NAME = os.getenv('VH_REPOSITORY_DIR', '/work') + "/models/%s.model" % args.name
LEARNING_RATE = args.learning_rate
IMAGE_SIZE = args.image_size
EPOCHS = args.epochs
BATCH_SIZE = args.batch_size
DROP_OUT = args.drop_out
FILTER_COUNT = args.filter_count
DENSE_SIZE = args.dense_size
IMAGES_COUNT = args.images_count
VALIDATION_COUNT = args.validation_count

def label_image(img):
    img_name = img.split(".")[-3]
    if img_name == "cat":
        return [1,0]
    elif img_name == "dog":
        return [0,1]

def train_data_loader():
    cachepath = CACHE_DIR + '/training_data.npy'
    if os.path.exists(cachepath):
        return np.load(cachepath, allow_pickle=True)
    else:
        training_data = []
        for img in tqdm(os.listdir(path=TRAIN_DIR,)[:IMAGES_COUNT]):
            print(img, img.endswith('jpg'))
            if img.endswith('jpg'):
                img_label = label_image(img)
                path_to_img = os.path.join(TRAIN_DIR,img)
                img = cv2.resize(cv2.imread(path_to_img,cv2.IMREAD_COLOR),(IMAGE_SIZE,IMAGE_SIZE))
                training_data.append([img,img_label])
        
        # np.save(cachepath, training_data)
        return training_data

full_data = train_data_loader()

model = get_model(
    learning_rate=LEARNING_RATE,
    image_size=IMAGE_SIZE,
    drop_out=DROP_OUT,
    dense_size=DENSE_SIZE,
    filter_count=FILTER_COUNT)

train_data = full_data[:-VALIDATION_COUNT]
test_data = full_data[-VALIDATION_COUNT:]

# for i, im in enumerate(test_data):
#     cv2.imwrite('/work/debug/test_img_%s_%s.jpg' % (i, im[1]), im[0])

train_images = np.array([i[0] for i in train_data]).reshape(-1,IMAGE_SIZE,IMAGE_SIZE,3)
train_labels = np.array([i[1] for i in train_data])
test_images = np.array([i[0] for i in test_data]).reshape(-1,IMAGE_SIZE,IMAGE_SIZE,3)
test_labels = np.array([i[1] for i in test_data])

datagen = ImageDataGenerator(
    rotation_range=15,
    shear_range=0.1,
    zoom_range=0.2,
    horizontal_flip=True,
    width_shift_range=0.1,
    height_shift_range=0.1
)

logdir = os.getenv('VH_REPOSITORY_DIR', '/work') + '/logs' + datetime.now().strftime("%Y%m%d-%H%M%S")
tensorboard_callback = callbacks.TensorBoard(log_dir=logdir, profile_batch=0)

# model.fit(x=train_images,
#         y=train_labels,
#         epochs=EPOCHS,
#         batch_size=BATCH_SIZE,
#         validation_data=(test_images, test_labels),
#         callbacks=[tensorboard_callback],
#         shuffle=True,
#         )

model.fit_generator(
    datagen.flow(train_images, train_labels, batch_size=100),
    steps_per_epoch=len(train_images) / 100,
    epochs=EPOCHS,
    validation_data=(test_images, test_labels),
    callbacks=[tensorboard_callback],
    shuffle=True,
    verbose=True,
    )