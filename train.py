import os
import stat
import json
import shutil
import tensorflow as tf
import numpy as np
from random import shuffle
import cv2
from tqdm import tqdm
import tensorflow as tf
from tensorflow.keras import datasets, layers, models, callbacks, optimizers
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from datetime import datetime
from model import get_model
import argparse

tf.random.set_seed(1234)

parser = argparse.ArgumentParser()
parser.add_argument('-learning_rate', type=float, default=0.001, help="Learning rate")
parser.add_argument('-image_size', type=int, default=50, help="Image size")
parser.add_argument('-filter_count', type=int, default=32, help="Filter count")
parser.add_argument('-dense_size', type=int, default=1024, help="Dense size")
parser.add_argument('-epochs', type=int, default=1, help="Epochs")
parser.add_argument('-batch_size', type=int, default=500, help="Batch size")
parser.add_argument('-images_count', type=int, default=100000, help="Image count limit")
parser.add_argument('-validation_count', type=int, default=500, help="Validation count")
parser.add_argument('-rotation', type=float, default="10", help="Augmented rotation")
parser.add_argument('-shear', type=float, default="0.1", help="Augmented shear")
parser.add_argument('-zoom', type=float, default="0.2", help="Augmented zoom")
parser.add_argument('-shift', type=float, default="0.1", help="Augmented scale shift")
parser.add_argument('-fill_mode', type=str, default="reflect", help="Augmented fillmode for edges")
args = parser.parse_args()

TRAIN_DIR = os.getenv('VH_REPOSITORY_DIR', '/work') + '/training_data'
MODEL_DIR = os.getenv('VH_OUTPUTS_DIR', '/work') + '/models'   

def label_image(img):
    img_name = img.split(".")[-3]
    if img_name == "cat":
        return [1,0]
    elif img_name == "dog":
        return [0,1]

full_data = []
for img in tqdm(os.listdir(path=TRAIN_DIR,)[:args.images_count]):
    if img.endswith('jpg'):
        img_label = label_image(img)
        path_to_img = os.path.join(TRAIN_DIR,img)
        img = cv2.resize(cv2.imread(path_to_img,cv2.IMREAD_COLOR),(args.image_size,args.image_size))
        full_data.append([img,img_label])

train_data = full_data[:-args.validation_count]
test_data = full_data[-args.validation_count:]

train_images = np.array([i[0] for i in train_data])
train_labels = np.array([i[1] for i in train_data])
test_images = np.array([i[0] for i in test_data])
test_labels = np.array([i[1] for i in test_data])

model = models.Sequential()
model.add(layers.Conv2D(args.filter_count, (3, 3), activation='relu', input_shape=(args.image_size, args.image_size, 3)))
model.add(layers.BatchNormalization())
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Dropout(0.25))

model.add(layers.Conv2D(args.filter_count * 2, (3, 3), activation='relu'))
model.add(layers.BatchNormalization())
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Dropout(0.25))

model.add(layers.Conv2D(args.filter_count * 4, (3, 3), activation='relu'))
model.add(layers.BatchNormalization())
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Dropout(0.25))

model.add(layers.Flatten())
model.add(layers.Dense(args.dense_size, activation='relu'))
model.add(layers.BatchNormalization())
model.add(layers.Dropout(0.5))
model.add(layers.Dense(2, activation='softmax'))

model.compile(
    optimizer=optimizers.RMSprop(lr=args.learning_rate),
    loss='categorical_crossentropy',
    metrics=['accuracy'],
    batch_size=args.batch_size,
    )
model.summary()

class EpochCallback(callbacks.Callback):
    best_accuracy = 0.0

    def on_epoch_end(self, epoch, logs={}):
        print(json.dumps({
            'epoch':epoch,
            'training_accuracy': str(logs['accuracy']),
            'training_loss': str(logs['loss']),
            'validated_accuracy': str(logs['val_accuracy']),
            'validated_loss': str(logs['val_loss'])
            }))
        if EpochCallback.best_accuracy < logs['val_accuracy']:
            if not os.path.exists(MODEL_DIR):
                os.makedirs(MODEL_DIR)
            filepath = MODEL_DIR + '/model-%s-acc-%s.h5' % (datetime.now().strftime("%Y%m%d-%H%M%S"), str(logs['val_accuracy']))
            model.save(filepath)
            EpochCallback.best_accuracy = logs['val_accuracy']  

epoch_callback = EpochCallback()

datagen = ImageDataGenerator(
    rotation_range=args.rotation,
    shear_range=args.shear,
    zoom_range=args.zoom,
    horizontal_flip=True,
    width_shift_range=args.shift,
    height_shift_range=args.shift,
    fill_mode="reflect"
)
model.fit_generator(
    datagen.flow(train_images, train_labels, batch_size=args.batch_size),
    steps_per_epoch=len(train_images) / args.batch_size,
    epochs=args.epochs,
    validation_data=(test_images, test_labels),
    callbacks=[epoch_callback],
    shuffle=True,
    verbose=False,
    )