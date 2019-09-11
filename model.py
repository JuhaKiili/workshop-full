import tensorflow as tf
from tflearn.layers.conv import conv_2d,max_pool_2d
from tflearn.layers.core import input_data,dropout,fully_connected
from tflearn.layers.estimator import regression
import tflearn

def get_model(learning_rate, image_size, drop_out, brain_size, filters):
    tf.reset_default_graph()

    convnet = input_data(shape=[None, image_size, image_size, 1], name='input')

    convnet = conv_2d(convnet, filters, 5, activation='relu')
    convnet = max_pool_2d(convnet, 5)
    convnet = conv_2d(convnet, filters * 2, 5, activation='relu')
    convnet = max_pool_2d(convnet, 5)
    convnet = conv_2d(convnet, filters * 4, 5, activation='relu')
    convnet = max_pool_2d(convnet, 5)
    convnet = conv_2d(convnet, filters * 2, 5, activation='relu')
    convnet = max_pool_2d(convnet, 5)
    convnet = conv_2d(convnet, filters, 5, activation='relu')
    convnet = max_pool_2d(convnet, 5)
    convnet = fully_connected(convnet, brain_size, activation='relu')
    convnet = dropout(convnet, drop_out)
    convnet = fully_connected(convnet, 2, activation='softmax')
    convnet = regression(
        convnet,
        optimizer='adam',
        learning_rate=learning_rate,
        loss='categorical_crossentropy',
        name='targets')
    return tflearn.DNN(convnet, tensorboard_dir='/work/log')
