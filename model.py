import tensorflow as tf
from tensorflow.keras import datasets, layers, models, metrics

def get_model(
    learning_rate=0.001,
    batch_size=32,
    image_size=50,
    drop_out=0.8,
    dense_size=1024,
    filter_count=32):

    model = models.Sequential()
    model.add(layers.Conv2D(filter_count, (3, 3), activation='relu', input_shape=(image_size, image_size, 3)))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(filter_count, (3, 3), activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(filter_count, (3, 3), activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(filter_count, (3, 3), activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(filter_count, (3, 3), activation='relu'))

    model.add(layers.Flatten())
    model.add(layers.Dense(dense_size, activation='relu'))
    model.add(layers.Dense(1, activation='sigmoid'))
    model.compile(
        optimizer='adam',
        loss='binary_crossentropy',
        metrics=['binary_accuracy'],
        batch_size=batch_size,
        learning_rate=learning_rate,
        drop_out=drop_out,
        )
    return model
