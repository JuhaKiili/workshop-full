import tensorflow as tf
from tensorflow.keras import datasets, layers, models, metrics, optimizers

def get_model(
    learning_rate=0.001,
    batch_size=32,
    image_size=50,
    drop_out=0.8,
    dense_size=1024,
    filter_count=32):

    model = models.Sequential()

    model.add(layers.Conv2D(filter_count, (3, 3), activation='relu', input_shape=(image_size, image_size, 3)))
    model.add(layers.BatchNormalization())
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Dropout(0.25))

    model.add(layers.Conv2D(filter_count * 2, (3, 3), activation='relu'))
    model.add(layers.BatchNormalization())
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Dropout(0.25))

    model.add(layers.Conv2D(filter_count * 4, (3, 3), activation='relu'))
    model.add(layers.BatchNormalization())
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Dropout(0.25))

    model.add(layers.Flatten())
    model.add(layers.Dense(dense_size, activation='relu'))
    model.add(layers.BatchNormalization())
    model.add(layers.Dropout(0.5))
    model.add(layers.Dense(2, activation='softmax'))

    model.compile(
        optimizer=optimizers.RMSprop(lr=learning_rate),
        loss='categorical_crossentropy',
        metrics=['accuracy'],
        batch_size=batch_size,
        )
    model.summary()
    return model
