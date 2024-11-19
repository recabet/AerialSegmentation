import keras
import numpy as np

def unet_model(input_shape):
    inputs = keras.layers.Input(shape=input_shape)

    c1 = keras.layers.Conv2D(64, (3, 3), activation='relu', padding='same')(inputs)
    c1 = keras.layers.Conv2D(64, (3, 3), activation='relu', padding='same')(c1)
    p1 = keras.layers.MaxPooling2D((2, 2))(c1)

    c2 = keras.layers.Conv2D(128, (3, 3), activation='relu', padding='same')(p1)
    c2 = keras.layers.Conv2D(128, (3, 3), activation='relu', padding='same')(c2)
    p2 = keras.layers.MaxPooling2D((2, 2))(c2)

    c3 = keras.layers.Conv2D(256, (3, 3), activation='relu', padding='same')(p2)
    c3 = keras.layers.Conv2D(256, (3, 3), activation='relu', padding='same')(c3)
    p3 = keras.layers.MaxPooling2D((2, 2))(c3)

    c4 = keras.layers.Conv2D(512, (3, 3), activation='relu', padding='same')(p3)
    c4 = keras.layers.Conv2D(512, (3, 3), activation='relu', padding='same')(c4)
    p4 = keras.layers.MaxPooling2D((2, 2))(c4)

    c5 = keras.layers.Conv2D(1024, (3, 3), activation='relu', padding='same')(p4)
    c5 = keras.layers.Conv2D(1024, (3, 3), activation='relu', padding='same')(c5)

    u6 = keras.layers.Conv2DTranspose(512, (3, 3), strides=(2, 2), padding='same')(c5)
    u6 = keras.layers.concatenate([u6, c4])
    c6 = keras.layers.Conv2D(512, (3, 3), activation='relu', padding='same')(u6)
    c6 = keras.layers.Conv2D(512, (3, 3), activation='relu', padding='same')(c6)

    u7 = keras.layers.Conv2DTranspose(256, (3, 3), strides=(2, 2), padding='same')(c6)
    u7 = keras.layers.concatenate([u7, c3])
    c7 = keras.layers.Conv2D(256, (3, 3), activation='relu', padding='same')(u7)
    c7 = keras.layers.Conv2D(256, (3, 3), activation='relu', padding='same')(c7)

    u8 = keras.layers.Conv2DTranspose(128, (3, 3), strides=(2, 2), padding='same')(c7)
    u8 = keras.layers.concatenate([u8, c2])
    c8 = keras.layers.Conv2D(128, (3, 3), activation='relu', padding='same')(u8)
    c8 = keras.layers.Conv2D(128, (3, 3), activation='relu', padding='same')(c8)

    u9 = keras.layers.Conv2DTranspose(64, (3, 3), strides=(2, 2), padding='same')(c8)
    u9 = keras.layers.concatenate([u9, c1])
    c9 = keras.layers.Conv2D(64, (3, 3), activation='relu', padding='same')(u9)
    c9 = keras.layers.Conv2D(64, (3, 3), activation='relu', padding='same')(c9)

    outputs = keras.layers.Conv2D(1, (1, 1), activation='sigmoid')(c9)

    _model = keras.models.Model(inputs, outputs)
    _model.compile(optimizer='adam',
                   loss='binary_crossentropy',
                   metrics=
                   ['accuracy',
                    keras.metrics.BinaryIoU(),
                    keras.metrics.Precision(),
                    keras.metrics.Recall()])

    return _model


def filter_pixels(threshold:float,predictions:np.array):
    for i, pred in enumerate(predictions):
        for j, row in enumerate(pred):
            for k, col in enumerate(row):
                if predictions[i][j][k] > threshold:
                    predictions[i][j][k] = 1
                else:
                    predictions[i][j][k] = 0