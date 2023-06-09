from tensorflow import keras
from tensorflow.keras import layers
from config import ConstantConfig
INPUT_SHAPE = (50, 50, 1)
ENCODER = ConstantConfig().ENCODER


def old_model():
    model = keras.Sequential([
        keras.Input(shape=INPUT_SHAPE),
        layers.Conv2D(32, kernel_size=3, activation='relu'),
        layers.MaxPooling2D(pool_size=2, strides=2),
        layers.Dropout(0.2),
        layers.Conv2D(64, kernel_size=3, activation='relu'),
        layers.MaxPooling2D(pool_size=2, strides=2),
        layers.Dropout(0.2),
        layers.Flatten(),
        layers.Dense(len(ENCODER)+1, activation='softmax')
    ])
    optimizer = keras.optimizers.Adam(learning_rate=0.001)
    model.compile(loss='sparse_categorical_crossentropy',
                  optimizer=optimizer, metrics=['accuracy'])
    return model


def custom_model():
    model = keras.Sequential(
        [keras.Input(shape=INPUT_SHAPE),
         layers.Conv2D(filters=32, kernel_size=3, activation='relu'),
         layers.MaxPool2D(pool_size=2, strides=2),
         layers.Conv2D(filters=64, kernel_size=3, activation='relu', padding='same'),
         layers.MaxPool2D(pool_size=2, strides=2),
         layers.Conv2D(filters=128, kernel_size=3, activation='relu', padding='valid'),
         layers.MaxPool2D(pool_size=2, strides=2),
         layers.Flatten(),
         layers.Dense(len(ENCODER)+1, activation="softmax")
         ]
    )
    model.compile(optimizer=keras.optimizers.Adam(learning_rate=0.001),
                  loss='categorical_crossentropy', metrics=['accuracy'])
    return model
