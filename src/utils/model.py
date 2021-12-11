import tensorflow as tf
from tensorflow import keras
import os
import logging
from functools import partial
from tensorflow.python.keras.backend import flatten
from src.utils.common import get_timestamp


def get_prepare_model(CLASSES, learning_rate, dropout_rate, kernel_size, input_shape, num_filters, num_neurons, model_path):
    DefaultConv2D = partial(keras.layers.Conv2D, kernel_size=kernel_size, activation='relu', padding="SAME") #a thin wrapper class for Convolution layer
    model = keras.models.Sequential([
        DefaultConv2D(filters=num_filters, kernel_size=kernel_size, input_shape=input_shape),
        keras.layers.MaxPooling2D(pool_size=2),
        DefaultConv2D(filters=num_filters*2),
        DefaultConv2D(filters=num_filters*2),
        keras.layers.MaxPooling2D(pool_size=2),
        DefaultConv2D(filters=num_filters*4),
        DefaultConv2D(filters=num_filters*4),
        keras.layers.MaxPooling2D(pool_size=2),
        keras.layers.Flatten(),
        keras.layers.Dense(units=num_neurons, activation='relu'),
        keras.layers.Dropout(dropout_rate),
        keras.layers.Dense(units=(num_neurons/2), activation='relu'),
        keras.layers.Dropout(dropout_rate),
        keras.layers.Dense(units=CLASSES, activation='softmax'),
    ])
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),loss=tf.keras.losses.categorical_crossentropy,metrics=['accuracy'])
    model.save(model_path)
    logging.info(f"custom model is saved at {model_path}")
    logging.info("custom model is compiled and ready to be trained")
    model.summary()
    return model

def load_full_model(untrained_full_model_path):
    model = tf.keras.models.load_model(untrained_full_model_path)
    logging.info(f"untrained model is read from {untrained_full_model_path}")
    return model

def get_unique_path_to_save_model(trained_model_dir, model_name="model"):
    timestamp = get_timestamp(model_name)
    unique_model_name = f"{timestamp}_.h5"
    unique_model_path = os.path.join(trained_model_dir, unique_model_name)
    return unique_model_path
