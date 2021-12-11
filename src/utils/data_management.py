import tensorflow as tf
import os
import gzip
import numpy as np
import logging

'''
def train_valid_generator(data_dir,IMAGE_SIZE,BATCH_SIZE, do_data_augmentation):
    
    datagenerator_kwargs = dict(
        rescale = 1./255, 
        validation_split=0.20
    )

    dataflow_kwargs = dict(
        target_size = IMAGE_SIZE,
        batch_size = BATCH_SIZE,
        interpolation = "bilinear"
    )

    valid_datagenerator = tf.keras.preprocessing.image.ImageDataGenerator(**datagenerator_kwargs)

    valid_generator = valid_datagenerator.flow_from_directory(
        directory=data_dir,
        subset="validation",
        shuffle=False,
        **dataflow_kwargs
    )

    if do_data_augmentation:
        train_datagenerator = tf.keras.preprocessing.image.ImageDataGenerator(
            rotation_range=40,
            horizontal_flip=True,
            width_shift_range=0.2, height_shift_range=0.2, shear_range=0.2, zoom_range=0.2, **datagenerator_kwargs
        )
        logging.info("data augmentation is used for training")
    else:
        train_datagenerator = valid_datagenerator
        logging.info("data augmentation is NOT used for training")

    train_generator = train_datagenerator.flow_from_directory(
        directory=data_dir,
        subset="training",
        shuffle=True, 
        **dataflow_kwargs
    )

    logging.info("train and valid generator is created.")
    return train_generator, valid_generator
    '''
    
def load_data(path, kind='train'):

    """Load MNIST data from `path`"""
    labels_path = os.path.join(path,
                               '%s-labels-idx1-ubyte.gz'
                               % kind)
    images_path = os.path.join(path,
                               '%s-images-idx3-ubyte.gz'
                               % kind)

    with gzip.open(labels_path, 'rb') as lbpath:
        labels = np.frombuffer(lbpath.read(), dtype=np.uint8,
                               offset=8)

    with gzip.open(images_path, 'rb') as imgpath:
        images = np.frombuffer(imgpath.read(), dtype=np.uint8,
                               offset=16).reshape(len(labels), 28,28,1)
    logging.info("data is loaded.") 
    return images, labels

def preprocess_data(train, test):
    train = train/255.0
    test = test/255.0
    logging.info("data is preprocessed.")
    return train, test

def one_hot_encode(train, test, classes):
    train = tf.keras.utils.to_categorical(train, classes)
    test = tf.keras.utils.to_categorical(test, classes)
    logging.info("data is one-hot encoded.")
    return train, test

#X_train, y_train = load_mnist('/content/sample_data', kind='train')
#X_test, y_test = load_mnist('/content/sample_data', kind='t10k')