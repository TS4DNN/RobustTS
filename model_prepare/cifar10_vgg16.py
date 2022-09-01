import numpy as np
# np.random.seed(698686)
# print("Set Random Seed 698686")
from tensorflow.keras.layers import Convolution2D, MaxPooling2D, Dropout, BatchNormalization
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras import backend as K
from tensorflow.keras.layers import Lambda
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.layers import Flatten, Dense
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import optimizers
from tensorflow.keras import callbacks
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.datasets import cifar10
# import keras
import tensorflow
from tensorflow import keras


def VGG16_clipped(input_shape=None, rate=0.2, nb_classes=10, drop=False):
    # Block 1
    model = Sequential()
    model.add(Convolution2D(64, (3, 3),
                            activation='relu',
                            padding='same',
                            name='block1_conv1', input_shape=input_shape)) #1
    model.add(BatchNormalization(name="batch_normalization_1"))     #2
    if drop:
        model.add(Lambda(lambda x: K.dropout(x, level=rate)))
    model.add(Convolution2D(64, (3, 3),
                            activation='relu',
                            padding='same',
                            name='block1_conv2')) #3
    model.add(BatchNormalization(name="batch_normalization_2"))
    if drop:
        model.add(Lambda(lambda x: K.dropout(x, level=rate)))

    model.add(MaxPooling2D((2, 2), strides=(2, 2), name='block1_pool'))#4

    # Block 2
    model.add(Convolution2D(128, (3, 3),
                            activation='relu',
                            padding='same',
                            name='block2_conv1'))#5
    model.add(BatchNormalization(name="batch_normalization_3"))
    if drop:
        model.add(Lambda(lambda x: K.dropout(x, level=rate)))
    model.add(Convolution2D(128, (3, 3),
                            activation='relu',
                            padding='same',
                            name='block2_conv2'))#6
    model.add(BatchNormalization(name="batch_normalization_4"))#7
    if drop:
        model.add(Lambda(lambda x: K.dropout(x, level=rate)))
    model.add(MaxPooling2D((2, 2), strides=(2, 2), name='block2_pool'))#8

    # Block 3
    model.add(Convolution2D(256, (3, 3),
                            activation='relu',
                            padding='same',
                            name='block3_conv1'))#9
    model.add(BatchNormalization(name="batch_normalization_5")) #10
    if drop:
        model.add(Lambda(lambda x: K.dropout(x, level=rate)))
    model.add(Convolution2D(256, (3, 3),
                            activation='relu',
                            padding='same',
                            name='block3_conv2'))#11
    model.add(BatchNormalization(name="batch_normalization_6"))

    if drop:
        model.add(Lambda(lambda x: K.dropout(x, level=rate)))
    model.add(Convolution2D(256, (3, 3),
                            activation='relu',
                            padding='same',
                            name='block3_conv3'))#12
    model.add(BatchNormalization(name="batch_normalization_7")) #13
    if drop:
        model.add(Lambda(lambda x: K.dropout(x, level=rate)))
    model.add(MaxPooling2D((2, 2), strides=(2, 2), name='block3_pool')) #14
    model.add(Flatten())#15
    model.add(Dense(256, activation='relu', name='dense_1')) #16
    model.add(BatchNormalization(name="batch_normalization_8"))#17
    model.add(Dense(256, activation='relu', name='dense_2'))#18
    model.add(BatchNormalization(name="batch_normalization_9"))#19
    model.add(Dense(nb_classes, activation='softmax', name='dense_3')) #20
    return model


def VGG16_clipped_dropout(input_shape=None, drop_rate=0.2, nb_classes=10, drop=False):
    # Block 1
    inputs = keras.Input(shape=input_shape)
    x = Convolution2D(64, (3, 3),
                            activation='relu',
                            padding='same',
                            name='block1_conv1', input_shape=input_shape)(inputs)
    x = BatchNormalization(name="batch_normalization_1")(x)
    x = Convolution2D(64, (3, 3),
                            activation='relu',
                            padding='same',
                            name='block1_conv2')(x)
    x = BatchNormalization(name="batch_normalization_2")(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block1_pool')(x)
    # Block 2
    x = Convolution2D(128, (3, 3),
                            activation='relu',
                            padding='same',
                            name='block2_conv1')(x)
    x = BatchNormalization(name="batch_normalization_3")(x)
    x = Convolution2D(128, (3, 3),
                            activation='relu',
                            padding='same',
                            name='block2_conv2')(x)
    x = BatchNormalization(name="batch_normalization_4")(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block2_pool')(x)

    # Block 3
    x = Convolution2D(256, (3, 3),
                            activation='relu',
                            padding='same',
                            name='block3_conv1')(x)
    x = BatchNormalization(name="batch_normalization_5")(x)
    x = Convolution2D(256, (3, 3),
                            activation='relu',
                            padding='same',
                            name='block3_conv2')(x)
    x = BatchNormalization(name="batch_normalization_6")(x)
    x = Convolution2D(256, (3, 3),
                            activation='relu',
                            padding='same',
                            name='block3_conv3')(x)
    x = BatchNormalization(name="batch_normalization_7")(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block3_pool')(x)
    x = Flatten()(x)
    x = Dense(256, activation='relu', name='dense_1')(x)
    x = Dropout(drop_rate)(x, training=True)
    x = BatchNormalization(name="batch_normalization_8")(x)
    x = Dense(256, activation='relu', name='dense_2')(x)
    x = BatchNormalization(name="batch_normalization_9")(x)
    outputs = Dense(nb_classes, activation='softmax', name='dense_3')(x)
    model = keras.Model(inputs, outputs)
    return model


if __name__ == '__main__':
    drop_model = VGG16_clipped_dropout(input_shape=(32, 32, 3))
    drop_model.compile(loss='categorical_crossentropy',
                  optimizer=tensorflow.optimizers.SGD(lr=1e-2, momentum=0.9),
                  metrics=['accuracy'])
    ori_model = tensorflow.keras.models.load_model("../models/cifar10/VGG16.h5")
    weights = ori_model.get_weights()
    drop_model.set_weights(weights)
    drop_model.summary()
    drop_model.save("../models/cifar10/vgg16_drop.h5")
