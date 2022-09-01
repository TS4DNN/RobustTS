import tensorflow as tf
from tensorflow.keras.layers import Input, Dense, Dropout, GlobalAveragePooling2D, Activation, Flatten, Reshape
from tensorflow.keras.models import Model
from tensorflow_addons.optimizers import SGDW
tf.keras.optimizers.SGDW = SGDW
from data_prepare_tiny import *


def DenseNet_drop(drop_rate=0.5):
    base_model = tf.keras.applications.DenseNet121(
            include_top=False,
            weights="imagenet",
            input_tensor=None,
            input_shape=None,
            pooling=None,
            classes=200
        )
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dropout(drop_rate)(x, training=True)
    x = Dense(200)(x)
    predictions = Activation('softmax', name='predictions')(x)
    model = Model(inputs=base_model.input, outputs=predictions)
    return model


def DenseNet_model():
    base_model = tf.keras.applications.DenseNet121(
        include_top=False,
        weights="imagenet",
        input_tensor=None,
        input_shape=None,
        pooling=None,
        classes=200
    )
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(200)(x)
    predictions = Activation('softmax', name='predictions')(x)
    model = Model(inputs=base_model.input, outputs=predictions)
    return model


if __name__ == '__main__':
    ori_model = tf.keras.models.load_model("../models/imagenet/densenet.h5")
    weights = ori_model.get_weights()

    model = DenseNet_drop()
    # model.summary()
    model.compile(optimizer='adam',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    model.set_weights(weights)
    model.summary()
    model.save("../models/imagenet/densenet_drop.h5")

