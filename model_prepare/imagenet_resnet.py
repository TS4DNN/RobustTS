import tensorflow as tf
from tensorflow.keras.layers import Input, Dense, Dropout, GlobalAveragePooling2D, Activation, Flatten, Reshape
from tensorflow.keras.models import Model
from tensorflow_addons.optimizers import SGDW
tf.keras.optimizers.SGDW = SGDW


def ResNet101_model():
    base_model = tf.keras.applications.ResNet101V2(
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


def ResNet101_drop(drop_rate=0.5):
    base_model = tf.keras.applications.ResNet101V2(
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


if __name__ == '__main__':
    ori_model = tf.keras.models.load_model("../models/imagenet/resnet101.h5")
    weights = ori_model.get_weights()

    model = ResNet101_drop()
    # model.summary()
    model.compile(optimizer='adam',
                          loss='categorical_crossentropy',
                          metrics=['accuracy'])
    model.set_weights(weights)
    model.summary()
    model.save("../models/imagenet/resnet101_drop.h5")

