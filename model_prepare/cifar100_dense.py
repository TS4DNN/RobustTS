import tensorflow as tf
from tensorflow.keras.layers import Input, Dense, Dropout, GlobalAveragePooling2D, Activation, Flatten, Reshape
from tensorflow.keras.models import Model
from tensorflow_addons.optimizers import SGDW
tf.keras.optimizers.SGDW = SGDW
from tensorflow.keras.datasets import cifar100
import tensorflow_addons as tfa
import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator


def compute_mean_var(image):
    # image.shape: [image_num, w, h, c]
    mean = []
    var  = []
    for c in range(image.shape[-1]):
        mean.append(np.mean(image[:, :, :, c]))
        var.append(np.std(image[:, :, :, c]))
    return mean, var


def schedule(epoch_idx):
    if (epoch_idx + 1) < 10:
        # return 1e-03
        return 5e-04
    elif (epoch_idx + 1) < 20:
        return 5e-03  # lr_decay_ratio = 0.2
    elif (epoch_idx + 1) < 30:
        return 1e-04
    elif (epoch_idx + 1) < 40:
        return 5e-04
    return 1e-05


def norm_images(image):
    # image.shape: [image_num, w, h, c]
    image = image.astype('float32')
    mean, var = compute_mean_var(image)
    image[:, :, :, 0] = (image[:, :, :, 0] - mean[0]) / var[0]
    image[:, :, :, 1] = (image[:, :, :, 1] - mean[1]) / var[1]
    image[:, :, :, 2] = (image[:, :, :, 2] - mean[2]) / var[2]
    return image


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
    x = Dense(100)(x)
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
        classes=100
    )
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(100)(x)
    predictions = Activation('softmax', name='predictions')(x)
    model = Model(inputs=base_model.input, outputs=predictions)
    return model


def DenseNet_model_dropout():
    base_model = tf.keras.applications.DenseNet121(
        include_top=False,
        weights="imagenet",
        input_tensor=None,
        input_shape=None,
        pooling=None,
        classes=100
    )
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dropout(0.2)(x, training=True)
    x = Dense(100)(x)
    predictions = Activation('softmax', name='predictions')(x)
    model = Model(inputs=base_model.input, outputs=predictions)
    return model


if __name__ == '__main__':
    # ori_model = tf.keras.models.load_model("../models/imagenet/densenet.h5")
    # weights = ori_model.get_weights()
    #
    # model = DenseNet_drop()
    # # model.summary()
    # model.compile(optimizer='adam',
    #               loss='categorical_crossentropy',
    #               metrics=['accuracy'])
    # model.set_weights(weights)
    # model.summary()
    # model.save("../models/imagenet/densenet_drop.h5")
    save_path = "../models/cifar100/densenet.h5"
    # model = DenseNet_model()
    (x_train, y_train), (x_test, y_test) = cifar100.load_data()
    # x_train = x_train.astype("float32") / 255
    # x_train_mean = np.mean(x_train, axis=0)
    # x_train -= x_train_mean
    # x_test = x_test.astype("float32") / 255
    # x_test -= x_train_mean
    x_train = norm_images(x_train)
    x_test = norm_images(x_test)
    y_test = tf.keras.utils.to_categorical(y_test, 100)
    y_train = tf.keras.utils.to_categorical(y_train, 100)

    # train_datagen = ImageDataGenerator(
    #     featurewise_center=False,  # set input mean to 0 over the dataset
    #     samplewise_center=False,  # set each sample mean to 0
    #     featurewise_std_normalization=False,  # divide inputs by std of the dataset
    #     samplewise_std_normalization=False,  # divide each input by its std
    #     zca_whitening=True,  # apply ZCA whitening
    #     rotation_range=15,  # randomly rotate images in the range (degrees, 0 to 180)
    #     width_shift_range=0.1,  # randomly shift images horizontally (fraction of total width)
    #     height_shift_range=0.1,  # randomly shift images vertically (fraction of total height)
    #     horizontal_flip=True,  # randomly flip images
    #     vertical_flip=True)  # randomly flip images
    #
    # train_datagen.fit(x_train)
    # train_generator = train_datagen.flow(x_train, y_train, batch_size=128)


    checkPoint = tf.keras.callbacks.ModelCheckpoint(save_path, monitor="val_accuracy",
                                                    save_best_only=True, verbose=0)
    lr_schedule = tf.keras.callbacks.LearningRateScheduler(schedule=schedule)
    # model.compile(loss='categorical_crossentropy',
    #               optimizer=tfa.optimizers.SGDW(lr=1e-03, momentum=0.9, weight_decay=1e-5),
    #               metrics=['accuracy'])
    # cbs = [lr_schedule, checkPoint]
    cbs = [lr_schedule]
    # model.fit(
    #           x_train,
    #           y_train,
    #           batch_size=32,
    #           shuffle=True,
    #           epochs=50,
    #           validation_data=(x_test, y_test),
    #           verbose=1,
    #           callbacks=cbs
    #           )
    model = tf.keras.models.load_model("../models/cifar100/densenet.h5")
    score = model.evaluate(x_test, y_test)
    print(score)
    # x_train = np.concatenate((x_train, x_test[:1000]))
    # y_train = np.concatenate((y_train, y_test[:1000]))
    # model.fit(x_train,
    #           y_train,
    #           batch_size=32,
    #           shuffle=True,
    #           epochs=1,
    #           validation_data=(x_test, y_test),
    #           verbose=1,
    #           callbacks=cbs
    #           )
    # model.save("../models/cifar100/densenet_2.h5")


