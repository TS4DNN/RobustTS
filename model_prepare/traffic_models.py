import tensorflow as tf
from cifar10_vgg16 import *
from ResNet import *
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import argparse


class CustomImageDataGen(ImageDataGenerator):  # Inheriting class ImageDataGenerator and manually standardize each input image (x)
    def standardize(self, x):
        if self.featurewise_center:
            x /= 255.
        return x


def traffic_generator(test_folder, batch_size):
    data_generator = CustomImageDataGen(
        horizontal_flip=True,
        featurewise_center=True
    )
    test_data_generator = data_generator.flow_from_directory(test_folder,
                                                             target_size=(32, 32),
                                                             batch_size=batch_size,
                                                             seed=1,
                                                             shuffle=True)
    return test_data_generator


def traffic_lenet5():
    # ori acc 0.9889
    nb_classes = 43
    # convolution kernel size
    kernel_size = (5, 5)
    img_rows, img_cols = 32, 32
    input_tensor = Input(shape=(img_rows, img_cols, 3))

    # block1
    x = Convolution2D(6, kernel_size, activation='relu', padding='same', name='block1_conv1')(input_tensor)
    x = MaxPooling2D(pool_size=(2, 2), name='block1_pool1')(x)

    # block2
    x = Convolution2D(16, kernel_size, activation='relu', padding='same', name='block2_conv1')(x)
    x = MaxPooling2D(pool_size=(2, 2), name='block2_pool1')(x)

    x = Flatten(name='flatten')(x)
    x = Dense(120, activation='relu', name='fc1')(x)
    x = Dense(84, activation='relu', name='fc2')(x)
    x = Dense(nb_classes, name='before_softmax')(x)
    x = Activation('softmax', name='predictions')(x)

    model = Model(input_tensor, x)
    return model


def traffic_lenet5_dropout():
    # ori acc 0.9889
    nb_classes = 43
    # convolution kernel size
    kernel_size = (5, 5)
    img_rows, img_cols = 32, 32
    input_tensor = Input(shape=(img_rows, img_cols, 3))

    # block1
    x = Convolution2D(6, kernel_size, activation='relu', padding='same', name='block1_conv1')(input_tensor)
    x = MaxPooling2D(pool_size=(2, 2), name='block1_pool1')(x)

    # block2
    x = Convolution2D(16, kernel_size, activation='relu', padding='same', name='block2_conv1')(x)
    x = MaxPooling2D(pool_size=(2, 2), name='block2_pool1')(x)

    x = Flatten(name='flatten')(x)
    x = Dense(120, activation='relu', name='fc1')(x)
    x = Dense(84, activation='relu', name='fc2')(x)
    x = Dropout(0.2)(x, training=True)
    x = Dense(nb_classes, name='before_softmax')(x)
    x = Activation('softmax', name='predictions')(x)

    model = Model(input_tensor, x)
    return model


def traffic_vgg16():
    input_shape = (32, 32, 3)
    model = VGG16_clipped(input_shape=input_shape, rate=0.2, nb_classes=43, drop=False)
    return model


def model_train(model_type, save_path):
    train_folder_path = "../datasets/traffic/Train/"
    test_folder_path = "../datasets/traffic/Test_2/"
    train_ds = traffic_generator(train_folder_path, 128)
    test_ds = traffic_generator(test_folder_path, 128)
    steps_per_epoch = 39209 // 128
    # x_test = np.load("../datasets/traffic/x_test.npy")
    # y_test = np.load("../datasets/traffic/y_test.npy")
    # x_test = x_test.astype("float32") / 255
    # y_test = tf.keras.utils.to_categorical(y_test, 43)

    if model_type == "lenet5":
        model = traffic_lenet5()

        model.compile(loss='categorical_crossentropy',
                      optimizer='adam',
                      metrics=['accuracy'])
    elif model_type == "vgg16":
        model = traffic_vgg16()
        model.compile(loss='categorical_crossentropy',
                      optimizer='adam',
                      metrics=['accuracy'])
    model.fit(
        train_ds,
        steps_per_epoch=steps_per_epoch,
        epochs=10,
    )
    model.save(save_path)
    score = model.evaluate(test_ds)
    print(score)


if __name__ == "__main__":
    # folder_path = "../datasets/traffic/Train/"
    # train_ds = traffic_generator(folder_path, 128)
    # # print(dir(train_ds))
    # for x, y in train_ds:
    #     print(x.shape)
    #     print(y.shape)
    #     print(x[0])
    #     break
    # parser = argparse.ArgumentParser()
    # parser.add_argument("--model",
    #                     "-model",
    #                     type=str,
    #                     default='lenet5'
    #                     )
    # parser.add_argument("--save_path",
    #                     "-save_path",
    #                     type=str
    #                     )
    # args = parser.parse_args()
    # model_train(args.model, args.model)
    model = tf.keras.models.load_model("../models/traffic/lenet5.h5")
    x_test = np.load("../datasets/traffic/ori_test_x.npy")
    y_test = np.load("../datasets/traffic/ori_test_y.npy")
    # x_test = x_test.astype("float32") / 255
    # y_test = y_test.astype("float32") / 255
    y_test = tf.keras.utils.to_categorical(y_test, 43)
    score = model.evaluate(x_test, y_test)
    print(score)

# VGG16 0.9286619424819946
# LeNet5 0.8352335691452026
