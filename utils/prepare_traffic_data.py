import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import argparse
import numpy as np

class CustomImageDataGen(ImageDataGenerator):  # Inheriting class ImageDataGenerator and manually standardize each input image (x)
    def standardize(self, x):
        if self.featurewise_center:
            x /= 255.
        return x


def traffic_generator(test_folder, batch_size):
    data_generator = CustomImageDataGen(
        horizontal_flip=True,
        featurewise_center=False
    )
    test_data_generator = data_generator.flow_from_directory(test_folder,
                                                             target_size=(32, 32),
                                                             batch_size=batch_size,
                                                             seed=1,
                                                             shuffle=True)
    return test_data_generator


def data2npy():
    test_folder_path = "../datasets/traffic/Train/"
    test_ds = traffic_generator(test_folder_path, 200)
    # epochs = 12630 // 30
    epochs = 39209 // 200
    count = 0
    x_test = []
    y_test = []
    for x, y in test_ds:
        x_test.append(x)
        y_test.append(y.argmax(axis=1))
        count += 1
        if count == epochs:
            break
    x_test = np.asarray(x_test)
    y_test = np.asarray(y_test)
    x_test = x_test.reshape(-1, 32, 32, 3)
    y_test = y_test.reshape(-1)
    print(x_test.shape)
    print(y_test.shape)
    np.save("../datasets/traffic/x_train.npy", x_test)
    np.save("../datasets/traffic/y_train.npy", y_test)
    # x_test = np.load("../datasets/traffic/x_test.npy")
    # y_test = np.load("../datasets/traffic/y_test.npy")
    # x_test = x_test.astype("float32") / 255
    # y_test = tf.keras.utils.to_categorical(y_test, 43)


if __name__ == "__main__":
    data2npy()

# VGG16 0.9286619424819946
# LeNet5 0.8352335691452026
