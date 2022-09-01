import tensorflow as tf
import numpy as np
from scipy.stats import entropy
import random
from svhn_utils import *
from tensorflow.keras.datasets import cifar100
from tensorflow_addons.optimizers import SGDW
tf.keras.optimizers.SGDW = SGDW
from seed_prepare import *


def generate_from_ori_data(datasets, save_path="../datasets/seeds/"):
    if datasets == 'mnist':
        (_, _), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
        x_test = x_test.astype("float32") / 255
        x_test = x_test.reshape(-1, 28, 28, 1)
        # model = tf.keras.models.load_model("../models/mnist/lenet5.h5")
        model = tf.keras.models.load_model("../models/mnist/lenet1.h5")
        (_, _), (x_raw, y_raw) = tf.keras.datasets.mnist.load_data()
        class_num = 10
    elif datasets == 'cifar10':
        (x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()
        x_train = x_train.astype('float32') / 255
        x_test = x_test.astype('float32') / 255
        x_train_mean = np.mean(x_train, axis=0)
        x_train -= x_train_mean
        x_test -= x_train_mean
        # model = tf.keras.models.load_model("../models/cifar10/resnet20.h5")
        model = tf.keras.models.load_model("../models/cifar10/vgg16.h5")
        y_test = y_test.reshape(10000, )
        (_, _), (x_raw, y_raw) = tf.keras.datasets.cifar10.load_data()
        y_raw = y_raw.reshape(10000, )
        class_num = 10
    elif datasets == 'svhn':
        (x_train, y_train), (x_test, y_test) = load_data_before_processing()  # 32*32
        x_test = x_test[:10000]
        y_test = y_test[:10000]
        x_test = x_test.astype('float32') / 255
        y_test = y_test.argmax(axis=1)
        # model = tf.keras.models.load_model("../models/svhn/resnet20.h5")
        model = tf.keras.models.load_model("../models/svhn/lenet5.h5")
        (_, _), (x_raw, y_raw) = load_data_before_processing()
        x_raw = x_raw[:10000]
        y_raw = y_raw[:10000]
        class_num = 10
    elif datasets == 'traffic':
        x_test = np.load("../datasets/traffic/ori_test_x.npy")
        y_test = np.load("../datasets/traffic/ori_test_y.npy")
        # x_test = x_test.astype("float32") / 255
        # y_test = y_test.astype("float32") / 255
        # model = tf.keras.models.load_model("../models/traffic/vgg16.h5")
        model = tf.keras.models.load_model("../models/svhn/lenet5.h5")
        x_raw = np.load("../datasets/traffic/ori_test_x.npy") * 255
        y_raw = np.load("../datasets/traffic/ori_test_y.npy")
        x_raw = x_raw.astype("int")
        class_num = 43
    else:
        # model = tf.keras.models.load_model("../models/cifar100/densenet.h5")
        model = tf.keras.models.load_model("../models/cifar100/resnet.h5")
        (_, _), (x_test, y_test) = cifar100.load_data()
        x_test = norm_images(x_test)
        (_, _), (x_raw, y_raw) = cifar100.load_data()
        class_num = 100
    y_test = y_test.reshape(-1)
    predictions = model.predict(x_test)
    predicted_labels = np.argmax(predictions, axis=1)
    # first_max = np.argsort(predictions, axis=1)[:, -1]
    second_max = np.argsort(predictions, axis=1)[:, -2]
    # print(predicted_labels)
    # print(first_max)
    # print(second_max)
    corrected_index = np.where(predicted_labels == y_test)[0]
    print("predicted_labels shape: ", predicted_labels.shape)
    print("y_test shape", y_test.shape)
    print("corrected_index shape: ", corrected_index.shape)
    x_test = x_test[corrected_index]
    y_test = y_test[corrected_index]

    x_raw = x_raw[corrected_index]
    y_raw = y_raw[corrected_index]
    seed_data = np.array([])
    seed_label = np.array([])

    for i in range(class_num):
        label_index = np.where(y_test == i)[0]
        # print(y_test.shape)
        # print(label_index)
        second_index = []
        for m in range(class_num):
            if m != i:
                second_index.append(m)
        for j in second_index:
            second_label_index = np.where(second_max == j)[0]
            # print(second_label_index)
            final_candidate_index = np.intersect1d(label_index, second_label_index)
            print("1st: {}, 2nd: {}, num: {}".format(i, j, len(final_candidate_index)))
            if len(final_candidate_index) == 0:
                continue
            class_prediction = model.predict(x_test[final_candidate_index])
            entropy_list = entropy(class_prediction, base=2, axis=1)
            sorted_index = np.argsort(entropy_list)
            selected = sorted_index[-1:]
            seed_data = np.append(seed_data, x_raw[final_candidate_index[selected]])
            seed_label = np.append(seed_label, i)
    if datasets == "mnist":
        seed_data = seed_data.reshape(-1, 28, 28, 1)
    else:
        seed_data = seed_data.reshape(-1, 32, 32, 3)
    print(seed_data.shape)
    print(seed_label.shape)
    # np.save("../datasets/seeds/mnist_diverse_x.npy", seed_data)
    # np.save("../datasets/seeds/mnist_diverse_y.npy", seed_label)
    np.save(save_path + datasets + "_diverse_x.npy", seed_data)
    np.save(save_path + datasets + "_diverse_y.npy", seed_label)


def generate_from_transformed_data():
    (_, _), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
    x_test = x_test.astype("float32") / 255
    x_test = x_test.reshape(-1, 28, 28, 1)
    model = tf.keras.models.load_model("../models/mnist/lenet5.h5")


if __name__ == '__main__':
    # generate_from_ori_data("mnist", save_path="../datasets/seeds/mnist/lenet1/")
    # generate_from_ori_data("cifar10", save_path="../datasets/seeds/cifar10/vgg16/")
    # generate_from_ori_data("svhn", save_path="../datasets/seeds/svhn/lenet5/")
    generate_from_ori_data("traffic", save_path="../datasets/seeds/traffic/lenet5/")
    # generate_from_ori_data("cifar100", save_path="../datasets/seeds/cifar100/resnet/")
