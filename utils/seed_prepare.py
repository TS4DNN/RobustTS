import tensorflow as tf
import numpy as np
from scipy.stats import entropy
import random
from svhn_utils import *
from tensorflow.keras.datasets import cifar100
from tensorflow_addons.optimizers import SGDW
tf.keras.optimizers.SGDW = SGDW
import os


def compute_mean_var(image):
    # image.shape: [image_num, w, h, c]
    mean = []
    var  = []
    for c in range(image.shape[-1]):
        mean.append(np.mean(image[:, :, :, c]))
        var.append(np.std(image[:, :, :, c]))
    return mean, var


def norm_images(image):
    # image.shape: [image_num, w, h, c]
    image = image.astype('float32')
    mean, var = compute_mean_var(image)
    image[:, :, :, 0] = (image[:, :, :, 0] - mean[0]) / var[0]
    image[:, :, :, 1] = (image[:, :, :, 1] - mean[1]) / var[1]
    image[:, :, :, 2] = (image[:, :, :, 2] - mean[2]) / var[2]
    return image


def generate_from_ori_data(datasets, save_path="../datasets/seeds/"):
    if datasets == 'mnist':
        (_, _), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
        x_test = x_test.astype("float32") / 255
        x_test = x_test.reshape(-1, 28, 28, 1)
        model = tf.keras.models.load_model("../models/mnist/lenet1.h5")
        # model = tf.keras.models.load_model("../models/mnist/lenet5.h5")
        (_, _), (x_raw, y_raw) = tf.keras.datasets.mnist.load_data()
        class_num = 10
    elif datasets == 'cifar10':
        (x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()
        x_train = x_train.astype('float32') / 255
        x_test = x_test.astype('float32') / 255
        x_train_mean = np.mean(x_train, axis=0)
        # x_train -= x_train_mean
        # x_test -= x_train_mean
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
        model = tf.keras.models.load_model("../models/traffic/lenet5.h5")
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
    predictions = model.predict(x_test)
    predicted_labels = np.argmax(predictions, axis=1)
    corrected_index = np.where(predicted_labels == y_test)[0]
    x_test = x_test[corrected_index]
    y_test = y_test[corrected_index]
    # (_, _), (x_raw, y_raw) = tf.keras.datasets.mnist.load_data()
    x_raw = x_raw[corrected_index]
    y_raw = y_raw[corrected_index]
    # need to select uncertain data

    for i in range(class_num):
        label_index = np.where(y_test == i)[0]
        # print(i)
        if len(label_index) == 0:
            continue
        class_prediction = model.predict(x_test[label_index])
        entropy_list = entropy(class_prediction, base=2, axis=1)
        sorted_index = np.argsort(entropy_list)
        # sorted_entropy = entropy_list[sorted_index]
        # print(sorted_entropy)
        # print(len(entropy_list))
        selected = sorted_index[-5:]
        # print(class_prediction[selected])
        # selected = np.random.choice(label_index, 10, replace=False)
        # print(model.predict(x_test[label_index[selected]]))
        np.save(save_path + datasets + "_" + str(i) + ".npy", x_raw[label_index[selected]])
        # np.save(save_path + datasets + "_diverse_x.npy", seed_data)
        # np.save(save_path + datasets + "_diverse_y.npy", seed_label)
        # break


def seed_combine(data_type, model_type):
    # for i in range(10):
    #     single_data = np.load("../datasets/seeds/cifar10_" + str(i) + ".npy")
    #     single_label = np.load("../datasets/seeds/cifar10_" + str(i) + ".npy")
    x_combined = np.array([])
    y_combined = np.array([])
    if data_type == "mnist":
        class_num = 10
        shape = (-1, 28, 28, 1)
    elif data_type == "cifar10":
        class_num = 10
        shape = (-1, 32, 32, 3)
    elif data_type == "svhn":
        class_num = 10
        shape = (-1, 32, 32, 3)
    elif data_type == "traffic":
        class_num = 43
        shape = (-1, 32, 32, 3)
    else:
        class_num = 100
        shape = (-1, 32, 32, 3)

    for gt in range(class_num):
        if os.path.isfile("../datasets/seeds/" + data_type + "/" + model_type + "/" + data_type + "_" + str(gt) + ".npy"):
            data = np.load("../datasets/seeds/" + data_type + "/" + model_type + "/" + data_type + "_" + str(gt) + ".npy")
            x_combined = np.append(x_combined, data).reshape(shape)
            y_combined = np.append(y_combined, np.array([gt for i in range(len(data))]))
    print(x_combined.shape)
    print(y_combined.shape)
    # model = tf.keras.models.load_model("models/mnist/lenet5.h5")
    # x_combined = x_combined.astype("float32") / 255
    # print(model.predict(x_combined).argmax(axis=1))
    np.save("../datasets/seeds/" + data_type + "/" + model_type + "/" + data_type + "_x_all.npy", x_combined)
    np.save("../datasets/seeds/" + data_type + "/" + model_type + "/" + data_type + "_y_all.npy", y_combined)


if __name__ == '__main__':
    # generate_from_ori_data("mnist", save_path="../datasets/seeds/mnist/lenet1/")
    # generate_from_ori_data("cifar10", save_path="../datasets/seeds/cifar10/vgg16/")
    # generate_from_ori_data("svhn", save_path="../datasets/seeds/svhn/lenet5/")
    # generate_from_ori_data("traffic", save_path="../datasets/seeds/traffic/lenet5/")
    # generate_from_ori_data("cifar100", save_path="../datasets/seeds/cifar100/resnet/")
    # seed_combine()
    #
    # seed_combine("mnist", "lenet1")
    seed_combine("mnist", "lenet5")
    seed_combine("cifar10", "resnet")
    seed_combine("cifar10", "vgg16")
    seed_combine("svhn", "lenet5")
    seed_combine("svhn", "resnet20")
    seed_combine("traffic", "lenet5")
    seed_combine("traffic", "vgg16")
    seed_combine("cifar100", "densenet")
    seed_combine("cifar100", "resnet")
    # seed_combine("traffic")
    # seed_combine("cifar100")

