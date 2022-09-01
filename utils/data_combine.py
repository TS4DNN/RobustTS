import numpy as np
import tensorflow as tf


def combine_ori_new(data_type="ori"):
    (_, _), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
    if data_type == "ori":
        x_test = np.load("datasets/mnist/RT_test_x.npy")
        y_test = np.load("datasets/mnist/RT_test_y.npy")
        return x_test, y_test
    elif data_type == "new":
        x_test = x_test.reshape(-1, 28, 28, 1)
        # x_new = np.load("datasets/generated/diverse/mnist_x_all.npy")
        # y_new = np.load("datasets/generated/diverse/mnist_y_all.npy")
        all_index = np.arange(974)
        selected_index = np.random.choice(all_index, 100, replace=False)
        x_new = np.load("datasets/generated/mnist_x_all.npy")[selected_index]
        y_new = np.load("datasets/generated/mnist_y_all.npy")[selected_index]

        print(x_test.shape)
        print(x_new.shape)
        x_final = np.concatenate((x_test, x_new))
        y_final = np.concatenate((y_test, y_new))
        return x_final, y_final
    else:
        # x_test = x_test.reshape(-1, 28, 28, 1)
        # x_test = np.load("/Users/qiang.hu/PycharmProjects/TS4code/datasets/mnist/" + data_type + "/test_images.npy")
        # y_test = np.load("/Users/qiang.hu/PycharmProjects/TS4code/datasets/mnist/" + data_type + "/test_labels.npy")
        x_test = np.load("datasets/mnist/RT_test_x.npy")
        y_test = np.load("datasets/mnist/RT_test_y.npy")
        # x_new = np.load("datasets/generated/diverse/mnist_x_all.npy")
        # y_new = np.load("datasets/generated/diverse/mnist_y_all.npy")
        all_index = np.arange(974)
        selected_index = np.random.choice(all_index, 100, replace=False)
        x_new = np.load("datasets/generated/mnist_x_all.npy")[selected_index]
        y_new = np.load("datasets/generated/mnist_y_all.npy")[selected_index]
        x_final = np.concatenate((x_test, x_new))
        y_final = np.concatenate((y_test, y_new))
        return x_final, y_final


def combine_clean_adv(x_ori, y_ori):
    # all_index = np.arange(974)
    all_index = np.arange(81)
    selected_index = np.random.choice(all_index, 81, replace=False)
    x_new = np.load("datasets/generated/mnist_x_all.npy")[selected_index]
    y_new = np.load("datasets/generated/mnist_y_all.npy")[selected_index]
    # x_new = np.load("datasets/generated/diverse/mnist_x_all.npy")[selected_index]
    # y_new = np.load("datasets/generated/diverse/mnist_y_all.npy")[selected_index]
    x_final = np.concatenate((x_ori, x_new))
    y_final = np.concatenate((y_ori, y_new))
    return x_final, y_final


def combine_clean_adv_cifar10(x_ori, y_ori, diverse=False):
    # all_index = np.arange(89)
    # selected_index = np.random.choice(all_index, 89, replace=False)
    if diverse:
        x_new = np.load("datasets/generated/cifar10/diverse/cifar10_x_all.npy")
        y_new = np.load("datasets/generated/cifar10/diverse/cifar10_y_all.npy")
    else:
        x_new = np.load("datasets/generated/cifar10/cifar10_x_all.npy")
        y_new = np.load("datasets/generated/cifar10/cifar10_y_all.npy")
    # x_new = np.load("datasets/generated/mnist_x_all.npy")[selected_index]
    # y_new = np.load("datasets/generated/mnist_y_all.npy")[selected_index]
    x_final = np.concatenate((x_ori, x_new))
    y_final = np.concatenate((y_ori, y_new))
    return x_final, y_final


if __name__ == "__main__":
    a = np.load("../datasets/generated/cifar10/diverse/cifar10_x_all.npy")
    print(a.shape)
