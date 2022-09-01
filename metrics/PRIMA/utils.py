import pickle
import os
import numpy as np
import tensorflow as tf

def cos_distribution(cos_array):
    cos_distribute = [0 for i in range(10)]
    for i in cos_array:
        if i >= 0 and i < 0.1:
            cos_distribute[0] += 1
        elif i >= 0.1 and i < 0.2:
            cos_distribute[1] += 1
        elif i >= 0.2 and i < 0.3:
            cos_distribute[2] += 1
        elif i >= 0.3 and i < 0.4:
            cos_distribute[3] += 1
        elif i >= 0.4 and i < 0.5:
            cos_distribute[4] += 1
        elif i >= 0.5 and i < 0.6:
            cos_distribute[5] += 1
        elif i >= 0.6 and i < 0.7:
            cos_distribute[6] += 1
        elif i >= 0.7 and i < 0.8:
            cos_distribute[7] += 1
        elif i >= 0.8 and i < 0.9:
            cos_distribute[8] += 1
        elif i >= 0.9 and i <= 1.0:
            cos_distribute[9] += 1
    return cos_distribute


def read_kill_rate_dict(file_name):
    dictfile = open(file_name + '.dict', 'rb')
    kill_rate_file = pickle.load(dictfile)
    if type(kill_rate_file) == dict:
        kill_rate_dict = kill_rate_file
    else:
        kill_rate_dict = {score: letter for score, letter in kill_rate_file}
    return kill_rate_dict


def walkFile(file):
    file_list = []
    for root, dirs, files in os.walk(file):
        for f in files:
            file_list.append(os.path.join(root, f))
    return file_list


def count_wrong_prediction(given_list):
    set01 = set(given_list)
    dict01 = {}
    for item in set01:
        dict01.update({item: given_list.count(item)})
    return dict01


def save_dict(filename,dictionary):
    dictfile = open(filename + '.dict', 'wb')
    pickle.dump(dictionary, dictfile)
    dictfile.close()



def load_dict(filename):
    dictfile = open(filename + '.dict', 'rb')
    a = pickle.load(dictfile)
    dictfile.close()
    return a


def combine_ori_new(data_type="ori"):
    (_, _), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
    if data_type == "ori":
        x_test = np.load("../../datasets/mnist/RT_test_x.npy")
        y_test = np.load("../../datasets/mnist/RT_test_y.npy")
        return x_test, y_test
    elif data_type == "new":
        x_test = x_test.reshape(-1, 28, 28, 1)
        # x_new = np.load("datasets/generated/diverse/mnist_x_all.npy")
        # y_new = np.load("datasets/generated/diverse/mnist_y_all.npy")
        all_index = np.arange(974)
        selected_index = np.random.choice(all_index, 100, replace=False)
        x_new = np.load("../../datasets/generated/mnist_x_all.npy")[selected_index]
        y_new = np.load("../../datasets/generated/mnist_y_all.npy")[selected_index]

        print(x_test.shape)
        print(x_new.shape)
        x_final = np.concatenate((x_test, x_new))
        y_final = np.concatenate((y_test, y_new))
        return x_final, y_final
    else:
        # x_test = x_test.reshape(-1, 28, 28, 1)
        # x_test = np.load("/Users/qiang.hu/PycharmProjects/TS4code/datasets/mnist/" + data_type + "/test_images.npy")
        # y_test = np.load("/Users/qiang.hu/PycharmProjects/TS4code/datasets/mnist/" + data_type + "/test_labels.npy")
        x_test = np.load("../../datasets/mnist/RT_test_x.npy")
        y_test = np.load("../../datasets/mnist/RT_test_y.npy")
        # x_new = np.load("datasets/generated/diverse/mnist_x_all.npy")
        # y_new = np.load("datasets/generated/diverse/mnist_y_all.npy")
        all_index = np.arange(974)
        selected_index = np.random.choice(all_index, 100, replace=False)
        x_new = np.load("../../datasets/generated/mnist_x_all.npy")[selected_index]
        y_new = np.load("../../datasets/generated/mnist_y_all.npy")[selected_index]
        x_final = np.concatenate((x_test, x_new))
        y_final = np.concatenate((y_test, y_new))
        return x_final, y_final


def combine_clean_adv(x_ori, y_ori):
    # all_index = np.arange(974)
    all_index = np.arange(81)
    selected_index = np.random.choice(all_index, 81, replace=False)
    x_new = np.load("../../datasets/generated/mnist_x_all.npy")[selected_index]
    y_new = np.load("../../datasets/generated/mnist_y_all.npy")[selected_index]
    # x_new = np.load("datasets/generated/diverse/mnist_x_all.npy")[selected_index]
    # y_new = np.load("datasets/generated/diverse/mnist_y_all.npy")[selected_index]
    x_final = np.concatenate((x_ori, x_new))
    y_final = np.concatenate((y_ori, y_new))
    return x_final, y_final


def combine_clean_adv_cifar10(x_ori, y_ori, diverse=False):
    # all_index = np.arange(89)
    # selected_index = np.random.choice(all_index, 89, replace=False)
    if diverse:
        x_new = np.load("../../datasets/generated/cifar10/diverse/cifar10_x_all.npy")
        y_new = np.load("../../datasets/generated/cifar10/diverse/cifar10_y_all.npy")
    else:
        x_new = np.load("../../datasets/generated/cifar10/cifar10_x_all.npy")
        y_new = np.load("../../datasets/generated/cifar10/cifar10_y_all.npy")
    # x_new = np.load("datasets/generated/mnist_x_all.npy")[selected_index]
    # y_new = np.load("datasets/generated/mnist_y_all.npy")[selected_index]
    x_final = np.concatenate((x_ori, x_new))
    y_final = np.concatenate((y_ori, y_new))
    return x_final, y_final

