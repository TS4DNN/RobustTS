import os
from Dau import Dau
import numpy as np
from tensorflow.keras.datasets import cifar10, mnist, cifar100


class SvhnDau(Dau):
    def __init__(self):
        super().__init__("svhn")
        self.train_size = 73257
        self.test_size = 26032
        self.nb_classes = 10

    def load_data(self, use_norm=False):
        from utils import SVNH_DatasetUtil
        (x_train, y_train), (x_test, y_test) = SVNH_DatasetUtil.load_data()
        self.train_size = len(x_train)
        self.test_size = len(x_test)
        print(np.max(x_test), np.min(x_test), x_test.dtype)
        if use_norm:
            x_test = x_test.astype('float32')
            x_train = x_train.astype('float32')
            x_train /= 255
            x_test /= 255
        y_test = np.argmax(y_test, axis=1)
        y_train = np.argmax(y_train, axis=1)
        return (x_train, y_train), (x_test, y_test)

    def get_dau_params(self):
        params = {
            # "SF": [(0, 0.15), (0, 0.15)],
            # "RT": (5, 20),  # rotation
            # "ZM": ((0.8, 1.5), (0.8, 1.5)),  # zoom
            # "BR": 0.3,
            # "SR": [10, 30],  # sheer
            "CT": [0.5, 1.5],
            "BL": None,  # blur
        }
        return params




class CifarDau(Dau):
    def __init__(self):
        super().__init__("cifar")
        self.train_size = 50000  # 写死了
        self.test_size = 10000
        self.nb_classes = 10

    # 用于扩增图片
    # 加载原始数据
    def load_data(self, use_norm=False):  # use_norm=False 用于图片扩增  # use_norm=True 用于训练数据和实验
        (x_train, y_train), (x_test, y_test) = cifar10.load_data()
        y_train = y_train.reshape(-1)
        y_test = y_test.reshape(-1)
        x_test = x_test.reshape(-1, 32, 32, 3)
        x_train = x_train.reshape(-1, 32, 32, 3)
        self.train_size = len(x_train)
        self.test_size = len(x_test)
        if use_norm:
            x_test = x_test.astype('float32')
            x_train = x_train.astype('float32')
            x_train /= 255
            x_test /= 255
        return (x_train, y_train), (x_test, y_test)

    def get_dau_params(self):
        params = {
            "SF": [(0.05, 0.15), (0.05, 0.15)],
            "RT": (5, 25),  # rotation
            "ZM": ((0.7, 1.5), (0.7, 1.5)),  #
            "BR": 0.5,  #
            "SR": [15, 30],  #
            "BL": "easy",  #
            "CT": [0.5, 1.5],
        }
        return params


class MnistDau(Dau):
    def __init__(self):
        super().__init__("mnist")
        self.train_size = 60000
        self.test_size = 10000
        self.nb_classes = 10

    def load_data(self, use_norm=False):
        (x_train, y_train), (x_test, y_test) = mnist.load_data()
        x_test = x_test.reshape(-1, 28, 28, 1)
        x_train = x_train.reshape(-1, 28, 28, 1)
        self.train_size = len(x_train)
        self.test_size = len(x_test)
        if use_norm:
            x_test = x_test.astype('float32')
            x_train = x_train.astype('float32')
            x_train /= 255
            x_test /= 255
        return (x_train, y_train), (x_test, y_test)

    def get_dau_params(self):
        params = {
            "SF": [(0, 0.15), (0, 0.15)],
            "RT": (0, 40),  # rotation
            "ZM": ((0.5, 2.0), (0.5, 2.0)),  # zoom
            "BR": 0.5,
            "SR": [10, 40.0],  # sheer
            "BL": "hard",  # blur
            "CT": [0.5, 1.5],
        }
        print(params)
        return params


class Cifar100Dau(Dau):
    def __init__(self):
        super().__init__("cifar")
        self.train_size = 50000  # 写死了
        self.test_size = 10000
        self.nb_classes = 10

    # 用于扩增图片
    # 加载原始数据
    def load_data(self, use_norm=False):  # use_norm=False 用于图片扩增  # use_norm=True 用于训练数据和实验
        (x_train, y_train), (x_test, y_test) = cifar100.load_data()
        y_train = y_train.reshape(-1)
        y_test = y_test.reshape(-1)
        x_test = x_test.reshape(-1, 32, 32, 3)
        x_train = x_train.reshape(-1, 32, 32, 3)
        self.train_size = len(x_train)
        self.test_size = len(x_test)
        if use_norm:
            x_test = x_test.astype('float32')
            x_train = x_train.astype('float32')
            x_train /= 255
            x_test /= 255
        return (x_train, y_train), (x_test, y_test)

    def get_dau_params(self):
        params = {
            "SF": [(0.05, 0.15), (0.05, 0.15)],
            "RT": (5, 25),  # rotation
            "ZM": ((0.7, 1.5), (0.7, 1.5)),  #
            "BR": 0.5,  #
            "SR": [15, 30],  #
            "BL": "easy",  #
            "CT": [0.5, 1.5],
        }
        return params


if __name__ == '__main__':
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    dau = Cifar100Dau()
    dau.run("test")

