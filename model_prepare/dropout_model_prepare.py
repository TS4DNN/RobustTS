import tensorflow as tf
from tensorflow.keras.optimizers import Adam
from mnist_lenet import *
from cifar10_resnet20 import *
from cifar10_vgg16 import *
from svhn_model import *
from traffic_models import *
from cifar100_resnet import *
from cifar100_dense import *


def model_transfer(data_type, model_type):
    if data_type == "mnist":
        if model_type == "lenet1":
            ori_model = tf.keras.models.load_model("../models/mnist/lenet1.h5")
            dropout_model = Lenet1_dropout()
            dropout_model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
            weights = ori_model.get_weights()
            dropout_model.set_weights(weights)
            dropout_model.save("../models/mnist/lenet1_dropout.h5")
        else:
            ori_model = tf.keras.models.load_model("../models/mnist/lenet5.h5")
            dropout_model = Lenet5_dropout()
            dropout_model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
            weights = ori_model.get_weights()
            dropout_model.set_weights(weights)
            dropout_model.save("../models/mnist/lenet5_dropout.h5")
    elif data_type == "cifar10":
        if model_type == "resnet20":
            ori_model = tf.keras.models.load_model("../models/cifar10/resnet20.h5")
            dropout_model = resnet20_drop((32, 32, 3), 10, drop=True)
            dropout_model.compile(loss='categorical_crossentropy',
                          optimizer=Adam(lr=1e-3),
                          metrics=['accuracy'])
            weights = ori_model.get_weights()
            dropout_model.set_weights(weights)
            dropout_model.save("../models/cifar10/resnet20_dropout.h5")
        else:
            drop_model = VGG16_clipped_dropout(input_shape=(32, 32, 3))
            drop_model.compile(loss='categorical_crossentropy',
                               optimizer=tensorflow.optimizers.SGD(lr=1e-2, momentum=0.9),
                               metrics=['accuracy'])
            ori_model = tensorflow.keras.models.load_model("../models/cifar10/vgg16.h5")
            weights = ori_model.get_weights()
            drop_model.set_weights(weights)
            drop_model.summary()
            drop_model.save("../models/cifar10/vgg16_drop.h5")
    elif data_type == "svhn":
        if model_type == "lenet5":
            ori_model = tf.keras.models.load_model("../models/svhn/lenet5.h5")
            dropout_model = svhn_lenet5_drop()
            sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
            dropout_model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])
            weights = ori_model.get_weights()
            dropout_model.set_weights(weights)
            dropout_model.save("../models/svhn/lenet5_dropout.h5")
        else:
            ori_model = tf.keras.models.load_model("../models/svhn/resnet20.h5")
            dropout_model = resnet20_drop((32, 32, 3), 10, drop=True)
            dropout_model.compile(loss='categorical_crossentropy',
                                  optimizer=Adam(lr=1e-3),
                                  metrics=['accuracy'])
            weights = ori_model.get_weights()
            dropout_model.set_weights(weights)
            dropout_model.save("../models/svhn/resnet20_dropout.h5")
    elif data_type == "traffic":
        if model_type == "lenet5":
            ori_model = tf.keras.models.load_model("../models/traffic/lenet5.h5")
            dropout_model = traffic_lenet5_dropout()
            dropout_model.compile(loss='categorical_crossentropy',
                          optimizer='adam',
                          metrics=['accuracy'])
            weights = ori_model.get_weights()
            dropout_model.set_weights(weights)
            dropout_model.save("../models/traffic/lenet5_dropout.h5")
        else:
            ori_model = tf.keras.models.load_model("../models/traffic/vgg16.h5")
            dropout_model = VGG16_clipped_dropout((32, 32, 3), nb_classes=43)
            dropout_model.compile(loss='categorical_crossentropy',
                                  optimizer='adam',
                                  metrics=['accuracy'])
            weights = ori_model.get_weights()
            dropout_model.set_weights(weights)
            dropout_model.save("../models/traffic/vgg16_dropout.h5")
    else:
        if model_type == "densenet":
            ori_model = tf.keras.models.load_model("../models/cifar100/densenet.h5")
            dropout_model = DenseNet_model_dropout()
            dropout_model.compile(loss='categorical_crossentropy',
                          optimizer=tfa.optimizers.SGDW(lr=1e-03, momentum=0.9, weight_decay=1e-5),
                          metrics=['accuracy'])
            weights = ori_model.get_weights()
            dropout_model.set_weights(weights)
            dropout_model.save("../models/cifar100/densenet_dropout.h5")
        else:
            ori_model = tf.keras.models.load_model("../models/cifar100/resnet.h5")
            # ori_model.summary()
            dropout_model = ResNet101_drop()
            dropout_model.compile(loss='categorical_crossentropy',
                                  optimizer=tfa.optimizers.SGDW(lr=1e-03, momentum=0.9, weight_decay=1e-5),
                                  metrics=['accuracy'])
            weights = ori_model.get_weights()
            # dropout_model.summary()
            dropout_model.set_weights(weights)
            dropout_model.save("../models/cifar100/resnet_dropout.h5")


if __name__ == "__main__":
    # model_transfer("mnist", "lenet1")
    # model_transfer("mnist", "lenet5")
    #
    # model_transfer("cifar10", "resnet20")
    # model_transfer("cifar10", "vgg16")
    #
    # model_transfer("svhn", "lenet5")
    # model_transfer("svhn", "resnet20")
    #
    # model_transfer("traffic", "lenet5")
    # model_transfer("traffic", "vgg16")
    #
    # model_transfer("cifar100", "densenet")
    model_transfer("cifar100", "resnet")

