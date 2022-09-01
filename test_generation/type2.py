import sys
sys.path.append('.')
from GA import Population2
from PIL import Image
from keras.applications.vgg16 import preprocess_input
import random
import time
import argparse
import numpy as np
import os
import shutil
import tensorflow as tf
import keras
from img_mutators import Mutators
from scipy.stats import entropy


def imagenet_preprocessing(input_img_data):
    temp = np.copy(input_img_data)
    temp = np.float32(temp)
    qq = preprocess_input(temp)  # final input shape = (1,224,224,3)
    return qq


def mnist_preprocessing(x_test):
    temp = np.copy(x_test)
    temp = temp.reshape(temp.shape[0], 28, 28, 1)
    temp = temp.astype('float32')
    temp /= 255
    return temp


def cifar_preprocessing(x_test):
    (x_train, _), (_, _) = tf.keras.datasets.cifar10.load_data()
    x_train = x_train.astype('float32') / 255
    x_train_mean = np.mean(x_train, axis=0)
    temp = np.copy(x_test)
    temp = temp.astype('float32') / 255
    temp -= x_train_mean
    return temp


shape_dic = {
    'vgg16': (32, 32, 3),
    'resnet20': (32, 32, 3),
    'lenet1': (28, 28, 1),
    'lenet4': (28, 28, 1),
    'lenet5': (28, 28, 1),
    'mobilenet': (224, 224, 3),
    'vgg19': (224, 224, 3),
    'resnet50': (224, 224, 3)
}
preprocess_dic = {
    'cifar10': cifar_preprocessing,
    'mnist': mnist_preprocessing,
    'imagenet': imagenet_preprocessing
}


def create_image_indvs(img, num):
    indivs = []
    indivs.append(img)
    mu = Mutators()
    for i in range(num - 1):
        indivs.append(mu.mutate(img, img))
    return np.array(indivs)


def predict(input_data, model):
    inp = model.inputs
    # ensure that layers[-2] is the logit output
    logits_layer = model.layers[-2]
    sotfmax_layer = model.layers[-1]
    model = keras.Model(inp, [logits_layer.output, sotfmax_layer.output])
    outputs = model.predict(input_data)
    return outputs


def untarget_diff_object_func(model, preprocess, target_ratio=0.9):
    def func(indvs, ground_truth):
        # define target
        array = np.array(indvs)
        preprocessed = preprocess(array)
        outputs = predict(preprocessed, model)

        class_results = np.argmax(outputs[-1], axis=1)
        is_correct = class_results == ground_truth
        correct_index = np.where(class_results == ground_truth)[0]
        prob_sort_outputs = np.sort(outputs[-1], axis=1)
        top1_results = prob_sort_outputs[:, -1]
        top2_results = prob_sort_outputs[:, -2]
        fitness1 = top1_results
        fitness1[correct_index] = top2_results
        fitness2 = fitness1
        top2_ind = top1_results >= target_ratio

        final_index_2 = np.logical_and(top2_ind, is_correct)
        final_index = final_index_2
        candidate_index = np.where(final_index)[0]
        output_results = indvs[candidate_index]
        return fitness1, fitness2, output_results, top1_results, top2_results, class_results
    return func


def build_mutate_func(ref_img):
    def func(indv):
        return Mutators.mutate(indv, ref_img)
    return func


def build_save_func(npy_output, seedname):
    def save_func(indvs, round):
        name = seedname + '_' + str(round)
        np.save(os.path.join(npy_output, name + '.npy'), indvs)
    return save_func


def generation_mnist_run(model_type):
    if model_type == "lenet1":
        model = tf.keras.models.load_model("../models/mnist/lenet1.h5")
    else:
        model = tf.keras.models.load_model("../models/mnist/lenet5.h5")
    seed_data = np.load("../datasets/seeds/mnist/mnist_x_all.npy")
    seed_labels = np.load("../datasets/seeds/mnist/mnist_y_all.npy")
    for ite in range(10):
        for i in range(len(seed_data)):
            ground_truth = seed_labels[i]
            pop_num = 200
            img = seed_data[i]
            inds = create_image_indvs(img, pop_num)
            mutation_function = build_mutate_func(img)
            save_function = build_save_func("../datasets/generated/mnist/",
                                            "mnist_" + str(ground_truth) + "_" + str(i) + "_" + str(ite))
            preprocess = preprocess_dic["mnist"]
            fitness_compute_function = untarget_diff_object_func(model, preprocess, target_ratio=0.95)
            pop = Population2(inds, mutation_function, fitness_compute_function, save_function, ground_truth,
                             max_time=1000000000, seed=img, max_iteration=100)


def generation_cifar10_run(model_type):
    if model_type == "resnet":
        model = tf.keras.models.load_model("../models/cifar10/resnet20.h5")
    else:
        model = tf.keras.models.load_model("../models/cifar10/VGG16.h5")
    # print(seed.shape)
    seed_data = np.load("../datasets/seeds/cifar10_x_all.npy")
    seed_labels = np.load("../datasets/seeds/cifar10_y_all.npy")
    for ite in range(10):
        for i in range(len(seed_data)):
            ground_truth = seed_labels[i]
            pop_num = 200
            img = seed_data[i]
            inds = create_image_indvs(img, pop_num)
            mutation_function = build_mutate_func(img)
            save_function = build_save_func("../datasets/generated/cifar10/",
                                            "cifar10_" + str(ground_truth) + "_" + str(i) + "_" + str(ite))
            preprocess = preprocess_dic["cifar10"]
            fitness_compute_function = untarget_diff_object_func(model, preprocess, target_ratio=0.95)
            pop = Population2(inds, mutation_function, fitness_compute_function, save_function, ground_truth,
                             max_time=1000000000, seed=img, max_iteration=100)


def generation_svhn_run(model_type):
    if model_type == "lenet5":
        model = tf.keras.models.load_model("../models/svhn/lenet5.h5")
    else:
        model = tf.keras.models.load_model("../models/svhn/nin.h5")
    # print(seed.shape)
    seed_data = np.load("../datasets/seeds/svhn/svhn_x_all.npy")
    seed_labels = np.load("../datasets/seeds/svhn/svhn_y_all.npy")
    for ite in range(10):
        for i in range(len(seed_data)):
            ground_truth = seed_labels[i]
            pop_num = 200
            img = seed_data[i]
            inds = create_image_indvs(img, pop_num)
            mutation_function = build_mutate_func(img)
            save_function = build_save_func("../datasets/generated/svhn/",
                                            "svhn_" + str(ground_truth) + "_" + str(i) + "_" + str(ite))
            preprocess = preprocess_dic["svhn"]
            fitness_compute_function = untarget_diff_object_func(model, preprocess, target_ratio=0.95)
            pop = Population2(inds, mutation_function, fitness_compute_function, save_function, ground_truth,
                             max_time=1000000000, seed=img, max_iteration=100)


def generation_traffic_run(model_type):
    if model_type == "lenet5":
        model = tf.keras.models.load_model("../models/traffic/lenet5.h5")
    else:
        model = tf.keras.models.load_model("../models/traffic/vgg16.h5")
    seed_data = np.load("../datasets/seeds/traffic/traffic_x_all.npy")
    seed_labels = np.load("../datasets/seeds/traffic/traffic_y_all.npy")
    for ite in range(10):
        for i in range(len(seed_data)):
            ground_truth = seed_labels[i]
            pop_num = 200
            img = seed_data[i]
            inds = create_image_indvs(img, pop_num)
            mutation_function = build_mutate_func(img)
            save_function = build_save_func("../datasets/generated/traffic/",
                                            "traffic_" + str(ground_truth) + "_" + str(i) + "_" + str(ite))
            preprocess = preprocess_dic["traffic"]
            fitness_compute_function = untarget_diff_object_func(model, preprocess, target_ratio=0.95)
            pop = Population2(inds, mutation_function, fitness_compute_function, save_function, ground_truth,
                             max_time=1000000000, seed=img, max_iteration=100)


def generation_cifar100_run(model_type):
    if model_type == "densenet":
        model = tf.keras.models.load_model("../models/cifar100/densenet.h5")
    else:
        model = tf.keras.models.load_model("../models/cifar100/resnet.h5")
    seed_data = np.load("../datasets/seeds/cifar100/cifar100_x_all.npy")
    seed_labels = np.load("../datasets/seeds/cifar100/cifar100_y_all.npy")
    for ite in range(10):
        for i in range(len(seed_data)):
            ground_truth = seed_labels[i]
            pop_num = 200
            img = seed_data[i]
            inds = create_image_indvs(img, pop_num)
            mutation_function = build_mutate_func(img)
            save_function = build_save_func("../datasets/generated/cifar100/",
                                            "cifar100_" + str(ground_truth) + "_" + str(i) + "_" + str(ite))
            preprocess = preprocess_dic["cifar100"]
            fitness_compute_function = untarget_diff_object_func(model, preprocess, target_ratio=0.5)
            pop = Population2(inds, mutation_function, fitness_compute_function, save_function, ground_truth,
                             max_time=1000000000, seed=img, max_iteration=100)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset",
                        "-dataset",
                        type=str,
                        default='mnist'
                        )
    parser.add_argument("--model_type",
                        "-model_type",
                        type=str
                        )
    args = parser.parse_args()
    dataset = args.dataset
    model_type = args.model_type
    if dataset == "mnist":
        generation_mnist_run(model_type)
    elif dataset == "cifar10":
        generation_cifar10_run(model_type)
    elif dataset == "svhn":
        generation_svhn_run(model_type)
    elif dataset == "traffic":
        generation_traffic_run(model_type)
    else:
        generation_cifar100_run(model_type)


if __name__ == '__main__':

    main()
