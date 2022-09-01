import sys
sys.path.append('.')
from GA import Population
from PIL import Image
from tensorflow.keras.applications.vgg16 import preprocess_input
import random
import time
import argparse
import numpy as np
import os
import shutil
import tensorflow as tf
from tensorflow import keras
from img_mutators import Mutators
from scipy.stats import entropy
from tensorflow_addons.optimizers import SGDW
tf.keras.optimizers.SGDW = SGDW


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


def cifar_preprocessing_2(x_test):
    temp = np.copy(x_test)
    temp = temp.astype('float32') / 255
    return temp


def svhn_preprocessing(x_test):
    temp = np.copy(x_test)
    temp = temp.reshape(temp.shape[0], 32, 32, 3)
    temp = temp.astype('float32')
    temp /= 255
    return temp


def traffic_preprocessing(x_test):
    temp = np.copy(x_test)
    temp = temp.reshape(temp.shape[0], 32, 32, 3)
    temp = temp.astype('float32')
    temp /= 255
    return temp


def cifar100_preprocessing(x_test):
    temp = np.copy(x_test)
    temp = norm_images(temp)
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
    'cifar10_2': cifar_preprocessing_2,
    'mnist': mnist_preprocessing,
    'imagenet': imagenet_preprocessing,
    'svhn': svhn_preprocessing,
    'traffic': traffic_preprocessing,
    'cifar100': cifar100_preprocessing
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


def untarget_diff_object_func(model, preprocess, target_ratio=0.01, target_ratio_2=0.5):
    def func(indvs, ground_truth):
        # define target
        array = np.array(indvs)
        preprocessed = preprocess(array)
        outputs = predict(preprocessed, model)
        class_results = np.argmax(outputs[-1], axis=1)
        prob_sort_outputs = np.sort(outputs[-1], axis=1)
        top2_diff = prob_sort_outputs[:, -1] - prob_sort_outputs[:, -2]
        top2_diff2 = prob_sort_outputs[:, -1] - prob_sort_outputs[:, -3]

        fitness1 = -1 * top2_diff
        fitness2 = -1 * prob_sort_outputs[:, -1]

        top2_ind = top2_diff <= target_ratio
        top3_ind = top2_diff2 <= target_ratio_2
        is_correct = class_results == ground_truth


        final_index_1 = np.logical_and(top2_ind, top3_ind)
        final_index_2 = np.logical_and(top2_ind, is_correct)
        final_index = np.logical_and(final_index_1, final_index_2)

        candidate_index = np.where(final_index)[0]
        output_results = indvs[candidate_index]
        return fitness1, fitness2, output_results, top2_diff, top2_diff2, class_results
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
    seed_data = np.load("../datasets/seeds/mnist/" + model_type + "/" + "mnist_x_all.npy")
    seed_labels = np.load("../datasets/seeds/mnist/" + model_type + "/" + "mnist_y_all.npy")
    for ite in range(10):
        for i in range(len(seed_data)):
            ground_truth = seed_labels[i]
            pop_num = 200
            img = seed_data[i]
            inds = create_image_indvs(img, pop_num)
            mutation_function = build_mutate_func(img)
            save_function = build_save_func("/scratch/users/qihu/generated/type1/mnist/" + model_type + "/",
                                            "mnist_" + str(ground_truth) + "_" + str(i) + "_" + str(ite))
            preprocess = preprocess_dic["mnist"]
            fitness_compute_function = untarget_diff_object_func(model, preprocess, target_ratio=0.01, target_ratio_2=0.5)
            pop = Population(inds, mutation_function, fitness_compute_function, save_function, ground_truth,
                             max_time=1000000000, seed=img, max_iteration=200)


def generation_cifar10_run(model_type):
    if model_type == "resnet":
        model = tf.keras.models.load_model("../models/cifar10/resnet20.h5")
    else:
        model = tf.keras.models.load_model("../models/cifar10/VGG16.h5")
    # print(seed.shape)
    seed_data = np.load("../datasets/seeds/cifar10/" + model_type + "/" + "cifar10_x_all.npy")
    seed_labels = np.load("../datasets/seeds/cifar10/" + model_type + "/" + "cifar10_y_all.npy")
    for ite in range(10):
        for i in range(len(seed_data)):
            ground_truth = seed_labels[i]
            pop_num = 200
            img = seed_data[i]
            inds = create_image_indvs(img, pop_num)
            mutation_function = build_mutate_func(img)
            save_function = build_save_func("/scratch/users/qihu/generated/type1/cifar10/" + model_type + "/",
                                            "cifar10_" + str(ground_truth) + "_" + str(i) + "_" + str(ite))
            if model_type == "resnet":
                preprocess = preprocess_dic["cifar10"]
            else:
                preprocess = preprocess_dic["cifar10_2"]
            fitness_compute_function = untarget_diff_object_func(model, preprocess, target_ratio=0.01, target_ratio_2=0.5)
            pop = Population(inds, mutation_function, fitness_compute_function, save_function, ground_truth,
                             max_time=1000000000, seed=img, max_iteration=200)


def generation_svhn_run(model_type):
    if model_type == "lenet5":
        model = tf.keras.models.load_model("../models/svhn/lenet5.h5")
    else:
        model = tf.keras.models.load_model("../models/svhn/resnet20.h5")
    # print(seed.shape)
    seed_data = np.load("../datasets/seeds/svhn/" + model_type + "/" + "svhn_x_all.npy")
    seed_labels = np.load("../datasets/seeds/svhn/" + model_type + "/" + "svhn_y_all.npy")
    for ite in range(10):
        for i in range(len(seed_data)):
            ground_truth = seed_labels[i]
            pop_num = 200
            img = seed_data[i]
            inds = create_image_indvs(img, pop_num)
            mutation_function = build_mutate_func(img)
            save_function = build_save_func("/scratch/users/qihu/generated/type1/svhn/" + model_type + "/",
                                            "svhn_" + str(ground_truth) + "_" + str(i) + "_" + str(ite))
            preprocess = preprocess_dic["svhn"]
            fitness_compute_function = untarget_diff_object_func(model, preprocess, target_ratio=0.01, target_ratio_2=0.5)
            pop = Population(inds, mutation_function, fitness_compute_function, save_function, ground_truth,
                             max_time=1000000000, seed=img, max_iteration=200)


def generation_traffic_run(model_type):
    if model_type == "lenet5":
        model = tf.keras.models.load_model("../models/traffic/lenet5.h5")
    else:
        model = tf.keras.models.load_model("../models/traffic/vgg16.h5")
    seed_data = np.load("../datasets/seeds/traffic/" + model_type + "/" + "traffic_x_all.npy")
    seed_labels = np.load("../datasets/seeds/traffic/" + model_type + "/" + "traffic_y_all.npy")
    for ite in range(10):
        for i in range(len(seed_data)):
            ground_truth = seed_labels[i]
            pop_num = 200
            img = seed_data[i]
            inds = create_image_indvs(img, pop_num)
            mutation_function = build_mutate_func(img)
            save_function = build_save_func("/scratch/users/qihu/generated/type1/traffic/" + model_type + "/",
                                            "traffic_" + str(ground_truth) + "_" + str(i) + "_" + str(ite))
            preprocess = preprocess_dic["traffic"]
            fitness_compute_function = untarget_diff_object_func(model, preprocess, target_ratio=0.05, target_ratio_2=0.3)
            pop = Population(inds, mutation_function, fitness_compute_function, save_function, ground_truth,
                             max_time=1000000000, seed=img, max_iteration=200)


def generation_cifar100_run(model_type):
    if model_type == "densenet":
        model = tf.keras.models.load_model("../models/cifar100/densenet.h5")
    else:
        model = tf.keras.models.load_model("../models/cifar100/resnet.h5")
    seed_data = np.load("../datasets/seeds/cifar100/" + model_type + "/" + "cifar100_x_all.npy")
    seed_labels = np.load("../datasets/seeds/cifar100/" + model_type + "/" + "cifar100_y_all.npy")
    for ite in range(10):
        for i in range(len(seed_data)):
            ground_truth = seed_labels[i]
            pop_num = 200
            img = seed_data[i]
            inds = create_image_indvs(img, pop_num)
            mutation_function = build_mutate_func(img)
            save_function = build_save_func("/scratch/users/qihu/generated/type1/cifar100/" + model_type + "/",
                                            "cifar100_" + str(ground_truth) + "_" + str(i) + "_" + str(ite))
            preprocess = preprocess_dic["cifar100"]
            fitness_compute_function = untarget_diff_object_func(model, preprocess, target_ratio=0.05, target_ratio_2=0.1)
            pop = Population(inds, mutation_function, fitness_compute_function, save_function, ground_truth,
                             max_time=1000000000, seed=img, max_iteration=200)


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
    # diverse_generation_run()
    # generation_cifar10_run()
    # diverse_generation_cifar10_run()
    main()
