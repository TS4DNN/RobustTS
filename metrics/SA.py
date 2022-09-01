#!/usr/bin/env python2
# -*- coding: utf-8 -*-

import numpy as np
import time
import os

from multiprocessing import Pool
from tqdm import tqdm
from tensorflow.keras.models import load_model, Model
from scipy.stats import gaussian_kde

import utils
class Colors:
    HEADER = "\033[95m"
    OKBLUE = "\033[94m"
    OKGREEN = "\033[92m"
    WARNING = "\033[93m"
    FAIL = "\033[91m"
    ENDC = "\033[0m"
    BOLD = "\033[1m"


def info(msg):
    return Colors.OKBLUE + msg + Colors.ENDC


def infog(msg):
    return Colors.OKGREEN + msg + Colors.ENDC

def warn(msg):
    return Colors.WARNING + msg + Colors.ENDC


def fail(msg):
    return Colors.FAIL + msg + Colors.ENDC


def _aggr_output(x):
    return [np.mean(x[..., j]) for j in range(x.shape[-1])]


def _get_saved_path(base_path, dataset, dtype, layer_names, model_name):
    """Determine saved path of ats and pred

    Args:
        base_path (str): Base save path.
        dataset (str): Name of dataset.
        dtype (str): Name of dataset type (e.g., train, test, fgsm, ...).
        layer_names (list): List of layer names.

    Returns:
        ats_path: File path of ats.
        pred_path: File path of pred (independent of layers)
    """

    joined_layer_names = "_".join(layer_names)
    return (
        os.path.join(
            base_path,
            dataset + "_" + model_name + "_" + dtype + "_" + joined_layer_names + "_ats" + ".npy",
        ),
        os.path.join(base_path, dataset + "_" + model_name + "_" + dtype + "_pred" + ".npy"),
    )


def get_ats(
        model,
        dataset,
        name,
        layer_names,
        save_path=None,
        batch_size=128,
        is_classification=True,
        num_classes=10,
        num_proc=10,
):
    """Extract activation traces of dataset from model.

    Args:
        model (keras model): Subject model.
        dataset (list): Set of inputs fed into the model.
        name (str): Name of input set.
        layer_names (list): List of selected layer names.
        save_path (tuple): Paths of being saved ats and pred.
        batch_size (int): Size of batch when serving.
        is_classification (bool): Task type, True if classification task or False.
        num_classes (int): The number of classes (labels) in the dataset.
        num_proc (int): The number of processes for multiprocessing.

    Returns:
        ats (list): List of (layers, inputs, neuron outputs).
        pred (list): List of predicted classes.
    """

    temp_model = Model(
        inputs=model.input,
        outputs=[model.get_layer(layer_name).output for layer_name in layer_names],
    )

    prefix = info("[" + name + "] ")
    if is_classification:
        p = Pool(num_proc)
        print(prefix + "Model serving")
        print(dataset.shape)
        pred = model.predict(dataset, batch_size=batch_size, verbose=1)
        pred = np.argmax(pred, axis=1)
        if len(layer_names) == 1:
            layer_outputs = [
                temp_model.predict(dataset, batch_size=batch_size, verbose=1)
            ]
        else:
            layer_outputs = temp_model.predict(
                dataset, batch_size=batch_size, verbose=1
            )

        print(prefix + "Processing ATs")
        ats = None
        for layer_name, layer_output in zip(layer_names, layer_outputs):
            print("Layer: " + layer_name)
            if layer_output[0].ndim == 3:
                # For convolutional layers
                layer_matrix = np.array(
                    p.map(_aggr_output, [layer_output[i] for i in range(len(dataset))])
                )
            else:
                layer_matrix = np.array(layer_output)

            if ats is None:
                ats = layer_matrix
            else:
                ats = np.append(ats, layer_matrix, axis=1)
                layer_matrix = None

    if save_path is not None:
        np.save(save_path[0], ats)
        np.save(save_path[1], pred)

    return ats, pred


def find_closest_at(at, train_ats):
    """The closest distance between subject AT and training ATs.

    Args:
        at (list): List of activation traces of an input.
        train_ats (list): List of activation traces in training set (filtered)

    Returns:
        dist (int): The closest distance.
        at (list): Training activation trace that has the closest distance.
    """

    dist = np.linalg.norm(at - train_ats, axis=1)
    return (min(dist), train_ats[np.argmin(dist)])


def _get_train_target_ats(model, x_train, x_target, target_name, layer_names, dataset, num_classes, model_name):
    """Extract ats of train and target inputs. If there are saved files, then skip it.

    Args:
        model (keras model): Subject model.
        x_train (list): Set of training inputs.
        x_target (list): Set of target (test or adversarial) inputs.
        target_name (str): Name of target set.
        layer_names (list): List of selected layer names.
        args: keyboard args.

    Returns:
        train_ats (list): ats of train set.
        train_pred (list): pred of train set.
        target_ats (list): ats of target set.
        target_pred (list): pred of target set.
    """
    save_path = "for_dsa/"
    saved_train_path = _get_saved_path(save_path, dataset, "train", layer_names, model_name)

    if os.path.exists(saved_train_path[0]):
        print(infog("Found saved {} ATs, skip serving".format("train")))
        # In case train_ats is stored in a disk
        train_ats = np.load(saved_train_path[0])
        train_pred = np.load(saved_train_path[1])
    else:
        train_ats, train_pred = get_ats(
            model,
            x_train,
            "train",
            layer_names,
            num_classes=num_classes,
            is_classification=True,
            save_path=saved_train_path,
        )
        print(infog("train ATs is saved at " + saved_train_path[0]))

    saved_target_path = _get_saved_path(
        save_path, dataset, target_name, layer_names, model_name
    )
    if os.path.exists(saved_target_path[0]):
        print(infog("Found saved {} ATs, skip serving").format(target_name))
        # In case target_ats is stored in a disk
        target_ats = np.load(saved_target_path[0])
        target_pred = np.load(saved_target_path[1])
    else:
        target_ats, target_pred = get_ats(
            model,
            x_target,
            target_name,
            layer_names,
            num_classes=num_classes,
            is_classification=True,
            save_path=saved_target_path,
        )
        print(infog(target_name + " ATs is saved at " + saved_target_path[0]))

    return train_ats, train_pred, target_ats, target_pred


def fetch_dsa(model, x_train, x_target, target_name, layer_names, num_classes, dataset, model_name):
    """Distance-based SA

    Args:
        model (keras model): Subject model.
        x_train (list): Set of training inputs.
        x_target (list): Set of target (test or adversarial) inputs.
        target_name (str): Name of target set.
        layer_names (list): List of selected layer names.
        args: keyboard args.

    Returns:
        dsa (list): List of dsa for each target input.
    """


    prefix = info("[" + target_name + "] ")
    train_ats, train_pred, target_ats, target_pred = _get_train_target_ats(
        model, x_train, x_target, target_name, layer_names, dataset, num_classes, model_name
    )

    class_matrix = {}
    all_idx = []
    for i, label in enumerate(train_pred):
        if label not in class_matrix:
            class_matrix[label] = []
        class_matrix[label].append(i)
        all_idx.append(i)

    dsa = []

    print(prefix + "Fetching DSA")
    if os.path.exists(target_name + dataset + model_name + '_dsaats.npy'):
        print(infog("Found saved {} DSAs, skip serving").format(target_name))
        # In case target_ats is stored in a disk
        dsa = np.load(target_name + dataset + model_name + '_dsaats.npy')
        return list(dsa)
    for i, at in enumerate(tqdm(target_ats)):
        print(i)
        label = target_pred[i]
        a_dist, a_dot = find_closest_at(at, train_ats[class_matrix[label]])
        b_dist, _ = find_closest_at(
            a_dot, train_ats[list(set(all_idx) - set(class_matrix[label]))]
        )
        dsa.append(a_dist / b_dist)

    np.save(target_name + dataset + model_name + '_dsaats.npy', dsa)
    return dsa


# 每一个kdes，都是一个类的实例

def _get_lsa(kde, at, removed_cols):
    refined_at = np.delete(at, removed_cols, axis=0)
    # 达不到要求的神经元被删除了
    return np.asscalar(-kde.logpdf(np.transpose(refined_at)))


# 转换成Python类型的数 np.asscalar
# np.asscalar(np.array([24]))  返回 24



def find_index(target_lsa, selected_lst, max_lsa):
    for i in range(len(target_lsa)):
        if max_lsa==target_lsa[i] and i not in selected_lst:
            return i
    return 0


def order_output(target_lsa, select_amount):
    lsa_lst = []

    tmp_lsa_lst = target_lsa[:]
    selected_lst = []
    while len(selected_lst) < select_amount:
        max_lsa = max(tmp_lsa_lst)
        selected_lst.append(find_index(target_lsa, selected_lst, max_lsa))
        lsa_lst.append(max_lsa)
        tmp_lsa_lst.remove(max_lsa)
    return selected_lst, lsa_lst


def select_from_large(select_amount, target_lsa):
    selected_lst, lsa_lst = order_output(target_lsa, select_amount)
    # print(lsa_lst)
    print(selected_lst)
    selected_index = []
    for i in range(select_amount):
        selected_index.append(selected_lst[i])
    return selected_index
