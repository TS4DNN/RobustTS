from metrics.ATS.ATS import *
import tensorflow as tf
import numpy as np
from metrics.DeepGini import *
from metrics.MaxP import *
from metrics.Random import *
from metrics.MCP import *
from metrics.MC import *
tf_version = tf.__version__
print(tf_version)
if tf_version == '2.3.0':
    from metrics.TestRank.ts_selection import *
    from metrics.PRIMA.prima_selection import *
from metrics.SA import *
from utils.data_combine import *
from model_prepare.svhn_utils import *
from tensorflow.keras.datasets import cifar100
import csv
import argparse
from tensorflow_addons.optimizers import SGDW
from model_prepare.cifar100_resnet import *
tf.keras.optimizers.SGDW = SGDW
import time


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


def test_prior(model, x_candidate, y_candidate, metric, budget, x_train=None, y_train=None, data_type=None, dropout_model=None, class_num=10):
    prediction = model.predict(x_candidate)
    if metric == "gini":
        gini_rank = deep_metric(prediction)
        fault_index = gini_rank[:budget]
    elif metric == "maxp":
        maxp_rank = max_p_rank(prediction)
        fault_index = maxp_rank[:budget]
    elif metric == "mcp":
        fault_index = select_only(model, budget, x_candidate)
    elif metric == "mc":
        fault_index = MC_selection(dropout_model, x_candidate, budget)
    elif metric == "ats":
        ats = ATS_metric()
        y_sel_psedu, predictions = get_psedu_label(model, x_candidate)
        div_rank, _, _ = ats.get_priority_sequence(x_candidate, y_sel_psedu, class_num, model, th=0.001)
        fault_index = div_rank[:budget]
    else:
        # random
        fault_index = random_selection(x_candidate, budget)
    selected_prediction = prediction[fault_index]
    selected_predicted_label = np.argmax(selected_prediction, axis=1)
    selected_ground_truth = y_candidate[fault_index]
    fault_num = np.sum(selected_predicted_label != selected_ground_truth)
    print("fault num : {}".format(fault_num))
    return fault_num, fault_index


def test_prior_testrank(model, x_candidate, y_candidate, metric, budget, x_train=None, y_train=None, data_type=None, dropout_model=None):
    fault_index_ls = test_rank_selection(x_train[:20000], y_train[:20000], x_candidate, y_candidate, model, budget, data_type=data_type)
    # fault_index_ls = test_rank_selection(x_candidate[5000:], y_candidate[5000:], x_candidate[:5000], y_candidate[:5000], model, budget,
    #                                      data_type=data_type)
    prediction = model.predict(x_candidate)
    fault_nums = []
    for fault_index in fault_index_ls:
        selected_prediction = prediction[fault_index]
        selected_predicted_label = np.argmax(selected_prediction, axis=1)
        selected_ground_truth = y_candidate[fault_index]
        fault_num = np.sum(selected_predicted_label != selected_ground_truth)
        print("fault num : {}".format(fault_num))
        fault_nums.append(fault_num)
    return fault_nums, fault_index_ls


def test_prior_prima(model, x_candidate, y_candidate, dataset, budgets, data_type, model_type=None):
    if dataset == "mnist":
        if model_type == "lenet1":
            mutant_path = "mutant_models/mnist/lenet1/GF_0.1_"
        else:
            mutant_path = "mutant_models/mnist/lenet5/GF_0.1_"
    elif dataset == "cifar10":
        if model_type == "resnet20":
            mutant_path = "mutant_models/cifar10/resnet20/GF_0.1_"
        else:
            mutant_path = "mutant_models/cifar10/vgg16/GF_0.1_"
    elif dataset == "svhn":
        if model_type == "lenet5":
            mutant_path = "mutant_models/svhn/lenet5_attack/GF_0.1_"
        else:
            mutant_path = "mutant_models/svhn/resnet20/GF_0.1_"
    elif dataset == "traffic":
        if model_type == "lenet5":
            mutant_path = "mutant_models/traffic/lenet5/GF_0.1_"
        else:
            mutant_path = "mutant_models/traffic/vgg16/GF_0.1_"
    else:
        if model_type == "densenet":
            mutant_path = "mutant_models/cifar100/densenet/GF_0.1_"
        else:
            mutant_path = "mutant_models/cifar100/resnet/GF_0.01_"
    fault_nums, index = prima_pri_index(model, dataset, x_candidate, y_candidate, mutant_path, budgets, data_type, model_type)
    print(fault_nums)
    return fault_nums, index


def DSA_selection(model, x_candidate, y_candidate, budget, dataset, data_type, model_name, x_train=None, class_num=10, dsa_layer=None):
    DSA_index = fetch_dsa(model, x_train, x_candidate, data_type, dsa_layer, class_num, dataset, model_name)
    fault_index = select_from_large(budget, DSA_index)
    prediction = model.predict(x_candidate)
    selected_prediction = prediction[fault_index]
    selected_predicted_label = np.argmax(selected_prediction, axis=1)
    selected_ground_truth = y_candidate[fault_index]
    fault_num = np.sum(selected_predicted_label != selected_ground_truth)
    print("fault num : {}".format(fault_num))
    return fault_num, fault_index


def prioritization_before_attack_run_all(dataset, data_name, model_name, save_path):
    metrics = ["gini", "maxp", "mcp", "random", "mc", "ats", "dsa", "testrank", "prima"]
    budgets_ratio = np.array([i / 100 for i in range(1, 11)])
    if dataset == "mnist":
        if data_name == "ori":
            (x_train, y_train), (x_final, y_final) = tf.keras.datasets.mnist.load_data()
        else:
            (x_train, y_train), (_, _) = tf.keras.datasets.mnist.load_data()
            x_final = np.load("datasets/mnist/" + data_name + "_test_x.npy")
            y_final = np.load("datasets/mnist/" + data_name + "_test_y.npy")
        x_train = x_train.astype("float32") / 255
        x_final = x_final.astype("float32") / 255
        budgets = budgets_ratio * len(x_final)
        budgets = budgets.astype("int")
        model = tf.keras.models.load_model("models/mnist/" + model_name + ".h5")
        drop_model = tf.keras.models.load_model("models/mnist/" + model_name + "_dropout.h5")
        class_num = 10
        dsa_layer = ["before_softmax"]

    elif dataset == "cifar10":
        if data_name == "ori":
            (x_train, y_train), (x_final, y_final) = tf.keras.datasets.cifar10.load_data()
            x_train = x_train.astype('float32') / 255
            x_train_mean = np.mean(x_train, axis=0)
            y_final = y_final.reshape(10000, )
        else:
            (x_train, y_train), (_, _) = tf.keras.datasets.cifar10.load_data()
            x_train = x_train.astype('float32') / 255
            x_train_mean = np.mean(x_train, axis=0)
            x_final = np.load("datasets/cifar10/" + data_name + "_test_x.npy")
            y_final = np.load("datasets/cifar10/" + data_name + "_test_y.npy")
            y_final = y_final.reshape(10000, )
        # x_train = x_train.astype("float32") / 255
        x_final = x_final.astype("float32") / 255
        if model_name == "resnet20":
            x_train -= x_train_mean
            x_final -= x_train_mean
            dsa_layer = ["global_average_pooling2d"]
        else:
            dsa_layer = ["batch_normalization_9"]
        model = tf.keras.models.load_model("models/cifar10/" + model_name + ".h5")
        drop_model = tf.keras.models.load_model("models/cifar10/" + model_name + "_dropout.h5")
        budgets = budgets_ratio * len(x_final)
        budgets = budgets.astype("int")
        class_num = 10
    elif dataset == "svhn":
        if data_name == "ori":
            (x_train, y_train), (x_test, y_test) = load_data()  # 32*32
            x_train = x_train[:-10000]
            y_train = y_train[:-10000]
            x_final = x_test[:10000]
            y_final = y_test[:10000]
            y_final = y_final.argmax(axis=1)
        else:
            (x_train, y_train), (_, _) = load_data()  # 32*32
            x_train = x_train[:-10000]
            y_train = y_train[:-10000]
            x_final = np.load("datasets/svhn/" + data_name + "_test_x.npy")
            y_final = np.load("datasets/svhn/" + data_name + "_test_y.npy")
            x_final = x_final.astype("float32") / 255
        y_train = y_train.argmax(axis=1)
        # x_train = x_train.astype("float32") / 255
        model = tf.keras.models.load_model("models/svhn/" + model_name + ".h5")
        drop_model = tf.keras.models.load_model("models/svhn/" + model_name + "_dropout.h5")
        budgets = budgets_ratio * len(x_final)
        budgets = budgets.astype("int")
        class_num = 10
        if model_name == "lenet5":
            dsa_layer = ["dense_2"]
        else:
            dsa_layer = ["before_softmax"]

    elif dataset == "cifar100":
        if data_name == "ori":
            (x_train, y_train), (x_final, y_final) = cifar100.load_data()
            x_final = norm_images(x_final)
            x_train = norm_images(x_train)
        else:
            (x_train, y_train), (_, _) = cifar100.load_data()
            x_final = np.load("datasets/cifar100/" + data_name + "_test_x.npy")
            y_final = np.load("datasets/cifar100/" + data_name + "_test_y.npy")
            x_final = norm_images(x_final)
            x_train = norm_images(x_train)
        y_train = y_train.reshape(-1)
        y_final = y_final.reshape(-1)
        model = tf.keras.models.load_model("models/cifar100/" + model_name + ".h5")
        drop_model = tf.keras.models.load_model("models/cifar100/" + model_name + "_dropout.h5")
        budgets = budgets_ratio * len(x_final)
        budgets = budgets.astype("int")
        class_num = 100
        dsa_layer = ['dense']

    else:
        x_train = np.load("datasets/traffic/x_train.npy")
        y_train = np.load("datasets/traffic/y_train.npy")
        x_final = np.load("datasets/traffic/" + data_name + "_test_x.npy")
        y_final = np.load("datasets/traffic/" + data_name + "_test_y.npy")
        x_train = x_train.astype("float32") / 255
        x_final = x_final.astype("float32") / 255
        model = tf.keras.models.load_model("models/traffic/" + model_name + ".h5")
        drop_model = tf.keras.models.load_model("models/traffic/" + model_name + "_dropout.h5")
        budgets = budgets_ratio * len(x_final)
        budgets = budgets.astype("int")
        class_num = 43
        if model_name == "lenet5":
            dsa_layer = ["before_softmax"]
        else:
            dsa_layer = ["batch_normalization_9"]
    for metric in metrics:
        print("metric: ", metric)
        if metric == "testrank":
            start_time = time.clock()
            results, selected_index = test_prior_testrank(model, x_final, y_final, metric, budgets, x_train=x_train, y_train=y_train, data_type=dataset)
            elapsed = (time.clock() - start_time)
            print("running time: ", elapsed)
            results.append(elapsed)
        elif metric == "prima":
            start_time = time.clock()
            results, selected_index = test_prior_prima(model, x_final, y_final, dataset, budgets, data_name, model_type=model_name)
            elapsed = (time.clock() - start_time)
            print("running time: ", elapsed)
            results.append(elapsed)
        elif metric == "dsa":
            results = []
            selected_index = []
            start_time = time.clock()
            for budget in budgets:
                print("budget: ", budget)
                fault_num, selected_index_single = DSA_selection(model, x_final, y_final, budget, dataset, data_name, model_name, x_train=x_train, class_num=class_num, dsa_layer=dsa_layer)
                results.append(fault_num)
                selected_index.append(selected_index_single)
            elapsed = (time.clock() - start_time)
            print("running time: ", elapsed)
            results.append(elapsed)
        else:
            results = []
            selected_index = []
            start_time = time.clock()
            for budget in budgets:
                print("budget: ", budget)
                fault_num, selected_index_single = test_prior(model, x_final, y_final, metric, budget, x_train=x_train, y_train=y_train, data_type=dataset, class_num=class_num, dropout_model=drop_model)
                results.append(fault_num)
                selected_index.append(selected_index_single)
            elapsed = (time.clock() - start_time)
            print("running time: ", elapsed)
            results.append(elapsed)
        results = [metric] + results
        csv_file = open(save_path, "a")
        try:
            writer = csv.writer(csv_file)
            writer.writerow(results)
        finally:
            csv_file.close()

        index_save_path = "datasets/selected_index/" + metric + "_" + dataset + "_" + model_name + "_" + data_name + ".csv"
        csv_file = open(index_save_path, "a")
        try:
            writer = csv.writer(csv_file)
            for _ in selected_index:
                writer.writerow(_)
        finally:
            csv_file.close()



def prioritization_after_attack_run_all(dataset, data_name, model_name, save_path, attack_type):
    metrics = ["gini", "maxp", "mcp", "random", "mc", "ats", "dsa", "testrank", "prima"]
    budgets_ratio = np.array([i / 100 for i in range(1, 11)])

    if dataset == "mnist":
        if data_name == "ori":
            (x_train, y_train), (x_final, y_final) = tf.keras.datasets.mnist.load_data()
        else:
            (x_train, y_train), (_, _) = tf.keras.datasets.mnist.load_data()
            x_final = np.load("datasets/mnist/" + data_name + "_test_x.npy")
            y_final = np.load("datasets/mnist/" + data_name + "_test_y.npy")
        x_train = x_train.astype("float32") / 255
        x_final = x_final.astype("float32") / 255
        x_train = x_final.reshape(-1, 28, 28, 1)
        x_final = x_final.reshape(-1, 28, 28, 1)
        budgets = budgets_ratio * len(x_final)
        budgets = budgets.astype("int")
        model = tf.keras.models.load_model("models/mnist/" + model_name + ".h5")
        drop_model = tf.keras.models.load_model("models/mnist/" + model_name + "_dropout.h5")
        class_num = 10

    elif dataset == "cifar10":
        if data_name == "ori":
            (x_train, y_train), (x_final, y_final) = tf.keras.datasets.cifar10.load_data()
            x_train = x_train.astype('float32') / 255
            x_train_mean = np.mean(x_train, axis=0)
            y_final = y_final.reshape(-1)
            y_train = y_train.reshape(-1)
        else:
            (x_train, y_train), (_, _) = tf.keras.datasets.cifar10.load_data()
            x_train = x_train.astype('float32') / 255
            x_train_mean = np.mean(x_train, axis=0)
            x_final = np.load("datasets/cifar10/" + data_name + "_test_x.npy")
            y_final = np.load("datasets/cifar10/" + data_name + "_test_y.npy")
            y_final = y_final.reshape(-1)
            y_train = y_train.reshape(-1)
        # x_train = x_train.astype("float32") / 255
        x_final = x_final.astype("float32") / 255
        if model_name == "resnet20":
            x_train -= x_train_mean
            x_final -= x_train_mean
        model = tf.keras.models.load_model("models/cifar10/" + model_name + ".h5")
        drop_model = tf.keras.models.load_model("models/cifar10/" + model_name + "_dropout.h5")
        budgets = budgets_ratio * len(x_final)
        budgets = budgets.astype("int")
        class_num = 10
    elif dataset == "svhn":
        if data_name == "ori":
            (x_train, y_train), (x_test, y_test) = load_data()  # 32*32
            x_train = x_train[:-10000]
            y_train = y_train[:-10000]
            x_final = x_test[:10000]
            y_final = y_test[:10000]
            y_final = y_final.argmax(axis=1)
        else:
            (x_train, y_train), (_, _) = load_data()  # 32*32
            x_train = x_train[:-10000]
            y_train = y_train[:-10000]
            x_final = np.load("datasets/svhn/" + data_name + "_test_x.npy")
            y_final = np.load("datasets/svhn/" + data_name + "_test_y.npy")
            x_final = x_final.astype("float32") / 255
        y_train = y_train.argmax(axis=1)
        # x_train = x_train.astype("float32") / 255
        model = tf.keras.models.load_model("models/svhn/" + model_name + ".h5")
        drop_model = tf.keras.models.load_model("models/svhn/" + model_name + "_dropout.h5")
        budgets = budgets_ratio * len(x_final)
        budgets = budgets.astype("int")
        class_num = 10

    elif dataset == "cifar100":
        if data_name == "ori":
            (x_train, y_train), (x_final, y_final) = cifar100.load_data()
            x_final = norm_images(x_final)
            x_train = norm_images(x_train)
        else:
            (x_train, y_train), (_, _) = cifar100.load_data()
            x_final = np.load("datasets/cifar100/" + data_name + "_test_x.npy")
            y_final = np.load("datasets/cifar100/" + data_name + "_test_y.npy")
            x_final = norm_images(x_final)
            x_train = norm_images(x_train)
        y_train = y_train.reshape(-1)
        y_final = y_final.reshape(-1)
        model = tf.keras.models.load_model("models/cifar100/" + model_name + ".h5")
        drop_model = tf.keras.models.load_model("models/cifar100/" + model_name + "_dropout.h5")
        budgets = budgets_ratio * len(x_final)
        budgets = budgets.astype("int")
        class_num = 100
    else:
        x_train = np.load("datasets/traffic/x_train.npy")
        y_train = np.load("datasets/traffic/y_train.npy")
        x_final = np.load("datasets/traffic/" + data_name + "_test_x.npy")
        y_final = np.load("datasets/traffic/" + data_name + "_test_y.npy")
        x_train = x_train.astype("float32") / 255
        x_final = x_final.astype("float32") / 255
        model = tf.keras.models.load_model("models/traffic/" + model_name + ".h5")
        drop_model = tf.keras.models.load_model("models/traffic/" + model_name + "_dropout.h5")
        budgets = budgets_ratio * len(x_final)
        budgets = budgets.astype("int")
        class_num = 43
    attack_data_path = "datasets/generated/" + attack_type + "/" + dataset + "/" + model_name + "/x.npy"
    attack_label_path = "datasets/generated/" + attack_type + "/" + dataset + "/" + model_name + "/y.npy"
    attack_data = np.load(attack_data_path)
    if attack_type == "type1":
        if dataset == "cifar100":
            attack_data = norm_images(attack_data)
        else:
            attack_data = attack_data.astype("float32") / 255
        if dataset == "cifar10" and model_name == "resnet20":
            # attack_data = attack_data.astype("float32") / 255
            attack_data -= x_train_mean
    attack_label = np.load(attack_label_path)
    x_final = np.concatenate((x_final, attack_data))
    y_final = np.concatenate((y_final, attack_label))
    len_un_attack = len(y_final[:-len(attack_data)])
    for metric in metrics:
        print("metric: ", metric)
        if metric == "testrank":
            start_time = time.clock()
            results, selected_index = test_prior_testrank(model, x_final, y_final, metric, budgets, x_train=x_train,
                                                          y_train=y_train, data_type=dataset)
            elapsed = (time.clock() - start_time)
            print("running time: ", elapsed)
            # results.append(elapsed)
        elif metric == "prima":
            start_time = time.clock()
            results, selected_index = test_prior_prima(model, x_final, y_final, dataset, budgets, data_name,
                                                       model_type=model_name)
            elapsed = (time.clock() - start_time)
            print("running time: ", elapsed)
            # results.append(elapsed)
        else:
            results = []
            selected_index = []
            start_time = time.clock()
            for budget in budgets:
                print("budget: ", budget)
                fault_num, selected_index_single = test_prior(model, x_final, y_final, metric, budget, x_train=x_train,
                                                              y_train=y_train, data_type=dataset, class_num=class_num,
                                                              dropout_model=drop_model)
                results.append(fault_num)
                selected_index.append(selected_index_single)
            elapsed = (time.clock() - start_time)
            print("running time: ", elapsed)
            # results.append(elapsed)
        attack_selected_num = []
        for _index in selected_index:
            _index = np.asarray(_index)
            attack_selected_num.append(len(np.where(_index >= len_un_attack)[0]))
        results = [metric] + results + attack_selected_num + [elapsed]
        # results.append(elapsed)
        csv_file = open(save_path, "a")
        try:
            writer = csv.writer(csv_file)
            writer.writerow(results)
        finally:
            csv_file.close()
        if attack_type != "no":
            index_save_path = "datasets/selected_index/" + attack_type + "/" + metric + "_" + dataset + "_" + model_name + "_" + data_name + ".csv"
        else:
            index_save_path = "datasets/selected_index/" + metric + "_" + dataset + "_" + model_name + "_" + data_name + ".csv"
        csv_file = open(index_save_path, "a")
        try:
            writer = csv.writer(csv_file)
            for _ in selected_index:
                writer.writerow(_)
        finally:
            csv_file.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--attack_type",
                        "-attack_type",
                        type=str,
                        default='no'
                        )
    parser.add_argument("--dataset",
                        "-dataset",
                        type=str,
                        default='mnist'
                        )
    parser.add_argument("--model_name",
                        "-model_name",
                        type=str
                        )
    parser.add_argument("--save_path",
                        "-save_path",
                        type=str
                        )
    parser.add_argument("--data_name",
                        "-data_name",
                        type=str
                        )

    args = parser.parse_args()
    save_path = args.save_path + args.dataset + "_" + args.model_name + "_" + args.data_name + ".csv"
    if args.attack_type == "no":
        prioritization_before_attack_run_all(args.dataset, args.data_name, args.model_name, save_path)
    else:
        prioritization_after_attack_run_all(args.dataset, args.data_name, args.model_name, save_path, args.attack_type)


if __name__ == "__main__":
    main()
