from metrics.ATS.ATS import *
import tensorflow as tf
from test_prioritization import *
tf_version = tf.__version__
# print(tf_version)
# if tf_version == '2.3.0':
#     from metrics.TestRank.ts_selection import *
# from metrics.PRIMA.prima_selection import *
import tensorflow.keras.backend as K
from utils.data_combine import *
from model_prepare.svhn_utils import *
from tensorflow.keras.datasets import cifar100
import csv
import argparse
from tensorflow_addons.optimizers import SGDW
from model_prepare.cifar100_resnet import *
tf.keras.optimizers.SGDW = SGDW
import time
from tensorflow.keras.utils import to_categorical


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


def retrain_run_before_attack(dataset, data_name, model_name, save_path):
    budgets_ratio = np.array([i / 100 for i in range(1, 11)])
    metrics = ["gini", "maxp", "mcp", "random", "mc", "ats", "dsa", "testrank", "prima"]
    if dataset == "mnist":
        epochs = 5
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
        batch_size = 256

    elif dataset == "cifar10":
        epochs = 10
        if data_name == "ori":
            (x_train, y_train), (x_final, y_final) = tf.keras.datasets.cifar10.load_data()
            x_train = x_train.astype('float32') / 255
            x_train_mean = np.mean(x_train, axis=0)
            y_final = y_final.reshape(-1)
        else:
            (x_train, y_train), (_, _) = tf.keras.datasets.cifar10.load_data()
            x_train = x_train.astype('float32') / 255
            x_train_mean = np.mean(x_train, axis=0)
            x_final = np.load("datasets/cifar10/" + data_name + "_test_x.npy")
            y_final = np.load("datasets/cifar10/" + data_name + "_test_y.npy")
            y_final = y_final.reshape(-1)
        y_train = y_train.reshape(-1)
        x_final = x_final.astype("float32") / 255
        if model_name == "resnet20":
            x_train -= x_train_mean
            x_final -= x_train_mean
        budgets = budgets_ratio * len(x_final)
        budgets = budgets.astype("int")
        model = tf.keras.models.load_model("models/cifar10/" + model_name + ".h5")
        drop_model = tf.keras.models.load_model("models/cifar10/" + model_name + "_dropout.h5")
        class_num = 10
        batch_size = 128

    elif dataset == "svhn":
        epochs = 5
        if data_name == "ori":
            (x_train, y_train), (x_test, y_test) = load_data()  # 32*32
            x_train = x_train[:-10000]
            y_train = y_train[:-10000]
            x_final = x_test[:10000]
            y_final = y_test[:10000]
            # y_train = y_train.argmax(axis=1)
            y_final = y_final.argmax(axis=1)
        else:
            (x_train, y_train), (_, _) = load_data()  # 32*32
            x_train = x_train[:-10000]
            y_train = y_train[:-10000]
            x_final = np.load("datasets/svhn/" + data_name + "_test_x.npy")
            y_final = np.load("datasets/svhn/" + data_name + "_test_y.npy")
            x_final = x_final.astype("float32") / 255
        y_train = y_train.argmax(axis=1)
        budgets = budgets_ratio * len(x_final)
        budgets = budgets.astype("int")
        model = tf.keras.models.load_model("models/svhn/" + model_name + ".h5")
        drop_model = tf.keras.models.load_model("models/svhn/" + model_name + "_dropout.h5")
        class_num = 10
        batch_size = 64

    elif dataset == "cifar100":
        epochs = 10
        if data_name == "ori":
            (x_train, y_train), (x_final, y_final) = cifar100.load_data()
            x_final = norm_images(x_final)
            x_train = norm_images(x_train)
        else:
            (x_train, y_train), (_, _) = cifar100.load_data()
            x_final = np.load("datasets/" + dataset + "/" + data_name + "_test_x.npy")
            y_final = np.load("datasets/" + dataset + "/" + data_name + "_test_y.npy")
            x_final = norm_images(x_final)
            x_train = norm_images(x_train)
        y_train = y_train.reshape(-1)
        y_final = y_final.reshape(-1)
        budgets = budgets_ratio * len(x_final)
        budgets = budgets.astype("int")
        model = tf.keras.models.load_model("models/cifar100/" + model_name + ".h5")
        drop_model = tf.keras.models.load_model("models/cifar100/" + model_name + "_dropout.h5")
        class_num = 100
        batch_size = 64
    else:
        epochs = 5
        x_train = np.load("datasets/traffic/x_train.npy")
        y_train = np.load("datasets/traffic/y_train.npy")
        x_final = np.load("datasets/traffic/" + data_name + "_test_x.npy")
        y_final = np.load("datasets/traffic/" + data_name + "_test_y.npy")
        x_train = x_train.astype("float32") / 255
        x_final = x_final.astype("float32") / 255

        budgets = budgets_ratio * len(x_final)
        budgets = budgets.astype("int")
        model = tf.keras.models.load_model("models/traffic/" + model_name + ".h5")
        drop_model = tf.keras.models.load_model("models/traffic/" + model_name + "_dropout.h5")
        class_num = 43
        batch_size = 128

    candidate_index = np.random.choice(np.arange(len(x_final)), int(len(x_final) / 2), replace=False)
    candidate_data = x_final[candidate_index]
    candidate_label = y_final[candidate_index]
    new_test_index = np.delete(np.arange(len(x_final)), candidate_index)
    new_test_data = x_final[new_test_index]
    new_test_label = y_final[new_test_index]

    for metric in metrics:
        print("metric: ", metric)
        if metric == "testrank":
            start_time = time.clock()
            results, selected_index = test_prior_testrank(model, candidate_data, candidate_label, metric, budgets, x_train=x_train,
                                                          y_train=y_train, data_type=dataset)
            elapsed = (time.clock() - start_time)
            print("running time: ", elapsed)
            # results.append(elapsed)
        elif metric == "prima":
            start_time = time.clock()
            results, selected_index = test_prior_prima(model, candidate_data, candidate_label, dataset, budgets, data_name,
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
                fault_num, selected_index_single = test_prior(model, candidate_data, candidate_label, metric, budget, x_train=x_train,
                                                              y_train=y_train, data_type=dataset, class_num=class_num,
                                                              dropout_model=drop_model)
                results.append(fault_num)
                selected_index.append(selected_index_single)
            elapsed = (time.clock() - start_time)
            print("running time: ", elapsed)
        count = 0

        y_final_1 = to_categorical(new_test_label, class_num)
        ori_acc = model.evaluate(new_test_data, y_final_1)[1]
        results = selected_index
        for selected_index in results:
            model = tf.keras.models.load_model("models/" + dataset + "/" + model_name + ".h5")
            print("round: {}".format(count))
            count += 1
            selected_x = candidate_data[selected_index]
            selected_y = candidate_label[selected_index]
            total_x = np.concatenate((x_train, selected_x))
            total_y = np.concatenate((y_train, selected_y))
            # print(total_y)
            total_y = to_categorical(total_y, class_num)
            his = model.fit(total_x,
                            total_y,
                            validation_data=(new_test_data, y_final_1),
                            batch_size=batch_size,
                            shuffle=True,
                            epochs=epochs,
                            verbose=1,
                            # callbacks=[checkpoint]
                            )
            # print(his.history["val_accuracy"])
            accs = his.history["val_accuracy"]
            accs = np.asarray(accs)
            accs_diff = accs - ori_acc
            accs_diff = accs_diff * 100
            accs_diff = accs_diff.tolist()
            save_results = [metric] + accs_diff
            print(save_results)
            csv_file = open(save_path, "a")
            try:
                writer = csv.writer(csv_file)
                writer.writerow(save_results)
            finally:
                csv_file.close()
            K.clear_session()
            del model


def retrain_run_after_attack(dataset, data_name, model_name, save_path, attack_type):
    budgets_ratio = np.array([i / 100 for i in range(1, 11)])
    metrics = ["gini", "maxp", "mcp", "random", "mc", "ats", "dsa", "testrank", "prima"]

    if dataset == "mnist":
        epochs = 5
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
        batch_size = 256

    elif dataset == "cifar10":
        epochs = 10
        if data_name == "ori":
            (x_train, y_train), (x_final, y_final) = tf.keras.datasets.cifar10.load_data()
            x_train = x_train.astype('float32') / 255
            x_train_mean = np.mean(x_train, axis=0)
            y_final = y_final.reshape(-1)
        else:
            (x_train, y_train), (_, _) = tf.keras.datasets.cifar10.load_data()
            x_train = x_train.astype('float32') / 255
            x_train_mean = np.mean(x_train, axis=0)
            x_final = np.load("datasets/cifar10/" + data_name + "_test_x.npy")
            y_final = np.load("datasets/cifar10/" + data_name + "_test_y.npy")
            y_final = y_final.reshape(-1)
        y_train = y_train.reshape(-1)
        x_final = x_final.astype("float32") / 255
        if model_name == "resnet20":
            x_train -= x_train_mean
            x_final -= x_train_mean
        budgets = budgets_ratio * len(x_final)
        budgets = budgets.astype("int")
        model = tf.keras.models.load_model("models/cifar10/" + model_name + ".h5")
        drop_model = tf.keras.models.load_model("models/cifar10/" + model_name + "_dropout.h5")
        class_num = 10
        batch_size = 128

    elif dataset == "svhn":
        epochs = 5
        if data_name == "ori":
            (x_train, y_train), (x_test, y_test) = load_data()  # 32*32
            x_train = x_train[:-10000]
            y_train = y_train[:-10000]
            x_final = x_test[:10000]
            y_final = y_test[:10000]
            # y_train = y_train.argmax(axis=1)
            y_final = y_final.argmax(axis=1)
        else:
            (x_train, y_train), (_, _) = load_data()  # 32*32
            x_train = x_train[:-10000]
            y_train = y_train[:-10000]
            x_final = np.load("datasets/svhn/" + data_name + "_test_x.npy")
            y_final = np.load("datasets/svhn/" + data_name + "_test_y.npy")
            x_final = x_final.astype("float32") / 255
        y_train = y_train.argmax(axis=1)
        budgets = budgets_ratio * len(x_final)
        budgets = budgets.astype("int")
        model = tf.keras.models.load_model("models/svhn/" + model_name + ".h5")
        drop_model = tf.keras.models.load_model("models/svhn/" + model_name + "_dropout.h5")
        class_num = 10
        batch_size = 64

    elif dataset == "cifar100":
        epochs = 10
        if data_name == "ori":
            (x_train, y_train), (x_final, y_final) = cifar100.load_data()
            x_final = norm_images(x_final)
            x_train = norm_images(x_train)
        else:
            (x_train, y_train), (_, _) = cifar100.load_data()
            x_final = np.load("datasets/" + dataset + "/" + data_name + "_test_x.npy")
            y_final = np.load("datasets/" + dataset + "/" + data_name + "_test_y.npy")
            x_final = norm_images(x_final)
            x_train = norm_images(x_train)
        y_train = y_train.reshape(-1)
        y_final = y_final.reshape(-1)
        budgets = budgets_ratio * len(x_final)
        budgets = budgets.astype("int")
        model = tf.keras.models.load_model("models/cifar100/" + model_name + ".h5")
        drop_model = tf.keras.models.load_model("models/cifar100/" + model_name + "_dropout.h5")
        class_num = 100
        batch_size = 64
    else:
        epochs = 5
        x_train = np.load("datasets/traffic/x_train.npy")
        y_train = np.load("datasets/traffic/y_train.npy")
        x_final = np.load("datasets/traffic/" + data_name + "_test_x.npy")
        y_final = np.load("datasets/traffic/" + data_name + "_test_y.npy")
        x_train = x_train.astype("float32") / 255
        x_final = x_final.astype("float32") / 255

        budgets = budgets_ratio * len(x_final)
        budgets = budgets.astype("int")
        model = tf.keras.models.load_model("models/traffic/" + model_name + ".h5")
        drop_model = tf.keras.models.load_model("models/traffic/" + model_name + "_dropout.h5")
        class_num = 43
        batch_size = 128

    attack_data_path = "datasets/generated/" + attack_type + "/" + dataset + "/" + model_name + "/x.npy"
    attack_label_path = "datasets/generated/" + attack_type + "/" + dataset + "/" + model_name + "/y.npy"
    attack_data = np.load(attack_data_path)
    attack_label = np.load(attack_label_path)
    if attack_type == "type1":
        if dataset == "cifar100":
            attack_data = norm_images(attack_data)
        else:
            attack_data = attack_data / 255
        if dataset == "cifar10" and model_name == "resnet20":
            attack_data -= x_train_mean
    #
    x_final = np.concatenate((x_final, attack_data))
    y_final = np.concatenate((y_final, attack_label))
    candidate_index = np.random.choice(np.arange(len(x_final)), int(len(x_final) / 2), replace=False)
    candidate_data = x_final[candidate_index]
    candidate_label = y_final[candidate_index]
    new_test_index = np.delete(np.arange(len(x_final)), candidate_index)
    new_test_data = x_final[new_test_index]
    new_test_label = y_final[new_test_index]
    for metric in metrics:
        print("metric: ", metric)
        if metric == "testrank":
            start_time = time.clock()
            results, selected_index = test_prior_testrank(model, candidate_data, candidate_label, metric, budgets,
                                                          x_train=x_train,
                                                          y_train=y_train, data_type=dataset)
            elapsed = (time.clock() - start_time)
            print("running time: ", elapsed)
            # results.append(elapsed)
        elif metric == "prima":
            start_time = time.clock()
            results, selected_index = test_prior_prima(model, candidate_data, candidate_label, dataset, budgets,
                                                       data_name,
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
                fault_num, selected_index_single = test_prior(model, candidate_data, candidate_label, metric, budget,
                                                              x_train=x_train,
                                                              y_train=y_train, data_type=dataset, class_num=class_num,
                                                              dropout_model=drop_model)
                results.append(fault_num)
                selected_index.append(selected_index_single)
            elapsed = (time.clock() - start_time)
            print("running time: ", elapsed)
        count = 0

        y_final_1 = to_categorical(new_test_label, class_num)
        ori_acc = model.evaluate(new_test_data, y_final_1)[1]
        results = selected_index
        for selected_index in results:
            model = tf.keras.models.load_model("models/" + dataset + "/" + model_name + ".h5")
            print("round: {}".format(count))
            count += 1
            selected_x = candidate_data[selected_index]
            selected_y = candidate_label[selected_index]
            total_x = np.concatenate((x_train, selected_x))
            total_y = np.concatenate((y_train, selected_y))
            # print(total_y)
            total_y = to_categorical(total_y, class_num)
            his = model.fit(total_x,
                            total_y,
                            validation_data=(new_test_data, y_final_1),
                            batch_size=batch_size,
                            shuffle=True,
                            epochs=epochs,
                            verbose=1,
                            # callbacks=[checkpoint]
                            )
            # print(his.history["val_accuracy"])
            accs = his.history["val_accuracy"]
            accs = np.asarray(accs)
            accs_diff = accs - ori_acc
            accs_diff = accs_diff * 100
            accs_diff = accs_diff.tolist()
            save_results = [metric] + accs_diff
            print(save_results)
            csv_file = open(save_path, "a")
            try:
                writer = csv.writer(csv_file)
                writer.writerow(save_results)
            finally:
                csv_file.close()
            K.clear_session()
            del model


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
    # for data_name in data_names:
    save_path = args.save_path + args.dataset + "_" + args.model_name + "_" + args.data_name + ".csv"
    if args.attack_type == "no":
        retrain_run_before_attack(args.dataset, args.data_name, args.model_name, save_path)
    else:
        retrain_run_after_attack(args.dataset, args.data_name, args.model_name, save_path, args.attack_type)


if __name__ == "__main__":
    main()

