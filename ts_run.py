from metrics.CES import *
from metrics.Random import *
from metrics.PACE import *
import csv
import argparse
from tensorflow_addons.optimizers import SGDW
import tensorflow_model_optimization as tfmot
tf.keras.optimizers.SGDW = SGDW
from model_prepare.cifar100_resnet import *
import time
from model_prepare.svhn_utils import *
from tensorflow.keras.utils import to_categorical


def test_selection_main(model, metric, candidate_data, candidate_label, select_size, selected_layer, cluster_data=None):
    if metric == 'ces':
        selected_index = CES_selection(model, candidate_data, select_size, selected_layer)
    elif metric == 'pace':
        selected_index = PACE_selection(model, candidate_data, select_size, selected_layer)
    elif metric == 'random':
        selected_index = random_selection(candidate_data, select_size)
    selected_x = candidate_data[selected_index]
    selected_y = candidate_label[selected_index]
    predictions = model.predict(selected_x)
    predict_label = np.argmax(predictions, axis=1)
    acc = np.sum(predict_label == selected_y) / len(predict_label)
    return acc, selected_index


def selection_run_before_attack(dataset, data_type, model_type, save_path, selected_layer):
    select_sizes = [50, 60, 70, 80, 90, 100, 120, 120, 130, 140, 150, 160, 170, 180]
    selection_metrics = ['ces', 'pace']
    if dataset == "mnist":
        if data_type == "ori":
            (x_train, y_train), (x_final, y_final) = tf.keras.datasets.mnist.load_data()
        else:
            (x_train, y_train), (_, _) = tf.keras.datasets.mnist.load_data()
            x_final = np.load("datasets/mnist/" + data_type + "_test_x.npy")
            y_final = np.load("datasets/mnist/" + data_type + "_test_y.npy")
        x_train = x_train.astype("float32") / 255
        x_final = x_final.astype("float32") / 255
        model = tf.keras.models.load_model("models/mnist/" + model_type + ".h5")
        class_num = 10
    elif dataset == "cifar10":
        if data_type == "ori":
            (x_train, y_train), (x_final, y_final) = tf.keras.datasets.cifar10.load_data()
            x_train = x_train.astype('float32') / 255
            x_train_mean = np.mean(x_train, axis=0)
            y_final = y_final.reshape(-1)
        else:
            (x_train, y_train), (_, _) = tf.keras.datasets.cifar10.load_data()
            x_train = x_train.astype('float32') / 255
            x_train_mean = np.mean(x_train, axis=0)
            x_final = np.load("datasets/cifar10/" + data_type + "_test_x.npy")
            y_final = np.load("datasets/cifar10/" + data_type + "_test_y.npy")
            y_final = y_final.reshape(-1)
        x_final = x_final.astype("float32") / 255
        if model_type == "resnet20":
            x_train -= x_train_mean
            x_final -= x_train_mean
        model = tf.keras.models.load_model("models/cifar10/" + model_type + ".h5")
        class_num = 10

    elif dataset == "svhn":
        if data_type == "ori":
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
            x_final = np.load("datasets/svhn/" + data_type + "_test_x.npy")
            y_final = np.load("datasets/svhn/" + data_type + "_test_y.npy")
            x_final = x_final.astype("float32") / 255
        y_train = y_train.argmax(axis=1)
        model = tf.keras.models.load_model("models/svhn/" + model_type + ".h5")
        class_num = 10

    elif dataset == "cifar100":
        if data_type == "ori":
            (x_train, y_train), (x_final, y_final) = cifar100.load_data()
            x_final = norm_images(x_final)
            x_train = norm_images(x_train)
        else:
            (x_train, y_train), (_, _) = cifar100.load_data()
            x_final = np.load("datasets/cifar100/" + data_type + "_test_x.npy")
            y_final = np.load("datasets/cifar100/" + data_type + "_test_y.npy")
            x_final = norm_images(x_final)
            x_train = norm_images(x_train)
        y_train = y_train.reshape(-1)
        y_final = y_final.reshape(-1)
        model = tf.keras.models.load_model("models/cifar100/" + model_type + ".h5")
        class_num =100
    else:
        x_train = np.load("datasets/traffic/x_train.npy")
        y_train = np.load("datasets/traffic/y_train.npy")
        x_final = np.load("datasets/traffic/" + data_type + "_test_x.npy")
        y_final = np.load("datasets/traffic/" + data_type + "_test_y.npy")
        x_train = x_train.astype("float32") / 255
        x_final = x_final.astype("float32") / 255
        model = tf.keras.models.load_model("models/traffic/" + model_type + ".h5")
        class_num = 43
    y_final_1 = to_categorical(y_final, class_num)
    ori_acc = model.evaluate(x_final, y_final_1)[1]
    for metric in selection_metrics:
        accs = []
        selected_index = []
        start_time = time.clock()
        for select_size in select_sizes:
            acc, _index = test_selection_main(model, metric, x_final, y_final, select_size, selected_layer)
            accs.append(abs(ori_acc - acc) * 100)
            selected_index.append(_index)
        elapsed = (time.clock() - start_time)
        print("running time: ", elapsed)
        print(accs)
        mean_value = np.mean(accs)
        accs.append(mean_value)
        accs.append(elapsed)
        results = [metric] + accs
        csv_file = open(save_path, "a")
        try:
            writer = csv.writer(csv_file)
            writer.writerow(results)
        finally:
            csv_file.close()

        index_save_path = "datasets/selected_index/es/" + metric + "_" + dataset + "_" + model_type + "_" + data_type + "_" + str(selected_layer) + ".csv"
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

    parser.add_argument("--selected_layer",
                        "-selected_layer",
                        type=int
                        )

    args = parser.parse_args()
    save_path = args.save_path
    selected_layer = -args.selected_layer
    if args.attack_type == "no":
        selection_run_before_attack(args.dataset, args.data_name, args.model_name, save_path, selected_layer)



if __name__ == "__main__":
    main()
    # selection_run_combine()

# python ts_main_all.py -dataset mnist -data_type ori -save_path results/RQ1/mnist/ori_acc.csv
