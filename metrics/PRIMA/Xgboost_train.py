import xgboost as xgb
from numpy import genfromtxt
import tensorflow as tf
import numpy as np
from utils import *


def xgb_train(dataset, data_name, model_name):
    budgets = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
    model = xgb.XGBRanker(
        # tree_method='gpu_hist',
        booster='gbtree',
        objective='rank:pairwise',
        random_state=42,
        learning_rate=0.05,
        colsample_bytree=0.5,
        eta=0.05,
        max_depth=5,
        n_estimators=110,
        subsample=0.75
    )
    if dataset == "mnist":
        if data_name == "ori":
            (x_train, y_train), (x_final, y_final) = tf.keras.datasets.mnist.load_data()

        elif data_name == "RT":
            (x_train, y_train), (_, _) = tf.keras.datasets.mnist.load_data()
            x_train = x_train.reshape(-1, 28, 28, 1)
            x_final = np.load("../../datasets/mnist/" + data_name + "_test_x.npy")
            y_final = np.load("../../datasets/mnist/" + data_name + "_test_y.npy")
            model.load_model("models/mnist.json")
        elif data_name == "ori_adv" or data_name == "ori_adv_d":
            (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
            x_train = x_train.reshape(-1, 28, 28, 1)
            x_test = x_test.reshape(-1, 28, 28, 1)
            x_final, y_final = combine_clean_adv(x_test, y_test)
            model.load_model("models/mnist.json")
        else:
            (x_train, y_train), (_, _) = tf.keras.datasets.mnist.load_data()
            x_test = np.load("../../datasets/mnist/RT_test_x.npy")
            y_test = np.load("../../datasets/mnist/RT_test_y.npy")
            x_train = x_train.reshape(-1, 28, 28, 1)
            x_test = x_test.reshape(-1, 28, 28, 1)
            x_final, y_final = combine_clean_adv(x_test, y_test)
            model.load_model("models/mnist.json")

        dl_model = tf.keras.models.load_model("../../models/mnist/" + model_name + ".h5")

    x_candidate = x_final.astype("float32") / 255
    x_candidate = x_candidate.reshape(-1, 28, 28, 1)
    y_candidate = y_final

    x_train = genfromtxt("tem_model_" + dataset + "_" + data_name + "_feature.csv",
                         delimiter=',')
    x_train = x_train[:, 1:]
    x_train = x_train[1:, :]
    x_train = x_train / x_train.max(axis=0)

    x_train_2 = genfromtxt("tem_input_" + dataset + "_" + data_name + "_feature.csv",
                           delimiter=',')
    x_train_2 = x_train_2[:, 1:]
    x_train_2 = x_train_2[1:, :]
    x_train_2 = x_train_2 / x_train_2.max(axis=0)
    x_train = np.concatenate((x_train, x_train_2), axis=1)
    predicted_labels = dl_model.predict(x_candidate).argmax(axis=1)
    corrected_index = np.where(predicted_labels == y_candidate)[0]
    wrong_index = np.where(predicted_labels != y_candidate)[0]
    correct_num = len(corrected_index)
    predicted_labels[corrected_index] = 1
    predicted_labels[wrong_index] = 0
    X_features = np.concatenate((x_train[corrected_index], x_train[wrong_index]))
    y = np.concatenate((predicted_labels[corrected_index], predicted_labels[wrong_index]))
    X = np.nan_to_num(X_features)
    num_per_group = len(x_candidate) // 2
    if data_name == "ori":
        model.fit(X, y, group=[num_per_group, len(x_candidate) - num_per_group], verbose=True)
        model.save_model("models/mnist_" + data_name + ".json")
    y_predict = model.predict(X)
    sorted_index = np.argsort(y_predict)
    results = []
    for i in budgets:
        s = sorted_index[:i]
        fault_num = len(np.where(s >= correct_num)[0])
        # print(fault_num)
        results.append(fault_num)
    print(results)
    #     delete temporary folders
    print("finished!!!!")


if __name__ == '__main__':
    dataset = "mnist"
    data_name = "RT_adv_d"
    model_name = "lenet5"
    # dataset, data_name, model_name = "mnist", "ori", "lenet5"
    xgb_train(dataset, data_name, model_name)

# model = xgb.XGBRanker(
#     # tree_method='gpu_hist',
#     booster='gbtree',
#     objective='rank:pairwise',
#     random_state=42,
#     learning_rate=0.05,
#     colsample_bytree=0.5,
#     eta=0.05,
#     max_depth=5,
#     n_estimators=110,
#     subsample=0.75
#     )
#
# x_train = genfromtxt('tem_model_mnist_feature.csv', delimiter=',')
# x_train = x_train[:, 1:]
# x_train = x_train[1:, :]
# # print(x_train.shape)
# # x_train = (x_train - np.min(x_train))/np.ptp(x_train)
# x_train = x_train / x_train.max(axis=0)
#
# x_train_2 = genfromtxt('tem_input_mnist_feature.csv', delimiter=',')
# x_train_2 = x_train_2[:, 1:]
# x_train_2 = x_train_2[1:, :]
# x_train_2 = x_train_2 / x_train_2.max(axis=0)
#
# x_train = np.concatenate((x_train, x_train_2), axis=1)
#
# # print(x_train[0])
#
# keras_model = tf.keras.models.load_model("../../models/mnist/lenet5.h5")
# (_, _), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
# x_test = x_test.astype("float32") / 255
# predicted_labels = keras_model.predict(x_test).argmax(axis=1)
# corrected_index = np.where(predicted_labels == y_test)[0]
# wrong_index = np.where(predicted_labels != y_test)[0]
# predicted_labels[corrected_index] = 1
# predicted_labels[wrong_index] = 0
# # print(len(wrong_index))
#
# # Generate some sample data to illustrate ranking
# X_features = np.concatenate((x_train[corrected_index], x_train[wrong_index]))
# y = np.concatenate((predicted_labels[corrected_index], predicted_labels[wrong_index]))
#
# # X = np.concatenate([X_groups[:,None], X_features], axis=1)
#
# X = np.nan_to_num(X_features)
# # ranker = XGBRanker(n_estimators=150, learning_rate=0.1, subsample=0.9)
# print(y)
# # model.fit(X, y, group=[9887, 113], verbose=True)
# # model.fit(X[-226:], y[-226:], group=[113, 113], verbose=True)
# model.fit(X, y, group=[5000, 5000], verbose=True)
#
#
# y_predict = model.predict(X)
# sorted_index = np.argsort(y_predict)
# results = []
# for i in [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]:
#     s = sorted_index[:i]
#     print(s)
#     # results.append(len(np.where(s >= 9887)[0]))
#     print(len(np.where(s >= len(corrected_index))[0]))
#     results.append(s)
# # print(results)

# prediction = keras_model.predict(x_test)
# fault_nums = []
# for fault_index in results:
#     selected_prediction = prediction[fault_index]
#     selected_predicted_label = np.argmax(selected_prediction, axis=1)
#     selected_ground_truth = y_test[fault_index]
#     fault_num = np.sum(selected_predicted_label != selected_ground_truth)
#     print("fault num : {}".format(fault_num))
#     fault_nums.append(fault_num)

# print(np.argsort(y_predict)[:100])
# print(y_predict[:10])
# print(y_predict[-10:])
# print("predict:"+str(y_predict))
# print("type(y_predict):"+str(type(y_predict)))
