import os
import numpy as np
from tensorflow.keras.models import Model
import hdbscan
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import FastICA
from .mmd_critic.run_digits_new import run
import math
import tensorflow as tf


def get_score(x_test, y_test, model):
    #计算准确率
    score = model.evaluate(x_test, y_test, verbose=0)

    print('Test loss:', score[0])
    print('Test accuracy:', score[1])
    print('预测错的数目：', len(x_test)*(1-score[1]))
    return score


def get_ds(countlist, res_get_s, sample_size, X_test, res, selected_index):
    #在非异常点中采样
    len_nonnoise = len(X_test) - countlist[0]
    for key in res:
        b = []
        if len(res[key]) > (len_nonnoise/sample_size):
            #mmd方法采样
            for num in range(int(round(len(res[key]) / (len_nonnoise/sample_size)))):
                b.append(res[key][res_get_s[key][num]])
        else:
            b.append(res[key][res_get_s[key][0]])

        for i in range(len(b)):
            # X_test2.append(X_test[b[i]])
            # Y_test2.append(Y_test[b[i]])
            selected_index.append(b[i])


def get_std1(X_test, a_unoise, countlist, res, label_noise, first_noise, res_get_s, dis, select_size):
    #存储所有标准差
    selected_index = []
    for j in select_size:

        #选择的异常点数目
        len_noise = j*(1-a_unoise)
        #adaptive random
        selected_index.append(label_noise[first_noise])
        pre_num = []
        pre_num.append(first_noise)
        pre_num.append(np.argmax(dis[first_noise]))
        while len(selected_index) < len_noise:
            mins = []
            for i in range(len(label_noise)):
                if i not in set(pre_num):
                    min_info = [float('inf'), 0, 0]
                    for l in pre_num:
                        if dis[i][l] < min_info[0]:
                            min_info[0] = dis[i][l]
                            min_info[1] = i
                            min_info[2] = l
                    mins.append(min_info)
            maxnum = 0

            selected_index.append(label_noise[mins[0][1]])
            pre_num.append(mins[0][1])
            for i in mins:
                if i[0] > maxnum:
                    selected_index[-1] = label_noise[i[1]]
                    pre_num[-1] = i[1]
                    # pre_num.append(i[1])

        # print("异常点挑选个数：", len(selected_index))
        # print(selected_index)
        get_ds(countlist, res_get_s, j*a_unoise, X_test, res, selected_index)

    return selected_index


def PACE_selection(model, candidate_data, select_size, selected_layer):
    basedir = os.path.abspath(os.path.dirname(__file__))
    select_layer_idx = selected_layer
    dec_dim = 8
    min_cluster_size = 80
    min_samples = 4
    dense_layer_model = Model(inputs=model.input, outputs=model.layers[select_layer_idx].output)
    dense_output = dense_layer_model.predict(candidate_data)
    minMax = MinMaxScaler()
    dense_output = minMax.fit_transform(dense_output)
    fica = FastICA(n_components=dec_dim)
    dense_output = fica.fit_transform(dense_output)
    clusterer = hdbscan.HDBSCAN(min_cluster_size=min_cluster_size, min_samples=min_samples, gen_min_span_tree=True)
    r = clusterer.fit(dense_output)
    labels = r.labels_
    y_pred_list = labels.tolist()
    countlist = []

    for i in range(np.min(labels), np.max(labels) + 1):
        countlist.append(y_pred_list.count(i))

    label_noise = []
    for i, l in enumerate(labels):
        if l == -1:
            label_noise.append(i)

    res = {}
    for i, l in enumerate(labels):
        if l != -1:
            if l not in res:
                res[l] = []
            res[l].append(i)

    # 计算异常点每两点之间的距离
    dis = np.zeros((len(label_noise), len(label_noise)))
    for i in range(len(label_noise)):
        for j in range(len(label_noise)):
            if j != i:
                dis[i][j] = math.sqrt(np.power(dense_output[label_noise[i]] - dense_output[label_noise[j]], 2).sum())

    noise_score = []
    # print(r.outlier_scores_)
    for i, l in enumerate(r.outlier_scores_):
        if labels[i] == -1:
            noise_score.append(l)
    noise_score = np.array(noise_score)
    first_noise = np.argsort(-noise_score)[0]
    # print(first_noise)
    # print(noise_score[first_noise])
    # 非异常点每一类的排序，key类别号
    res_get_s = {}
    for key in res:
        temp_dense = []
        for l in res[key]:
            temp_dense.append(dense_output[l])
        temp_dense = np.array(temp_dense)
        temp_label = np.full((len(temp_dense)), key)
        mmd_res, _ = run(temp_dense, temp_label, gamma=0.026, m=min(len(temp_dense), 1000), k=0, ktype=0, outfig=None,
                         critoutfig=None, testfile=os.path.join(basedir, 'mmd_critic/data/a.txt'))
        res_get_s[key] = mmd_res
    # print(res)
    # print("#########")
    # print(res_get_s)
    select_size = [select_size]
    a_unoise = 0.6
    selected_index = get_std1(X_test=candidate_data, a_unoise=a_unoise, countlist=countlist, res=res,
                              label_noise=label_noise, first_noise=first_noise, res_get_s=res_get_s, dis=dis,
                              select_size=select_size)
    return selected_index


def PACE_selection_attack(model, candidate_data, select_size, selected_layer):
    basedir = os.path.abspath(os.path.dirname(__file__))
    select_layer_idx = selected_layer
    dec_dim = 8
    min_cluster_size = 80
    min_samples = 4
    dense_layer_model = Model(inputs=model.input, outputs=model.layers[select_layer_idx].output)
    dense_output = dense_layer_model.predict(candidate_data)
    attacked_dense_output = np.maximum(dense_output) * 1.5

    minMax = MinMaxScaler()
    dense_output = minMax.fit_transform(dense_output)
    fica = FastICA(n_components=dec_dim)
    dense_output = fica.fit_transform(dense_output)
    clusterer = hdbscan.HDBSCAN(min_cluster_size=min_cluster_size, min_samples=min_samples, gen_min_span_tree=True)
    r = clusterer.fit(dense_output)
    labels = r.labels_
    y_pred_list = labels.tolist()
    countlist = []

    for i in range(np.min(labels), np.max(labels) + 1):
        countlist.append(y_pred_list.count(i))

    label_noise = []
    for i, l in enumerate(labels):
        if l == -1:
            label_noise.append(i)

    res = {}
    for i, l in enumerate(labels):
        if l != -1:
            if l not in res:
                res[l] = []
            res[l].append(i)

    # 计算异常点每两点之间的距离
    dis = np.zeros((len(label_noise), len(label_noise)))
    for i in range(len(label_noise)):
        for j in range(len(label_noise)):
            if j != i:
                dis[i][j] = math.sqrt(np.power(dense_output[label_noise[i]] - dense_output[label_noise[j]], 2).sum())

    noise_score = []
    # print(r.outlier_scores_)
    for i, l in enumerate(r.outlier_scores_):
        if labels[i] == -1:
            noise_score.append(l)
    noise_score = np.array(noise_score)
    first_noise = np.argsort(-noise_score)[0]
    # print(first_noise)
    # print(noise_score[first_noise])
    # 非异常点每一类的排序，key类别号
    res_get_s = {}
    for key in res:
        temp_dense = []
        for l in res[key]:
            temp_dense.append(dense_output[l])
        temp_dense = np.array(temp_dense)
        temp_label = np.full((len(temp_dense)), key)
        mmd_res, _ = run(temp_dense, temp_label, gamma=0.026, m=min(len(temp_dense), 1000), k=0, ktype=0, outfig=None,
                         critoutfig=None, testfile=os.path.join(basedir, 'mmd_critic/data/a.txt'))
        res_get_s[key] = mmd_res
    # print(res)
    # print("#########")
    # print(res_get_s)
    select_size = [select_size]
    a_unoise = 0.6
    selected_index = get_std1(X_test=candidate_data, a_unoise=a_unoise, countlist=countlist, res=res,
                              label_noise=label_noise, first_noise=first_noise, res_get_s=res_get_s, dis=dis,
                              select_size=select_size)
    return selected_index


if __name__ == '__main__':
    x_test = np.load("datasets/mnist/RT_test_x.npy")
    y_test = np.load("datasets/mnist/RT_test_y.npy")
    x_test = x_test.reshape(-1, 28, 28, 1)
    x_test = x_test.astype("float32") / 255
    model = tf.keras.models.load_model("models/mnist/lenet5.h5")
    selected_index = PACE_selection(model, x_test, 50)
    print(selected_index)

