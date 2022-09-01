import numpy as np


def deep_metric(pred_test_prob):
    metrics = np.sum(pred_test_prob ** 2, axis=1)  # 值越小,1-值就越大,因此值越小越好
    rank_lst = np.argsort(metrics)  # 按照值从小到大排序,因此序号越小代表值越小代表越好
    return rank_lst

