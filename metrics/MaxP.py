import numpy as np


def max_p_rank(prediction):
    metrics = np.max(prediction, axis=1)
    max_p_rank = np.argsort(metrics)  # 有小到大
    return max_p_rank
