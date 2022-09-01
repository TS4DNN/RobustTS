import numpy as np
from scipy import stats


def MC_selection(dropout_model, target_data, select_size):
    BALD_list = []
    mode_list = []
    data_len = len(target_data)
    print("Prepare...")
    for _ in range(20):
        prediction = np.argmax(dropout_model.predict(target_data), axis=1)
        BALD_list.append(prediction)
    BALD_list = np.asarray(BALD_list)
    for _ in range(data_len):
        mode_num = stats.mode(BALD_list[:, _:(_ + 1), ].reshape(-1,))[1][0]

        mode_list.append(1 - mode_num / 50)
    sorted_index = np.argsort(mode_list)
    select_index = sorted_index[-(select_size):]
    # selected_data = target_data[select_index]
    # selected_label = target_label[select_index]
    # print("target len, ", len(target_data))
    # print("select size, ", select_size)
    # remain_data = np.delete(target_data, select_index, axis=0)
    # remain_label = np.delete(target_label, select_index, axis=0)
    return select_index
