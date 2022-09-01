import os
import sys
import numpy as np
import datetime
from .img_mutation import *
from .datautils import data_proprecessing
import time
import pandas as pd
from .utils import *
import xgboost as xgb
from numpy import genfromtxt
import tensorflow as tf
import glob
import gc
import tensorflow.keras.backend as K
import shutil


def mutated_input_generation(dataset, x_candidate, y_candidate):
    print("preparing input mutants...")
    start = time.time()
    # making needed directory
    basedir = "/home/qhu/qhu-data/tem_data/"
    if not os.path.exists(os.path.join(basedir, 'tem_input')):
        os.mkdir(os.path.join(basedir, 'tem_input'))
    basedir = os.path.join(basedir, 'tem_input')
    basedir = os.path.join(basedir, dataset)
    if not os.path.exists(basedir):
        os.mkdir(basedir)
    basedir = os.path.join(basedir, dataset)
    if not os.path.exists(basedir):
        os.mkdir(basedir)
    if not os.path.exists(os.path.join(basedir, 'gauss')):
        os.mkdir(os.path.join(basedir, 'gauss'))
    if not os.path.exists(os.path.join(basedir, 'reverse')):
        os.mkdir(os.path.join(basedir, 'reverse'))
    if not os.path.exists(os.path.join(basedir, 'black')):
        os.mkdir(os.path.join(basedir, 'black'))
    if not os.path.exists(os.path.join(basedir, 'white')):
        os.mkdir(os.path.join(basedir, 'white'))
    if not os.path.exists(os.path.join(basedir, 'shuffle')):
        os.mkdir(os.path.join(basedir, 'shuffle'))
    perturbate_types = ['gauss', 'reverse', 'black', 'white', 'shuffle']
    # perturbate_types = ['gauss', 'reverse']
    # perturbate_types = ['gauss']
    if dataset == "mnist":
        shape_1, shape_2, shape_3 = 28, 28, 1
    else:
        shape_1, shape_2, shape_3 = 32, 32, 3
    for perturbate_type in perturbate_types:
        print("type: ", perturbate_type)
        image_id = 0
        for image in x_candidate:
            tt_temp = []
            for i in range(16):
                for j in range(16):
                    if perturbate_type == 'gauss':
                        tt = gauss_noise(image, i, j, ratio=1.0, var=0.01)
                    elif perturbate_type == 'white':
                        tt = white(image, i, j)
                    elif perturbate_type == 'black':
                        tt = black(image, i, j)
                    elif perturbate_type == 'reverse':
                        tt = reverse_color(image, i, j)
                    elif perturbate_type == 'shuffle':
                        tt = shuffle_pixel(image, i, j)
                    tt_temp.append(data_proprecessing(dataset)(tt))

            np.save(os.path.join(basedir, str(perturbate_type), str(image_id) + '.npy'),
                    np.array(tt_temp).reshape(-1, shape_1, shape_2, shape_3))
            image_id += 1
        print(time.time() - start)
        # print('finish generating...')


def accquire_prob(dataset, model, x_candidate, y_candidate):
    print("get prediction of input mutants...")
    start_ = time.time()
    basedir = "/home/qhu/qhu-data/tem_data/"
    basedir = os.path.join(basedir, 'tem_input')
    basedir = os.path.join(basedir, dataset)
    predicting_file_path = os.path.join(basedir, 'predict_probability_vector_' + str(dataset) + '.npy')
    X_test, Y_test = x_candidate, y_candidate
    X_test = data_proprecessing(dataset)(X_test)
    samples = len(X_test)
    origin_model = model
    if not os.path.exists(predicting_file_path):
        a = origin_model.predict(X_test)
        # a = np.argmax(a, axis=1)
        np.save(predicting_file_path, a)
        ori_prob = a
    else:
        ori_prob = np.load(predicting_file_path)
    perturbate_types = ['gauss', 'reverse', 'black', 'white', 'shuffle']
    # perturbate_types = ['gauss']
    for ptype in perturbate_types:
        print("type: ", ptype)
        file_name = 'image_perturbation_' + dataset + '_' + ptype
        file_name = os.path.join(basedir, file_name)
        result_recording_file = open(file_name + '.txt', 'w')
        origin_model_temp_result = ori_prob
        origin_model_result = np.argmax(origin_model_temp_result, axis=1)
        # print('origin_prediction:', origin_model_result)
        result_recording_file.write(str(origin_model_result))
        kill_num_dict = {}

        perturbate_image_path = os.path.join(basedir, dataset, ptype)
        nlp_tasks = ['imdb_bilstm', 'sst5_bilstm', 'trec_bilstm', 'spam_bilstm']
        form_tasks = ['kddcup99']
        if dataset in nlp_tasks + form_tasks:
            file_list = [os.path.join(perturbate_image_path, str(i) + '.csv') for i in range(samples)]
        else:
            file_list = [os.path.join(perturbate_image_path, str(i) + '.npy') for i in range(samples)]
        image_id = 0
        for file in file_list:
            start = datetime.datetime.now()
            if dataset in nlp_tasks:
                x_all = pd.read_csv(file, header=None, names=['review'])
                x_all = x_all.review
                x_all = data_proprecessing(dataset)(x_all)
            elif dataset in form_tasks:
                x_all = pd.read_csv(file).values
                x_all = data_proprecessing(dataset)(x_all)
            else:
                x_all = np.load(file)

            my_model = origin_model
            temp_result = my_model.predict(x_all)
            spath = str(dataset) + '_' + ptype + '_prob'
            if not os.path.exists(os.path.join(basedir, spath)):
                os.mkdir(os.path.join(basedir, spath))
            spath = os.path.join(basedir, spath)
            np.save(os.path.join(spath, str(image_id) + '.npy'), temp_result)
            result = np.argmax(temp_result, axis=1)
            kill_num = 0
            for r in result:
                if r != origin_model_result[image_id]:
                    kill_num += 1

            kill_num_dict.update({image_id: kill_num})
            # print('image_id:' + str(image_id))
            # print('kill_rate:', kill_num)
            result_recording_file.write('image_id:' + str(image_id))
            result_recording_file.write('\n')
            result_recording_file.write('kill_num:' + str(kill_num))
            result_recording_file.write('\n')
            image_id += 1

        d2 = sorted(kill_num_dict.items(), key=lambda x: x[1], reverse=True)
        kill_num_dict = {score: letter for score, letter in d2}
        import pickle
        dictfile = open(file_name + '.dict', 'wb')
        pickle.dump(kill_num_dict, dictfile)
        dictfile.close()
        result_recording_file.close()
        print(time.time() - start_)


def feature_extraction(dataset, model, x_candidate, y_candidate):
    print("feature extraction of input mutants...")
    basedir = "/home/qhu/qhu-data/tem_data/"
    basedir = os.path.join(basedir, 'tem_input')
    basedir = os.path.join(basedir, dataset)

    predicting_file_path = os.path.join(basedir, 'predict_probability_vector_' + str(dataset) + '.npy')
    X_test, Y_test = x_candidate, y_candidate
    X_test = data_proprecessing(dataset)(X_test)
    origin_model = model
    if not os.path.exists(predicting_file_path):
        a = origin_model.predict(X_test)
        # a = np.argmax(a, axis=1)
        np.save(predicting_file_path, a)
        ori_prob = a
    else:
        ori_prob = np.load(predicting_file_path)
    perturbate_types = ['gauss', 'reverse', 'black', 'white', 'shuffle']
    # perturbate_types = ['gauss']
    samples = len(x_candidate)
    for ptype in perturbate_types:
        print("type: ", ptype)
        file_name = dataset + '_' + ptype + '_feature'
        file_name = os.path.join(basedir, file_name)
        prob_path = dataset + '_' + ptype + '_prob'
        prob_path = os.path.join(basedir, prob_path)
        for i in range(0, samples):
            a = ori_prob[i]
            max_value = np.max(a)
            max_value_pos = np.argmax(a)
            file_path = os.path.join(prob_path, str(i) + '.npy')
            # if not os.path.exists(file_path):
            # continue
            perturbated_prediction = np.load(file_path)
            result_recording_file = open(file_name + '.txt', 'a+')
            euler = 0
            mahat = 0
            qube = 0
            cos = 0
            difference = 0
            different_class = []
            cos_list = []
            for pp in perturbated_prediction:
                pro = pp
                opro = a
                # if np.argmax(ii) != result[i]:
                difference += abs(max_value - pp[max_value_pos])
                euler += np.linalg.norm(pro - opro)
                mahat += np.linalg.norm(pro - opro, ord=1)
                qube += np.linalg.norm(pro - opro, ord=np.inf)
                co = (1 - (np.dot(pro, opro.T) / (np.linalg.norm(pro) * (np.linalg.norm(opro)))))
                if co < 0:
                    co = 0
                elif co > 1:
                    co = 1
                cos += co
                cos_list.append(co)
                if np.argmax(pp) != max_value_pos:
                    different_class.append(np.argmax(pp))
            cos_dis = cos_distribution(cos_list)
            dic = {}
            for key in different_class:
                dic[key] = dic.get(key, 0) + 1
            wrong_class_num = len(dic)
            if len(dic) > 0:
                max_class_num = max(dic.values())
            else:
                max_class_num = 0
            result_recording_file.write('image_id:' + str(i))
            result_recording_file.write('\n')
            result_recording_file.write('euler:' + str(euler))
            result_recording_file.write('\n')
            result_recording_file.write('mahat:' + str(mahat))
            result_recording_file.write('\n')
            result_recording_file.write('qube:' + str(qube))
            result_recording_file.write('\n')
            result_recording_file.write('cos:' + str(cos))
            result_recording_file.write('\n')
            result_recording_file.write('difference:' + str(difference))
            result_recording_file.write('\n')
            result_recording_file.write('wnum:' + str(wrong_class_num))
            result_recording_file.write('\n')
            result_recording_file.write('num_mc:' + str(max_class_num))
            result_recording_file.write('\n')
            result_recording_file.write('fenbu:' + str(cos_dis))
            result_recording_file.write('\n')
            result_recording_file.close()


def feature_csv_conclusion(dataset, ptypes, data_type, x_candidate, y_candidate, model_type):
    print("save results to csv...")
    if ptypes == 'tem_input':
        input_types = ['gauss', 'reverse']
        # input_types = ['gauss']
    else:
        # input_types = ['GF', 'NAI', 'NEB', 'WS']
        input_types = ['GF']
    all = []
    all_title = []
    basedir = "/home/qhu/qhu-data/tem_data/"
    basedir = os.path.join(basedir, ptypes)
    # basedir = ptypes
    basedir = os.path.join(basedir, dataset)
    sample = len(x_candidate)
    if dataset == "traffic":
        class_num = 43
    elif dataset == "cifar100":
        class_num = 100
    else:
        class_num = 10
    for mt in input_types:
        print("type: ", mt)
        types = ptypes + '_' + str(mt) + '_'
        f = open(os.path.join(basedir, dataset + '_' + mt + '_feature.txt'), 'r')
        if ptypes == 'tem_input':
            kill_rate_dict = read_kill_rate_dict(os.path.join(basedir, 'image_perturbation_' + dataset + '_' + mt))
        else:
            kill_rate_dict = read_kill_rate_dict(os.path.join(basedir, 'model_perturbation_' + dataset + '_' + mt))
        a = f.readlines()
        sh = []
        cos = []
        difference = []
        wrong_class_num = []
        max_class_num = []
        cos_distribution = []
        for i in a:
            if i[0] == 'c':
                x = float(i[i.find(':') + 1:-1].strip())
                cos.append(x)
            elif i[0] == 'd':
                x = float(i[i.find(':') + 1:-1].strip())
                difference.append(x)
            elif i[0] == 'n':
                x = int(i[i.find(':') + 1:-1].strip())
                max_class_num.append(x)
            elif i[0] == 'w':
                x = int(i[i.find(':') + 1:-1].strip())
                wrong_class_num.append(x)
            elif i[0] == 'f':
                x = eval(i[i.find(':') + 1:-1].strip())
                cos_distribution.append(x)

        kill_num_list = []
        for i in range(sample):
            kill_num_list.append(kill_rate_dict[i])

        all_vector = []
        all_vector.append(kill_num_list)
        all_vector.append(cos)
        all_vector.append(difference)
        all_vector.append(max_class_num)
        all_vector.append(wrong_class_num)
        cd = list(np.asarray(cos_distribution).T)
        # print(len(cd))
        for i in range(10):
            all_vector.append(cd[i].tolist())

        title = [str(types) + 'kill_num', str(types) + 'cos',
                 str(types) + 'difference',
                 str(types) + 'max_class_num',
                 str(types) + 'wrong_class_num']
        title.extend([str(types) + 'cos_distribution' + str(i) for i in range(10)])

        all_title.extend(title)
        all.extend(all_vector)
    # print(all_title)
    # print(len(all_vector[1]))
    # print(len(all))

    # print(all_vector[1])
    # print(np.asarray(all).T.shape)
    test_me = all

    for i in range(len(all_title)):
        test_me[i] = test_me[i][:sample]
        # print(len(test_me[i]))
    # print(test_me.shape)
    # test_me = test_me.T
    # print(test_me.shape)
    test_me = np.asarray(test_me).T
    # print(test_me.shape)
    pd_data_all = pd.DataFrame(test_me, columns=all_title)
    # if ptypes == 'tem_input':
    #     pd_data_all = pd.DataFrame(test_me, columns=all_title)
    # else:
    #     # save_data = np.asarray(all).T
    #     # pd_data_all = pd.DataFrame([save_data.reshape(save_data.shape[-2], save_data.shape[-1])], columns=all_title)
    #     pd_data_all = pd.DataFrame([np.asarray(all).T], columns=all_title)
    X_test, Y_test = x_candidate, y_candidate
    y_predict_prob = np.load(os.path.join(basedir, 'predict_probability_vector_' + str(dataset) + '.npy'))
    y_predict = np.argmax(y_predict_prob, axis=1)
    right_or_wrong = []
    for i in range(sample):
        if Y_test[i] != y_predict[i]:
            right_or_wrong.append(0)
        else:
            right_or_wrong.append(1)
    rightness_pd = pd.DataFrame(np.array(right_or_wrong), columns=['rightness'])
    result = pd.concat([pd_data_all, rightness_pd], axis=1)
    result.to_csv(ptypes + '_' + dataset + '_' + data_type + '_' + model_type + '_feature.csv')


def model_prioritization(dataset, model_save_path, model, x_candidate, y_candidate):
    print("model prioritization...")
    start = time.time()
    """Parser of command args"""
    # ptypes = ['GF', 'NEB', 'NAI', 'WS']
    ptypes = ['GF']
    for ptype in ptypes:
        sample = len(x_candidate)
        file_name = 'model_perturbation_' + str(dataset) + '_' + str(ptype)
        basedir = "/home/qhu/qhu-data/tem_data/"
        basedir = os.path.join(basedir, 'tem_model')
        if not os.path.exists(basedir):
            os.mkdir(basedir)
        basedir = os.path.join(basedir, dataset)
        if not os.path.exists(basedir):
            os.mkdir(basedir)
        file_name = os.path.join(basedir, file_name)
        kill_num_dict = {i: 0 for i in range(int(float(sample)))}
        save_dict(dictionary=kill_num_dict, filename=file_name)
        # model_save_path = "model/mutant/GF_0.1_"
        file_list = [model_save_path + str(i) + '.h5' for i in range(1, 51)]
        X_test, Y_test = x_candidate, y_candidate
        X_test = data_proprecessing(dataset)(X_test)
        origin_model = model
        file_id = 0
        for file in file_list:
            # print(file_name,file)
            basedir = "/home/qhu/qhu-data/tem_data/"
            basedir = os.path.join(basedir, 'tem_model')
            basedir = os.path.join(basedir, dataset)

            predicting_file_path = os.path.join(basedir, 'predict_probability_vector_' + str(dataset) + '.npy')

            if not os.path.exists(predicting_file_path):
                a = origin_model.predict(X_test)
                np.save(predicting_file_path, a)
                origin_model_result = a
            else:
                origin_model_result = np.load(predicting_file_path)

            origin_model_result = np.argmax(origin_model_result, axis=1)

            kill_num_dict = load_dict(file_name)
            result_recording_file = open(file_name + '.txt', 'a')
            start = datetime.datetime.now()
            my_model = tf.keras.models.load_model(file)
            print('file:', file)
            result_recording_file.write(str('file:' + str(file)))
            result_recording_file.write('\n')
            temp_result = my_model.predict(X_test)
            new_name = str(file_id)
            savepath = os.path.join(basedir, dataset + '_temp_result_' + ptype)
            if not os.path.exists(savepath):
                os.mkdir(savepath)
            np.save(savepath + '/' + new_name + '.npy', temp_result)
            result = np.argmax(temp_result, axis=1)
            wrong_predict = []
            for count in range(len(X_test)):
                if result[count] != origin_model_result[count]:
                    kill_num_dict[count] += 1
                    wrong_predict.append(count)

            result_recording_file.write('diff_num:' + str(len(wrong_predict)))
            result_recording_file.write('\n')
            result_recording_file.write('different_pred:' + str(wrong_predict))
            result_recording_file.write('\n')

            elapsed = (datetime.datetime.now() - start)
            print("Time used: ", elapsed)
            result_recording_file.close()
            save_dict(dictionary=kill_num_dict, filename=file_name)
            # os.system("python /home/qhu/projects/TSattack/metrics/PRIMA/core_prioritization_unit.py %s %s %s %s %s" % (file_name, file, str(file_id), dataset, ptype))
            file_id += 1
            K.clear_session()
            del my_model
            gc.collect()
        # print(time.time() - start)


def model_feature_extraction(dataset, model, x_candidate, y_candidate):
    print("model feature extraction...")
    sample = len(x_candidate)

    basedir = "/home/qhu/qhu-data/tem_data/"
    basedir = os.path.join(basedir, 'tem_model')
    basedir = os.path.join(basedir, dataset)

    predicting_file_path = os.path.join(basedir, 'predict_probability_vector_' + str(dataset) + '.npy')
    X_test, Y_test = x_candidate, y_candidate
    origin_model = model
    X_test = data_proprecessing(dataset)(X_test)
    if not os.path.exists(predicting_file_path):
        a = origin_model.predict(X_test)
        np.save(predicting_file_path, a)
        ori_prob = a
    else:
        ori_prob = np.load(predicting_file_path)

    # ptypes = ['GF', 'NEB', 'NAI', 'WS']
    ptypes = ['GF']
    for ptype in ptypes:
        print("type: ", ptype)
        file_name = 'image_perturbation_' + dataset + '_' + ptype
        result = np.argmax(ori_prob, axis=1)

        samples = int(float(sample))
        file_name = dataset + '_' + ptype + '_feature'
        file_name = os.path.join(basedir, file_name)
        euler = [0 for i in range(samples)]
        mahat = [0 for i in range(samples)]
        qube = [0 for i in range(samples)]
        cos = [0 for i in range(samples)]
        difference = [0 for i in range(samples)]
        cos_list = [[] for i in range(samples)]
        different_class = [[] for i in range(samples)]

        for i in range(0, 50):
            file_path = dataset + '_temp_result_' + ptype + '/' + str(i) + '.npy'
            file_path = os.path.join(basedir, file_path)

            perturbated_prediction = np.load(file_path)
            for ii in range(samples):
                pro = perturbated_prediction[ii]
                opro = ori_prob[ii]
                max_value_pos = np.argmax(opro)
                max_value = np.max(opro)
                difference[ii] += abs(max_value - pro[max_value_pos])
                euler[ii] += np.linalg.norm(pro - opro)
                mahat[ii] += np.linalg.norm(pro - opro, ord=1)
                qube[ii] += np.linalg.norm(pro - opro, ord=np.inf)
                co = (1 - (np.dot(pro, opro.T) / (np.linalg.norm(pro) * (np.linalg.norm(opro)))))
                if co < 0:
                    co = 0
                elif co > 1:
                    co = 1
                cos[ii] += co
                cos_list[ii].append(co)

                if np.argmax(pro) != max_value_pos:
                    different_class[ii].append(np.argmax(pro))

        result_recording_file = open(file_name + '.txt', 'a+')
        for i in range(samples):
            dic = {}
            for key in different_class[i]:
                dic[key] = dic.get(key, 0) + 1
            wrong_class_num = len(dic)
            if len(dic) > 0:
                max_class_num = max(dic.values())
            else:
                max_class_num = 0
            cos_dis = cos_distribution(cos_list[i])
            # print('id:', i)
            result_recording_file.write('image_id:' + str(i))
            result_recording_file.write('\n')
            result_recording_file.write('euler:' + str(euler[i]))
            result_recording_file.write('\n')
            result_recording_file.write('mahat:' + str(mahat[i]))
            result_recording_file.write('\n')
            result_recording_file.write('qube:' + str(qube[i]))
            result_recording_file.write('\n')
            result_recording_file.write('cos:' + str(cos[i]))
            result_recording_file.write('\n')
            result_recording_file.write('difference:' + str(difference[i]))
            result_recording_file.write('\n')
            result_recording_file.write('wnum:' + str(wrong_class_num))
            result_recording_file.write('\n')
            result_recording_file.write('num_mc:' + str(max_class_num))
            result_recording_file.write('\n')
            result_recording_file.write('fenbu:' + str(cos_dis))
            result_recording_file.write('\n')
        result_recording_file.close()


def prima_pri(dl_model, dataset, x_candidate, y_candidate, mutant_path, budgets, data_type, model_type):
    # input mutants generation
    mutated_input_generation(dataset, x_candidate, y_candidate)
    # get prediction of input mutants
    accquire_prob(dataset, dl_model, x_candidate, y_candidate)
    # feature extraction of input mutants
    feature_extraction(dataset, dl_model, x_candidate, y_candidate)
    feature_csv_conclusion(dataset, "tem_input", data_type, x_candidate, y_candidate, model_type)
    # print(aaaa)
    # feature extraction of model mutants
    model_prioritization(dataset, mutant_path, dl_model, x_candidate, y_candidate)
    model_feature_extraction(dataset, dl_model, x_candidate, y_candidate)
    feature_csv_conclusion(dataset, "tem_model", data_type, x_candidate, y_candidate, model_type)
    # train learning-to-rank model
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
    x_train = genfromtxt("/home/qhu/projects/TSattack/tem_model_" + dataset + '_' + data_type + '_' + model_type + "_feature.csv", delimiter=',')
    x_train = x_train[:, 1:]
    x_train = x_train[1:, :]
    x_train = x_train / x_train.max(axis=0)

    x_train_2 = genfromtxt("/home/qhu/projects/TSattack/tem_input_" + dataset + '_' + data_type + '_' + model_type + "_feature.csv", delimiter=',')
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
    print(len(predicted_labels))
    print(X.shape)
    print(y.shape)
    model.load_model("/home/qhu/projects/TSattack/metrics/PRIMA/models/" + dataset + "_" + model_type + ".json")
    # model.fit(X, y, group=[num_per_group, len(x_candidate) - num_per_group], verbose=True)
    # model.save_model("/home/qhu/projects/TSattack/metrics/PRIMA/models/" + dataset + "_" + model_type + ".json")
    y_predict = model.predict(X)
    sorted_index = np.argsort(y_predict)
    results = []
    for i in budgets:
        s = sorted_index[:i]
        results.append(len(np.where(s >= correct_num)[0]))
    #     delete temporary folders
    basedir = "/home/qhu/qhu-data/tem_data/"
    basedir = os.path.join(basedir, 'tem_model')
    shutil.rmtree(basedir)

    basedir = "/home/qhu/qhu-data/tem_data/"
    basedir = os.path.join(basedir, 'tem_input')
    shutil.rmtree(basedir)
    # os.remove("/home/qhu/projects/TSattack/tem_model_" + dataset + '_' + data_type + '_' + model_type + "_feature.csv")
    # os.remove("/home/qhu/projects/TSattack/tem_input_" + dataset + '_' + data_type + '_' + model_type + "_feature.csv")
    print("finished!!!!")
    return results


def prima_pri_index(dl_model, dataset, x_candidate, y_candidate, mutant_path, budgets, data_type, model_type):
    # input mutants generation
    mutated_input_generation(dataset, x_candidate, y_candidate)
    # get prediction of input mutants
    accquire_prob(dataset, dl_model, x_candidate, y_candidate)
    # feature extraction of input mutants
    feature_extraction(dataset, dl_model, x_candidate, y_candidate)
    feature_csv_conclusion(dataset, "tem_input", data_type, x_candidate, y_candidate, model_type)
    # print(aaaa)
    # feature extraction of model mutants
    model_prioritization(dataset, mutant_path, dl_model, x_candidate, y_candidate)
    model_feature_extraction(dataset, dl_model, x_candidate, y_candidate)
    feature_csv_conclusion(dataset, "tem_model", data_type, x_candidate, y_candidate, model_type)
    # train learning-to-rank model
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
    x_train = genfromtxt("/home/qhu/projects/TSattack/tem_model_" + dataset + '_' + data_type + '_' + model_type + "_feature.csv", delimiter=',')
    x_train = x_train[:, 1:]
    x_train = x_train[1:, :]
    x_train = x_train / x_train.max(axis=0)

    x_train_2 = genfromtxt("/home/qhu/projects/TSattack/tem_input_" + dataset + '_' + data_type + '_' + model_type + "_feature.csv", delimiter=',')
    x_train_2 = x_train_2[:, 1:]
    x_train_2 = x_train_2[1:, :]
    x_train_2 = x_train_2 / x_train_2.max(axis=0)
    x_train = np.concatenate((x_train, x_train_2), axis=1)
    predicted_labels = dl_model.predict(x_candidate).argmax(axis=1)
    corrected_index = np.where(predicted_labels == y_candidate)[0]
    wrong_index = np.where(predicted_labels != y_candidate)[0]
    predicted_labels[corrected_index] = 1
    predicted_labels[wrong_index] = 0
    total_index = np.append(corrected_index, wrong_index)
    X_features = np.concatenate((x_train[corrected_index], x_train[wrong_index]))
    y = np.concatenate((predicted_labels[corrected_index], predicted_labels[wrong_index]))
    X = np.nan_to_num(X_features)
    num_per_group = len(x_candidate) // 2
    model.load_model("/home/qhu/projects/TSattack/metrics/PRIMA/models/" + dataset + "_" + model_type + ".json")
    # model.fit(X, y, group=[num_per_group, len(x_candidate) - num_per_group], verbose=True)
    y_predict = model.predict(X)
    sorted_index = np.argsort(y_predict)
    results = []
    faults = []
    correct_num = len(corrected_index)
    for i in budgets:
        s = sorted_index[:i]
        results.append(total_index[s])
        faults.append(len(np.where(s >= correct_num)[0]))
    #     delete temporary folders
    basedir = "/home/qhu/qhu-data/tem_data/"
    basedir = os.path.join(basedir, 'tem_model')
    shutil.rmtree(basedir)

    basedir = "/home/qhu/qhu-data/tem_data/"
    basedir = os.path.join(basedir, 'tem_input')
    shutil.rmtree(basedir)
    os.remove("/home/qhu/projects/TSattack/tem_model_" + dataset + '_' + data_type + '_' + model_type + "_feature.csv")
    os.remove("/home/qhu/projects/TSattack/tem_input_" + dataset + '_' + data_type + '_' + model_type + "_feature.csv")
    print("finished!!!!")
    return faults, results
