from liblinearutil import *
from statistics import mean

import data_process
import config
import logging
import os
import multiprocessing
import pandas as pd
import numpy as np

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s [%(levelname)s] <%(processName)s> (%(threadName)s) %(message)s')
logger = logging.getLogger(__name__)
# with open('liblinear.log', 'w') as f:
#     pass
# f_handler = logging.FileHandler('liblinear.log')
# f_handler.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] <%(processName)s> (%(threadName)s) %(message)s"))
# logger.addHandler(f_handler)

columns = config.columns
class_types = [config.svm_01_name, config.svm_3_name, config.svm_4_name]


def get_macro_f1_score_values(class_type, actual_label, predict_label, precision_v_01=None, recall_v_01=None, f1_score_v_01=None):
    actual_label = pd.Series(actual_label)
    predict_label = pd.Series(predict_label)
    # marco 方式计算
    pl = []
    rl = []
    fl = []

    if precision_v_01 is not None:
        pl.append(precision_v_01)
    if recall_v_01 is not None:
        rl.append(recall_v_01)
    if f1_score_v_01 is not None:
        fl.append(f1_score_v_01)
    for label in np.arange(min(actual_label), max(actual_label) + 1):
        tp, fp, fn, precision, recall, f1_score = data_process.get_ml_score_values(actual_label == label,
                                                                                   predict_label == label)
        logger.info("Label %d with recall %f, precision %f, f1_score: %f", label, recall, precision, f1_score)
        if precision + recall == 0:
            logger.info("Label %d with zero f1_score!", label)
        # 特殊处理 2/3 分类
        if label == 0 and class_type == config.svm_01_name:
            return precision, recall, f1_score
        pl.append(precision)
        rl.append(recall)
        fl.append(f1_score)
    return mean(pl), mean(rl), mean(fl)


def process_column(column, f1_map, mkey='', local=False, best_param=''):
    logger.info("[%s] Processing subset..." % column)
    acc_df = pd.DataFrame(
        columns=['subset', 'class_type', 'param', 'V_f1score', 'V_precision', 'V_recall', 'V_acc',
                 'V_MSE', 'V_SCC',
                 'f1_score_list'])
    p_labels_t_01 = None
    precision_v_01 = None
    recall_v_01 = None
    f1_score_v_01 = None
    for class_type in class_types:
        logger.info("[%s] class: %s" % (column, class_type))
        # 交叉验证选参
        train_label, train_value = svm_read_problem(
            config.svm_dataset_path + mkey + "/" + class_type + "/train/" + column)
        validation_label, validation_value = svm_read_problem(
            config.svm_dataset_path + mkey + "/" + class_type + "/validation/" + column)
        test_label, test_value = svm_read_problem(
            config.svm_dataset_path + mkey + "/" + class_type + "/test/" + column)
        pram = '-q -e 0.001 -s %d -c %f'
        best_model = None
        best_f1_score = 0.0
        f1_score_list = []
        if not local:
            logger.info('[%s] - [%s] Running CV optimize with %s...' % (column, class_type, class_type))
            # 安静模式自动优化
            # pram = '-e 0.001 -s %d -c %f'
            for s in [2, 1]:
                for c in [0.0001, 0.001, 0.002, 0.003, 0.1, 0.5, 1]:
                    curr_parm = pram % (s, c)
                    logger.info('[%s] - [%s] Params: %s...' % (column, class_type, curr_parm))
                    model = train(train_label, train_value, curr_parm)
                    p_labels, p_acc, p_vals = predict(validation_label, validation_value, model, '-q')
                    acc, mse, scc = evaluations(validation_label, p_labels)
                    precision, recall, f1_score = get_macro_f1_score_values(class_type, validation_label, p_labels)
                    logger.info(
                        '[%s] - [%s] Optimize with param %s, Result with f1_score: %.5f, precision:%.5f, '
                        'recall: %.5f, ACC:%.5f%%, MSE:%.5f, SCC:%.5f.' % (
                            column, class_type, curr_parm, f1_score,
                            precision, recall, acc, mse, scc))
                    f1_score_list.append(f1_score)
                    if f1_score > best_f1_score:
                        best_f1_score = f1_score
                        best_param = curr_parm
                        best_model = model
            logger.info(
                '[%s] - [%s] Done Optimize, best macro f1_score: %.5f, with Params: %s.' % (
                    column, class_type, best_f1_score, best_param))

        # 模型训练
        logger.info('[%s] - [%s] Training...' % (column, class_type))
        if best_model:
            model = best_model
        else:
            model = train(train_label, train_value, best_param)
        save_model(config.predict_result_path + mkey + '/model/[%s]%s' % (column, class_type), model)
        logger.info('[%s] - [%s] Done training n-class: %d...' % (column, class_type, model.nr_class))

        # 验证集指标
        logger.info('[%s] - [%s] Validating...' % (column, class_type))
        p_labels, p_acc, p_vals = predict(validation_label, validation_value, model, '-q')
        acc, mse, scc = evaluations(validation_label, p_labels)
        # 特殊处理 2-3 分类
        if class_type == config.svm_3_name:
            precision, recall, f1_score = get_macro_f1_score_values(class_type, validation_label, p_labels, precision_v_01,
                                                                    recall_v_01, f1_score_v_01)
        else:
            precision, recall, f1_score = get_macro_f1_score_values(class_type, validation_label, p_labels)
        # 特殊处理 2-3 分类
        if class_type == config.svm_01_name:
            precision_v_01 = precision
            recall_v_01 = recall
            f1_score_v_01 = f1_score
        f1_map[class_type].append(f1_score)
        logger.info(
            '[%s] - [%s] Done validation with f1_score: %.5f, precision:%.5f, '
            'recall: %.5f, ACC:%.5f%%, MSE:%.5f, SCC:%.5f.' % (
                column, class_type, f1_score,
                precision, recall, acc, mse, scc))

        # 结果生成
        logger.info('[%s] - [%s] Testing...' % (column, class_type))
        p_labels, p_acc, p_vals = predict(test_label, test_value, model, '-q')
        # 特殊处理 2-3 分类
        if class_type == config.svm_01_name:
            p_labels_t_01 = p_labels
        elif class_type == config.svm_3_name:
            for i in range(len(p_labels)):
                if p_labels_t_01[i] == 0:
                    p_labels[i] = -2
        test_df = pd.DataFrame(columns=['y'])
        test_df['y'] = p_labels
        save_path = config.predict_result_path + mkey + '/' + class_type + '/'
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        test_df.to_csv(save_path + column + '.csv', index=False,
                       sep=',',
                       encoding='utf_8_sig')
        logger.info('[%s] - [%s] Done test...' % (column, class_type))
        if not local:
            # 性能指标存档
            acc_df.loc[len(acc_df)] = [column, class_type, best_param,
                                       f1_score, precision, recall, acc, mse, scc,
                                       f1_score_list]
            save_path = config.predict_result_path + mkey + '/indicate/'
            if not os.path.exists(save_path):
                os.makedirs(save_path)
            acc_df.to_csv(save_path + column + '.csv', index=False, sep=',',
                          encoding='utf_8_sig')
    logger.info('[%s] Done subset.' % column)
    logger.info("")


def svm_predict_test(mkey='default', local=True):
    manager = multiprocessing.Manager()
    f1_map = manager.dict()
    f1_map[config.svm_01_name] = manager.list()
    f1_map[config.svm_3_name] = manager.list()
    if local:
        for column in columns:
            process_column(column, f1_map, mkey, local, '-q -e 0.001 -s 2 -c 0.001')
    else:
        jobs = []
        for column in columns:
            p = multiprocessing.Process(target=process_column,
                                        args=(column, f1_map, mkey, local))
            jobs.append(p)
            p.start()

        for j in jobs:
            j.join()
    logger.info(list(f1_map[config.svm_3_name]))
    logger.info(mean(f1_map[config.svm_3_name]))
    logger.info("*Done svm predict test")
