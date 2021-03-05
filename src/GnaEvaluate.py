# -*- coding: UTF-8 -*-

from scipy.optimize import linear_sum_assignment as linear_assignment
from sklearn.cluster import KMeans
from sklearn.metrics import f1_score, normalized_mutual_info_score, adjusted_rand_score
import numpy as np
import pandas as pd


def reassign_cluster(y_pred, index):
    y_ = np.zeros_like(y_pred)
    for i in range(y_pred.size):
        for j in range(index[1].size):
            if y_pred[i] == index[0][j]:
                y_[i] = index[1][j]
    return y_


# not used
def reassign_cluster_with_ref(Y_pred, Y):
    """
    Reassign cluster to reference labels
    Inputs:
        Y_pred: predict y classes
        Y: true y classes
    Return:
        f1_score: clustering f1 score
        y_pred: reassignment index predict y classes
        indices: classes assignment
    """
    assert Y_pred.size == Y.size
    D = max(Y_pred.max(), Y.max()) + 1
    w = np.zeros((D, D), dtype=np.int64)
    for i in range(Y_pred.size):
        w[Y_pred[i], Y[i]] += 1
    ind = linear_assignment(w.max() - w)

    return reassign_cluster(Y_pred, ind)


# 评估
def evaluate_score():
    # 每行作为一个细胞数据
    matrix_file = 'bias_output/scIGANs-data.txt-data.label.txt.csv'
    label_file = '../dataset/data.label.txt'
    data_set = pd.read_csv(matrix_file, header=0, index_col=0)
    data_frames = pd.DataFrame(data_set)

    np_array = np.array(data_frames.values)
    kmeans = KMeans(n_clusters=4, init='k-means++', algorithm='elkan', random_state=0).fit(np_array)
    # print('[KMeans label] len:', len(kmeans.labels_), ' iters:', kmeans.n_iter_, ' content:', kmeans.labels_)

    # 获取 label 并转化为 数字 表示
    label_set = pd.read_table(label_file, header=None, index_col=False)
    src_label = pd.Categorical(label_set.iloc[:, 1]).codes

    print('pre_label:', kmeans.labels_, '\nsrc_label:', src_label)

    # 两个 label 均是数据即可, 不需要一一对应 ?
    pre_label = kmeans.labels_

    # adjust score;
    score_a = adjusted_rand_score(src_label, pre_label)
    print(score_a)

    # normalized score
    score_b = normalized_mutual_info_score(src_label, pre_label)
    print(score_b)


if __name__ == '__main__':
    evaluate_score()
