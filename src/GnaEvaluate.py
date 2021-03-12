# -*- coding: UTF-8 -*-

from scipy.optimize import linear_sum_assignment as linear_assignment
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE
from umap import UMAP
from sklearn.metrics import f1_score, normalized_mutual_info_score, adjusted_rand_score
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import scanpy as sc
import anndata as ad
import argparse
import os

os.environ['MPLCONFIGDIR'] = "../graph/"

parser = argparse.ArgumentParser()
parser.add_argument('--matrix_file', type=str, default='', help='path of gene matrix file')
parser.add_argument('--label_file', type=str, default='', help='path of source label file')
parser.add_argument('--cluster_fig_path', type=str, default='./my_cluster.png', help='path of plt picture')
parser.add_argument('--need_transpose', type=bool, default=False, help='the gene matrix file need to be transposed')
parser.add_argument('--n_clusters', type=int, default=4, help='the number of cell clusters')
parser.add_argument('--label_index', type=int, default=0, help='for label column indexing')
parser.add_argument('--skip_label_first', type=bool, default=False, help='skip label file s first row')
opt = parser.parse_args()


def read_file():
    global data_set, src_label
    # gene data
    if opt.need_transpose:
        data_set = pd.read_table(opt.matrix_file, header=0, index_col=0)
    else:
        data_set = pd.read_csv(opt.matrix_file, header=0, index_col=0)
    # 获取 label 并转化为 数字 表示
    if opt.skip_label_first:
        label_set = pd.read_table(opt.label_file, header=0, index_col=False)
    else:
        label_set = pd.read_table(opt.label_file, header=None, index_col=False)
    src_label = pd.Categorical(label_set.iloc[:, opt.label_index]).codes


# not used
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


def plot_embedding(np_data, labels, classname=None, classes=None, method='tSNE', cmap='tab20', figsize=(4, 4), markersize=4,
                   marker=None,
                   return_emb=False, save=False, save_emb=False, show_legend=True, show_axis_label=False,
                   **legend_params):
    if marker is not None:
        np_data = np.concatenate([np_data, marker], axis=0)
    label_len = len(labels)
    # 降维
    if np_data.shape[1] != 2:
        if method == 'tSNE':
            np_data = TSNE(n_components=2, random_state=124).fit_transform(np_data)
        if method == 'UMAP':
            np_data = UMAP(n_neighbors=30, min_dist=0.3, metric='correlation').fit_transform(np_data)
        if method == 'PCA':
            from sklearn.decomposition import PCA
            np_data = PCA(n_components=2, random_state=124).fit_transform(np_data)

    plt.figure(figsize=figsize)
    if classes is None:
        classes = np.unique(labels)

    if cmap is not None:
        cmap = cmap
    elif len(classes) <= 10:
        cmap = 'tab10'
    elif len(classes) <= 20:
        cmap = 'tab20'
    else:
        cmap = 'husl'
    colors = sns.color_palette(cmap, n_colors=len(classes))

    for i, c in enumerate(classes):
        plt.scatter(np_data[:label_len][labels == c, 0], np_data[:label_len][labels == c, 1], s=markersize, color=colors[i], label=classname[c])
    if marker is not None:
        plt.scatter(np_data[label_len:, 0], np_data[label_len:, 1], s=10 * markersize, color='black', marker='*')
    #     plt.axis("off")

    legend_params_ = {'loc': 'center left',
                      'bbox_to_anchor': (1.0, 0.45),
                      'fontsize': 10,
                      'ncol': 1,
                      'frameon': False,
                      'markerscale': 1.5
                      }
    legend_params_.update(**legend_params)
    if show_legend:
        plt.legend(**legend_params_)
    sns.despine(offset=10, trim=True)
    if show_axis_label:
        plt.xlabel(method + ' dim 1', fontsize=12)
        plt.ylabel(method + ' dim 2', fontsize=12)

    plt.savefig(save, bbox_inches='tight', dpi = 600)
    plt.show()

    if save_emb:
        np.savetxt(save_emb, np_data)
    if return_emb:
        return np_data


def my_plot(np_data, pre_label):
    plt.figure(figsize=(4, 4))
    # 降维
    # np_data = TSNE(n_components=2, random_state=124).fit_transform(np_data)
    np_data = UMAP(n_neighbors=opt.n_clusters, min_dist=0.3, metric='correlation').fit_transform(np_data)
    mark = ['or', 'ob', 'og', 'ok', 'oc', 'om', 'oy', 'ow', '+r', '+g', '+y']
    # 这里'or'代表中的'o'代表画圈，'r'代表颜色为红色，后面的依次类推
    j = 0
    for i in pre_label:
        plt.plot([np_data[j:j + 1, 0]], [np_data[j:j + 1, 1]],
                 mark[i], markersize=8)
        j += 1
    # plt.show()  # 画出聚类结果简易图
    plt.savefig(opt.cluster_fig_path, bbox_inches='tight')


# 评估
def evaluate_score():
    data_frames = pd.DataFrame(data_set)
    np_array = np.array(data_frames.values)
    if opt.need_transpose:
        np_array = np.transpose(np_array)
    kmeans = KMeans(n_clusters=opt.n_clusters, init='k-means++', algorithm='elkan', random_state=0).fit(np_array)
    # print('[KMeans label] len:', len(kmeans.labels_), ' iters:', kmeans.n_iter_, ' content:', kmeans.labels_)

    print('[pre_label] len:', len(kmeans.labels_), ' data:', kmeans.labels_,
          '\n[src_label] len:', len(src_label), ' data:', src_label)

    pre_label = kmeans.labels_
    # centroids = kmeans.cluster_centers_     # 聚类中心
    # inertia = kmeans.inertia_               # 准则

    # adjust score;
    score_a = adjusted_rand_score(src_label, pre_label)
    print(score_a)

    # normalized score
    score_b = normalized_mutual_info_score(src_label, pre_label)
    print(score_b)

    # plot_embedding(np_array, src_label)
    # my_plot(np_array, pre_label)


def evaluate_data():
    data_frames = pd.DataFrame(data_set)
    np_array = np.array(data_frames.values)
    np_array = np.log2(np_array + 0.0001)

    plt.show(np_array.plot(kind = 'box', rot = 90))

    # print(src_label)
    # print(np_array.shape)


def scanpy_deal():
    if opt.need_transpose:
        src_data = sc.read(opt.matrix_file, first_column_names=True)
    else:
        src_data = sc.read_csv(opt.matrix_file, first_column_names=True)
    print('X:', src_data.X, ' \ncells:', src_data.obs, ' \ngenes:', src_data.var)
    print('cell name:', src_data.obs_names, '\ngene name:', src_data.var_names)

def scanpy_first():
    # results_file = 'scanpy_output\scanpy_output.h5ad'
    # file_path = '../dataset/scanpy_data/'
    # file_path = 'human_brain_output/scIGANs-brainTags.csv-src_label.txt-100-15-16-5.0-2.0.csv'
    file_path = '../dataset/human_brain/brainTags.csv'

    # label_path = '../dataset/pollen_labels.txt'
    # label_set = pd.read_table(label_path, header=None, index_col=False)
    # src_label = pd.Categorical(label_set.iloc[:, 1]).codes

    adata = sc.AnnData(pd.read_csv(file_path, header=0, index_col=0).transpose())

    # adata = sc.read_10x_mtx(file_path, var_names='gene_symbols', cache=True)
    # adata = ad.read_csv(file_path, first_column_names=True)
    # adata.var_names_make_unique()       # this is unnecessary if using `var_names='gene_ids'` in `sc.read_10x_mtx`

    # print('X:', adata.X, ' \ncells:', adata.obs, ' \ngenes:', adata.var)
    # print('cell name:', adata.obs_names, '\ngene name:', adata.var_names)
    # print(adata.obs.shape)
    # print(adata.var.shape)

    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)
    sc.pp.highly_variable_genes(adata, min_mean=0.0125, max_mean=3, min_disp=0.5)
    # sc.pl.highly_variable_genes(adata)
    # 保存原始数据
    adata.raw = adata

    sc.pp.neighbors(adata, n_neighbors=15, n_pcs=40)
    sc.tl.leiden(adata)
    sc.tl.umap(adata)
    sc.pl.umap(adata, color=['leiden'])

if __name__ == '__main__':
    read_file()
    evaluate_score()
    # evaluate_data()
    # scanpy_deal()
    # scanpy_first()
