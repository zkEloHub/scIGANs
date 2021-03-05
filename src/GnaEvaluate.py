# -*- coding: UTF-8 -*-

from sklearn.cluster import KMeans
import numpy as np
import pandas as pd

# 每行作为一个细胞数据
matrix_file = '../src/bias_output/scIGANs-data.txt-labels.txt.csv'
matrix = pd.read_csv(matrix_file, header=0, index_col=0)
print('[matrix] len:', len(matrix))

# X = np.array([[1, 2], [1, 4], [1, 0], [4, 2], [4, 4], [4, 0]])
# kmeans = KMeans(n_clusters=3, random_state=0).fit(X)
#
# print(kmeans.labels_)
