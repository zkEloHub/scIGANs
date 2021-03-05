# -*- coding: UTF-8 -*-

import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset

df = pd.DataFrame([
    ['green', 'M', 10.1, 'class1'],
    ['red', 'L', 13.5, 'class2'],
    ['blue', 'XL', 15.3, 'class1']])


# print(df)
#        0   1     2       3
# 0  green   M  10.1  class1
# 1    red   L  13.5  class2
# 2   blue  XL  15.3  class1

# print(df.iloc[0:1])

# print(df.iloc[:, 1].values[:, ])

d_file = "../test_data/data.txt"
data = pd.read_table(d_file, header=0, index_col=0)
print(data, '\n')
print(data.iloc[:, 0])

class DummyDataset(Dataset):
    """
    Dataset of numbers in [a,b] inclusive
    """

    def __init__(self, a=0, b=100):
        super(DummyDataset, self).__init__()
        self.a = a
        self.b = b

    def __len__(self):
        return self.b - self.a + 1

    def __getitem__(self, index):
        print(index)
        return index, "label_{}".format(index)


data_set = DummyDataset(0, 15)
# print(data_set.a, data_set.b)
dataloaders1 = DataLoader(data_set, batch_size=2, shuffle=True)

# dataloaders2 = DataLoader(DummyDataset(0, 9), batch_size=2, shuffle=True)

# for i, data in enumerate(dataloaders1):
    # print(data)
