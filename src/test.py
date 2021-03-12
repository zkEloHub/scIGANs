#-*-coding:utf-8 -*-
import pandas as pd
import numpy as np
import csv

def numpy_converse():
    txt = np.loadtxt('../dataset/pollen_data.txt', dtype='str')
    txt = txt[1:, 1:].astype('float')
    txt = np.transpose(txt)
    txtDF = pd.DataFrame(txt)
    txtDF.to_csv('pollen_output/src_matrix.csv')


def deal_human_cell():
    matrix_file = open('../dataset/human_brain/brainTags.csv')
    label_file = open('../dataset/human_brain/SraRunTable.txt')
    label_output = open('../dataset/human_brain/src_label.txt', 'w')
    head = label_file.readline()
    label_output.write(head)

    items = matrix_file.readline().split(',')

    for line in label_file.readlines():
        datas = line.split( )
        flag = False
        for item in items:
            if datas[7] == item.strip('"'):
                flag = True
                break
        if not flag:
            print(datas[7])
        else:
            label_output.write(line)

if __name__ == '__main__':
    # numpy_converse()
    deal_human_cell()
