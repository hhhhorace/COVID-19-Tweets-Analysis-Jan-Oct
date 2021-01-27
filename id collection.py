# -*- coding: utf-8 -*-
"""
Created on Sun Dec 27 03:11:15 2020

@author: lijin
"""

# Extract 5% of tweet id randomly and save as txt document
import random
import pandas as pd

data_list = ['data_01.csv','data_02.csv','data_03_01_06.csv',
             'data_03_07_12.csv','data_03_13_18.csv','data_03_19_24.csv',
             'data_03_25_30.csv','data_03_31.csv','data_04_1_6.csv',
             'data_04_7_12.csv','data_04_13_18.csv','data_04_19_24.csv',
             'data_04_25_30.csv','data_05_01_06.csv','data_05_07_12.csv',
            'data_05_13_18.csv','data_05_19_24.csv','data_05_25_31.csv',
            'data_06_01_06.csv','data_06_07_12.csv','data_06_13_18.csv',
            'data_06_19_24.csv','data_06_25_30.csv','data_07_01_06.csv',
            'data_07_07_12.csv','data_07_13_18.csv','data_07_19_24.csv',
            'data_07_25_31.csv','data_08_01_06.csv','data_08_07_12.csv',
            'data_08_13_18.csv','data_08_19_24.csv','data_08_25_31.csv',
            'data_09_01_06.csv','data_09_07_12.csv','data_09_13_18.csv',
            'data_09_19_24.csv','data_09_25_30.csv','data_10_01_06.csv',
            'data_10_07_12.csv','data_10_13_18.csv','data_10_19_24.csv',
            'data_10_25_31.csv']

for data in data_list:
    file = pd.read_csv(data)
    random_file = file.sample(frac = 0.05,random_state = None)
    random_file.reset_index(drop=True, inplace=True)
    txt_name = data[:-3] + 'txt'
    with open(txt_name,'w') as id_collect:
        for i in range(len(random_file)):
            t = str(int(random_file.at[i,'tweet_id']))
            id_collect.write(t)
            id_collect.write('\n')