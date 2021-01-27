# -*- coding: utf-8 -*-
"""
Created on Sun Dec 27 03:02:46 2020

@author: lijin
"""

# extract tweet info by date from full dataset

import pandas as pd

data = pd.read_csv('full_dataset_clean.tsv',sep = '\t', lineterminator = '\n',
                   header = 0, chunksize = 5000)

# January
data_wanted = pd.DataFrame(columns=['tweet_id','date','time','lang','country_place'])

for chunk in data:
    data_selected = chunk[chunk.date.str.contains('-01-')]
    if len(data_selected) != 0:
        data_en = data_selected[data_selected['lang'] == 'en']
        data_wanted = pd.concat([data_wanted,data_en])
    elif len(data_wanted) != 0:
        break
    
data_wanted.to_csv('data_01.csv')

# February
data_wanted = pd.DataFrame(columns=['tweet_id','date','time','lang','country_place'])

for chunk in data:
    data_selected = chunk[chunk.date.str.contains('-02-')]
    if len(data_selected) != 0:
        data_en = data_selected[data_selected['lang'] == 'en']
        data_wanted = pd.concat([data_wanted,data_en])
    elif len(data_wanted) != 0:
        break
    
data_wanted.to_csv('data_02.csv')

# From March to October, extract info each 6 or 7 days to avoid MemoryError
# March:
data_wanted = pd.DataFrame(columns=['tweet_id','date','time','lang','country_place'])

for chunk in data:
    data_selected = chunk[chunk.date.str.contains('03-01|03-02|03-03|03-04|03-05|03-06')]
    if len(data_selected) != 0:
        data_en = data_selected[data_selected['lang'] == 'en']
        data_wanted = pd.concat([data_wanted,data_en])
    elif len(data_wanted) != 0:
        break
    
data_wanted.to_csv('data_03_01_06.csv')


data_wanted = pd.DataFrame(columns=['tweet_id','date','time','lang','country_place'])

for chunk in data:
    data_selected = chunk[chunk.date.str.contains('03-07|03-08|03-09|03-10|03-11|03-12')]
    if len(data_selected) != 0:
        data_en = data_selected[data_selected['lang'] == 'en']
        data_wanted = pd.concat([data_wanted,data_en])
    elif len(data_wanted) != 0:
        break
    
data_wanted.to_csv('data_03_07_12.csv')


data_wanted = pd.DataFrame(columns=['tweet_id','date','time','lang','country_place'])

for chunk in data:
    data_selected = chunk[chunk.date.str.contains('03-13|03-14|03-15|03-16|03-17|03-18')]
    if len(data_selected) != 0:
        data_en = data_selected[data_selected['lang'] == 'en']
        data_wanted = pd.concat([data_wanted,data_en])
    elif len(data_wanted) != 0:
        break
    
data_wanted.to_csv('data_03_13_18.csv')


data_wanted = pd.DataFrame(columns=['tweet_id','date','time','lang','country_place'])

for chunk in data:
    data_selected = chunk[chunk.date.str.contains('03-19|03-20|03-21|03-22|03-23|03-24')]
    if len(data_selected) != 0:
        data_en = data_selected[data_selected['lang'] == 'en']
        data_wanted = pd.concat([data_wanted,data_en])
    elif len(data_wanted) != 0:
        break
    
data_wanted.to_csv('data_03_19_24.csv')


data_wanted = pd.DataFrame(columns=['tweet_id','date','time','lang','country_place'])

for chunk in data:
    data_selected = chunk[chunk.date.str.contains('03-25|03-26|03-27|03-28|03-29|03-30')]
    if len(data_selected) != 0:
        data_en = data_selected[data_selected['lang'] == 'en']
        data_wanted = pd.concat([data_wanted,data_en])
    elif len(data_wanted) != 0:
        break
    
data_wanted.to_csv('data_03_25_30.csv')


data_wanted = pd.DataFrame(columns=['tweet_id','date','time','lang','country_place'])

for chunk in data:
    data_selected = chunk[chunk.date.str.contains('03-31')]
    if len(data_selected) != 0:
        data_en = data_selected[data_selected['lang'] == 'en']
        data_wanted = pd.concat([data_wanted,data_en])
    elif len(data_wanted) != 0:
        break
    
data_wanted.to_csv('data_03_31.csv')

# April:
data_wanted = pd.DataFrame(columns=['tweet_id','date','time','lang','country_place'])

for chunk in data:
    data_selected = chunk[chunk.date.str.contains('04-01|04-02|04-03|04-04|04-05|04-06')]
    if len(data_selected) != 0:
        data_en = data_selected[data_selected['lang'] == 'en']
        data_wanted = pd.concat([data_wanted,data_en])
    elif len(data_wanted) != 0:
        break
    
data_wanted.to_csv('data_04_01_06.csv')


data_wanted = pd.DataFrame(columns=['tweet_id','date','time','lang','country_place'])

for chunk in data:
    data_selected = chunk[chunk.date.str.contains('04-07|04-08|04-09|04-10|04-11|04-12')]
    if len(data_selected) != 0:
        data_en = data_selected[data_selected['lang'] == 'en']
        data_wanted = pd.concat([data_wanted,data_en])
    elif len(data_wanted) != 0:
        break
    
data_wanted.to_csv('data_04_07_12.csv')


data_wanted = pd.DataFrame(columns=['tweet_id','date','time','lang','country_place'])

for chunk in data:
    data_selected = chunk[chunk.date.str.contains('04-13|04-14|04-15|04-16|04-17|04-18')]
    if len(data_selected) != 0:
        data_en = data_selected[data_selected['lang'] == 'en']
        data_wanted = pd.concat([data_wanted,data_en])
    elif len(data_wanted) != 0:
        break
    
data_wanted.to_csv('data_04_13_18.csv')


data_wanted = pd.DataFrame(columns=['tweet_id','date','time','lang','country_place'])

for chunk in data:
    data_selected = chunk[chunk.date.str.contains('04-19|04-20|04-21|04-22|04-23|04-24')]
    if len(data_selected) != 0:
        data_en = data_selected[data_selected['lang'] == 'en']
        data_wanted = pd.concat([data_wanted,data_en])
    elif len(data_wanted) != 0:
        break
    
data_wanted.to_csv('data_04_19_24.csv')


data_wanted = pd.DataFrame(columns=['tweet_id','date','time','lang','country_place'])

for chunk in data:
    data_selected = chunk[chunk.date.str.contains('04-25|04-26|04-27|04-28|04-29|04-30')]
    if len(data_selected) != 0:
        data_en = data_selected[data_selected['lang'] == 'en']
        data_wanted = pd.concat([data_wanted,data_en])
    elif len(data_wanted) != 0:
        break
    
data_wanted.to_csv('data_04_25_30.csv')

# May:
data_wanted = pd.DataFrame(columns=['tweet_id','date','time','lang','country_place'])

for chunk in data:
    data_selected = chunk[chunk.date.str.contains('05-01|05-02|05-03|05-04|05-05|05-06')]
    if len(data_selected) != 0:
        data_en = data_selected[data_selected['lang'] == 'en']
        data_wanted = pd.concat([data_wanted,data_en])
    elif len(data_wanted) != 0:
        break
    
data_wanted.to_csv('data_05_01_06.csv')


data_wanted = pd.DataFrame(columns=['tweet_id','date','time','lang','country_place'])

for chunk in data:
    data_selected = chunk[chunk.date.str.contains('05-07|05-08|05-09|05-10|05-11|05-12')]
    if len(data_selected) != 0:
        data_en = data_selected[data_selected['lang'] == 'en']
        data_wanted = pd.concat([data_wanted,data_en])
    elif len(data_wanted) != 0:
        break
    
data_wanted.to_csv('data_05_07_12.csv')


data_wanted = pd.DataFrame(columns=['tweet_id','date','time','lang','country_place'])

for chunk in data:
    data_selected = chunk[chunk.date.str.contains('05-13|05-14|05-15|05-16|05-17|05-18')]
    if len(data_selected) != 0:
        data_en = data_selected[data_selected['lang'] == 'en']
        data_wanted = pd.concat([data_wanted,data_en])
    elif len(data_wanted) != 0:
        break
    
data_wanted.to_csv('data_05_13_18.csv')


data_wanted = pd.DataFrame(columns=['tweet_id','date','time','lang','country_place'])

for chunk in data:
    data_selected = chunk[chunk.date.str.contains('05-19|05-20|05-21|05-22|05-23|05-24')]
    if len(data_selected) != 0:
        data_en = data_selected[data_selected['lang'] == 'en']
        data_wanted = pd.concat([data_wanted,data_en])
    elif len(data_wanted) != 0:
        break
    
data_wanted.to_csv('data_05_19_24.csv')


data_wanted = pd.DataFrame(columns=['tweet_id','date','time','lang','country_place'])

for chunk in data:
    data_selected = chunk[chunk.date.str.contains('05-25|05-26|05-27|05-28|05-29|05-30|05-31')]
    if len(data_selected) != 0:
        data_en = data_selected[data_selected['lang'] == 'en']
        data_wanted = pd.concat([data_wanted,data_en])
    elif len(data_wanted) != 0:
        break
    
data_wanted.to_csv('data_05_25_31.csv')

# June:
data_wanted = pd.DataFrame(columns=['tweet_id','date','time','lang','country_place'])

for chunk in data:
    data_selected = chunk[chunk.date.str.contains('06-01|06-02|06-03|06-04|06-05|06-06')]
    if len(data_selected) != 0:
        data_en = data_selected[data_selected['lang'] == 'en']
        data_wanted = pd.concat([data_wanted,data_en])
    elif len(data_wanted) != 0:
        break
    
data_wanted.to_csv('data_06_01_06.csv')


data_wanted = pd.DataFrame(columns=['tweet_id','date','time','lang','country_place'])

for chunk in data:
    data_selected = chunk[chunk.date.str.contains('06-07|06-08|06-09|06-10|06-11|06-12')]
    if len(data_selected) != 0:
        data_en = data_selected[data_selected['lang'] == 'en']
        data_wanted = pd.concat([data_wanted,data_en])
    elif len(data_wanted) != 0:
        break
    
data_wanted.to_csv('data_06_07_12.csv')


data_wanted = pd.DataFrame(columns=['tweet_id','date','time','lang','country_place'])

for chunk in data:
    data_selected = chunk[chunk.date.str.contains('06-13|06-14|06-15|06-16|06-17|06-18')]
    if len(data_selected) != 0:
        data_en = data_selected[data_selected['lang'] == 'en']
        data_wanted = pd.concat([data_wanted,data_en])
    elif len(data_wanted) != 0:
        break
    
data_wanted.to_csv('data_06_13_18.csv')


data_wanted = pd.DataFrame(columns=['tweet_id','date','time','lang','country_place'])

for chunk in data:
    data_selected = chunk[chunk.date.str.contains('06-19|06-20|06-21|06-22|06-23|06-24')]
    if len(data_selected) != 0:
        data_en = data_selected[data_selected['lang'] == 'en']
        data_wanted = pd.concat([data_wanted,data_en])
    elif len(data_wanted) != 0:
        break
    
data_wanted.to_csv('data_06_19_24.csv')


data_wanted = pd.DataFrame(columns=['tweet_id','date','time','lang','country_place'])

for chunk in data:
    data_selected = chunk[chunk.date.str.contains('06-25|06-26|06-27|06-28|06-29|06-30')]
    if len(data_selected) != 0:
        data_en = data_selected[data_selected['lang'] == 'en']
        data_wanted = pd.concat([data_wanted,data_en])
    elif len(data_wanted) != 0:
        break
    
data_wanted.to_csv('data_06_25_30.csv')

# July:
data_wanted = pd.DataFrame(columns=['tweet_id','date','time','lang','country_place'])

for chunk in data:
    data_selected = chunk[chunk.date.str.contains('07-01|07-02|07-03|07-04|07-05|07-06')]
    if len(data_selected) != 0:
        data_en = data_selected[data_selected['lang'] == 'en']
        data_wanted = pd.concat([data_wanted,data_en])
    elif len(data_wanted) != 0:
        break
    
data_wanted.to_csv('data_07_01_06.csv')


data_wanted = pd.DataFrame(columns=['tweet_id','date','time','lang','country_place'])

for chunk in data:
    data_selected = chunk[chunk.date.str.contains('07-07|07-08|07-09|07-10|07-11|07-12')]
    if len(data_selected) != 0:
        data_en = data_selected[data_selected['lang'] == 'en']
        data_wanted = pd.concat([data_wanted,data_en])
    elif len(data_wanted) != 0:
        break
    
data_wanted.to_csv('data_07_07_12.csv')


data_wanted = pd.DataFrame(columns=['tweet_id','date','time','lang','country_place'])

for chunk in data:
    data_selected = chunk[chunk.date.str.contains('07-13|07-14|07-15|07-16|07-17|07-18')]
    if len(data_selected) != 0:
        data_en = data_selected[data_selected['lang'] == 'en']
        data_wanted = pd.concat([data_wanted,data_en])
    elif len(data_wanted) != 0:
        break
    
data_wanted.to_csv('data_07_13_18.csv')


data_wanted = pd.DataFrame(columns=['tweet_id','date','time','lang','country_place'])

for chunk in data:
    data_selected = chunk[chunk.date.str.contains('07-19|07-20|07-21|07-22|07-23|07-24')]
    if len(data_selected) != 0:
        data_en = data_selected[data_selected['lang'] == 'en']
        data_wanted = pd.concat([data_wanted,data_en])
    elif len(data_wanted) != 0:
        break
    
data_wanted.to_csv('data_07_19_24.csv')


data_wanted = pd.DataFrame(columns=['tweet_id','date','time','lang','country_place'])

for chunk in data:
    data_selected = chunk[chunk.date.str.contains('07-25|07-26|07-27|07-28|07-29|07-30|07-31')]
    if len(data_selected) != 0:
        data_en = data_selected[data_selected['lang'] == 'en']
        data_wanted = pd.concat([data_wanted,data_en])
    elif len(data_wanted) != 0:
        break
    
data_wanted.to_csv('data_07_25_31.csv')

# August:
data_wanted = pd.DataFrame(columns=['tweet_id','date','time','lang','country_place'])

for chunk in data:
    data_selected = chunk[chunk.date.str.contains('08-01|08-02|08-03|08-04|08-05|08-06')]
    if len(data_selected) != 0:
        data_en = data_selected[data_selected['lang'] == 'en']
        data_wanted = pd.concat([data_wanted,data_en])
    elif len(data_wanted) != 0:
        break
    
data_wanted.to_csv('data_08_01_06.csv')


data_wanted = pd.DataFrame(columns=['tweet_id','date','time','lang','country_place'])

for chunk in data:
    data_selected = chunk[chunk.date.str.contains('08-07|08-08|08-09|08-10|08-11|08-12')]
    if len(data_selected) != 0:
        data_en = data_selected[data_selected['lang'] == 'en']
        data_wanted = pd.concat([data_wanted,data_en])
    elif len(data_wanted) != 0:
        break
    
data_wanted.to_csv('data_08_07_12.csv')


data_wanted = pd.DataFrame(columns=['tweet_id','date','time','lang','country_place'])

for chunk in data:
    data_selected = chunk[chunk.date.str.contains('08-13|08-14|08-15|08-16|08-17|08-18')]
    if len(data_selected) != 0:
        data_en = data_selected[data_selected['lang'] == 'en']
        data_wanted = pd.concat([data_wanted,data_en])
    elif len(data_wanted) != 0:
        break
    
data_wanted.to_csv('data_08_13_18.csv')


data_wanted = pd.DataFrame(columns=['tweet_id','date','time','lang','country_place'])

for chunk in data:
    data_selected = chunk[chunk.date.str.contains('08-19|08-20|08-21|08-22|08-23|08-24')]
    if len(data_selected) != 0:
        data_en = data_selected[data_selected['lang'] == 'en']
        data_wanted = pd.concat([data_wanted,data_en])
    elif len(data_wanted) != 0:
        break
    
data_wanted.to_csv('data_08_19_24.csv')


data_wanted = pd.DataFrame(columns=['tweet_id','date','time','lang','country_place'])

for chunk in data:
    data_selected = chunk[chunk.date.str.contains('08-25|08-26|08-27|08-28|08-29|08-30|08-31')]
    if len(data_selected) != 0:
        data_en = data_selected[data_selected['lang'] == 'en']
        data_wanted = pd.concat([data_wanted,data_en])
    elif len(data_wanted) != 0:
        break
    
data_wanted.to_csv('data_08_25_31.csv')

# September:
data_wanted = pd.DataFrame(columns=['tweet_id','date','time','lang','country_place'])

for chunk in data:
    data_selected = chunk[chunk.date.str.contains('09-01|09-02|09-03|09-04|09-05|09-06')]
    if len(data_selected) != 0:
        data_en = data_selected[data_selected['lang'] == 'en']
        data_wanted = pd.concat([data_wanted,data_en])
    elif len(data_wanted) != 0:
        break
    
data_wanted.to_csv('data_09_01_06.csv')


data_wanted = pd.DataFrame(columns=['tweet_id','date','time','lang','country_place'])

for chunk in data:
    data_selected = chunk[chunk.date.str.contains('09-07|09-08|09-09|09-10|09-11|09-12')]
    if len(data_selected) != 0:
        data_en = data_selected[data_selected['lang'] == 'en']
        data_wanted = pd.concat([data_wanted,data_en])
    elif len(data_wanted) != 0:
        break
    
data_wanted.to_csv('data_09_07_12.csv')


data_wanted = pd.DataFrame(columns=['tweet_id','date','time','lang','country_place'])

for chunk in data:
    data_selected = chunk[chunk.date.str.contains('09-13|09-14|09-15|09-16|09-17|09-18')]
    if len(data_selected) != 0:
        data_en = data_selected[data_selected['lang'] == 'en']
        data_wanted = pd.concat([data_wanted,data_en])
    elif len(data_wanted) != 0:
        break
    
data_wanted.to_csv('data_09_13_18.csv')


data_wanted = pd.DataFrame(columns=['tweet_id','date','time','lang','country_place'])

for chunk in data:
    data_selected = chunk[chunk.date.str.contains('09-19|09-20|09-21|09-22|09-23|09-24')]
    if len(data_selected) != 0:
        data_en = data_selected[data_selected['lang'] == 'en']
        data_wanted = pd.concat([data_wanted,data_en])
    elif len(data_wanted) != 0:
        break
    
data_wanted.to_csv('data_09_19_24.csv')


data_wanted = pd.DataFrame(columns=['tweet_id','date','time','lang','country_place'])

for chunk in data:
    data_selected = chunk[chunk.date.str.contains('09-25|09-26|09-27|09-28|09-29|09-30')]
    if len(data_selected) != 0:
        data_en = data_selected[data_selected['lang'] == 'en']
        data_wanted = pd.concat([data_wanted,data_en])
    elif len(data_wanted) != 0:
        break
    
data_wanted.to_csv('data_09_25_30.csv')

# October:
data_wanted = pd.DataFrame(columns=['tweet_id','date','time','lang','country_place'])

for chunk in data:
    data_selected = chunk[chunk.date.str.contains('10-01|10-02|10-03|10-04|10-05|10-06')]
    if len(data_selected) != 0:
        data_en = data_selected[data_selected['lang'] == 'en']
        data_wanted = pd.concat([data_wanted,data_en])
    elif len(data_wanted) != 0:
        break
    
data_wanted.to_csv('data_10_01_06.csv')


data_wanted = pd.DataFrame(columns=['tweet_id','date','time','lang','country_place'])

for chunk in data:
    data_selected = chunk[chunk.date.str.contains('10-07|10-08|10-09|10-10|10-11|10-12')]
    if len(data_selected) != 0:
        data_en = data_selected[data_selected['lang'] == 'en']
        data_wanted = pd.concat([data_wanted,data_en])
    elif len(data_wanted) != 0:
        break
    
data_wanted.to_csv('data_10_07_12.csv')


data_wanted = pd.DataFrame(columns=['tweet_id','date','time','lang','country_place'])

for chunk in data:
    data_selected = chunk[chunk.date.str.contains('10-13|10-14|10-15|10-16|10-17|10-18')]
    if len(data_selected) != 0:
        data_en = data_selected[data_selected['lang'] == 'en']
        data_wanted = pd.concat([data_wanted,data_en])
    elif len(data_wanted) != 0:
        break
    
data_wanted.to_csv('data_10_13_18.csv')


data_wanted = pd.DataFrame(columns=['tweet_id','date','time','lang','country_place'])

for chunk in data:
    data_selected = chunk[chunk.date.str.contains('10-19|10-20|10-21|10-22|10-23|10-24')]
    if len(data_selected) != 0:
        data_en = data_selected[data_selected['lang'] == 'en']
        data_wanted = pd.concat([data_wanted,data_en])
    elif len(data_wanted) != 0:
        break
    
data_wanted.to_csv('data_10_19_24.csv')


data_wanted = pd.DataFrame(columns=['tweet_id','date','time','lang','country_place'])

for chunk in data:
    data_selected = chunk[chunk.date.str.contains('10-25|10-26|10-27|10-28|10-29|10-30|10-31')]
    if len(data_selected) != 0:
        data_en = data_selected[data_selected['lang'] == 'en']
        data_wanted = pd.concat([data_wanted,data_en])
    elif len(data_wanted) != 0:
        break
    
data_wanted.to_csv('data_10_25_31.csv')
