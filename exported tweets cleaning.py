# -*- coding: utf-8 -*-
"""
Created on Sun Dec 27 03:29:35 2020

@author: lijin
"""

# Extract useful info after hydrating id and exporting tweets as .csv file
import pandas as pd

file_jan = 'tweet_id_jan_exported.csv'
file_feb = 'tweet_id_feb_exported.csv'
file_mar = ['tweet_id_mar_1_exported.csv','tweet_id_mar_2_exported.csv',
            'tweet_id_mar_3_exported.csv','tweet_id_mar_4_exported.csv',
            'tweet_id_mar_5_exported.csv','tweet_id_mar_6_exported.csv']
file_apr = ['tweet_id_apr_1_exported.csv','tweet_id_apr_2_exported.csv',
            'tweet_id_apr_3_exported.csv','tweet_id_apr_4_exported.csv',
            'tweet_id_apr_5_exported.csv']
file_may = ['tweet_id_may_1_exported.csv','tweet_id_may_2_exported.csv',
            'tweet_id_may_3_exported.csv','tweet_id_may_4_exported.csv',
            'tweet_id_may_5_exported.csv']
file_jun = ['tweet_id_jun_1_exported.csv','tweet_id_jun_2_exported.csv',
            'tweet_id_jun_3_exported.csv','tweet_id_jun_4_exported.csv',
            'tweet_id_jun_5_exported.csv']
file_jul = ['tweet_id_jul_1_exported.csv','tweet_id_jul_2_exported.csv',
            'tweet_id_jul_3_exported.csv','tweet_id_jul_4_exported.csv',
            'tweet_id_jul_5_exported.csv']
file_aug = ['tweet_id_aug_1_exported.csv','tweet_id_aug_2_exported.csv',
            'tweet_id_aug_3_exported.csv','tweet_id_aug_4_exported.csv',
            'tweet_id_aug_5_exported.csv']
file_sep = ['tweet_id_sep_1_exported.csv','tweet_id_sep_2_exported.csv',
            'tweet_id_sep_3_exported.csv','tweet_id_sep_4_exported.csv',
            'tweet_id_sep_5_exported.csv']
file_oct = ['tweet_id_oct_1_exported.csv','tweet_id_oct_2_exported.csv',
            'tweet_id_oct_3_exported.csv','tweet_id_oct_4_exported.csv',
            'tweet_id_oct_5_exported.csv']
# January
data = pd.read_csv(file_jan)

data = data[['created_at','user_location','text','favorite_count','retweet_count',
             'user_followers_count','user_friends_count','user_statuses_count',
             'user_verified']]

def normalize_date(date_string):
  alist = date_string.split()
  new_date = alist[5] + '-' + '01' + '-' + alist[2]
  return new_date

data['created_at'] = data['created_at'].apply(normalize_date)

data.to_csv('jan.csv')

# February
data = pd.read_csv(file_jan)

data = data[['created_at','user_location','text','favorite_count','retweet_count',
             'user_followers_count','user_friends_count','user_statuses_count',
             'user_verified']]

def normalize_date(date_string):
  alist = date_string.split()
  new_date = alist[5] + '-' + '02' + '-' + alist[2]
  return new_date

data['created_at'] = data['created_at'].apply(normalize_date)

data.to_csv('feb.csv')

# March
def normalize_date(date_string):
  alist = date_string.split()
  new_date = alist[5] + '-' + '03' + '-' + alist[2]
  return new_date

for file in file_mar:
    data = pd.read_csv(file)
    data = data[['created_at','user_location','text','favorite_count','retweet_count',
             'user_followers_count','user_friends_count','user_statuses_count',
             'user_verified']]
    data['created_at'] = data['created_at'].apply(normalize_date)
    new_name = 'mar' + str(file_mar.index(file) + 1) + '.csv'
    data.to_csv(new_name)
  
# April
def normalize_date(date_string):
  alist = date_string.split()
  new_date = alist[5] + '-' + '04' + '-' + alist[2]
  return new_date

for file in file_apr:
    data = pd.read_csv(file)
    data = data[['created_at','user_location','text','favorite_count','retweet_count',
             'user_followers_count','user_friends_count','user_statuses_count',
             'user_verified']]
    data['created_at'] = data['created_at'].apply(normalize_date)
    new_name = 'apr' + str(file_mar.index(file) + 1) + '.csv'
    data.to_csv(new_name)

# May
def normalize_date(date_string):
  alist = date_string.split()
  new_date = alist[5] + '-' + '05' + '-' + alist[2]
  return new_date

for file in file_may:
    data = pd.read_csv(file)
    data = data[['created_at','user_location','text','favorite_count','retweet_count',
             'user_followers_count','user_friends_count','user_statuses_count',
             'user_verified']]
    data['created_at'] = data['created_at'].apply(normalize_date)
    new_name = 'may' + str(file_mar.index(file) + 1) + '.csv'
    data.to_csv(new_name)

# June
def normalize_date(date_string):
  alist = date_string.split()
  new_date = alist[5] + '-' + '06' + '-' + alist[2]
  return new_date

for file in file_jun:
    data = pd.read_csv(file)
    data = data[['created_at','user_location','text','favorite_count','retweet_count',
             'user_followers_count','user_friends_count','user_statuses_count',
             'user_verified']]
    data['created_at'] = data['created_at'].apply(normalize_date)
    new_name = 'jun' + str(file_mar.index(file) + 1) + '.csv'
    data.to_csv(new_name)

# July
def normalize_date(date_string):
  alist = date_string.split()
  new_date = alist[5] + '-' + '07' + '-' + alist[2]
  return new_date

for file in file_jul:
    data = pd.read_csv(file)
    data = data[['created_at','user_location','text','favorite_count','retweet_count',
             'user_followers_count','user_friends_count','user_statuses_count',
             'user_verified']]
    data['created_at'] = data['created_at'].apply(normalize_date)
    new_name = 'jul' + str(file_mar.index(file) + 1) + '.csv'
    data.to_csv(new_name)

# August
def normalize_date(date_string):
  alist = date_string.split()
  new_date = alist[5] + '-' + '08' + '-' + alist[2]
  return new_date

for file in file_aug:
    data = pd.read_csv(file)
    data = data[['created_at','user_location','text','favorite_count','retweet_count',
             'user_followers_count','user_friends_count','user_statuses_count',
             'user_verified']]
    data['created_at'] = data['created_at'].apply(normalize_date)
    new_name = 'aug' + str(file_mar.index(file) + 1) + '.csv'
    data.to_csv(new_name)

# September
def normalize_date(date_string):
  alist = date_string.split()
  new_date = alist[5] + '-' + '09' + '-' + alist[2]
  return new_date

for file in file_sep:
    data = pd.read_csv(file)
    data = data[['created_at','user_location','text','favorite_count','retweet_count',
             'user_followers_count','user_friends_count','user_statuses_count',
             'user_verified']]
    data['created_at'] = data['created_at'].apply(normalize_date)
    new_name = 'sep' + str(file_mar.index(file) + 1) + '.csv'
    data.to_csv(new_name)

# October
def normalize_date(date_string):
  alist = date_string.split()
  new_date = alist[5] + '-' + '10' + '-' + alist[2]
  return new_date

for file in file_oct:
    data = pd.read_csv(file)
    data = data[['created_at','user_location','text','favorite_count','retweet_count',
             'user_followers_count','user_friends_count','user_statuses_count',
             'user_verified']]
    data['created_at'] = data['created_at'].apply(normalize_date)
    new_name = 'oct' + str(file_mar.index(file) + 1) + '.csv'
    data.to_csv(new_name)
    