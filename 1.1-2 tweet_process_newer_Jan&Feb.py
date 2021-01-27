# -*- coding: utf-8 -*-

'''

PART A Import Packages to be used

'''
# Data Processing
import re
import nltk
import string
import warnings
warnings.filterwarnings('ignore', category = DeprecationWarning)

from nltk.corpus import stopwords #nltk.download('stopwords')
from wordcloud import WordCloud

# Prediction model 
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.model_selection import cross_val_score

# Math & Statistics
import pandas as pd
from scipy import stats

# visualization
import matplotlib.pyplot as plt


'''
PART B Train the predicting model
    1. Here we used BoW (TF-IDF) as the feature and logistics regression as the 
       calssifier.
    2. We use L1 normalization to avoid overfitting/underfitting

'''
# get train data
train = pd.read_csv('train.csv')
train.dropna(axis = 0, how = 'any', inplace = True)

# Get 30% tweets from each category (neg, pos and neu) 
# Creat a new dataframe and use this dataframe as the new train data
# This is to avoid memoryerror

typicalNDict = {-1: 0.3, 0: 0.3, 1: 0.3}

def typicalsamling(group, typicalNDict):
    name = group.name
    frac= typicalNDict[name]
    return group.sample(frac = frac, random_state = 1)
 
select_train = train.groupby('category', group_keys = False
                             ).apply(typicalsamling, typicalNDict)
# clean text 
def clean_text(text):
    text = re.sub(r'@[\w]*', ' ', text) # remove twitter handle
    text = re.sub('http://[a-zA-Z0-9.?/&=:]*', ' ', text)
    text = re.sub('https://[a-zA-Z0-9.?/&=:]*', ' ', text)
    text = re.sub(r'[^a-zA-Z]', ' ', text)
    text = re.sub('[%s]' % re.escape(string.punctuation), ' ', text)
    text = text.encode('UTF-8', 'ignore').decode()
    text = text.strip()
    tokens = text.lower().split()
    text = [token for token in tokens if token not in stop_words]
    text = ' '.join(text)
    
    return text

stop_words = stopwords.words('english')
select_train['clean'] = select_train ['clean_text'].apply(clean_text)

# TF-IDF
# split 
validation_size = 0.30
y = select_train.category.values
X_train, X_test, y_train, y_test = train_test_split(select_train.clean.values, y, 
                                                    stratify = y, 
                                                    random_state = 1, 
                                                    test_size = validation_size, 
                                                    shuffle = True)

# vectorize clean tweets
vectorizer_tfidf = TfidfVectorizer(max_features = 5000)
train_vsm_tfidf = vectorizer_tfidf.fit_transform(select_train.clean).toarray()
X_train_tfidf = vectorizer_tfidf.transform(X_train)
X_test_tfidf = vectorizer_tfidf.transform(X_test)

model = LogisticRegression(penalty = 'l1', C = 0.1, max_iter = 1000)
model.fit(X_train_tfidf, y_train)

predictions = model.predict(X_test_tfidf)
print('\nconfusion matrix:\n', confusion_matrix(y_test,predictions)) 
print('\naccuracy:\n', accuracy_score(y_test, predictions)) # 0.73
print('\nF1:\n', classification_report(y_test, predictions)) # F1 0.71
print('\nmodel_eval\n\n', cross_val_score(model, X_test_tfidf, y_test, cv = 5))
#  [0.58508802 0.59668508 0.5955095  0.58914997 0.58984105]

# overfitting test
train_predictions = model.predict(X_train_tfidf)
overfit = accuracy_score(y_train, train_predictions) - accuracy_score(y_test, predictions)
print(overfit)
# 0.0037322164110831224
# after using L1 normalization, the accuracy of train and test sets roughly equal
# the model then is neither overfitting nor underfitting

'''
PART C Tweets processing

In this part we first define all functions to be used, then conduct data processing 
for the convinience of reusing these functions for tweets of other months.

'''
###### define funtions
# influency classification
# select 60% of the tweets to avoid memoryerror (3% in total)
def select(df, final_name):
    df = df.sample(frac = 0.6, random_state = 5, axis = 0)
    save = df
    save = df.to_csv(final_name, index = False)
    return df, save

# top 100 - opinion leader (1). others: majority (0)
def influency_classficiation(df):
    df = df.sort_values(by = ['influency_mark'], ascending = False)
    top = df[:99]
    other = df[100:]
    top['class'] = [1 for index in range(len(top))]
    other['class'] = [0 for index in range(len(other))]
    df = pd.concat([top, other], axis = 0)
    
    return df

# get column mean
def get_column_mean(df, column):
    mean = df[column].mean()
    
    return str('average of ' + column + ' is: ' + str('%.2f' %mean))

# normality check
def distribution(data):
    k2, p = stats.kstest(data, 'norm')
    alpha = 0.05
    if p > alpha:
        result = 'obey normal distribution'
    else:
        result = 'deviation from normal distribution'
    
    return '%.2f'%k2, '%.13f'%p, result

# U test
def u_test(group1, group2):
    u, p = stats.mannwhitneyu(group1, group2)
    alpha = 0.05
    if p < alpha:
        result = 'significant differences'
    else:
        result = 'no significant difference'
    
    return '%.2f'%u, '%.2f'%p, result

## spearman correlation
#def corre_spearman(var1, var2):
#    rho, p = stats.spearmanr(var1, var2)
#    alpha = 0.05
#    if p < alpha:
#        result = 'significant differences'
#    else:
#        result = 'no significant difference'
#    
#    return '%.2f'%rho, '%.2f'%p, result


'''============================ read data 202001 ==========================='''
tweet_202001 = pd.read_csv('hydrated_tweet/jan.csv')
tweet_202001, save_202001 = select(tweet_202001, 'collected_tweet/202001.csv')
#print(tweet_202001.head())
#print(tweet_202001.isnull().sum())
# we have too many lines with only location empty

tweet_202001['clean_text'] = tweet_202001['text'].apply(clean_text)
tweet_202001['influency_mark'] = 0.144 * tweet_202001['user_followers_count'] 
+ 0.330 * tweet_202001['retweet_count'] + 0.330 * tweet_202001['favorite_count']
+ 0.062 * tweet_202001['user_statuses_count'] + 0.026 * tweet_202001['user_friends_count']
+ 0.018 * tweet_202001['user_verified']
tweet_202001 = influency_classficiation(tweet_202001)

# describe
#print(tweet_202001.columns.tolist())
columns = ['Unnamed: 0', 'created_at', 'user_location', 'text', 'favorite_count', 
           'retweet_count', 'user_followers_count', 'user_friends_count', 
           'user_statuses_count', 'user_verified', 'clean_text', 'influency_mark', 
           'class']

for column in columns[4:10] + columns[-2:]:
    print(get_column_mean(tweet_202001, column))

#average of favorite_count is: 21.73
#average of retweet_count is: 6.31
#average of user_followers_count is: 107602.40
#average of user_friends_count is: 2421.12
#average of user_statuses_count is: 82083.71
#average of user_verified is: 0.09
#average of influency_mark is: 15494.75
#average of class is: 0.01

# Use TF-IDF to predict
vectorizer_202001 = TfidfVectorizer(max_features = 5000)
tfidf_202001 = vectorizer_202001.fit_transform(tweet_202001.clean_text).toarray()

tweet_202001['sentiment_value'] =  model.predict(tfidf_202001)
save = tweet_202001
save = save.to_csv('labelled_tweet/tweet_202001_labelled.csv', index = False)

# describe
print('negative:', sum(tweet_202001['sentiment_value'] == -1.0)) #1013
print('neutral:', sum(tweet_202001['sentiment_value'] == 0.0)) #13513
print('positive:', sum(tweet_202001['sentiment_value'] == 1.0)) #2274
print('average:', tweet_202001['sentiment_value'].mean()) # 0.075

# normality check
print('\nNormality Check:\n', distribution(tweet_202001['sentiment_value']))
#  ('0.44', '0.00', 'deviation from normal distribution')
print('\nNormality Check:\n', distribution(tweet_202001['influency_mark']))
# ('0.94', '0.00', 'deviation from normal distribution')

# difference check - U test
# verified vs unverified
group_verified = tweet_202001[tweet_202001['user_verified'] == True]['sentiment_value']
group_unverified = tweet_202001[tweet_202001['user_verified'] == False]['sentiment_value']
print('\nU test (Verified):\n', u_test(group_verified, group_unverified))
# ('10919575.00', '0.15', 'no significant difference')

# opiniton leader vs majority
group_ol = tweet_202001[tweet_202001['class'] == 1]['sentiment_value']
group_maj = tweet_202001[tweet_202001['class'] == 0]['sentiment_value']
print('\nU test (Verified):\n', u_test(group_ol, group_maj))
#  ('793635.00', '0.16', 'no significant difference')

'''============================ read data 202002 ==========================='''
 
tweet_202002 = pd.read_csv('hydrated_tweet/feb.csv')
tweet_202002, save_202002 = select(tweet_202002, 'collected_tweet/202002.csv')
#print(tweet_202002.head())
#print(tweet_202002.isnull().sum())
# we have too many lines with only location empty

tweet_202002['clean_text'] = tweet_202002['text'].apply(clean_text)
tweet_202002['influency_mark'] = 0.144 * tweet_202002['user_followers_count'] 
+ 0.330 * tweet_202002['retweet_count'] + 0.330 * tweet_202002['favorite_count']
+ 0.062 * tweet_202002['user_statuses_count'] + 0.026 * tweet_202002['user_friends_count']
+ 0.028 * tweet_202002['user_verified']
tweet_202002 = influency_classficiation(tweet_202002)

# describe
#print(tweet_202002.columns.tolist())
columns = ['Unnamed: 0', 'created_at', 'user_location', 'text', 'favorite_count', 
           'retweet_count', 'user_followers_count', 'user_friends_count', 
           'user_statuses_count', 'user_verified', 'clean_text', 'influency_mark', 
           'class']

for column in columns[4:10] + columns[-2:]:
    print(get_column_mean(tweet_202002, column))
    

#average of favorite_count is: 9.59
#average of retweet_count is: 3.88
#average of user_followers_count is: 113212.52
#average of user_friends_count is: 2899.80
#average of user_statuses_count is: 95520.56
#average of user_verified is: 0.08
#average of influency_mark is: 16302.60
#average of class is: 0.00

# Use TF-IDF to predict
vectorizer_202002 = TfidfVectorizer(max_features = 5000)
tfidf_202002 = vectorizer_202002.fit_transform(tweet_202002.clean_text).toarray()

tweet_202002['sentiment_value'] =  model.predict(tfidf_202002)
save = tweet_202002
save = save.to_csv('labelled_tweet/tweet_202002_labelled.csv', index = False)

# describe
print('negative:', sum(tweet_202002['sentiment_value'] == -1.0)) #2754
print('neutral:', sum(tweet_202002['sentiment_value'] == 0.0)) # 61969
print('positive:', sum(tweet_202002['sentiment_value'] == 1.0)) # 12589
print('average:', tweet_202002['sentiment_value'].mean()) # 0.127

# normality check
print('\nNormality Check:\n', distribution(tweet_202002['sentiment_value']))
#  ('0.46', '0.00', 'deviation from normal distribution')
print('\nNormality Check:\n', distribution(tweet_202002['influency_mark']))
# ('0.94', '0.00', 'deviation from normal distribution')

# difference check - U test
# verified vs unverified
group_verified = tweet_202002[tweet_202002['user_verified'] == True]['sentiment_value']
group_unverified = tweet_202002[tweet_202002['user_verified'] == False]['sentiment_value']
print('\nU test (Verified):\n', u_test(group_verified, group_unverified))
#  ('224295497.00', '0.01', 'significant differences')

# opiniton leader vs majority
group_ol = tweet_202002[tweet_202002['class'] == 1]['sentiment_value']
group_maj = tweet_202002[tweet_202002['class'] == 0]['sentiment_value']
print('\nU test (Verified):\n', u_test(group_ol, group_maj))
#  ('3556028.50', '0.04', 'significant differences')
