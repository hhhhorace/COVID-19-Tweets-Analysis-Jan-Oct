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
from sklearn.feature_extraction.text import CountVectorizer
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

# Wordfreq
# split 
validation_size = 0.30
y = select_train.category.values
X_train, X_test, y_train, y_test = train_test_split(select_train.clean.values, y, 
                                                    stratify = y, 
                                                    random_state = 1, 
                                                    test_size = validation_size, 
                                                    shuffle = True)

# vectorize clean tweets
vectorizer_wordfreq = CountVectorizer(max_features = 3000)
train_wordfreq = vectorizer_wordfreq.fit_transform(select_train.clean).toarray()
X_train_wordfreq = vectorizer_wordfreq.transform(X_train)
X_test_wordfreq = vectorizer_wordfreq.transform(X_test)

model = LogisticRegression(penalty = 'l1', C = 0.05, max_iter = 1000)
model.fit(X_train_wordfreq, y_train)


'''
PART C Tweets processing

In this part we first define all functions to be used, then conduct data processing 
for the convinience of reusing these functions for tweets of other months.

'''
###### define funtions
# influency classification
# select 40% of the tweets to avoid memoryerror (3% in total)
def select(df, final_name):
    df = df.sample(frac = 0.4, random_state = 5, axis = 0)
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
    
    return '%.2f'%u, '%.13f'%p, result


'''============================ read data 202009 ==========================='''

# 202009
tweet_202009_01 = pd.read_csv('hydrated_tweet/sep1.csv')
tweet_202009_02 = pd.read_csv('hydrated_tweet/sep2.csv')
tweet_202009_03 = pd.read_csv('hydrated_tweet/sep3.csv')
tweet_202009_04 = pd.read_csv('hydrated_tweet/sep4.csv')
tweet_202009_05 = pd.read_csv('hydrated_tweet/sep5.csv')
tweet_202009 = pd.concat([tweet_202009_01, tweet_202009_02, tweet_202009_03,
                          tweet_202009_04, tweet_202009_05])
tweet_202009, save_202009 = select(tweet_202009, 'collected_tweet/202009.csv')

tweet_202009['clean_text'] = tweet_202009['text'].apply(clean_text)
tweet_202009['influency_mark'] = 0.144 * tweet_202009['user_followers_count'] 
+ 0.330 * tweet_202009['retweet_count'] + 0.330 * tweet_202009['favorite_count']
+ 0.062 * tweet_202009['user_statuses_count'] + 0.026 * tweet_202009['user_friends_count']
+ 0.018 * tweet_202009['user_verified']
tweet_202009 = influency_classficiation(tweet_202009)

# describe
#print(tweet_202009.columns.tolist())
columns = ['Unnamed: 0', 'created_at', 'user_location', 'text', 'favorite_count', 
           'retweet_count', 'user_followers_count', 'user_friends_count', 
           'user_statuses_count', 'user_verified', 'clean_text', 'influency_mark', 
           'class']

for column in columns[4:10] + columns[-2:]:
    print(get_column_mean(tweet_202009, column))

#average of favorite_count is: 9.84
#average of retweet_count is: 3.20
#average of user_followers_count is: 61804.77
#average of user_friends_count is: 2270.23
#average of user_statuses_count is: 63193.05
#average of user_verified is: 0.09
#average of influency_mark is: 8899.89
#average of class is: 0.00

# Use wordfreq to predict
vectorizer_202009 = CountVectorizer(max_features = 3000)
wordfreq_202009 = vectorizer_202009.fit_transform(tweet_202009.clean_text).toarray()

tweet_202009['sentiment_value'] =  model.predict(wordfreq_202009)
save = tweet_202009
save = save.to_csv('labelled_tweet/tweet_202009_labelled.csv', index = False)

# describe
print('negative:', sum(tweet_202009['sentiment_value'] == -1.0)/
      len(tweet_202009['sentiment_value'])) 
print('neutral:', sum(tweet_202009['sentiment_value'] == 0.0)/
      len(tweet_202009['sentiment_value'])) 
print('positive:', sum(tweet_202009['sentiment_value'] == 1.0)/
      len(tweet_202009['sentiment_value'])) 
print('average:', tweet_202009['sentiment_value'].mean()) 

#negative: 0.07612491775067065
#neutral: 0.775876904388318
#positive: 0.1479981778610113
#average: 0.07187326011034063

# normality check
print('\nNormality Check:\n', distribution(tweet_202009['influency_mark']))
# ('0.92', '0.00', 'deviation from normal distribution')
print('\nNormality Check:\n', distribution(tweet_202009['sentiment_value']))
#  ('0.42', '0.00', 'deviation from normal distribution')

# difference check - U test
# verified vs unverified
group_verified = tweet_202009[tweet_202009['user_verified'] == True]['sentiment_value']
group_unverified = tweet_202009[tweet_202009['user_verified'] == False]['sentiment_value']
print('\nU test (Verified):\n', u_test(group_verified, group_unverified))
#('989066792.00', '0.3592572529010', 'no significant difference')

# opiniton leader vs majority
group_ol = tweet_202009[tweet_202009['class'] == 1]['sentiment_value']
group_maj = tweet_202009[tweet_202009['class'] == 0]['sentiment_value']
print('\nU test (Verified):\n', u_test(group_ol, group_maj))
# ('7790355.50', '0.4655874142778', 'no significant difference')

