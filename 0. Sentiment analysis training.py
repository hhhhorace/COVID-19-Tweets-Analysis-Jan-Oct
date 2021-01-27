# -*- coding: utf-8 -*-

#############  1. Import packages ############# 

import re
import numpy as np
import pandas as pd 
import nltk
import warnings
import string
warnings.filterwarnings('ignore', category = DeprecationWarning)

from nltk.corpus import stopwords

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.model_selection import cross_val_score

#############  2. Import and Explore the Data ############# 

data = pd.read_csv('train.csv')

# delete lines where there is/are null value(s)
data.dropna(axis=0, how='any', inplace=True)

print('negative:', sum(data['category'] == -1.0))

print('neutral:', sum(data['category'] == 0.0))

print('positive:', sum(data['category'] == 1.0))

print(data.isnull().sum())

# Get 30% tweets from each category (neg, pos and neu) 
# Creat a new dataframe and use this dataframe as the new train data
# This is to avoid memoryerror

typicalNDict = {-1: 0.3, 0: 0.3, 1: 0.3}

def typicalsamling(group, typicalNDict):
    name = group.name
    frac= typicalNDict[name]
    return group.sample(frac = frac, random_state = 1)
 
data = data.groupby('category', group_keys = False
                    ).apply(typicalsamling, typicalNDict)

print(data.isnull().sum())

#############  3. Clean and Explore texts ############# 
stop_words = stopwords.words('english')

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

data['clean'] = data['clean_text'].apply(clean_text)

#############  4. Model Evluation ############# 
def model_eval(test, predict):

    print('Confusion Matrix\n')
    print(confusion_matrix(test, predict))
    print('\nAccuracy\n')
    print(accuracy_score(test, predict))
    print('\nPrecision, Recall, and F1\n')
    print(classification_report(test, predict))

#split
validation_size = 0.30
y = data.category.values
X_train, X_test, y_train, y_test = train_test_split(data.clean.values, y, 
                                                    stratify = y, 
                                                    random_state = 1, 
                                                    test_size = validation_size, 
                                                    shuffle = True)

#############  5. Bags of word (wordfreq) ############# 
from sklearn.feature_extraction.text import CountVectorizer

#vectorize
vectorizer_wordfreq = CountVectorizer(max_features = 3000)
train_wordfreq = vectorizer_wordfreq.fit_transform(data.clean).toarray()
X_train_wordfreq = vectorizer_wordfreq.transform(X_train)
X_test_wordfreq = vectorizer_wordfreq.transform(X_test)

#fit and evaluate
model = LogisticRegression(penalty = 'l1', C = 0.05, max_iter = 1000)  
model.fit(X_train_wordfreq, y_train)
print('\n=========feature: word frequency=========\n')
model_eval(y_test, model.predict(X_test_wordfreq))

# overfitting test
print(accuracy_score(y_train, model.predict(X_train_wordfreq)) - 
      accuracy_score(y_test, model.predict(X_test_wordfreq)))

############# 6 Tf-iDF ############# 
from sklearn.feature_extraction.text import TfidfVectorizer

#vectorize
vectorizer_tfidf = TfidfVectorizer(max_features = 3000)
train_tfidf = vectorizer_tfidf.fit_transform(data.clean).toarray()
X_train_tfidf = vectorizer_tfidf.transform(X_train)
X_test_tfidf = vectorizer_tfidf.transform(X_test)

#fit and evaluate
model = LogisticRegression(penalty = 'l1', C = 0.05, max_iter = 1000) # 
model.fit(X_train_tfidf, y_train)
print('\n=========feature: TF-IDF=========\n')
model_eval(y_test, model.predict(X_test_tfidf))

# overfitting test
print(accuracy_score(y_train, model.predict(X_train_tfidf)) - 
      accuracy_score(y_test, model.predict(X_test_tfidf)))


#############  7. bi-gram ############# 
from sklearn.feature_extraction.text import CountVectorizer

vectorizer_bigram = CountVectorizer(ngram_range = (2, 2), 
                                    token_pattern = r'\b\w+\b', 
                                    max_features= 3000) 
train_bigram = vectorizer_bigram.fit_transform(data.clean).toarray()
X_train_bigram = vectorizer_bigram.transform(X_train)
X_test_bigram = vectorizer_bigram.transform(X_test)
 
#fit and evaluate
model = LogisticRegression(penalty = 'l1', C = 0.05, max_iter = 1000) # 
model.fit(X_train_bigram, y_train)
print('\n=========feature: Bigram=========\n')
model_eval(y_test, model.predict(X_test_bigram))

# overfitting test
print(accuracy_score(y_train, model.predict(X_train_bigram)) - 
      accuracy_score(y_test, model.predict(X_test_bigram)))

#############  8. word2vec (gensim) ############# 
# train a word2vec model
import os
import logging
from gensim import models
from nltk.tokenize import word_tokenize

data['tokens'] = data['clean'].apply(word_tokenize)

loggings = logging.basicConfig(format = '%(asctime)s : %(levelname)s : %(message)s', 
                               level = logging.INFO)

# set parameters for word2vec training
num_features = 300      # dimensionality of word2vec is 300
min_word_count = 40     # exclude words with frequency llower than 40
num_workers = 4         
context = 10            
model_ = 0              # use CBOW model

# train and save the model
sentences = []
for s in data['tokens']:
    sentences.append(s)

#model_name_w2v = '{}features_{}minwords_{}context.model'.format(num_features, 
#                  min_word_count, context)

model_w2v = models.Word2Vec(sentences, workers = num_workers, size = num_features, 
                          min_count = min_word_count, window = context, sg=model_)
#model_w2v.save(os.path.join('models', model_name_w2v))
model_w2v.wv.save_word2vec_format(os.path.join('models/word2vec_txt.txt'), binary = False)

# average the vectors of words in clean text
words = []
vectors = []
word2vec = {}
with open('models/word2vec_txt.txt', 'r') as file:
    for i, line in enumerate(file):
        if i == 0:
            continue
        line = line.split()
        word = line[0]
        vector = line[1:]
        vector = [float(s) for s in vector]
        
        words.append(word)
        vectors.append(vector)
        word2vec[word] = vector

def sentence_embedding(tokenized_sentence, word2vec):
    words_to_be_used = [word for word in tokenized_sentence if word in word2vec]
    word_embs = [word2vec[word] for word in words_to_be_used]
    if not word_embs:
        return np.random.normal(size = (300,))
    return np.mean(word_embs, axis = 0)

data_train, data_test = train_test_split(data, 
                                         random_state= 1,
                                         test_size = validation_size, 
                                         shuffle = True)

train_word2vec = [sentence_embedding(tokens, word2vec) for tokens in data_train['tokens']]
test_word2vec = [sentence_embedding(tokens, word2vec) for tokens in data_test['tokens']]

#fit and evaluate
model = LogisticRegression(penalty = 'l1', C = 0.1, max_iter = 1000) 
model.fit(train_word2vec, y_train)
print('\n=========feature: Word2vec(gensim)=========\n')
model_eval(data_test['category'], model.predict(test_word2vec))

############# 9. word2vec (glove) ############# 
#11.1 get sentence embeddign with glove vectors
from nltk.tokenize import word_tokenize
data['tokens'] = data['clean'].apply(word_tokenize)

words_glove = []
vectors_glove = []
word2vec_glove = {}
with open('models/glove.6B.300d.txt', 'r', encoding='utf-8') as file:
    for i, line in enumerate(file):
        if i == 0:
            continue
        line = line.split()
        word = line[0]
        vector = line[1:]
        vector = [float(s) for s in vector]
        
        words_glove.append(word)
        vectors_glove.append(vector)
        word2vec_glove[word] = vector

def sentence_embedding(tokenized_sentence, word2vec_glove):
    words_to_be_used_glove = [word for word in tokenized_sentence 
                              if word in word2vec_glove]
    word_embs_glove = [word2vec_glove[word] for word in words_to_be_used_glove]
    if not word_embs_glove:
        return np.random.normal(size = (300,))
    return np.mean(word_embs_glove, axis = 0)

train_word2vec_glove = [sentence_embedding(tokens, word2vec_glove) for tokens in data_train['tokens']]
test_word2vec_glove = [sentence_embedding(tokens, word2vec_glove) for tokens in data_test['tokens']]

#fit and evaluate
model = LogisticRegression(penalty = 'l1', C = 0.1, max_iter = 1000) 
model.fit(train_word2vec_glove, y_train)
print('\n=========feature: Word2vec(gensim)=========\n')
model_eval(data_test['category'], model.predict(test_word2vec_glove))
