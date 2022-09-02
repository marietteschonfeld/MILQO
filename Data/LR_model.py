import timeit

import pandas as pd, numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import cross_val_score
import re, string
import time
import pickle
import os

DB = pd.read_csv("NLP_modelDB.csv")
DB.set_index("model_name")

train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')
test_labels = pd.read_csv('test_labels.csv')

label_cols = ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']
train.describe()

COMMENT = 'comment_text'
train[COMMENT].fillna("unknown", inplace=True)
test[COMMENT].fillna("unknown", inplace=True)

test = test[test_labels["toxic"] != -1]
test_labels = test_labels[test_labels["toxic"] != -1]

re_tok = re.compile(f'([{string.punctuation}“”¨«»®´·º½¾¿¡§£₤‘’])')
def tokenize(s): return re_tok.sub(r' \1 ', s).split()

n = train.shape[0]
vec = TfidfVectorizer(ngram_range=(1,2), tokenizer=tokenize,
                      min_df=3, max_df=0.9, strip_accents='unicode', use_idf=1,
                      smooth_idf=1, sublinear_tf=1)
trn_term_doc = vec.fit_transform(train[COMMENT])
test_term_doc = vec.transform(test[COMMENT])

filename = 'models/LR_vectorizer.sav'
with open(filename, 'wb') as file:
    pickle.dump(vec, file)
vec_file_size = os.path.getsize(filename)

def pr(y_i, y):
    p = x[y==y_i].sum(0)
    return (p+1) / ((y==y_i).sum()+1)

x = trn_term_doc
test_x = test_term_doc

C = 1
def get_mdl(y):
    y = y.values
    r = np.log(pr(1,y) / pr(0,y))
    m = LogisticRegression(C=C, dual=False)
    x_nb = x.multiply(r)
    return m.fit(x_nb, y), r

def classify(m,r, comment):
    transformed = vec.transform((np.array([comment])))
    res = np.rint(m.predict_proba(transformed.multiply(r)))
    return res[0, 1]

preds = np.zeros((len(test), len(label_cols)))
for i, j in enumerate(label_cols):
    m, r = get_mdl(train[j])
    filename = 'LR_{}.sav'.format(j)
    with open(filename, 'wb') as file:
        pickle.dump(vec, file)
    model_size = os.path.getsize(filename)
    preds[:, i] = m.predict_proba(test_x.multiply(r))[:, 1]
    # accuracy = 0
    # count = 0
    # for index in range(test.values.shape[0]):
    #     if test_labels[j].iloc[index] != -1:
    #         accuracy += (preds[index, i] == test_labels[j].iloc[index])
    #         count += 1
    score = roc_auc_score(test_labels[j], preds[:, i])
    start = time.time()
    for comment in test[COMMENT]:
        classification = classify(m, r, comment)
    end = time.time()
    cost = ((end-start) * 1000) / len(test[COMMENT])
    print("For label {} roc_auc_score is {} and cost is {} ms.".format(j, score, cost))
    DB_entry = {
        "model_name": ["LR_{}".format(j)],
        "cost": [round(cost)],
        "memory": [vec_file_size+model_size],
        "toxic": [0],
        "severe_toxic": [0],
        "obscene": [0],
        "threat": [0],
        "insult": [0],
        "identity_hate": [0]
    }
    DB_entry[j] = round(score, 5)
    DB = pd.concat([DB, pd.DataFrame(DB_entry)])

DB.to_csv("NLP_modelDB.csv")

