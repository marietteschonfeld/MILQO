import pandas as pd, numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
import re, string
import time
import pickle
import os

# LR model. classifies only one class.

DB = pd.read_csv("NLP_modelDB.csv")

seed = 2022

# split train-test set
df = pd.read_csv('train_extend.csv')
train, test = train_test_split(df, test_size=0.4, random_state=seed)
train = pd.DataFrame(train, columns=df.columns)
test = pd.DataFrame(test, columns=df.columns)

label_cols = ['toxic', 'severe_toxic', 'obscene',
              'threat', 'insult', 'identity_hate',
              'negative', 'neutral', 'positive']

COMMENT = 'comment_text'
train[COMMENT].fillna("unknown", inplace=True)
test[COMMENT].fillna("unknown", inplace=True)

re_tok = re.compile(f'([{string.punctuation}“”¨«»®´·º½¾¿¡§£₤‘’])')
def tokenize(s): return re_tok.sub(r' \1 ', s).split()

n = train.shape[0]
vec = TfidfVectorizer(ngram_range=(1,2), tokenizer=tokenize,
                      min_df=3, max_df=0.9, strip_accents='unicode', use_idf=1,
                      smooth_idf=1, sublinear_tf=1)
trn_term_doc = vec.fit_transform(train[COMMENT])
test_term_doc = vec.transform(test[COMMENT])

filename = 'LR_vectorizer.pickle'
with open(filename, 'wb') as file:
    pickle.dump(vec, file)
vec_file_size = os.path.getsize(filename)

def pr(y_i, y):
    p = x[y==y_i].sum(0)
    return (p+1) / ((y==y_i).sum()+1)

x = trn_term_doc
test_x = test_term_doc

C = 1
def get_mdl(y, label):
    y = y.values
    r = np.log(pr(1,y) / pr(0,y))
    with open('y_{}.npy'.format(label), 'wb') as f:
        np.save(f, r)
    m = LogisticRegression(C=C, dual=False, random_state=seed)
    x_nb = x.multiply(r)
    return m.fit(x_nb, y), r

def classify(m,r, comment):
    transformed = vec.transform((np.array([comment])))
    res = np.rint(m.predict_proba(transformed.multiply(r)))
    return res[0, 1]

scores = [f1_score, accuracy_score, precision_score, recall_score]
preds = np.zeros((len(test), len(label_cols)))
for index in range(len(label_cols)):
    label = label_cols[index]
    m, r = get_mdl(train[label_cols[index]], label)
    filename = 'LR_{}.pickle'.format(label)
    with open(filename, 'wb') as file:
        pickle.dump(m, file)
    model_size = os.path.getsize(filename)
    y_size = os.path.getsize("y_{}.npy".format(label))

    preds[:, index] = np.rint(m.predict_proba(test_x.multiply(r))[:, 1])
    start = time.time()
    for comment in test[COMMENT][0:1000]:
        classification = classify(m, r, comment)
    end = time.time()
    cost = ((end-start) * 1000) / len(test[COMMENT][0:1000])
    for metric in scores:
       score = metric(test[label], preds[:, index])
       DB_entry = {
            "model_name": ["LR_{}".format(label)],
            "cost": [round(cost)],
            "memory": [vec_file_size+model_size+y_size],
            "score_type": metric.__name__,
            "toxic": [0],
            "severe_toxic": [0],
            "obscene": [0],
            "threat": [0],
            "insult": [0],
            "identity_hate": [0],
           "negative": [0],
           "neutral": [0],
           "positive": [0],
       }
       DB_entry[label] = round(score, 5)
       DB = pd.concat([DB, pd.DataFrame(DB_entry)])
       DB.to_csv("NLP_modelDB.csv")

DB.to_csv("NLP_modelDB.csv")

