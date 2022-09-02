import pandas as pd
import re
import string
from sklearn.feature_extraction.text import CountVectorizer
import numpy as np
import nltk
import json

nltk.download('wordnet')
nltk.download('stopwords')
from nltk.corpus import wordnet

data = pd.read_csv("train.csv")

def remove_punctuation(text):
    punctuationfree="".join([i for i in text if i not in string.punctuation])
    return punctuationfree

stopwords = nltk.corpus.stopwords.words('english')
def remove_stopwords(text):
    output = [i for i in text.split(" ") if i not in stopwords]
    return " ".join(output)

#storing the puntuation free text
data['clean_comment']= data['comment_text'].apply(lambda x:remove_punctuation(x))
data.head()

data['comment_lower']= data['clean_comment'].apply(lambda x: x.lower())

# TODO remove stopwords ornot?
# data['cleaned'] = data['comment_lower'].apply(lambda x: remove_stopwords(x))

vectorizer = CountVectorizer()
bag = vectorizer.fit_transform(data['comment_lower'])

counts = bag.sum(axis=0)

adv_size = 200
max_ind = np.argsort(counts)[0, -adv_size:].T
words = vectorizer.get_feature_names_out()
print(counts.shape)

# fuck this is ugly
empty = ["", 0]
counts_dict = np.array([empty for x in range(0, adv_size)])
print(counts_dict.shape)
for index in range(0, adv_size):
    ind = int(max_ind[index])
    a = str(words[ind])
    b = int(counts[0, ind])
    counts_dict[index, :] = a, b

counts_df = pd.DataFrame(counts_dict, columns=['word', 'count'])
counts_df = counts_df.set_index('word')

counts_df.to_csv("BoW.csv")