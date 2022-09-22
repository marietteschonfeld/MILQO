import os
import pandas as pd
from keras.preprocessing.text import Tokenizer
from keras_preprocessing.sequence import pad_sequences
from keras.layers import Dense, Input, LSTM, Embedding, Dropout
from keras.layers import GlobalMaxPool1D
from keras.models import Model
import time
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
import pickle
import random
import numpy as np
from sklearn.model_selection import train_test_split

df = pd.read_csv('train_extend.csv')
df = df.drop(['Unnamed: 0'], axis=1)
train, test = train_test_split(df, test_size=0.1)
train = pd.DataFrame(train, columns=df.columns)
test = pd.DataFrame(test, columns=df.columns)

DB = pd.read_csv("NLP_modelDB.csv")

list_classes = ["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate", "negative", "neutral", "positive"]
list_sentences_train = train["comment_text"]

list_sentences_test = test["comment_text"]

max_features = 20000
tokenizer = Tokenizer(num_words=max_features)
tokenizer.fit_on_texts(list(list_sentences_train))
list_tokenized_train = tokenizer.texts_to_sequences(list_sentences_train)
list_tokenized_test = tokenizer.texts_to_sequences(list_sentences_test)

with open('models/small_LSTM_tokenizer.pickle', 'wb') as handle:
    pickle.dump(tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)
tokenizer_file_size = os.path.getsize("models/small_LSTM_tokenizer.pickle")

maxlen = 200
X_t = pad_sequences(list_tokenized_train, maxlen=maxlen)
X_te = pad_sequences(list_tokenized_test, maxlen=maxlen)

inp = Input(shape=(maxlen, )) #maxlen=200 as defined earlier
embed_size = 128


x = Embedding(max_features, embed_size)(inp)

x = LSTM(125, return_sequences=True, name='lstm_layer')(x)
x = GlobalMaxPool1D()(x)
x = Dropout(0.1)(x)
x = Dense(80, activation="relu")(x)
x = Dropout(0.1)(x)
x = Dense(40, activation="relu")(x)
x = Dropout(0.1)(x)
x = Dense(len(list_classes), activation="sigmoid")(x)

model = Model(inputs=inp, outputs=x)
model.compile(loss='binary_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

batch_size = 32
epochs = 30

def classify(model, comment):
    token_comment = tokenizer.texts_to_sequences([comment])
    x_comment = pad_sequences(token_comment, maxlen=maxlen)
    return model.predict(x_comment, verbose=0)

def get_folder_size(start_path = '.'):
    total_size = 0
    for dirpath, dirnames, filenames in os.walk(start_path):
        for f in filenames:
            fp = os.path.join(dirpath, f)
            # skip if it is symbolic link
            if not os.path.islink(fp):
                total_size += os.path.getsize(fp)

    return total_size

scores = [f1_score, accuracy_score, precision_score, recall_score]

y = train[list_classes].values
model.fit(X_t, y, batch_size=batch_size, epochs=epochs, validation_split=0.1)
model_name = "LSTM_large_{}_{}_{}".format(list_classes[0], list_classes[1], list_classes[2])
model.save(model_name)
model_size = get_folder_size(model_name)
predictions = model.predict(X_te)
start = time.time()
for comment in test['comment_text'][0:1000]:
    classification = classify(model, comment)
end = time.time()
cost = ((end-start) * 1000) / len(test['comment_text'][0:1000])
print("for model trained on {} and {} cost is {} ms")

for score_type in scores:
    scorings = [score_type(test[list_classes[x]], np.rint(predictions[:, x]))
                for x in range(len(list_classes))]
    DB_entry = {
        "model_name": [model_name],
        "cost": [round(cost)],
        "memory": [model_size+tokenizer_file_size],
        "score_type": score_type.__name__,
        "toxic": [0],
        "severe_toxic": [0],
        "obscene": [0],
        "threat": [0],
        "insult": [0],
        "identity_hate": [0],
        "negative": [0],
        "neutral": [0],
        "positive": [0]
    }
    for index, label in enumerate(list_classes):
        DB_entry[label] = [scorings[index]]

    DB = pd.concat([DB, pd.DataFrame(DB_entry)])
    DB.to_csv("NLP_modelDB.csv")

DB.to_csv("NLP_modelDB.csv")