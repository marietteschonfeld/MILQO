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

train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')
test_labels = pd.read_csv('test_labels.csv')

DB = pd.read_csv("NLP_modelDB.csv")
DB.set_index("model_name")

list_classes = ["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]
list_sentences_train = train["comment_text"]

test = test[test_labels["toxic"] != -1]
test_labels = test_labels[test_labels["toxic"] != -1]
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

def create_model(size1, size2):
    x = Embedding(max_features, embed_size)(inp)

    x = LSTM(size1, return_sequences=True, name='lstm_layer')(x)
    x = GlobalMaxPool1D()(x)
    x = Dropout(0.1)(x)
    x = Dense(size2, activation="relu")(x)
    x = Dropout(0.1)(x)
    x = Dense(4, activation="sigmoid")(x)

    model = Model(inputs=inp, outputs=x)
    model.compile(loss='binary_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])
    return model

batch_size = 32
epochs = 5

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
for i, label1 in enumerate(list_classes[0: -3]):
    for j, label2 in enumerate(list_classes[i+1:-2]):
        for k, label3 in enumerate(list_classes[j+1:-1]):
            for m, label4 in enumerate(list_classes[k+1:]):
                y = train[[label1, label2, label3, label4]].values
                model = create_model(size1=random.randint(60, 65), size2=random.randint(35, 45))
                model.fit(X_t, y, batch_size=batch_size, epochs=epochs, validation_split=0.1)
                model_name = "LSTM_medium_{}_{}_{}_{}".format(label1, label2, label3, label4)
                model.save(model_name)
                model_size = get_folder_size(model_name)
                predictions = model.predict(X_te)
                start = time.time()
                for comment in test['comment_text'][0:1000]:
                    classification = classify(model, comment)
                end = time.time()
                cost = ((end-start) * 1000) / len(test['comment_text'][0:1000])
                print("for model trained on {} and {} and {} and {} cost is {} ms".
                      format(label1, label2, label3, label4, cost))

                for score in scores:
                    score1 = score(test_labels[label1], np.rint(predictions[:, 0]))
                    score2 = score(test_labels[label2], np.rint(predictions[:, 1]))
                    score3 = score(test_labels[label1], np.rint(predictions[:, 2]))
                    score4 = score(test_labels[label2], np.rint(predictions[:, 3]))
                    DB_entry = {
                        "model_name": [model_name],
                        "cost": [round(cost)],
                        "memory": [model_size+tokenizer_file_size],
                        "score_type": score.__name__,
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
                    DB_entry[label1] = round(score1, 5)
                    DB_entry[label2] = round(score2, 5)
                    DB_entry[label3] = round(score3, 5)
                    DB_entry[label4] = round(score4, 5)

                    DB = pd.concat([DB, pd.DataFrame(DB_entry)])
                    DB.to_csv("NLP_modelDB.csv")

DB.to_csv("NLP_modelDB.csv")