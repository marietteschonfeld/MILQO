import pickle
import numpy as np
import keras
from keras_preprocessing.sequence import pad_sequences
import re, string
#from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

# class to classify data using only a model name
class inference_model:
    def __init__(self, name):
        self.name = name
        # LSTM or LR
        self.type = name.split("_")[0]
        self.classes = []
        self.detect_classes()
        self.detect_task()

    # detects which classes can be classified based on name
    def detect_classes(self):
        total_classes = ['toxic', 'severe_toxic', 'obscene',
                                      'threat', 'insult', 'identity_hate',
                                      'negative', 'neutral', 'positive']
        split_name = self.name.split("_")
        for index, item in enumerate(split_name):
            if item == "severe":
                self.classes.append("severe_toxic")
            elif item == "toxic" and split_name[index-1] != "severe":
                self.classes.append("toxic")
            elif item == "hate":
                self.classes.append("identity_hate")
            elif item in total_classes and item != "toxic":
                self.classes.append(item)

    # detect whether model is a toxicity analysis or sentiment analysis model
    def detect_task(self):
        if "negative" in self.classes or "positive" in self.classes or "neutral" in self.classes:
            self.task = "sentiment_analysis"
        else:
            self.task = "toxicity_analysis"

    # classify data
    def classify(self, comments, filter, regression=False):
        classification = {}
        # only classify data according to the filter
        filtered_comments = comments[filter]
        if self.type == "LR":
            vec = pickle.load(open("LR_vectorizer.pickle", 'rb'))
            transformed = vec.transform((filtered_comments))
            m = pickle.load(open("LR_{}.pickle".format(self.classes[0]), 'rb'))
            r = np.load("y_{}.npy".format(self.classes[0]))
            res = m.predict_proba(transformed.multiply(r))
            if not regression:
                res = np.rint(res)
            classification[self.classes[0]] = 0.5*np.ones((comments.shape[0], 1))
            count = 0
            for i in range(len(comments)):
                if filter[i]:
                    classification[self.classes[0]][i] = res[count, 1]
                    count += 1
        else:
            size = self.name.split("_")[1]
            tokenizer = pickle.load(open("{}_LSTM_tokenizer.pickle".format(size), 'rb'))
            token_comments = tokenizer.texts_to_sequences(list(filtered_comments))
            lens = {'small': 200, 'medium': 200, 'large': 200}
            x_comments = pad_sequences(token_comments, maxlen=lens[size])
            model = keras.models.load_model(self.name)
            prediction = model.predict(x_comments, verbose=0)
            if not regression:
                prediction = np.rint(prediction)
            for index, pred in enumerate(self.classes):
                classification[pred] = 0.5*np.ones((comments.shape[0], 1))
                count = 0
                for i in range(len(comments)):
                    if filter[i]:
                        classification[pred][i] = prediction[count, index]
                        count += 1
        return classification



re_tok = re.compile(f'([{string.punctuation}“”¨«»®´·º½¾¿¡§£₤‘’])')
def tokenize(s): return re_tok.sub(r' \1 ', s).split()