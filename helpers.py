import numpy as np

# Keras
from keras.preprocessing import text, sequence
from keras.callbacks import Callback
import pandas as pd

from sklearn.metrics import roc_auc_score

import re
from replacer import repl


class RocAucEvaluation(Callback):
    def __init__(self, validation_data=(), interval=1):
        super(Callback, self).__init__()

        self.interval = interval
        self.X_val, self.y_val = validation_data 

    def on_epoch_end(self, epoch, logs={}):
        if epoch % self.interval == 0:
            y_pred = self.model.predict(self.X_val, verbose=0)
            score = roc_auc_score(self.y_val, y_pred)
            print("\n ROC-AUC - epoch: %d - score: %.6f \n" % (epoch+1, score))

def clean(data_matrix):
    # remove numbers
    #words = re.sub("(^|\W)\d+($|\W)", " ", words)
    #remove \n
    #words = re.sub("\\n","", words)
    # remove leaky elements like ip,user
    #words = re.sub("\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}","", words)
    
    #words = 
    
    keys = [i for i in repl.keys()]

    new_data = []
    ltr = data_matrix["comment_text"].tolist()

    for i in ltr:
        arr = str(i).split()
        xx = ""
        for j in arr:
            j = str(j).lower()
            if j[:4] == 'http' or j[:3] == 'www':
                continue
            if j in keys:
                # print("inn")
                j = repl[j]
            xx += j + " "
        new_data.append(xx)

    data_matrix["new_comment_text"] = new_data
    print("clean")
    trate = data_matrix["new_comment_text"].tolist()

    for i, c in enumerate(trate):
        # Remove everything that's not a-z, !, or ?
        trate[i] = re.sub('[^a-zA-Z ?!]+', '', str(trate[i]).lower())
        # Remove usernames
        trate[i] = re.sub("\[\[.*\]","", str(trate[i]).lower())
    data_matrix["comment_text"] = trate

    print("Text is clean")
    # TODO: Add normalization

    return data_matrix

def make_df(train_path, test_path, max_features, maxlen, list_classes):
    train = pd.read_csv(train_path)
    test = pd.read_csv(test_path)
    train = train.sample(frac=1)

    train = clean(train)
    test = clean(test)

    list_sentences_train = train["comment_text"].fillna("unknown").values
    y = train[list_classes].values
    list_sentences_test = test["comment_text"].fillna("unknown").values

    tokenizer = text.Tokenizer(num_words=max_features)
    tokenizer.fit_on_texts(list(list_sentences_train))
    list_tokenized_train = tokenizer.texts_to_sequences(list_sentences_train)
    list_tokenized_test = tokenizer.texts_to_sequences(list_sentences_test)
    X_t = sequence.pad_sequences(list_tokenized_train, maxlen=maxlen)
    X_te = sequence.pad_sequences(list_tokenized_test, maxlen=maxlen)
    
    word_index = tokenizer.word_index

    return X_t, X_te, y, word_index

def make_glovevec(glovepath, max_features, embed_size, word_index, veclen=300):
    embeddings_index = {}
    f = open(glovepath)
    for line in f:
        values = line.split()
        word = ' '.join(values[:-veclen])
        coefs = np.asarray(values[-veclen:], dtype='float32')
        embeddings_index[word] = coefs.reshape(-1)
    f.close()

    nb_words = min(max_features, len(word_index))
    embedding_matrix = np.zeros((nb_words, embed_size))
    for word, i in word_index.items():
        if i >= max_features:
            continue
        embedding_vector = embeddings_index.get(word)
        if embedding_vector is not None:
            embedding_matrix[i] = embedding_vector
    print(embedding_matrix)
    return embedding_matrix


def predict_and_save(model, test_data, epoch, filename, batch_size=1024):

    list_classes = ["toxic", "severe_toxic", "obscene", "threat", "insult",
                "identity_hate"]

    model.load_weights('./modelckpts/.model.'+ epoch + '.hdf5')
    print("Predicting with model...")
    y_test = model.predict(test_data, batch_size=batch_size)

    sample_submission = pd.read_csv("./input/sample_submission.csv")
    sample_submission[list_classes] = y_test
    print("Saving to submission file...")
    sample_submission.to_csv("./submissions/" + filename + ".csv", index=False)
    return