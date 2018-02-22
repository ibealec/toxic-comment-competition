import numpy as np

# Keras
from keras.preprocessing import text, sequence
import pandas as pd


def make_df(train_path, test_path, max_features, maxlen, list_classes):
    train = pd.read_csv(train_path)
    test = pd.read_csv(test_path)
    train = train.sample(frac=1)

    list_sentences_train = train["comment_text"].fillna("unknown").values
    y = train[list_classes].values
    list_sentences_test = test["comment_text"].fillna("unknown").values
    
    #list_sentences_train = hlp.clean(list_sentences_train)
    #list_sentences_test = hlp.clean(list_sentences_test)

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

def clean(words):
    # remove numbers
    words = re.sub("(^|\W)\d+($|\W)", " ", words)
    # turn to lowercase
    words = words.lower()
    #remove \n
    words = re.sub("\\n","", words)
    # remove leaky elements like ip,user
    words = re.sub("\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}","", words)
    #removing usernames
    words = re.sub("\[\[.*\]","", words)
    
    
    # TODO: Add normalization

    return words