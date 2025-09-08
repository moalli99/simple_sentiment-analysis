import re
import numpy as np
from collections import Counter
from nltk.corpus import stopwords
import nltk

nltk.download('stopwords')


# function to clean the text
def process_(text):
    # remove punctuation and special characters
    text = re.sub(r'[^\w\s]', '', text)
    # remove digits
    text = re.sub(r"[\d]", '', text)
    # replace all runs of whitespaces with a single space
    text = re.sub(r"\s+", ' ', text)
    # convert to lowercase
    return text.lower()


# main tokenization function
def tockenize(x_train, y_train, x_val, y_val,x_test,y_test, vocab_size):
    word_list = []
    stopwrds = set(stopwords.words('english'))

    # preprocess all sentences first
    x_train = [process_(s) for s in x_train]
    x_val = [process_(s) for s in x_val]
    x_test = [process_(s) for s in x_test]

    # build vocab from training set only
    word_list = []
    for sentence in x_train:
        for word in sentence.split():
            if word and word not in stopwrds:
                word_list.append(word)
    # count word frequencies
    corpus = Counter(word_list)
    # select most common words up to vocab_size
    corpus_ = {w for w, _ in corpus.most_common(vocab_size)}
    # create dictionary mapping word -> index (starting from 1)
    word_dic = {word: i + 1 for i, word in enumerate(corpus_)}

    # function to encode a list of sentences
    def encode_sentence(sentences):
        encoded = []
        for sent in sentences:
            encoded.append([
                word_dic[process_(word)]
                for word in sent.split()
                if process_(word) in word_dic
            ])
        return encoded

    # encode training and validation sentences
    list_x_train = encode_sentence(x_train)
    list_x_val = encode_sentence(x_val)
    list_x_test = encode_sentence(x_test)

    # encode labels
    y_train_enc = [1 if label == 'positive' else 0 for label in y_train]
    y_val_enc = [1 if label == 'positive' else 0 for label in y_val]
    y_test_enc = [1 if label == 'positive' else 0 for label in y_test]

    return list_x_train, y_train_enc, list_x_val, y_val_enc, list_x_test, y_test_enc, word_dic


# padding function
def padding(sentences, seq_len):
    features = np.zeros((len(sentences), seq_len), dtype=int)
    for i, rev in enumerate(sentences):
      if len(rev):
        features[i, -len(rev):] = np.array(rev[:seq_len])
    return features

def buile_embedding_matrix(word_dic,glove_path,embedding_dim):
    # load GloVe embeddings
    embedding_index = {}
    with open(glove_path,'r',encoding='utf8') as f:
        for line in f:
            values=line.split()
            word=values[0]
            vector=np.asarray(values[1:],dtype='float32')
            embedding_index[word]=vector
    # create embedding matrix
    embedding_matrix = np.zeros((len(word_dic) + 1, embedding_dim))
    for word,idx in word_dic.items():
        vector=embedding_index.get(word)
        if vector is not None:
            embedding_matrix[idx]=vector
        else:
            embedding_matrix[idx]=np.random.normal(scale=.6,size=(embedding_dim,))
    return embedding_matrix 
