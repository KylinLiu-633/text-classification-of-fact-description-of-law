import numpy as np
import re
import itertools
from collections import Counter
from keras.preprocessing import sequence
from keras.preprocessing.text import Tokenizer
from tflearn.data_utils import to_categorical
from keras.utils import np_utils
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.feature_extraction.text import CountVectorizer


def clean_str(string):
    """
    Tokenization/string cleaning for datasets.
    Original taken from https://github.com/yoonkim/CNN_sentence/blob/master/process_data.py
    """
    string = re.sub(r"[^A-Za-z0-9(),!?\'\`]", " ", string)
    string = re.sub(r"\'s", " \'s", string)
    string = re.sub(r"\'ve", " \'ve", string)
    string = re.sub(r"n\'t", " n\'t", string)
    string = re.sub(r"\'re", " \'re", string)
    string = re.sub(r"\'d", " \'d", string)
    string = re.sub(r"\'ll", " \'ll", string)
    string = re.sub(r",", " , ", string)
    string = re.sub(r"!", " ! ", string)
    string = re.sub(r"\(", " \( ", string)
    string = re.sub(r"\)", " \) ", string)
    string = re.sub(r"\?", " \? ", string)
    string = re.sub(r"\s{2,}", " ", string)
    return string.strip().lower()

def to_categorical(y,labels_num = 10000):
    y_n = np.zeros((labels_num))
    for k in y:
        if int(k) >= labels_num:continue
        y_n[int(k)] = 1
    return y_n

def load_text_data(l,r):
    x_file = list(open("data/converted_train_data_title_seg.txt", "r", encoding='utf8').readlines())
    labels = list(open("data/converted_train_data_id.csv", "r", encoding='utf8').readlines())

    x_dev = x_file[600000:610000]
    y_dev = labels[600000:610000]
    x_test = x_file[650000:660000]
    y_test = labels[650000:660000]
    x_train = x_file[l*100000:100000*l+100000]+x_file[r*100000:r*100000+100000]
    y_train = labels[l*100000:100000*l+100000]+x_file[r*100000:r*100000+100000]

    # 用来验证正文补充
    supplement = list(open("data/1w_all_detail.txt", "r", encoding='utf8').readlines())
    # print(supplement[:10])
    train_sup = supplement[l*100000:100000*l+100000]+supplement[r*100000:r*100000+100000]
    dev_sup = supplement[600000:610000]
    test_sup = supplement[650000:660000]
    for i in range(len(train_sup)):
        x_train[i] +=train_sup[i]
    for i in range(len(dev_sup)):
        x_dev[i]+= dev_sup[i]
    for i in range(len(test_sup)):
        x_test[i]+=test_sup[i]

    # 将标签转化为one-hot形式
    y_train = [s.strip() for s in y_train]
    for k in range(len(y_train)):
        y_train[k] = y_train[k].split("|")
        y_train[k] = list(map(int, y_train[k]))
    y_dev = [s.strip() for s in y_dev]
    for k in range(len(y_dev)):
        y_dev[k] = y_dev[k].split("|")
        y_dev[k] = list(map(int, y_dev[k]))
    y_test = [s.strip() for s in y_test]
    for k in range(len(y_test)):
        y_test[k] = y_test[k].split("|")
        y_test[k] = list(map(int, y_test[k]))

    # 将字符词语编译为数字
    tokenizer = Tokenizer(num_words=20000, lower=False)
    tokenizer.fit_on_texts(x_train)
    x_train = tokenizer.texts_to_sequences(x_train)
    x_dev = tokenizer.texts_to_sequences(x_dev)
    x_test = tokenizer.texts_to_sequences(x_test)

    return x_train, y_train, x_dev, y_dev, x_test, y_test,tokenizer
def load_data_and_labels():
    """
    Loads polarity data from files, splits the data into words and generates labels.
    Returns split sentences and labels.
    """
    # Load data from files
    # x_file = np.load('data/mini_train_data.npy')

    # x_file = np.load('data/mini_text.npy')
    x_file = np.load('data/text_50v.npy')
    # 阉割验证
    x_file = np.array(x_file[:5000])
    print(x_file.shape)

    x_file = sequence.pad_sequences(x_file, maxlen=100, dtype='float')
    # for i,k in enumerate(x_file):
    #     if x_file[1].shape[0]>2000


    # Generate labels
    # labels = list(open("./data/mini_label.txt", "r", encoding='utf8').readlines())
    labels = list(open("./data/label.txt", "r", encoding='utf8').readlines())
    # 阉割验证
    labels = labels[:5000]

    labels = [s.strip() for s in labels]
    labels = [s.split('|') for s in labels]
    labels = [to_categorical(s) for s in labels]

    return [x_file, labels]


def pad_sentences(sentences, padding_word="<PAD/>"):
    """
    Pads all sentences to the same length. The length is defined by the longest sentence.
    Returns padded sentences.
    """
    sequence_length = max(len(x) for x in sentences)
    padded_sentences = []
    for i in range(len(sentences)):
        sentence = sentences[i]
        num_padding = sequence_length - len(sentence)
        new_sentence = sentence + [padding_word] * num_padding
        padded_sentences.append(new_sentence)
    return padded_sentences


def build_vocab(sentences):
    """
    Builds a vocabulary mapping from word to index based on the sentences.
    Returns vocabulary mapping and inverse vocabulary mapping.
    """
    # Build vocabulary
    word_counts = Counter(itertools.chain(*sentences))
    # Mapping from index to word
    vocabulary_inv = [x[0] for x in word_counts.most_common()]
    vocabulary_inv = list(sorted(vocabulary_inv))
    # Mapping from word to index
    vocabulary = {x: i for i, x in enumerate(vocabulary_inv)}
    return [vocabulary, vocabulary_inv]


def build_input_data(sentences, labels, vocabulary):
    """
    Maps sentences and labels to vectors based on a vocabulary.
    """
    x = np.array([[vocabulary[word] for word in sentence] for sentence in sentences])
    y = np.array(labels)
    return [x, y]


def load_data():
    """
    Loads and preprocessed data for the dataset.
    Returns input vectors, labels, vocabulary, and inverse vocabulary.
    """
    # Load and preprocess data
    x_vec, labels = load_data_and_labels()


    return [x_vec, labels]