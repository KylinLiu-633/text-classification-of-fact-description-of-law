'''This example demonstrates the use of fasttext for text classification
Based on Joulin et al's paper:
Bags of Tricks for Efficient Text Classification
https://arxiv.org/abs/1607.01759

'''

from __future__ import print_function
import os
import numpy as np
from keras.callbacks import ModelCheckpoint
from keras.layers import Dense, Embedding
from keras.layers import GlobalAveragePooling1D
from keras.models import Sequential
from keras.optimizers import Adam
from keras.optimizers import rmsprop
from keras.preprocessing import sequence
from keras.models import load_model

from For_articles.data_helper import load_text_data, to_categorical


def create_ngram_set(input_list, ngram_value=2):
    return set(zip(*[input_list[i:] for i in range(ngram_value)]))


def add_ngram(sequences, token_indice, ngram_range=2):
    new_sequences = []
    for input_list in sequences:
        new_list = list(input_list[:])
        # new_list_2 = []
        for ngram_value in range(2, ngram_range + 1):
            for i in range(len(new_list) - ngram_value + 1):
                ngram = tuple(new_list[i:i + ngram_value])
                if ngram in token_indice:
                    new_list.append(token_indice[ngram])
                    # if len(new_list_2) < 200:
                        # new_list_2.append(token_indice[ngram])
        new_sequences.append(new_list)
        # new_sequences.append(new_list_2)

    return new_sequences


# Set parameters:
# ngram_range = 2 will add bi-grams features
ngram_range = 1
max_features = 20000
maxlen = 64
batch_size = 256
embedding_dims = 32
epochs = 20
final_f = 0.0
all_f_record = []

print('Loading data...')
# 枚举要训练的数据
train_lists = [0]
for train_list in train_lists:
    x_train, y_train, x_dev, y_dev, x_test, y_test = load_text_data(train_list)

    print(len(x_train), 'train sequences')
    print(len(x_dev), 'dev sequences')
    print(len(x_test), 'test sequences')
    print('Average train sequence length: {}'.format(np.mean(list(map(len, x_train)), dtype=int)))
    print('Average dev sequence length: {}'.format(np.mean(list(map(len, x_dev)), dtype=int)))
    print('Average test sequence length: {}'.format(np.mean(list(map(len, x_test)), dtype=int)))

    x_test = list(x_test)
    x_dev = list(x_dev)
    x_train = list(x_train)

    if ngram_range > 1:
        print('Adding {}-gram features'.format(ngram_range))
        # Create set of unique n-gram from the training set.
        ngram_set = set()
        for input_list in x_train:
            for i in range(2, ngram_range + 1):
                set_of_ngram = create_ngram_set(input_list, ngram_value=i)
                ngram_set.update(set_of_ngram)

        # Dictionary mapping n-gram token to a unique integer.
        # Integer values are greater than max_features in order
        # to avoid collision with existing features.
        start_index = max_features + 1
        token_indice = {v: k + start_index for k, v in enumerate(ngram_set)}
        indice_token = {token_indice[k]: k for k in token_indice}

        # max_features is the highest integer that could be found in the dataset.
        max_features = np.max(list(indice_token.keys())) + 1

        # Augmenting x_train and x_test with n-grams features
        x_train = add_ngram(list(x_train), token_indice, ngram_range)
        x_dev = add_ngram(list(x_dev), token_indice, ngram_range)
        x_test = add_ngram(list(x_test), token_indice, ngram_range)
        print('Average train sequence length: {}'.format(np.mean(list(map(len, x_train)), dtype=int)))
        print('Average dev sequence length: {}'.format(np.mean(list(map(len, x_dev)), dtype=int)))
        print('Average test sequence length: {}'.format(np.mean(list(map(len, x_test)), dtype=int)))

    print('Pad sequences (samples x time)')
    x_train = sequence.pad_sequences(x_train, maxlen=maxlen)
    x_dev = sequence.pad_sequences(x_dev, maxlen=maxlen)
    x_test = sequence.pad_sequences(x_test, maxlen=maxlen)
    testy = y_test
    y_train = np.array([to_categorical(s) for s in y_train])
    y_dev = np.array([to_categorical(s) for s in y_dev])
    y_test = np.array([to_categorical(s) for s in y_test])
    print('x_train shape:', x_train.shape)
    print('x_test shape:', x_dev.shape)
    print('x_test shape:', x_test.shape)

    print('Build model...')

    model = Sequential()

    # we start off with an efficient embedding layer which maps
    # our vocab indices into embedding_dims dimensions
    model.add(Embedding(max_features,
                        embedding_dims,
                        input_length=maxlen))

    # we add a GlobalAveragePooling1D, which will average the embeddings
    # of all words in the document
    model.add(GlobalAveragePooling1D())

    # We project onto a single unit output layer, and squash it with a sigmoid:
    model.add(Dense(len(y_train[1]), activation='sigmoid'))
    Rmsprop = rmsprop(lr=1e-4)
    adam = Adam(lr=1e-2, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
    model.compile(loss='categorical_crossentropy',
                  optimizer=adam,
                  metrics=['accuracy'])
    saved_path = 'model/'+str(train_list)+'/'
    if not os.path.exists(saved_path):
        os.makedirs(saved_path)
    checkpoint = ModelCheckpoint(saved_path+'weights.{epoch:03d}-{val_acc:.4f}.hdf5',
                                 monitor='val_acc', verbose=1,
                                 save_best_only=True, mode='max')
    # model.load_weights("model/0/weights.020-0.1982.hdf5")
    model.fit(x_train, y_train,
              batch_size=batch_size,
              epochs=epochs, shuffle=True,
              callbacks=[checkpoint],
              validation_data=(x_dev, y_dev))

    y_pre = model.predict(x_test)

    print("ended...")

    print(y_pre[0])

    predict_labels_list = list()
    predict_scores = list()

    real_label = y_test
    f_all = 0.0

    for ind_1 in range(y_pre.shape[0]):
        label_p = []
        tp = 0.0
        fp = 0.0
        tn = 0.0
        fn = 0.0
        for ind_2 in range(y_pre.shape[1]):
            if y_pre[ind_1][ind_2] > 0.5:
                y_pre[ind_1][ind_2] = 1
            else:
                y_pre[ind_1][ind_2] = 0
            pre_n = int(y_pre[ind_1][ind_2])
            real_n = int(real_label[ind_1][ind_2])
            if pre_n == 1:
                if pre_n == real_n:
                    tp += 1
                else:
                    fp += 1
            else:
                if pre_n == real_n:
                    tn += 1
                else:
                    fn += 1
                    label_p.append(int(ind_2))
        no_1 = float(tp)
        no_2 = float(tp + fn + fp)
        f_score = no_1 / no_2
        towrite = ""
        for index in label_p:
            towrite = towrite + str(index) + " "
        towrite = towrite + "\t" + str(f_score) + "\n"
        f_all += f_score
        all_f_record.append(towrite)

    f_all /= 24000
    final_f += f_all
    print("Precision: " + str(f_all))

# with open("pre_result_15.txt", "w", encoding="utf-8") as file_write:
#     for index in all_f_record:
#         file_write.write(index)
#     file_write.close()

# final_f /= 5.0
print("Final result: " + str(final_f))
