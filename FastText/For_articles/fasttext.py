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
from keras.metrics import top_k_categorical_accuracy
from keras.models import Sequential
from keras.optimizers import Adam
from keras.optimizers import rmsprop
from keras.preprocessing import sequence
from keras.models import load_model

from data_helpers import load_text_data, to_categorical


def create_ngram_set(input_list, ngram_value=2):
    return set(zip(*[input_list[i:] for i in range(ngram_value)]))


def add_ngram(sequences, token_indice, ngram_range=2):
    new_sequences = []
    for input_list in sequences:
        new_list = list(input_list[:])
        for ngram_value in range(2, ngram_range + 1):
            for i in range(len(new_list) - ngram_value + 1):
                ngram = tuple(new_list[i:i + ngram_value])
                if ngram in token_indice:
                    new_list.append(token_indice[ngram])
        new_sequences.append(new_list)

    return new_sequences


# Set parameters:
# ngram_range = 2 will add bi-grams features
ngram_range = 2
max_features = 20000
maxlen = 64
batch_size = 256
embedding_dims = 32
epochs = 20

print('Loading data...')
# 枚举要训练的数据
train_lists = [[0,2],[0,3],[0,4],[0,5],[1,3],[1,4],[1,5],[2,3],[2,4],[2,5],[3,4],[3,5],[4,5]]
for train_list in train_lists:
    x_train, y_train, x_dev, y_dev, x_test, y_test = load_text_data(train_list[0],train_list[1])

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
    print(x_train[1])
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
    saved_path = 'saved_model/fasttext/use'+str(train_list[0])+'-'+str(train_list[1])+'/'
    if not os.path.exists(saved_path):
        os.makedirs(saved_path)
    checkpoint = ModelCheckpoint(saved_path+'weights.{epoch:03d}-{val_acc:.4f}.hdf5',
                                 monitor='val_acc', verbose=1,
                                 save_best_only=True, mode='max')
    # model.load_weights("saved_model/fasttext/use20-40/weights.004-0.3084.hdf5")
    model.fit(x_train, y_train,
              batch_size=batch_size,
              epochs=epochs, shuffle=True,
              callbacks=[checkpoint],
              validation_data=(x_dev, y_dev))


# y_pre=model.predict(x_test)
# print("ended...")
# predict_labels_list = list()
# predict_scores = list()
# label_list = list(open('data/id2to.txt',"r",encoding='utf8').readlines())
# with open("an2.txt","w",encoding='utf8') as hh:
#     pass
#
# for i in range(y_pre.shape[0]):
#     predict_scores.append(y_pre[i])
#     predict_labels = np.array(y_pre[i]).argsort()[-1:-6:-1].tolist()
#
#     # for t in len(predict_labels):
#     #     predict_labels[t]=int(predict_labels[t])
#     predict_labels_list.append(predict_labels)
#     with open("an2.txt","a",encoding='utf8') as out:
#         for k in predict_labels:
#             out.write(str(label_list[k][:-1])+" ")
#         out.write("\t||||")
#         for k in testy[i]:
#             out.write(str(label_list[k][:-1]+" "))
#         out.write("\t||||"+str(text[i][:-1])+'\n')
#         # out.write(str(predict_labels)+","+str(testy[i])+","+str(text[i])+'\n')
# predict_label_and_marked_label_list = zip(predict_labels_list, testy)
# precision, recall, f1 = calculate_score.score_eval(predict_label_and_marked_label_list)
# print('Local valid p=%g, r=%g, f1=%g' % (precision, recall, f1))
