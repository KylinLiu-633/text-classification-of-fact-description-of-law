# 在天之靈保佑 順利跑出結果

import numpy as np
import gensim
import multiprocessing

from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from sklearn.cross_validation import train_test_split


def get_dataset(type_no, doc_vocb):
    pos_data = []
    neg_data = []
    with open("train_small_" + str(type_no) + "_1.txt", "r", encoding='utf-8') as readfile:
        while True:
            line = readfile.readline()
            if not line:
                break
                pass
            pos_data.append(line.split())

    with open("train_small_" + str(type_no) + "_0.txt", "r", encoding='utf-8') as readfile:
        while True:
            line = readfile.readline()
            if not line:
                break
                pass
            neg_data.append(line.split())

    label_data = np.concatenate((np.ones(len(pos_data)), np.zeros(len(neg_data))))

    x_train, x_test, y_train, y_test = train_test_split(np.concatenate((pos_data, neg_data)), label_data, test_size=0.2)

    # labelize data
    labelized = []
    for index, item in enumerate(x_train):
        document = TaggedDocument(words=item, tags=["TRAIN_" + str(index+1)])
        labelized.append(document)
        doc_vocb.append(document)
    x_train = labelized

    labelized = []
    for index, item in enumerate(x_test):
        document = TaggedDocument(words=item, tags=["TEST_" + str(index+1)])
        labelized.append(document)
        doc_vocb.append(document)
    x_test = labelized

    return x_train, x_test, y_train, y_test, doc_vocb

def train(train_data, test_data, doc_vocb, size, epoch_num):
    model_dm = gensim.models.Doc2Vec(min_count=1, window=10, vector_size=size, sample=1e-3, negative=5,
                                     workers=multiprocessing.cpu_count())
    model_dbow = gensim.models.Doc2Vec(min_count=1, window=10, vector_size=size, sample=1e-3, negative=5, dm=0,
                                       workers=multiprocessing.cpu_count())
    model_dm.build_vocab(doc_vocb)
    model_dbow.build_vocab(doc_vocb)

    print("build vocabulary success")

    new_train_data = train_data

    for epoch in range(epoch_num):
        model_dm.train(new_train_data, total_examples=model_dm.corpus_count, epochs=model_dm.epochs)
        model_dbow.train(new_train_data, total_examples=model_dbow.corpus_count, epochs=model_dbow.epochs)

    new_test_data = test_data

    for epoch in range(epoch_num):
        model_dm.train(new_test_data, total_examples=model_dm.corpus_count, epochs=model_dm.epochs)
        model_dbow.train(new_test_data, total_examples=model_dbow.corpus_count, epochs=model_dbow.epochs)

    return model_dm, model_dbow


# to get result of vectors
def get_Vector(model, data, size):
    vectors = [np.array(model.docvecs[item.tags[0]]).reshape((1, size)) for item in data]
    return np.concatenate(vectors)

def get_Vectors(model_dm, model_dbow, train_data, test_data):
    train_vector_dm = get_Vector(model_dm, train_data, size)
    train_vector_dbow = get_Vector(model_dbow, train_data, size)
    train_vector = np.hstack((train_vector_dm,train_vector_dbow))

    print("Get train vector success")

    test_vector_dm = get_Vector(model_dm, test_data, size)
    test_vector_dbow = get_Vector(model_dbow, test_data, size)
    test_vector = np.hstack((test_vector_dm, test_vector_dbow))

    print("Get test vector success")

    return train_vector, test_vector

def classifier(train_vector, train_label, test_vector, test_label):
    from sklearn.linear_model import SGDClassifier

    lr = SGDClassifier(loss='log', penalty="l1")
    lr.fit(train_vector, train_label)
    print('Test Accuracy: %.2f'%lr.score(test_vector, test_label))

    return lr

def ROC_curve(lr, test_vector, test_label):
    from sklearn.metrics import roc_curve, auc
    import matplotlib.pyplot as plt

    pred_probas = lr.predict_proba(test_vector)[:, 1]

    fpr, tpr, _ = roc_curve(test_label, pred_probas)
    roc_auc = auc(fpr, tpr)
    plt.plot(fpr, tpr, label='area = %.2f'%roc_auc)
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])

    plt.show()

if __name__ == "__main__":
    size, epoch_num = 200, 5
    for index in range(5):
        type_no = index + 1
        doc_vocb = []
        train_data, test_data, train_label, test_label, doc_vocb = get_dataset(type_no, doc_vocb)

        model_dm, model_dbow = train(train_data, test_data, doc_vocb, size, epoch_num)

        model_dm.save("doc2vec_model/train_doc2vec_dm_" + str(type_no) + ".model")
        model_dbow.save("doc2vec_model/train_doc2vec_dbow_" + str(type_no) + ".model")

        print("Save model success")
        train_vector, test_vector = get_Vectors(model_dm, model_dbow, train_data, test_data)

        model_dm.save_word2vec_format("doc2vec_vec/train_doc2vec_dm_" + str(type_no) + ".vector", binary=False)
        model_dbow.save_word2vec_format("doc2vec_vec/train_doc2vec_dbow_" + str(type_no) + ".vector", binary=False)
        print("Save vector success")

        print("For type" + str(type_no))
        lr = classifier(train_vector, train_label, test_vector, test_label)
    # ROC_curve(lr, test_vector, test_label)





