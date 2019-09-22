# 在天之靈保佑 順利跑出結果

import logging
import fastText
import random

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

from sklearn.cross_validation import train_test_split

def print_results(N, p, r):
    print("N\t" + str(N))
    print("P@{}\t{:.3f}".format(1, p))
    print("R@{}\t{:.3f}".format(1, r))

def build_dataset(i_no):
    text_data = []
    label_data = []
    with open("train_ft.txt", "r", encoding='utf-8') as readfile:
        while True:
            line = readfile.readline().strip()
            if not line:
                break
                pass
            (text, label_fine) = line.split("\t")
            text_data.append(text)
            label_data.append(label_fine)

    x_train, x_test, y_train, y_test = train_test_split(text_data, label_data, test_size=0.2)

    train_data = []
    for index in range(len(x_train)):
        tmp = str(x_train[index]) + "\t" + str(y_train[index]) + "\n"
        train_data.append(tmp)

    with open("train_ft_" + str(i_no) + ".txt", "w", encoding='utf-8') as file_to_write:
        for index in train_data:
            file_to_write.write(index)

    test_data = []
    for index in range(len(x_test)):
        tmp = str(x_test[index]) + "\t" + str(y_test[index]) + "\n"
        test_data.append(tmp)

    with open("test_ft_" + str(i_no) + ".txt", "w", encoding='utf-8') as file_to_write:
        for index in test_data:
            file_to_write.write(index)

    return

def build_dataset_cross():
    text_data = []
    label_data = []
    with open("train_p4.txt", "r", encoding='utf-8') as readfile:
        while True:
            line = readfile.readline().strip()
            if not line:
                break
                pass
            text_data.append(line)

    random.shuffle(text_data)

    ran = 0
    for i in range(5):
        train_data = []
        test_data = []
        if ran > 0:
            for index in range(ran):
                tmp = str(text_data[index]) + "\n"
                train_data.append(tmp)

        for index in range(ran, int(ran + 24000)):
            tmp = str(text_data[index]) + "\n"
            test_data.append(tmp)

        for index in range(int(ran + 24000), 120000):
            tmp = str(text_data[index]) + "\n"
            train_data.append(tmp)

        with open("train_p4_" + str(i) + ".txt", "w", encoding='utf-8') as file_to_write:
            for index in train_data:
                file_to_write.write(index)

        with open("test_p4_" + str(i) + ".txt", "w", encoding='utf-8') as file_to_write:
            for index in test_data:
                file_to_write.write(index)

        ran += 24000

    return

def get_dataset():
    text_data = []
    label_data = []
    with open("train_ft.txt", "r", encoding='utf-8') as readfile:
        while True:
            line = readfile.readline().strip()
            if not line:
                break
                pass
            (text, label_fine) = line.split("\t")
            text_data.append(text)
            label_data.append(label_fine)

    x_train, x_test, y_train, y_test = train_test_split(text_data, label_data, test_size=0.2)

    train_data = []
    for index in range(len(x_train)):
        tmp = str(x_train[index]) + "\t" + str(y_train[index]) + "\n"
        train_data.append(tmp)

    with open("train_ft_1.txt", "w", encoding='utf-8') as file_to_write:
        for index in train_data:
            file_to_write.write(index)

    test_data = []
    for index in range(len(x_test)):
        tmp = str(x_test[index]) + "\t" + str(y_test[index]) + "\n"
        test_data.append(tmp)

    with open("test_ft_1.txt", "w", encoding='utf-8') as file_to_write:
        for index in test_data:
            file_to_write.write(index)

    return

def train(i_no,dim, epoch, minCount, wordNgrams):
    train_data = "data/train_p3_" + str(i_no) + ".txt"
    # train_data = "data/train_p_t_1.txt"
    model = fastText.train_supervised(input=train_data, label="__label__", dim=dim, epoch=epoch, minCount=minCount,
                                      wordNgrams=wordNgrams, loss="softmax")
    return model

def classifier(model, i_no):
    test_data = "data/test_ft_" + str(i_no) + ".txt"
    print_results(*model.test(test_data))

    return

def classifier_1(model_name, i_no):
    test_file = "data/test_p3_" + str(i_no) + ".txt"
    model = fastText.load_model(model_name)

    r_text = []
    r_label = []
    with open(test_file, "r", encoding='utf-8') as readfile:
        while True:
            line = readfile.readline().strip()
            if not line:
                break
                pass
            (text, label_fine) = line.split("\t")
            r_text.append(text)
            # label_fine.replace("__label__", "")
            r_label.append(label_fine)

    # print(model.predict(r_text))
    # print("test file loadin success")

    p_label = [item[0] for item in model.predict(r_text)[0]]
    print("get predict result success")

    r_label_class = list(set(r_label))
    p_label_class = list(set(p_label))

    right_no = dict.fromkeys(r_label_class, 0)  # 预测正确的各个类的数目
    real_no = dict.fromkeys(r_label_class, 0)  # 测试数据集中各个类的数目
    predict_no = dict.fromkeys(p_label_class, 0)  # 预测结果中各个类的数目

    # print(right_no.keys())
    # print(real_no.keys())
    # print(predict_no.keys())

    for i in range(len(r_label)):
        real_no[r_label[i]] += 1
        predict_no[p_label[i]] += 1
        if r_label[i] == p_label[i]:
            right_no[r_label[i]] += 1

    # print(right_no)
    # print(real_no)
    print(predict_no)

    f1_score = 0.0
    r_no = re_no = p_no = 0.0
    for key in right_no:
        r_no += right_no[key]
    for key in real_no:
        re_no += real_no[key]
    for key in predict_no:
        p_no += predict_no[key]

    all_p = 0.0
    all_r = 0.0
    for key in real_no:
        try:
            if real_no[key] == 0:
                r = 1.0
            else:
                r = float(right_no[key]) / float(real_no[key])
            if predict_no[key] == 0:
                p = 1.0
            else:
                p = float(right_no[key]) / float(predict_no[key])
            all_p += p
            all_r += r
            f = p * r * 2 / (p + r)
            # print("%s:\t p:%f\t r:%f\t f:%f" % (key, p, r, f))
            f1_score += f * float(real_no[key])
        except:
            print("error:", key, "right:", right_no.get(key, 0), "real:", real_no.get(key, 0), "predict:",
                  predict_no.get(key, 0))

    f1_score = f1_score / 24000.0
    all_p /= 8.0
    all_r /= 8.0
    print("Precise:" + str(all_p))
    print("Recall: " + str(all_r))
    print("Macro_Average_F1_Score:" + str(f1_score))

    return all_p, all_r, f1_score


if __name__ == "__main__":

    # build_dataset_cross()

    F1_score = 0.00
    Precise = 0.00
    Recall = 0.00
    for index in range(5):
        print("Model" + str(index) + ":")
        dim = 150
        epoch = 5
        minCount = 1
        wordNgrams = 1

        # build_dataset(index)
        # 获取数据集
        # get_dataset()
        # print("DataSet Build Success")
        # 训练获得模型
        model = train(index, dim, epoch, minCount, wordNgrams)
        # print("Model Training Success")
        # 保存模型
        model_name = "model/ft_thu_model_1_" + str(index) + ".bin"
        model.save_model(model_name)
        # print("Save model success")
        # 获取测试结果
        # classifier(model)
        (allp, allr, fscore) = classifier_1(model_name, index)
        Precise += allp
        Recall += allr
        F1_score += fscore

    Precise /= 5
    Recall /= 5
    print("Precise:" + str(Precise))
    print("Recall: " + str(Recall))
    print("5_fold_corss_validation result:")
    F1_score /= 5
    print(F1_score)






