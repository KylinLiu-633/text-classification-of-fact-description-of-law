#!/usr/bin/env python
# coding=utf-8

import re

# 这是初步提取罪名的程序 生成第一步粗提取的结果

PATTERN = \
    r'[\u4e00-\u9fa5]{0,20}辩解[\u4e00-\u9fa5]{0,20}'

PATTERN1 = \
    r'犯[\u4e00-\u9fa5]{2,20}罪'

PATTERN2 = \
    r'[\u4e00-\u9fa5]{2,20}'

PATTERN3 = \
    r'判决以[\u4e00-\u9fa5]{2,35}犯([\u4e00-\u9fa5]{2,20})罪'

PATTERN4 = \
    r'指控被告人[\u4e00-\u9fa5]{2,35}犯([\u4e00-\u9fa5]{2,20})罪一案'

PATTERN5 = \
    r'于[\d]{4}年[\d]{1,2}月[\d]{1,2}日'

# PATTERN = \
#     r'起诉书指控被告人[\u4e00-\u9fa5]{2,35}犯([\u4e00-\u9fa5]{2,20})罪{1,2}'
#
# PATTERN1 = \
#     r'[\u4e00-\u9fa5]{2,20}罪'
#
# PATTERN2 = \
#     r'[\u4e00-\u9fa5]{2,20}'

def clean_str(string):
    """
    Tokenization/string cleaning for all datasets except for SST.
    Original taken from https://github.com/yoonkim/CNN_sentence/blob/master/process_data.py
    """
    string = re.sub(r"[0-9]", "", string)
    string = re.sub(r"年", "", string)
    string = re.sub(r"月", "", string)
    string = re.sub(r"日", "", string)
    # string = re.sub(r"\'s", " \'s", string)
    # string = re.sub(r"\'ve", " \'ve", string)
    # string = re.sub(r"n\'t", " n\'t", string)
    # string = re.sub(r"\'re", " \'re", string)
    # string = re.sub(r"\'d", " \'d", string)
    # string = re.sub(r"\'ll", " \'ll", string)
    string = re.sub(r"“", "", string)
    string = re.sub(r" ”", "", string)
    string = re.sub(r"；", "", string)
    string = re.sub(r"，", "", string)
    string = re.sub(r";", "", string)
    string = re.sub(r",", "", string)
    string = re.sub(r"&", "", string)
    string = re.sub(r"！", "", string)
    string = re.sub(r"。", "", string)
    string = re.sub(r"-", "", string)
    string = re.sub(r"（", "", string)
    string = re.sub(r"）", "", string)
    string = re.sub(r"？", "", string)
    string = re.sub(r"、", "", string)
    string = re.sub(r"x", "某", string)
    string = re.sub(r"X", "某", string)
    string = re.sub(r"×", "某", string)


    # string = re.sub(r"\s{2,}", " ", string)
    string = re.sub(r"  ", " ", string)
    return string.strip().lower()


def output_crime_name(filename, data):
    file_to_write = open(filename, "w+", encoding='utf-8')
    file_to_write.write(data)
    file_to_write.close()

def pattern_deal(tmp1):
    tmp2 = re.sub(r"被告人", "-", str(tmp1))
    tmp2 = re.sub(r"掩饰隐瞒犯罪", "", tmp2)
    tmp2 = re.sub(r"侵犯", "亲饭", tmp2)
    tmp2 = re.sub(r"与", "", tmp2)
    tmp2 = re.sub(r"和", "", tmp2)
    # print(tmp2)
    tmp3 = re.compile(PATTERN1).findall(tmp2)

    tmp2 = re.sub(r"犯", "", str(tmp3))
    # tmp2 = re.sub(r"犯", "", tmp2)
    tmp2 = re.sub(r"罪", "-", tmp2)
    tmp2 = re.sub(r"亲饭", "侵犯", tmp2)
    tmp2 = re.sub(r"所得", "掩饰隐瞒犯罪所得", tmp2)

    return tmp2


def load_data_and_labels(filename1, filename2):
    """
    Loads MR polarity data from files, splits the data into words and generates labels.
    Returns split sentences and labels.
    """

    x_text = []
    fi = []

    nocnt = 0
    with open(filename1, "r", encoding='utf-8') as file_to_read, open(filename2, "w+", encoding='utf-8') as file_to_write:
        while True:
            lines = file_to_read.readline()
            if not lines:
                break
                pass
            # train文件处理开始
            (number, text, fine, law) = lines.split('\t')
            # text = clean_str(text)          # train文件处理结束

            # tmp1 = re.compile(PATTERN).findall(text)

            # if len(tmp1) > 1:
            #     vwrite = tmp1[0]
            #     file_to_write.write(vwrite + "\n")
            #
            # tmp1 = re.compile(PATTERN).findall(text)

            if len(text) > 1000:
                file_to_write.write(text + "\n")

            pass

    print(nocnt)
    return x_text, fi

def text_3_error():
    # test_file = "BDCI_reproduce/train_p3" \
    #             ".txt"
    test_file = "train_p5.txt"

    r_text = []
    r_label = []
    with open(test_file, "r", encoding='utf-8') as readfile:
        while True:
            line = readfile.readline().strip()
            if not line:
                break
                pass
            if len(line) < 30:
                print(line)
            # (text, label_fine) = line.split("\t")
            # r_text.append(text)
            # r_label.append(label_fine)


# 读取train的数据，生成对应的罪名表
# load_data_and_labels(r"train.txt", r"defense.txt")
text_3_error()


