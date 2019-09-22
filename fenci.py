#encoding=utf-8

import jieba
import thulac
import re

import numpy as np

def func_jieba_readin(filename1, filename2):
    f_text = []
    newline = ""
    with open(filename1, "r", encoding='utf-8') as file_to_read:
        while True:
            lines = file_to_read.readline()
            if not lines:
                break
                pass
            # (number, text, fine, law) = lines.split('\t')
            (number, text) = lines.split('\t')
            print(number)
            text = jieba.cut(text, cut_all=False)
            text = " ".join(text)
            # newline = str(number) + "\t" + text + "\t" + str(fine) + "\t" + str(law)
            newline = str(number) + "\t" + text
            # print(newline)
            f_text.append(newline)
            pass
        with open(filename2, "w", encoding='utf-8') as file_to_write:
            for index in f_text:
                file_to_write.write(index)
    return


def func_thu_readin(filename1, filename2):
    f_text = []
    newline = ""
    thu1 = thulac.thulac(seg_only=True)
    with open(filename1, "r", encoding='utf-8') as file_to_read:
        while True:
            lines = file_to_read.readline()
            if not lines:
                break
                pass
            # (number, txt, fine, law) = lines.split('\t')
            (txt, fine) = lines.split('\t')
            txt = thu1.cut(txt, text=True)
            # txt = " ".join(txt)
            newline = txt + "\t" + str(fine) + "\n"
            # print(newline)
            f_text.append(newline)
            pass

        with open(filename2, "w", encoding='utf-8') as file_to_write:
            for index in f_text:
                file_to_write.write(index)
    return

def test_file(filename1):

    cnt = 0
    with open(filename1, "r", encoding='utf-8') as file_to_read:
        while True:
            lines = file_to_read.readline()
            if not lines:
                break
                pass
            if cnt > 100:
                break
                pass

            (txt, fine) = lines.split('\t')
            print(txt)
            cnt += 1
            # txt = " ".join(txt)
            # print(newline)
            pass

    return

# test_file("train_thu.txt")
# func_readin("train.txt", "fenciTrain.txt")
# func_thu_readin("train.txt", "train_thu.txt")
func_thu_readin("train_part_1.txt", "train_thu_part_1.txt")

