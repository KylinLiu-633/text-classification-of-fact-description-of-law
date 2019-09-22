#encoding=utf-8

import jieba
import thulac
import re

import numpy as np


stopkey = []
with open('stopWord.txt', "r", encoding='utf-8') as stopword:
    while True:
        line = stopword.readline()
        if not line:
            break
            pass
        stopkey.append(line.strip())



def func_readin(filename1, filename2):
    f_text = []
    newline = ""
    with open(filename1, "r", encoding='utf-8') as file_to_read:
        while True:
            lines = file_to_read.readline()
            if not lines:
                break
                pass
            (number, text, fine, law) = lines.split('\t')
            # (number, text) = lines.split('\t')
            print(number)
            text = text.strip()
            text = jieba.cut(text, cut_all=False)
            outstr = ''
            for word in text:
                if word not in stopkey:
                    outstr += word
                    outstr += ' '
            text = outstr
            newline = str(number) + "\t" + text + "\t" + str(fine) + "\t" + str(law)
            # newline = str(number) + "\t" + text
            # print(newline)
            f_text.append(newline)
            pass
        with open(filename2, "w", encoding='utf-8') as file_to_write:
            for index in f_text:
                file_to_write.write(index)
    return

def thu_func_readin(filename1, filename2):
    f_text = []
    thu1 = thulac.thulac(seg_only=True)
    newline = ""
    with open(filename1, "r", encoding='utf-8') as file_to_read:
        while True:
            lines = file_to_read.readline().strip()
            if not lines:
                break
                pass

            # (number, txt, fine, law) = lines.split('\t')
            (txt, fine) = lines.split('\t')
            # (number, text) = lines.split('\t')
            fine = str(fine).strip()
            txt = thu1.cut(txt)
            outstr = ''
            for word in txt:
                if word[0] not in stopkey:
                    outstr += word[0]
                    outstr += ' '
            txt = outstr
            if len(txt) < 5:
                txt = "予以 采纳"
            newline = txt + "\t__label__" + fine + "\n"
            # newline = str(number) + "\t" + text
            # print(newline)
            f_text.append(newline)
            pass
        print("Process End")
        with open(filename2, "w", encoding='utf-8') as file_to_write:
            for index in f_text:
                file_to_write.write(index)
    return

# func_readin("train.txt", "fenciTrain.txt")
# func_readin("test.txt", "fenciTestStop.txt")
# func_readin("train.txt", "fenciTrainStop.txt")
# thu_func_readin("train.txt", "train_thu_stop.txt")
thu_func_readin("train_part_1.txt", "train_p1.txt")
thu_func_readin("train_part_2.txt", "train_p2.txt")

thu_func_readin("train_part_3.txt", "train_p3.txt")
thu_func_readin("train_part_4.txt", "train_p4.txt")
