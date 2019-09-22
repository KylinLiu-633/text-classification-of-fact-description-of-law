#encoding=utf-8

import re

def pureTextExtraction(filename1, filename2, type):
    output = []
    with open(filename1, "r", encoding='utf-8') as readfile:
        while True:
            line = readfile.readline().strip()
            if not line:
                break
                pass
            if type == 1:
                (text, fine) = line.split("\t")
                text= re.sub(r"�", "某", text)
                output.append(text)
            else:
                (number, text) = line.split("\t")
                output.append(text)
    with open(filename2, "w", encoding='utf-8') as writefile:
        for index in output:
            tmp = index + "\n"
            writefile.write(tmp)
    return


def textForFastText(filename1, filename2, type):
    output = []
    with open(filename1, "r", encoding='utf-8') as readfile:
        while True:
            line = readfile.readline().strip()
            if not line:
                break
                pass
            if type == 1:
                (text, fine) = line.split("\t")
                text = text + "\t__label__" + fine + "\n"
                output.append(text)

    with open(filename2, "w", encoding='utf-8') as writefile:
        for index in output:
            tmp = index
            writefile.write(tmp)
    return


def textForFastText_noseg(filename1, filename2, type):
    output = []
    with open(filename1, "r", encoding='utf-8') as readfile:
        while True:
            line = readfile.readline().strip()
            if not line:
                break
                pass
            if type == 1:
                (text, fine ) = line.split("\t")
                text = text + "\t__label__" + fine
                output.append(text)
            else:
                (number, text) = line.split("\t")
                # text = text + "\t__label__"
                output.append(text)
    with open(filename2, "w", encoding='utf-8') as writefile:
        for index in output:
            tmp = index + "\n"
            writefile.write(tmp)
    return

# pureTextExtraction("train_stop_small.txt", "train_pure_small.txt", 1)
# pureTextExtraction("fenciTrainStop.txt", "train_pure.txt", 1)
# textForFastText("fenciTrainStop.txt", "train_ft.txt", 1)
# textForFastText_noseg("train.txt", "train_ft_noseg.txt", 1)
pureTextExtraction("train_thu_stop.txt", "train_p5.txt", 1)
