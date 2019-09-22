import re

import numpy as np


def pureTextExtraction(filename1, penalty):
    output_1 = []
    output_2 = []
    with open(filename1, "r", encoding='utf-8') as readfile:
        while True:
            line = readfile.readline().strip()
            if not line:
                break
                pass
            (text, fine) = line.split("\t")
            if int(fine) == penalty:
                output_1.append(text)
            else:
                output_2.append(text)
    writefile_name = "train" + "_" + str(penalty) + "_1.txt"
    with open(writefile_name, "w", encoding='utf-8') as writefile:
        for index in output_1:
            tmp = index + "\n"
            writefile.write(tmp)
    writefile_name = "train" + "_" + str(penalty) + "_0.txt"
    with open(writefile_name, "w", encoding='utf-8') as writefile:
        for index in output_2:
            tmp = index + "\n"
            writefile.write(tmp)
    print("Finish")

    return

def division(penalty):
    with open("train" + "_" + str(penalty) + "_1.txt", "r", encoding='utf-8') as readfile:
        output = []
        line_no = 1
        while True:
            line = readfile.readline().strip()
            if not line:
                break
                pass
            if line_no > 15000:
                break
                pass
            output.append(line)
            line_no += 1

        writefile_name = "train_small" + "_" + str(penalty) + "_1.txt"
        with open(writefile_name, "w", encoding='utf-8') as writefile:
            for index in output:
                tmp = index + "\n"
                writefile.write(tmp)
    with open("train" + "_" + str(penalty) + "_0.txt", "r", encoding='utf-8') as readfile:
        output = []
        line_no = 1
        while True:
            line = readfile.readline().strip()
            if not line:
                break
                pass
            if line_no > 15000:
                break
                pass
            output.append(line)
            line_no += 1

        writefile_name = "train_small" + "_" + str(penalty) + "_0.txt"
        with open(writefile_name, "w", encoding='utf-8') as writefile:
            for index in output:
                tmp = index + "\n"
                writefile.write(tmp)




for i in range(8):
    # pureTextExtraction("train_thu_stop.txt", i+1)
    print(i)
    division(i+1)

# division(7)
