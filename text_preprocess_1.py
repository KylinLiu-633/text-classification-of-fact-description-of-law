# coding=utf-8
import re
import numpy

PATTERN_2_1 = r"[\u4e00-\u9fa5]{0,30}起诉书指控[\u4e00-\u9fa5]{0,50}罪"
PATTERN_2_2 = r"[\u4e00-\u9fa5]{0,30}因{0,50}刑事拘留"
PATTERN_3 = r"现已审理终结[\u4e00-\u9fa5|0-9|.|A-Z|a-z|.]{0,1500}(上述){0,1}(全部){0,1}(犯罪){0,1}事实"
PATTERN_6 = r"现已审理终结.{20,}"
PATTERN_6_1 = r"程序，公开开庭。.{20,}"
PATTERN_6_2 = r"被取保候审.{20,}"
PATTERN_6_3 = r"公诉机关指控.{20,}"
PATTERN_8 = r"经审理查明.{20,}"
PATTERN_8_1 = r"另外查明.{20,}"
PATTERN_9 = r"上述犯罪事实.{20,}"
PATTERN_9_1 = r"针对上述指控.{20,}"


def for_part_1(text):

    tmp = text.split("。")[0]
    # print(tmp)

    return tmp

def for_part_2(text):
    tmp = re.sub(r"，", "", text)
    tmp = re.sub(r"、", "", tmp)
    # tmp = re.sub(r"。", "", tmp)
    tmp = re.sub(r"（", "", tmp)
    tmp = re.sub(r"）", "", tmp)
    tmp = re.sub(r"[0-9]", "", tmp)
    tmp = re.sub(r"x", "某", tmp)
    tmp = re.sub(r"X", "某", tmp)
    tmp = re.sub(r"×", "某", tmp)
    tmp = re.sub(r"&times;", "", tmp)
    tmp = re.sub(r"&middot;", "", tmp)
    # print(tmp)
    tmp2 = re.compile(PATTERN_2_1).findall(tmp)
    if len(tmp2) == 0:
        tmp2 = re.compile(PATTERN_2_2).findall(tmp)

    # print(tmp2)

    tmp = ""
    for index in tmp2:
        tmp += str(index)
        tmp += "。"
    print(tmp)

    return

def for_part_3(text):
    string = re.sub(r"：", "是", text)
    string = re.sub(r":", "", string)
    string = re.sub(r"“", "", string)
    string = re.sub(r" ”", "", string)
    string = re.sub(r"”", "", string)
    string = re.sub(r"%", "百分比", string)
    string = re.sub(r"／", "每", string)
    string = re.sub(r"；", "", string)
    string = re.sub(r"．", "", string)
    # string = re.sub(r"，", "", string)
    string = re.sub(r";", "", string)
    string = re.sub(r",", "", string)
    string = re.sub(r"&", "", string)
    string = re.sub(r"！", "", string)
    string = re.sub(r"-", "", string)
    string = re.sub(r"（", "", string)
    string = re.sub(r"）", "", string)
    string = re.sub(r"《", "", string)
    string = re.sub(r"》", "", string)
    string = re.sub(r"\[", "", string)
    string = re.sub(r"]", "", string)
    string = re.sub(r"\(", "", string)
    string = re.sub(r"\)", "", string)
    string = re.sub(r"？", "", string)
    string = re.sub(r"、", "，", string)
    string = re.sub(r"x", "某", string)
    string = re.sub(r"X", "某", string)
    string = re.sub(r"�", "某", string)
    tmp = re.sub(r"  ", " ", string)

    tmp2 = re.compile(PATTERN_3).findall(tmp)
    if len(tmp2) == 0:
        print(text)
        print(tmp)
        print(tmp2)

    # tmp = ""
    # for index in tmp2:
    #     tmp += str(index)
    #     tmp += "。"
    # print(tmp)

    return

def text_clear(text):
    string = re.sub(r"：", "是", text)
    string = re.sub(r":", "", string)
    string = re.sub(r"“", "", string)
    string = re.sub(r" ”", "", string)
    string = re.sub(r"”", "", string)
    string = re.sub(r"%", "百分比", string)
    string = re.sub(r"／", "每", string)
    string = re.sub(r"；", "", string)
    string = re.sub(r"．", "", string)
    string = re.sub(r";", "", string)
    string = re.sub(r",", "", string)
    string = re.sub(r"&", "", string)
    string = re.sub(r"！", "", string)
    string = re.sub(r"-", "", string)
    string = re.sub(r"（", "", string)
    string = re.sub(r"）", "", string)
    string = re.sub(r"《", "", string)
    string = re.sub(r"》", "", string)
    string = re.sub(r"\[", "", string)
    string = re.sub(r"]", "", string)
    string = re.sub(r"\(", "", string)
    string = re.sub(r"\)", "", string)
    string = re.sub(r"？", "", string)
    string = re.sub(r"、", "，", string)
    string = re.sub(r"x", "某", string)
    string = re.sub(r"X", "某", string)
    string = re.sub(r"�", "某", string)
    string = re.sub(r"  ", " ", string)
    string = re.sub(r"独任审理", "独任审判", string)
    string = re.sub(r"现己审理终结", "现已审理终结", string)
    string = re.sub(r"现已审理完毕", "现已审理终结", string)
    string = re.sub(r"现已经审理终结", "现已审理终结", string)
    string = re.sub(r"公开开庭进行了审理", "公开开庭", string)
    string = re.sub(r"公开开庭审理了本案", "公开开庭", string)
    string = re.sub(r"不公开开庭", "公开开庭", string)
    string = re.sub(r"审理本案", "审理了本案", string)
    string = re.sub(r"现审理终结", "现已审理终结", string)
    string = re.sub(r"现本案已审理终结", "现已审理终结", string)
    string = re.sub(r"实行独任审判，", "", string)
    string = re.sub(r"程序快速审理", "程序", string)
    string = re.sub(r"另查明", "另外查明", string)
    string = re.sub(r"以上事实", "上述事实", string)
    string = re.sub(r"以上指控", "上述指控", string)
    string = re.sub(r"以上的指控", "上述指控", string)
    string = re.sub(r"以上的事实", "上述事实", string)
    string = re.sub(r"上述犯罪", "上述犯罪事实", string)
    string = re.sub(r"以上犯罪事实", "上述犯罪事实", string)
    string = re.sub(r"对指控的这一事实", "上述犯罪事实", string)
    string = re.sub(r"上述指控的事实", "上述犯罪事实", string)
    string = re.sub(r"上述指控的犯罪事实", "上述犯罪事实", string)
    string = re.sub(r"上述指控", "针对上述指控", string)
    string = re.sub(r"上述事实", "上述犯罪事实", string)
    string = re.sub(r"上述已查明的事实", "上述犯罪事实", string)
    string = re.sub(r"就指控的事实", "上述犯罪事实", string)
    string = re.sub(r"上列事实", "上述犯罪事实", string)
    string = re.sub(r"为证实所指控的犯罪事实", "上述犯罪事实", string)
    string = re.sub(r"针对指控", "针对上述指控", string)
    tmp = re.sub(r"×", "某", string)
    return tmp

def for_part_6(text):
    tmp = text
    tmp2 = re.compile(PATTERN_6).findall(tmp)
    if len(tmp2) == 0:
        tmp2 = re.compile(PATTERN_6_1).findall(tmp)
    if len(tmp2) == 0:
        tmp2 = re.compile(PATTERN_6_2).findall(tmp)
    if len(tmp2) == 0:
        tmp2 = re.compile(PATTERN_6_3).findall(tmp)
    if len(tmp2) == 0:
        return ""
    else:
        return str(tmp2[0])


def for_part_8(text):
    tmp = text
    tmp2 = re.compile(PATTERN_8).findall(tmp)
    if len(tmp2) == 0:
        tmp2 = re.compile(PATTERN_8_1).findall(tmp)
    if len(tmp2) == 0:
        return ""
    else:
        return str(tmp2[0])

def for_part_9(text):
    tmp = text
    tmp2 = re.compile(PATTERN_9).findall(tmp)
    if len(tmp2) == 0:
        tmp2 = re.compile(PATTERN_9_1).findall(tmp)
    if len(tmp2) == 0:
        return ""
    else:
        return str(tmp2[0])

if __name__ == "__main__":
    text_file = "train.txt"
    cnt = 0
    text_part_1 = []
    text_part_2 = []
    text_part_3 = []
    text_part_4 = []

    with open(text_file, "r", encoding='utf-8') as readfile:
        while True:
            line = readfile.readline().strip()
            if not line:
                break
                pass
            # if cnt > 300:
            #     break
            #     pass
            (number, text, fine, law) = line.split("\t")

            text_1 = text

            text = text_clear(text)
            text_6 = for_part_6(text)
            text_8 = for_part_8(text)
            text_9 = for_part_9(text)
            tmp1 = ""
            tmp2 = ""
            tmp3 = ""
            tmp4 = ""

            if text_8 != "":
                tmp2 = text_8.replace(text_9, "")
            else:
                tmp2 = ""
            tmp1 = text.replace(text_6, "")
            if tmp1 == "":
                tmp1 = text_1
            text_part_1.append(tmp1 + "\t" + fine)
            if tmp2 == "":
                text_part_2.append(text_1 + "\t" + fine)
            else:
                text_part_2.append(tmp2 + "\t" + fine)
            tmp3 = tmp1 + tmp2
            if tmp3 == "":
                tmp3 = text_1
            text_part_3.append(tmp3 + "\t" + fine)
            if text_6 != "":
                tmp4 = text_6.replace(text_8, "")
            if tmp4 == "":
                tmp4 = text_1
            text_part_4.append(tmp4 + "\t" + fine)

            # print(tmp1)
            # print(tmp2)
            # print(tmp3)
            # print("=========================================")
            cnt += 1

        print(cnt)

    with open("train_part_1.txt", "w", encoding='utf-8') as file1:
        for index in text_part_1:
            file1.write(index + "\n")
    with open("train_part_2.txt", "w", encoding='utf-8') as file2:
        for index in text_part_2:
            file2.write(index + "\n")
    with open("train_part_3.txt", "w", encoding='utf-8') as file3:
        for index in text_part_3:
            file3.write(index + "\n")
    with open("train_part_4.txt", "w", encoding='utf-8') as file3:
        for index in text_part_4:
            file3.write(index + "\n")

