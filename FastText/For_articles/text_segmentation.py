#!/usr/bin/env tensorflow
# encoding=utf-8
import thulac

stopkey = []
with open('data/stopWord.txt', "r", encoding='utf-8') as stopword:
    while True:
        line = stopword.readline()
        if not line:
            break
            pass
        stopkey.append(line.strip())

def thu_func_readin(filename1, filename2):
    f_text = []
    thu1 = thulac.thulac(seg_only=True)
    with open(filename1, "r", encoding='utf-8') as file_to_read:
        while True:
            lines = file_to_read.readline().strip()
            if not lines:
                break
                pass

            txt = thu1.cut(lines)
            outstr = ''
            for word in txt:
                if word[0] not in stopkey:
                    outstr += word[0]
                    outstr += ' '
            txt = outstr
            if len(txt) < 5:
                txt = "予以 采纳"
            newline = txt + "\n"
            f_text.append(newline)
            pass
        print("Process End")
        with open(filename2, "w", encoding='utf-8') as file_to_write:
            for index in f_text:
                file_to_write.write(index)
    return

thu_func_readin("data/train_seg_1.txt", "data/train_1.txt")
thu_func_readin("data/train_seg_2.txt", "data/train_2.txt")
thu_func_readin("data/train_seg_3.txt", "data/train_3.txt")
thu_func_readin("data/train_seg_4.txt", "data/train_4.txt")
