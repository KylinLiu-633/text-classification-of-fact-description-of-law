
import fastText

def classifier_1(model_name):
    test_file = "data/test_ft_0.txt"
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
    print("test file loadin success")

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

    for i in range(0, len(r_label)):
        real_no[r_label[i]] += 1
        predict_no[p_label[i]] += 1
        if r_label[i] == p_label[i]:
            right_no[r_label[i]] += 1

    print(right_no)
    print(real_no)
    print(predict_no)

    f1_score = 0.0
    r_no = re_no = p_no = 0.0
    for key in right_no:
        r_no += right_no[key]
    for key in real_no:
        re_no += real_no[key]
    for key in predict_no:
        p_no += predict_no[key]

    all_p = float(r_no) / float(re_no)
    all_r = float(r_no) / float(p_no)

    print("Precise:" + str(all_p))
    print("Recall: " + str(all_r))

    for key in real_no:
        try:
            r = float(right_no[key]) / float(real_no[key])
            p = float(right_no[key]) / float(predict_no[key])
            f = p * r * 2 / (p + r)
            print("%s:\t p:%f\t r:%f\t f:%f" % (key, p, r, f))
            f1_score += float(right_no[key]) * f
        except:
            print("error:", key, "right:", right_no.get(key, 0), "real:", real_no.get(key, 0), "predict:",
                  predict_no.get(key, 0))

    f1_score = f1_score / len(r_label)
    print("Micro_Average_F1_Score" + str(f1_score))

    return f1_score


model_name = "model/ft_model_9_0.bin"
# 获取测试结果

classifier_1(model_name)
