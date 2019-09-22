
if __name__ == "__main__":
    text_file = "train.txt"
    cnt = 0
    t_law = []
    new_text = []

    with open(text_file, "r", encoding='utf-8') as readfile:
        while True:
            line = readfile.readline().strip()
            if not line:
                break
                pass
            (number, text, fine, law) = line.split("\t")
            x_law = law.split(",")
            # x_law = law
            t_law.append(x_law)

        print(t_law[0])

        with open("BDCI_reproduce/train_p1.txt", "r", encoding='utf-8') as file1:
            while True:
                line = file1.readline().strip()
                if not line:
                    break
                    pass
                # if cnt > 300:
                #     break
                #     pass
                (text, fine) = line.split("\t")
                tmp = str(text)
                for index in t_law[cnt]:
                    tmp = tmp + "\t__label__" + str(index)
                new_text.append(tmp)
                # print(tmp)
                # cnt += 1

        with open("train_p_t_1.txt", "w", encoding='utf-8') as file1:
            for index in new_text:
                file1.write(index + "\n")
