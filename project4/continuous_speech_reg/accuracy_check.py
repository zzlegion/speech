def accuracy_check(reg_results, labels, printmode=''):
    """输入的都是字符串的list"""
    word_right = 0
    word_wrong = 0
    sentence_right = 0
    sentence_wrong = 0
    if len(reg_results) != len(labels):
        print("Error.识别结果 与 答案 长度不一致!")
    else:
        for (reg_sentence, label_sentence) in zip(reg_results, labels):
            sentence_is_right = True
            if len(reg_sentence) != len(label_sentence):
                sentence_wrong += 1
                word_wrong += len(label_sentence)
                continue
            for (digit_in_reg, digit_in_label) in zip(reg_sentence, label_sentence):
                if digit_in_reg == digit_in_label:
                    word_right += 1
                else:
                    word_wrong += 1
                    sentence_is_right = False
            if sentence_is_right:
                sentence_right += 1
            else:
                sentence_wrong += 1
    if printmode == "print":
        print("accurcy统计:")
        print("word_accuracy=", word_right, "/", word_right+word_wrong, "=", word_right/(word_wrong+word_right))
        print("sentence accuracy=", sentence_right, "/", sentence_wrong+sentence_right, "=", sentence_right/(sentence_right+sentence_wrong))
    return word_right/(word_wrong+word_right), sentence_right/(sentence_right+sentence_wrong)


def word_dtw(sample_set, match_model_set):
    """输入的是两个字符串"""
    sample = "*" + sample_set
    match_model = "*" + match_model_set
    trellis_current = [0]*len(match_model)
    trellis_pre = [0]*len(match_model)
    # 初始化
    for i in range(len(trellis_current)):
        trellis_current[i] = i
    # 开始dtw
    for sample_digit in sample[1:]:
        for index in range(len(trellis_current)):
            trellis_pre[index] = trellis_current[index]
        for i in range(len(match_model)):
            if i == 0:
                trellis_current[i] = trellis_pre[i] + 1
            else:
                insertion_cost = trellis_pre[i] + 1
                deletion_cost = trellis_current[i-1] + 1
                substitution_cost = trellis_pre[i-1]
                if sample_digit != match_model:
                    substitution_cost += 1
                trellis_current[i] = min(insertion_cost, deletion_cost, substitution_cost)
    return trellis_current[-1], len(sample), len(match_model)
