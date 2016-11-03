# Bug 可能有多个父亲节点
import numpy as np
import sys
import mfcc
import record
import readwave

PENALTY = 200


class Node:
    def __init__(self, set_description="", set_is_non_emmit=False):
        self.father_ref = []    # fathers ref list
        self.cost = 0       # time t cost
        self.pre_cost = 0   # time t-1 cost
        self.backPointer = 0  # suppose all node derive from backPointerTable[0]
        self.pre_backPointer = 0  # suppose all node derive from backPointerTable[0]
        self.mean = []      # 39 dimension mfcc sequence
        self.var = []
        self.stay_trans_cost = 0
        self.leave_trans_cost = 0
        self.active = True  # puring attribute
        self.is_non_emmit = set_is_non_emmit
        self.description = set_description  # for debug


class BackPointerTableItem:
    def __init__(self, set_meaning="", set_previous_index=-1):
        self.meaning = set_meaning
        self.previous_index = set_previous_index


def trellis_generate(description_file):
    """根据描述文件description_file建立trellis"""
    trellis = []
    non_state_ref = {}
    file_read = open(description_file)
    file_read.readline()    # omit N_state info
    start_state = int(file_read.readline().split()[-1])
    end_state = int(file_read.readline().split()[-1])
    # create non-emmit state in trellis and save their references in a dictionary
    for i in range(start_state, end_state+1):
        node = Node("non_"+str(i), True)
        node.cost = sys.float_info.max
        node.pre_cost = sys.float_info.max
        node.backPointer = 0
        node.leave_trans_cost = 0
        non_state_ref[str(i)] = node
        trellis.append(node)
    # non-emmit states create done. Now begin insert isolate number speech model
    while True:
        line = file_read.readline()
        if line:
            result = line.split()
            if not result:
                continue
            if result[-1][0] != '\"':
                non_state_ref[result[2]].father_ref.append(non_state_ref[result[1]])
            else:
                # result = result[0]=edge | result[1]=from_state 0| result[2]=to_state 1| result[3]=model name "2"
                model = temp_iso_model_gen(result[1]+"_"+result[3]+"_"+result[2])
                model[0].father_ref.append(non_state_ref[result[1]])
                non_state_ref[result[2]].father_ref.append(model[-1])
                for state in model:
                    trellis.insert(trellis.index(non_state_ref[result[2]]), state)
        else:
            break
    # trellis create done.
    return trellis


def temp_iso_model_gen(name):
    """返回数字name对应的five state 模型" state_0暂无父节点 state_4 暂无孩子"""
    # construct model
    return_list = []
    digit_index = name.find('\"')
    digit_index += 1
    mean = np.loadtxt("./isolated model(100)./" + name[digit_index] + "hmm_mean.txt")
    stay_trans_cost = np.loadtxt("./isolated model(100)./" + name[digit_index] + "hmm_self_trans.txt")
    leave_trans_cost = np.loadtxt("./isolated model(100)./" + name[digit_index] + "hmm_trans_p.txt")
    var = np.loadtxt("./isolated model(100)./" + name[digit_index] + "hmm_var.txt")
    for i in range(5):
        node = Node(name+"_state_"+str(i))
        node.mean = mean[i]
        node.stay_trans_cost = stay_trans_cost[i]
        node.leave_trans_cost = leave_trans_cost[i]
        node.var = var[i]
        return_list.append(node)
    return_list[-1].stay_trans_cost = 0
    return_list[-1].leave_trans_cost = PENALTY

    for i in range(1, 5):
        return_list[i].father_ref.append(return_list[i-1])
    # return
    return return_list


def distance(vector, mean, var):
    """"Cal the Gaussian distance between input vector and the jth mean vector"""
    res = 0.5 * np.sum(np.log(2 * np.pi * var)) + 0.5 * np.sum((vector-mean)**2*1.0/var)
    return res


def back_pointer_dtw(mfcc_sequences, trellis):
    # initialize
    back_pointer_table = [BackPointerTableItem("", 0)]
    for node in trellis[1:]:    # the first node is "non_0"
        if node.father_ref[0].description == "non_0":
            node.cost = distance(mfcc_sequences[0], trellis[1].mean, trellis[1].var)
        else:
            node.cost = sys.float_info.max

    # job start from the second mfcc
    for mfcc in mfcc_sequences[1:]:
        # renew every node in trellis
        for node in trellis[1:]:
            node.pre_cost = node.cost
            node.pre_backPointer = node.backPointer
        # renew done, now cal the next incoming mfcc
        for node in trellis[1:]:
            if node.is_non_emmit:   # non-emmit state does not respond to any data
                min_father_ref = node.father_ref[0]
                for father_node in node.father_ref[1:]:
                    if min_father_ref.cost+min_father_ref.leave_trans_cost > father_node.cost+father_node.leave_trans_cost:
                        min_father_ref = father_node
                node.cost = min_father_ref.cost + min_father_ref.leave_trans_cost

                if node.cost != sys.float_info.max:
                    found = False
                    for item in back_pointer_table:
                        if item.meaning == min_father_ref.description and item.previous_index == min_father_ref.backPointer:
                            found = True
                            node.backPointer = back_pointer_table.index(item)
                            break
                    if not found:
                        back_pointer_table.append(BackPointerTableItem(min_father_ref.description, min_father_ref.backPointer))
                        node.backPointer = len(back_pointer_table)-1

                continue
            else:
                observation_cost = distance(mfcc, node.mean, node.var)
                insertion_cost = observation_cost + node.pre_cost + node.stay_trans_cost
                substitution_cost = observation_cost + node.father_ref[0].pre_cost + node.father_ref[0].leave_trans_cost
                if insertion_cost < substitution_cost:
                    # choose insertion
                    node.cost = insertion_cost
                    node.backPointer = node.pre_backPointer
                else:
                    # choose substitution
                    node.cost = substitution_cost
                    node.backPointer = node.father_ref[0].pre_backPointer

    # collect result
    pointer = trellis[-1].backPointer
    result = []
    while pointer != 0:
        result.insert(0, back_pointer_table[pointer].meaning)
        pointer = back_pointer_table[pointer].previous_index
    for i in range(len(result)):
        ob = result[i].find('\"')
        result[i] = result[i][ob+1]
    return result

# phone number check
if __name__ == '__main__':
        # labels load
        labels = []
        labels_file = open('numbers.txt')
        line = labels_file.readline()
        while line:
            line = line[:-1]
            labels.append(line)
            line = labels_file.readline()
        labels_file.close()

        # test_samples load
        test_samples = []
        for i in range(24):
            sample_file = "./jianwei_short/jianwei_short_" + str(i + 1) + ".txt"
            np_array = np.loadtxt(sample_file)
            test_samples.append(np_array.tolist())
        for i in range(26):
            sample_file = "./jianwei_long/jianwei_long_" + str(i + 1) + ".txt"
            np_array = np.loadtxt(sample_file)
            test_samples.append(np_array.tolist())

        # check accuracy
        results = []
        for sample in test_samples:
            generated_trellis = trellis_generate("network_gen.txt")
            getresult = back_pointer_dtw(sample, generated_trellis)
            format_result = ""
            for digit in getresult:
                format_result = format_result + digit
            results.append(format_result)
        import accuracy_check
        print(results)
        print(labels)
        accuracy_check.accuracy_check(results, labels, "print")

# record
# if __name__ == '__main__':
#     record.record()
# online
# if __name__ == '__main__':
#     generated_trellis = trellis_generate("network_gen.txt")
#     print('trellis construct done')
#     continuous_speech = mfcc.mfcc(record.record(), "fm")
#     mfcc_sequence = np.array(continuous_speech)
#     np.savetxt('junyo_all_.txt',mfcc_sequence)
#     continuous_speech = mfcc.mfcc(readwave.read_wave("./jianwei_long/jianwei_long_1.wav"),"fm")
#     continuous_speech += mfcc.mfcc(readwave.read_wave("./fake/1.wav"),"fm")
#     continuous_speech += mfcc.mfcc(readwave.read_wave("./fake/0.wav"),"fm")
#     continuous_speech += mfcc.mfcc(readwave.read_wave("./fake/9.wav"),"fm")
#     continuous_speech += mfcc.mfcc(readwave.read_wave("./fake/6.wav"), "fm")
#     continuous_speech += mfcc.mfcc(readwave.read_wave("./fake/7.wav"), "fm")
#     continuous_speech += mfcc.mfcc(readwave.read_wave("./fake/8.wav"), "fm")
#     getresult = back_pointer_dtw(continuous_speech, generated_trellis)
#     print(getresult)

# offline
# if __name__ == '__main__':
#     generated_trellis = trellis_generate("network_gen.txt")
#     print('trellis construct done')
#     sample_file = "./junyo_short/junyo_short_" + str(0 + 1) + ".txt"
#     continuous_speech = np.loadtxt(sample_file).tolist()
#     getresult = back_pointer_dtw(continuous_speech, generated_trellis)
#     print(getresult)
