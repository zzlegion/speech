# coding: -utf8
import sys
import numpy as np
import profile
from time import time
import os
import mfcc
import copy


def distance_GMM(vector, digit, state, current_gauss_num):  ## digit 某个数字，第j个state
    """"Cal the Gaussian distance between input vector and the jth mean vector"""
    # res = 0.5 * np.sum(np.log(2 * np.pi * var[digit][j]))+ 0.5* np.sum((vector-mean[digit][j])**2*1.0/var[digit][j])
    sum = 0
    for gauss in xrange(current_gauss_num):
        part1 = GMM_weight[digit][state][gauss] * 1.0 / np.sqrt(np.prod(2 * np.pi * GMM_var[gauss][digit][state]))
        part2 = np.exp(
            -0.5 * np.sum((vector - GMM_mean[gauss][digit][state]) ** 2 * 1.0 / GMM_var[gauss][digit][state]))
        sum += part1 * part2
    if sum == 0:
        return sys.maxint
    return - np.log(sum)


def initialize(initial_model):  ### 传入参数为初始化model的文件夹名称。定义各个变量 segment_info self_trans trans_p mean var
    ####################### segment_info key:数字i value:数字i的分段信息，大小为训练数据数量*5的list ###################
    global segment_info
    segment_info = {}
    for i in xrange(11):  ###　数字ｉ，最后一个是oh
        ### segment_info不需要初始化, segment_info[0-4]为第0-4段的起始index，segment_info[5]为属于数字i的结束index
        matrix = np.arange(training_sequence_num * 6, dtype=int).reshape(training_sequence_num, 6)
        matrix.fill(0)
        segment_info.setdefault(i, matrix)

    ####################### mean key: 数字i  value:数字i的5个state的mean，5*39的list ######################
    initial_mean = {}
    for i in xrange(11):  ###　数字ｉ，最后一个是oh
        ### 用训好的模型初始化
        matrix = np.loadtxt(initial_model + "/" + str(i) + "hmm_mean.txt")
        initial_mean.setdefault(i, matrix)

    global GMM_mean
    GMM_mean = []
    GMM_mean.append(initial_mean)

    ####################### var key: 数字i  value:数字i的5个state的var，5*39的list ######################
    initial_var = {}
    for i in xrange(11):  ###　数字ｉ
        ### 用训好的模型初始化
        matrix = np.loadtxt(initial_model + "/" + str(i) + "hmm_var.txt")
        initial_var.setdefault(i, matrix)

    global GMM_var
    GMM_var = []
    GMM_var.append(initial_var)

    ####################### GMM_weight key: 数字i  value:数字i的5个state的weight ######################
    global GMM_weight
    GMM_weight = {}
    for i in xrange(11):
        matrix = np.arange(5 * 1.0).reshape(5, 1)
        matrix.fill(1)  ### 似乎np.fill只能传进整数，传入0.5则全为0
        GMM_weight.setdefault(i, matrix)

    ####################### trans_p key: 数字i  value:数字i的transition p，1*5的list ######################
    global trans_p
    trans_p = {}
    for i in xrange(11):  ###　数字ｉ
        ### 用训好的模型初始化
        matrix = np.loadtxt(initial_model + "/" + str(i) + "hmm_trans_p.txt")
        trans_p.setdefault(i, matrix)
    ####################### self_trans key: 数字i  value:数字i的transition p，1*5的list ######################
    global self_trans
    self_trans = {}
    for i in xrange(11):  ###　数字ｉ
        ### 用训好的模型初始化
        matrix = np.loadtxt(initial_model + "/" + str(i) + "hmm_self_trans.txt")
        self_trans.setdefault(i, matrix)

def initialize_silence(initial_model):
    

def split(epsilon):  ## 设定epsilon为mean的百分之多少

    low_ratio = 1 - epsilon
    high_ratio = 1 + epsilon

    current_gauss_num = len(GMM_mean)
    for gauss in xrange(current_gauss_num):
        # 左边高斯的mean
        # key: 数字i
        # value:数字i的5个state的mean，5 * 39的list
        split_mean_left = {}
        for i in xrange(11):  ###　数字ｉ
            matrix = GMM_mean[gauss * 2][i] * low_ratio
            split_mean_left.setdefault(i, matrix)

        # 右边高斯的mean
        # key: 数字i
        # value:数字i的5个state的mean，5 * 39的list
        split_mean_right = {}
        for i in xrange(11):  ###　数字ｉ
            matrix = GMM_mean[gauss * 2][i] * high_ratio
            split_mean_right.setdefault(i, matrix)

        # 左边高斯的var，初始化时var和initial_var是一样的
        # key: 数字i
        # value:数字i的5个state的var，5 * 39的list
        split_var_left = {}
        for i in xrange(11):  ###　数字ｉ
            matrix = GMM_var[gauss * 2][i] * 1
            split_var_left.setdefault(i, matrix)

        # 右边高斯的var
        # key: 数字i
        # value:数字i的5个state的var，5 * 39的list
        split_var_right = {}
        for i in xrange(11):  ###　数字ｉ
            matrix = GMM_var[gauss * 2][i] * 1
            split_var_right.setdefault(i, matrix)

        ############################ 把原来的高斯删掉，插入两个新的分裂后的高斯 ################
        del GMM_mean[gauss * 2]
        GMM_mean.insert(gauss * 2, split_mean_right)
        GMM_mean.insert(gauss * 2, split_mean_left)

        del GMM_var[gauss * 2]
        GMM_var.insert(gauss * 2, split_var_right)
        GMM_var.insert(gauss * 2, split_var_left)

        del split_mean_left
        del split_var_left
        del split_mean_right
        del split_var_right

    ####### 把原来的高斯weight分成两半 ###############
    for i in xrange(11):
        matrix = np.arange(5 * current_gauss_num * 2 * 1.0).reshape(5, current_gauss_num * 2)
        for gauss in xrange(current_gauss_num):
            matrix[:, gauss * 2] = GMM_weight[i][:, gauss] * 0.5
            matrix[:, gauss * 2 + 1] = GMM_weight[i][:, gauss] * 0.5
        del GMM_weight[i]
        GMM_weight.setdefault(i, matrix)


def clustering():
    current_gauss_num = len(GMM_mean)
    for digit in xrange(11):  ### 对于每个数字
        for state in xrange(5):  ### 对于每个state
            ################################## 聚类 ########################################
            distance = [0] * current_gauss_num
            mean_local = []  ## 防止重复访问global变量影响效率
            var_local = []
            weight_local = []
            for gauss in xrange(current_gauss_num):
                mean_local.append(GMM_mean[gauss][digit][state])
                var_local.append(GMM_var[gauss][digit][state])
                weight_local.append(GMM_weight[digit][state][gauss])

            changed = True
            clusters = []  ## clusters[i]为一个list，里面包含第i个cluster的index, (sequence_index,index)
            pre_clusters = []
            for gauss in xrange(current_gauss_num):
                clusters.append([])
                pre_clusters.append([])

            while (changed):
                for gauss in xrange(current_gauss_num):
                    pre_clusters[gauss] = clusters[gauss][:]
                    clusters[gauss][:] = []

                for sequence_index in xrange(training_sequence_num):  ### 对于每个training sequence
                    # 判断这个sequence里面有没有这个digit，如果没有，continue
                    label = labels[sequence_index]
                    if digit not in label:
                        continue

                    start = int(segment_info[digit][sequence_index][state])  ## state开始的index
                    if state == 4:  ## 如果是最后一个state
                        end = int(segment_info[digit][sequence_index][state + 1])  ## state的最后一个index, end 属于state
                    else:
                        end = int(segment_info[digit][sequence_index][state + 1]) - 1

                    for index in xrange(start, end + 1):  # [start,end]
                        vector = all_training_sequences[sequence_index][index]
                        for gauss in xrange(current_gauss_num):
                            distance[gauss] = 0.5 * np.sum(np.log(2 * np.pi * var_local[gauss])) + \
                                              0.5 * np.sum((vector - mean_local[gauss]) ** 2 * 1.0 / var_local[gauss]) - \
                                              np.log(weight_local[gauss])
                        # distance1 = 0.5 * np.sum(np.log(2 * np.pi * var1))+ 0.5* np.sum((vector - mean1)**2*1.0/var1)- np.log(weight1)
                        # distance2 = 0.5 * np.sum(np.log(2 * np.pi * var2)) + 0.5 * np.sum((vector - mean2) ** 2 * 1.0 / var2) - np.log(weight2)
                        cluster_index = distance.index(min(distance))
                        clusters[cluster_index].append((sequence_index, index))

                changed = False
                for gauss in xrange(current_gauss_num):
                    if pre_clusters[gauss] != clusters[gauss]:
                        changed = True
                # print 'pre',pre_clusters
                # print 'len',len(pre_clusters[0]),len(pre_clusters[1])
                # print 'cur',clusters

                size_all = 0
                for gauss in xrange(current_gauss_num):
                    size_all += len(clusters[gauss])

                for gauss in xrange(current_gauss_num):
                    size = len(clusters[gauss])
                    if size == 0:
                        print 'wrong gauss', gauss, 'has no frames'
                        sys.exit()
                    mean_tmp = np.arange(39, dtype=float)
                    mean_tmp.fill(0.)
                    for tuple in clusters[gauss]:
                        sequence_index = tuple[0]
                        index = tuple[1]
                        mean_tmp += all_training_sequences[sequence_index][index]

                    mean_tmp = mean_tmp * 1.0 / size
                    GMM_mean[gauss][digit][state] = mean_tmp

                    var_tmp = np.arange(39, dtype=float)
                    var_tmp.fill(0.)
                    for tuple in clusters[gauss]:
                        sequence_index = tuple[0]
                        index = tuple[1]
                        var_tmp += (all_training_sequences[sequence_index][index] - mean_tmp) ** 2

                    var_tmp = var_tmp * 1.0 / size
                    GMM_var[gauss][digit][state] = var_tmp
                    del mean_tmp
                    del var_tmp

                    GMM_weight[digit][state][gauss] = size * 1.0 / size_all

    print 'cluster done'


def segment():
    cur_paths = []  # 每个node都关联一条path，每条path用list表示，paths是5个path的集合
    pre_paths = []
    current_gauss_num = len(GMM_mean)
    for sequence_index in xrange(training_sequence_num):
        ###################### 对第sequence_index条训练录音操作 ################################
        length = num_of_frames[sequence_index]
        label = labels[sequence_index]  ###### 保存label为局部变量，一直访问global labels影响效率
        num_of_digit = len(label)
        trellis = np.zeros(5 * num_of_digit * 2).reshape(2, 5 * num_of_digit)
        ####################### 初始化path #########################################
        pre_paths[:] = []
        cur_paths[:] = []
        for i in xrange(num_of_digit * 5):
            pre_paths.append([i])
            cur_paths.append([])
        ####################### 初始化trellis ######################################
        trellis.fill(sys.maxint)
        vector = all_training_sequences[sequence_index][0]  # 第i个sequence的第0个mfcc向量
        trellis[0][0] = distance_GMM(vector, label[0], 0, current_gauss_num)  # trellis[0][0]为vector和state0的距离
        ####################### 计算trellis ########################################
        for index in xrange(1, length):
            vector = all_training_sequences[sequence_index][index]
            digit0 = label[0]  ############### 第1个数字
            trellis[1][0] = trellis[0][0] + distance_GMM(vector, digit0, 0, current_gauss_num) + self_trans[digit0][
                0]  # 计算每一列的第一个元素,只能从上一列的第一个元素得到
            cur_paths[0] = pre_paths[0][:]  # 用slice 新建一个list 并copy prepaths[0]的内容。copy list 方法里 slice最快
            cur_paths[0].append(0)  # 点trellis[1][0]的path 只能是 [0 0]

            #################### 计算trellis 的一列 ###############################
            for node_index in xrange(1, num_of_digit * 5):  # state 1 -- 49
                state = node_index % 5  ####### node_index 0-4 表示第1个数字的5个状态，所以node_index % 5 表示当前处理的数字的第几个状态
                digit = node_index / 5  ####### node_index/5 表示处理到第几个数字
                digit = label[digit]  ####### 从label中读入当前处理的是数字几
                node_cost = distance_GMM(vector, digit, state, current_gauss_num)
                stay_edge_cost = self_trans[digit][state]
                move_edge_cost = trans_p[digit][state - 1]
                stay_cost = trellis[0][node_index] + node_cost + stay_edge_cost
                move_cost = trellis[0][node_index - 1] + node_cost + move_edge_cost
                if stay_cost < move_cost:
                    trellis[1][node_index] = stay_cost
                    cur_paths[node_index] = pre_paths[node_index][:]
                    cur_paths[node_index].append(state)  ## 把当前节点加入path中
                else:
                    trellis[1][node_index] = move_cost
                    cur_paths[node_index] = pre_paths[node_index - 1][:]  ## 复制上一节点的path
                    cur_paths[node_index].append(state)  ## 把当前节点加入path中
            #################### 将 trellis[1][:] 复制到 trellis[0][:] ################
            trellis[0][:] = trellis[1][:]
            pre_paths[:] = cur_paths[:]

        ##################### cur_paths[49]即为分段的best path ################################
        it = iter(label)  #### 用迭代器访问label
        digit = it.next()  #####从第1个label开始
        segment_info[digit][sequence_index][0] = 0
        for j in xrange(1, length):
            if j == length - 1:  ###　j为sequence的结尾，保存j为最后一个数字的结尾index
                segment_info[digit][sequence_index][5] = j

            if cur_paths[num_of_digit * 5 - 1][j] != cur_paths[num_of_digit * 5 - 1][j - 1]:  #### 如果 j 是state的起始index
                state = cur_paths[num_of_digit * 5 - 1][j]

                if state == 0:  ###　说明是下一个数字的开始
                    segment_info[digit][sequence_index][5] = j - 1  ### 将j-1保存为上一个数字的结尾index
                    digit = it.next()
                segment_info[digit][sequence_index][state] = j  ### 用segment info来记录state变化的地方

    print 'segment done'


def update_parameters():
    Nj = [[0 for col in xrange(5)] for row in xrange(11)]  ### Nj[digit][i] 为数字digit的第i个segment的帧数

    for k in xrange(training_sequence_num):
        label = labels[k]
        num_digit = len(label)
        for digit_index in xrange(num_digit):
            digit = label[digit_index]
            for i in xrange(5):
                if i < 4:
                    Nj[digit][i] += segment_info[digit][k][i + 1] - segment_info[digit][k][i]
                else:
                    Nj[digit][i] += segment_info[digit][k][5] - segment_info[digit][k][4] + 1

    #### 根据segment_info 更新 transition cost
    for digit in xrange(11):
        for i in xrange(5):
            trans_p[digit][i] = sequence_num_digit[digit] * 1.0 / Nj[digit][i]
            trans_p[digit][i] = - np.log(trans_p[digit][i])
    for digit in xrange(11):
        for i in xrange(5):
            self_trans[digit][i] = 1 - sequence_num_digit[digit] * 1.0 / Nj[digit][i]
            self_trans[digit][i] = -np.log(self_trans[digit][i])

    print 'update parameters done'


if __name__ == '__main__':
    # global num_of_frames
    num_of_frames = []
    # global all_training_sequences
    all_training_sequences = []
    # global labels
    labels = []  ### labels[i]是一个list，对应第i条语音的label
    # global training_sequence_num
    training_sequence_num = 0
    sequence_num_digit = [0] * 11  # 包含每个digit的sequence数量。sequence_num_digit[i]代表包含i的sequence数量

    f_label = open('../hwdata/smallTRAIN.transcripts', 'r')
    f_name = open('../hwdata/smallTRAIN.filelist', 'r')
    #### 将oh 标记为 10 ############
    pattern = {'zero': 0, 'oh': 10, 'one': 1, 'two': 2, 'three': 3, 'four': 4, 'five': 5,
               'six': 6, 'seven': 7, 'eight': 8, 'nine': 9}
    
    for file in f_name:
        file = file.split()[0]
        sequence = np.loadtxt('../train_mfcc/' + file + '.txt')
        num_of_frames.append(len(sequence))
        all_training_sequences.append(sequence)
        label_tmp = f_label.readline()
        label_tmp = label_tmp.split()
        del label_tmp[0]
        del label_tmp[-1]
        del label_tmp[-1]
        label_tmp = [pattern[x] if x in pattern else x for x in label_tmp]

        # 计算包含某个digit的sequence数量
        length_label = len(label_tmp)
        for i in xrange(length_label):
            digit = label_tmp[i]
            sequence_num_digit[digit] += 1

        labels.append(label_tmp)
        print file
        training_sequence_num += 1

    initialize('../speech model/hmm continuous')
    iteration = 10

    split(0.01)
    for ite in xrange(iteration):
        print iteration, ite
        segment()
        update_parameters()
        clustering()
    split(0.01)
    for ite in xrange(iteration):
        print iteration, ite
        segment()
        update_parameters()
        clustering()

    for digit in xrange(11):
        np.savetxt(str(digit) + "gmm_mean1.txt", GMM_mean[0][digit])
        np.savetxt(str(digit) + "gmm_mean2.txt", GMM_mean[1][digit])
        np.savetxt(str(digit) + "gmm_mean3.txt", GMM_mean[2][digit])
        np.savetxt(str(digit) + "gmm_mean4.txt", GMM_mean[3][digit])
        np.savetxt(str(digit) + "gmm_var1.txt", GMM_var[0][digit])
        np.savetxt(str(digit) + "gmm_var2.txt", GMM_var[1][digit])
        np.savetxt(str(digit) + "gmm_var3.txt", GMM_var[2][digit])
        np.savetxt(str(digit) + "gmm_var4.txt", GMM_var[3][digit])
        np.savetxt(str(digit) + "gmm_self_trans.txt", self_trans[digit])
        np.savetxt(str(digit) + "gmm_trans_p.txt", trans_p[digit])
        np.savetxt(str(digit) + "gmm_segment_info.txt", segment_info[digit])
        np.savetxt(str(digit) + "gmm_weight.txt", GMM_weight[digit])
