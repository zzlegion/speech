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
    for i in xrange(10):  ###　数字ｉ
        ### segment_info不需要初始化, segment_info[0-4]为第0-4段的起始index，segment_info[5]为属于数字i的结束index
        matrix = np.arange(training_sequence_num * 6, dtype=int).reshape(training_sequence_num, 6)
        matrix.fill(0)
        segment_info.setdefault(i, matrix)

    ####################### mean key: 数字i  value:数字i的5个state的mean，5*39的list ######################
    initial_mean = {}
    for i in xrange(10):  ###　数字ｉ
        ### 用训好的模型初始化
        matrix = np.loadtxt(initial_model + "/" + str(i) + "hmm_mean.txt")
        initial_mean.setdefault(i, matrix)

    global GMM_mean
    GMM_mean = []
    GMM_mean.append(initial_mean)

    ####################### var key: 数字i  value:数字i的5个state的var，5*39的list ######################
    initial_var = {}
    for i in xrange(10):  ###　数字ｉ
        ### 用训好的模型初始化
        matrix = np.loadtxt(initial_model + "/" + str(i) + "hmm_var.txt")
        initial_var.setdefault(i, matrix)

    global GMM_var
    GMM_var = []
    GMM_var.append(initial_var)

    ####################### GMM_weight key: 数字i  value:数字i的5个state的weight ######################
    global GMM_weight
    GMM_weight = {}
    for i in xrange(10):
        matrix = np.arange(5 * 1.0).reshape(5, 1)
        matrix.fill(1)  ### 似乎np.fill只能传进整数，传入0.5则全为0
        GMM_weight.setdefault(i, matrix)

    ####################### trans_p key: 数字i  value:数字i的transition p，1*5的list ######################
    global trans_p
    trans_p = {}
    for i in xrange(10):  ###　数字ｉ
        ### 用训好的模型初始化
        matrix = np.loadtxt(initial_model + "/" + str(i) + "hmm_trans_p.txt")
        trans_p.setdefault(i, matrix)
    ####################### self_trans key: 数字i  value:数字i的transition p，1*5的list ######################
    global self_trans
    self_trans = {}
    for i in xrange(10):  ###　数字ｉ
        ### 用训好的模型初始化
        matrix = np.loadtxt(initial_model + "/" + str(i) + "hmm_self_trans.txt")
        self_trans.setdefault(i, matrix)


def split(epsilon):  ## 设定epsilon为mean的百分之多少

    low_ratio = 1 - epsilon
    high_ratio = 1 + epsilon

    current_gauss_num = len(GMM_mean)
    for gauss in xrange(current_gauss_num):
        # 左边高斯的mean
        # key: 数字i
        # value:数字i的5个state的mean，5 * 39的list
        split_mean_left = {}
        for i in xrange(10):  ###　数字ｉ
            matrix = GMM_mean[gauss * 2][i] * low_ratio
            split_mean_left.setdefault(i, matrix)

        # 右边高斯的mean
        # key: 数字i
        # value:数字i的5个state的mean，5 * 39的list
        split_mean_right = {}
        for i in xrange(10):  ###　数字ｉ
            matrix = GMM_mean[gauss * 2][i] * high_ratio
            split_mean_right.setdefault(i, matrix)

        # 左边高斯的var，初始化时var和initial_var是一样的
        # key: 数字i
        # value:数字i的5个state的var，5 * 39的list
        split_var_left = {}
        for i in xrange(10):  ###　数字ｉ
            matrix = GMM_var[gauss * 2][i] * 1
            split_var_left.setdefault(i, matrix)

        # 右边高斯的var
        # key: 数字i
        # value:数字i的5个state的var，5 * 39的list
        split_var_right = {}
        for i in xrange(10):  ###　数字ｉ
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
    for i in xrange(10):
        matrix = np.arange(5 * current_gauss_num * 2 * 1.0).reshape(5, current_gauss_num * 2)
        for gauss in xrange(current_gauss_num):
            matrix[:, gauss * 2] = GMM_weight[i][:, gauss] * 0.5
            matrix[:, gauss * 2 + 1] = GMM_weight[i][:, gauss] * 0.5
        del GMM_weight[i]
        GMM_weight.setdefault(i, matrix)


def clustering():
    current_gauss_num = len(GMM_mean)
    for digit in xrange(10):  ### 对于每个数字
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
                        print 'wrong gauss',gauss,'has no frames'
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
    trellis = np.zeros(50 * 2).reshape(2, 50)  ###############  trellis y轴有5*10个state，10个数字， x轴有2列
    cur_paths = []  # 每个node都关联一条path，每条path用list表示，paths是5个path的集合
    pre_paths = []
    changed = False  # 这次迭代是否改变了segment_info
    current_gauss_num = len(GMM_mean)
    for sequence_index in xrange(training_sequence_num):
        ###################### 对第sequence_index条训练录音操作 ################################
        length = num_of_frames[sequence_index]
        label = labels[sequence_index]  ###### 保存label为局部变量，一直访问global labels影响效率
        ####################### 初始化path #########################################
        pre_paths[:] = []
        cur_paths[:] = []
        for i in xrange(50):
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
            for node_index in xrange(1, 50):  # state 1 -- 49
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
                if segment_info[digit][sequence_index][5] != j:
                    segment_info[digit][sequence_index][5] = j
                    changed = True
            if cur_paths[49][j] != cur_paths[49][j - 1]:  #### 如果 j 是state的起始index
                state = cur_paths[49][j]

                if state == 0:  ###　说明是下一个数字的开始
                    if segment_info[digit][sequence_index][5] != j - 1:  ### 将j-1保存为上一个数字的结尾index
                        segment_info[digit][sequence_index][5] = j - 1
                        changed = True
                    digit = it.next()
                if segment_info[digit][sequence_index][state] != j:  ### 用segment info来记录state变化的地方
                    segment_info[digit][sequence_index][state] = j
                    changed = True
    np.savetxt("trellis.txt", trellis)
    print 'segment done'
    return changed


def align(train_data, initial_model, iteration=10):
    global num_of_frames
    num_of_frames = []
    global all_training_sequences
    all_training_sequences = []
    global labels
    labels = []  ### labels[i]是一个list，对应第i条语音的label

    ### 俊优的train sequence
    for i in xrange(1, 7):
        if i == 1:
            labeltmp = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
        elif i == 2:
            labeltmp = [9, 8, 7, 6, 5, 4, 3, 2, 1, 0]
        elif i == 3:
            labeltmp = [1, 2, 3, 4, 5, 6, 7, 8, 9, 0]
        elif i == 4:
            labeltmp = [0, 9, 8, 7, 6, 5, 4, 3, 2, 1, ]
        elif i == 5:
            labeltmp = [1, 3, 5, 7, 9, 0, 2, 4, 6, 8]
        elif i == 6:
            labeltmp = [8, 6, 4, 2, 0, 9, 7, 5, 3, 1]
        for index in train_data[0]:
            sequence = np.loadtxt("..\\speech data\\new_record\\junyo_train\\junyo_" + str(i) + "_" + str(index) + ".txt")
            num_of_frames.append(len(sequence))
            all_training_sequences.append(sequence)
            labels.append(labeltmp)

    ### 健炜的train sequence
    for i in xrange(1, 7):
        if i == 1:
            labeltmp = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
        elif i == 2:
            labeltmp = [9, 8, 7, 6, 5, 4, 3, 2, 1, 0]
        elif i == 3:
            labeltmp = [1, 2, 3, 4, 5, 6, 7, 8, 9, 0]
        elif i == 4:
            labeltmp = [0, 9, 8, 7, 6, 5, 4, 3, 2, 1, ]
        elif i == 5:
            labeltmp = [1, 3, 5, 7, 9, 0, 2, 4, 6, 8]
        elif i == 6:
            labeltmp = [8, 6, 4, 2, 0, 9, 7, 5, 3, 1]
        for index in train_data[0]:
            sequence = np.loadtxt(
                "..\\speech data\\new_record\\jianwei_train\\jianwei_" + str(i) + "_" + str(index) + ".txt")
            num_of_frames.append(len(sequence))
            all_training_sequences.append(sequence)
            labels.append(labeltmp)

    initialize(initial_model)
    split(0.01)
    for ite in xrange(iteration):
        print iteration, ite
        segment()
        clustering()
    split(0.01)
    for ite in xrange(iteration):
        print iteration, ite
        segment()
        clustering()

    for digit in xrange(10):
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
    del num_of_frames
    del all_training_sequences
    del labels


if __name__ == '__main__':
    train_data = []
    train_data.append([1, 2, 3])
    train_data.append([])
    training_sequence_num = len(train_data[0]) * 6 + len(train_data[1])*6
    align(train_data, "../speech model/hmm continuous", 10)
