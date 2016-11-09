#coding: -utf8
import sys
import numpy as np
import profile
from time import time
import os
import mfcc
import copy

def distance(vector,digit,j):   ## digit 某个数字，第j个state
    """"Cal the Gaussian distance between input vector and the jth mean vector"""
    res = 0.5 * np.sum(np.log(2 * np.pi * var[digit][j]))+ 0.5* np.sum((vector-mean[digit][j])**2*1.0/var[digit][j])
    return res

def initialize(initial_model): ### 定义各个变量 segment_info self_trans trans_p mean var
    ####################### segment_info key:数字i value:数字i的分段信息，大小为训练数据数量*5的list ###################
    global segment_info
    segment_info = {}
    for i in xrange(11):  ###　数字ｉ
        ### segment_info不需要初始化, segment_info[0-4]为第0-4段的起始index，segment_info[5]为属于数字i的结束index
        matrix = np.arange(6 * training_sequence_num).reshape(training_sequence_num, 6)
        matrix = matrix % 6
        matrix.dtype = int
        segment_info.setdefault(i,matrix)

    ####################### mean key: 数字i  value:数字i的5个state的mean，5*39的list ######################
    global mean
    mean = {}
    for i in xrange(11):  ###　数字ｉ
        ### 用训好的模型初始化
        matrix = np.loadtxt(initial_model+"/"+str(i)+"hmm_mean.txt")
        mean.setdefault(i, matrix)
    ####################### var key: 数字i  value:数字i的5个state的var，5*39的list ######################
    global var
    var = {}
    for i in xrange(11):  ###　数字ｉ
        ### 用训好的模型初始化
        matrix = np.loadtxt(initial_model+"/"+str(i)+"hmm_var.txt")
        var.setdefault(i, matrix)
    ####################### trans_p key: 数字i  value:数字i的transition p，1*5的list ######################
    global trans_p
    trans_p = {}
    for i in xrange(11):  ###　数字ｉ
        ### 用训好的模型初始化
        matrix = np.loadtxt(initial_model+"/"+str(i)+"hmm_trans_p.txt")
        trans_p.setdefault(i, matrix)
    ####################### self_trans key: 数字i  value:数字i的transition p，1*5的list ######################
    global self_trans
    self_trans = {}
    for i in xrange(11):  ###　数字ｉ
        ### 用训好的模型初始化
        matrix = np.loadtxt(initial_model+"/"+str(i)+"hmm_self_trans.txt")
        self_trans.setdefault(i, matrix)

def update_parameters():
    global Nj
    Nj = [[0 for col in xrange(5)] for row in xrange(11)] ### Nj[digit][i] 为数字digit的第i个segment的帧数
    ####################### number of vectors belonging to the jth segment. #######################
    # for digit in xrange(10):
    #     for i in xrange(5):
    #         for k in xrange(training_sequence_num):
    #             if i < 4:
    #                 Nj[digit][i] += segment_info[digit][k][i + 1] - segment_info[digit][k][i]
    #             else:
    #                 Nj[digit][i] += segment_info[digit][k][5] - segment_info[digit][k][4]+1   ### segmentinfo[digit][k][5]为数字digit的第k个sequence的结束index

    for k in xrange(training_sequence_num):
        label = labels[k]
        num_digit = len(label)
        for digit_index in xrange(num_digit):
            digit = label[digit_index]
            for i in xrange(5):
                if i < 4:
                    Nj[digit][i] += segment_info[digit][k][i+1] - segment_info[digit][k][i]
                else:
                    Nj[digit][i] += segment_info[digit][k][5] - segment_info[digit][k][4] + 1

    print "Nj", Nj
    #######################   initialize means  ##################################################
    for digit in xrange(11):
        mean[digit].fill(0)

    for k in xrange(training_sequence_num):
        label = labels[k]
        num_digit = len(label)
        for digit_index in xrange(num_digit):
            digit = label[digit_index]
            start = segment_info[digit][k][0] # 第k个training sequence的数字digit的起始段和结束段
            end = segment_info[digit][k][5]
            for i in xrange(start,end+1):
                if i >=segment_info[digit][k][0] and i<segment_info[digit][k][1]:
                        mean[digit][0] += all_training_sequences[k][i]
                elif i >=segment_info[digit][k][1] and i<segment_info[digit][k][2]:
                        mean[digit][1] +=all_training_sequences[k][i]
                elif i >=segment_info[digit][k][2] and i<segment_info[digit][k][3]:
                        mean[digit][2] +=all_training_sequences[k][i]
                elif i >=segment_info[digit][k][3] and i<segment_info[digit][k][4]:
                        mean[digit][3] +=all_training_sequences[k][i]
                elif i >=segment_info[digit][k][4]:
                        mean[digit][4] +=all_training_sequences[k][i]

    for digit in xrange(11):
        for i in xrange(5):
            mean[digit][i] = mean[digit][i]*1.0/Nj[digit][i]
 #######################   initialize variances  ###################################
    for digit in xrange(11):
        var[digit].fill(0)

    for k in xrange(training_sequence_num):
        label = labels[k]
        num_digit = len(label)
        for digit_index in xrange(num_digit):
            digit = label[digit_index]
            start = segment_info[digit][k][0] # 第k个training sequence的数字digit的起始段和结束段
            end = segment_info[digit][k][5]
            for i in xrange(start,end+1):
                if i >=segment_info[digit][k][0] and i<segment_info[digit][k][1]:
                        var[digit][0] += (all_training_sequences[k][i]-mean[digit][0])**2
                elif i >=segment_info[digit][k][1] and i<segment_info[digit][k][2]:
                        var[digit][1] += (all_training_sequences[k][i] - mean[digit][1]) ** 2
                elif i >=segment_info[digit][k][2] and i<segment_info[digit][k][3]:
                        var[digit][2] += (all_training_sequences[k][i] - mean[digit][2]) ** 2
                elif i >=segment_info[digit][k][3] and i<segment_info[digit][k][4]:
                        var[digit][3] += (all_training_sequences[k][i] - mean[digit][3]) ** 2
                elif i >=segment_info[digit][k][4]:
                        var[digit][4] += (all_training_sequences[k][i] - mean[digit][4]) ** 2

    for digit in xrange(11):
        for i in xrange(5):
            var[digit][i] = var[digit][i]*1.0/Nj[digit][i]

############################## compute transition scores ##########################
    for digit in xrange(11):
        for i in xrange(5):
            trans_p[digit][i] = sequence_num_digit[digit]*1.0/Nj[digit][i]
            trans_p[digit][i] = - np.log(trans_p[digit][i])
    for digit in xrange(11):
        for i in xrange(5):
            self_trans[digit][i] = 1- sequence_num_digit[digit]*1.0/Nj[digit][i]
            self_trans[digit][i] = -np.log(self_trans[digit][i])
    print trans_p
    print self_trans

def segment():
    #trellis = np.zeros(50 * 2).reshape(2, 50) ###############  trellis y轴有5*10个state，10个数字， x轴有2列
    cur_paths = []  # 每个node都关联一条path，每条path用list表示，paths是5个path的集合
    pre_paths = []
    changed = False # 这次迭代是否改变了segment_info
    for sequence_index in xrange(training_sequence_num):
        ###################### 对第sequence_index条训练录音操作 ################################
        length = num_of_frames[sequence_index]
        label = labels[sequence_index]  ###### 保存label为局部变量，一直访问global labels影响效率
        num_of_digit = len(label)
        trellis = np.zeros(5*num_of_digit*2).reshape(2,5*num_of_digit)
        ####################### 初始化path #########################################
        pre_paths[:] = []
        cur_paths[:] = []
        for i in xrange(num_of_digit*5):
            pre_paths.append([i])
            cur_paths.append([])
        ####################### 初始化trellis ######################################
        trellis.fill(sys.maxint)
        vector = all_training_sequences[sequence_index][0]  # 第i个sequence的第0个mfcc向量
        trellis[0][0] = distance(vector,label[0],0)  # trellis[0][0]为vector和state0的距离
        ####################### 计算trellis ########################################
        for index in xrange(1, length):
            vector = all_training_sequences[sequence_index][index]
            digit0 = label[0] ############### 第1个数字
            trellis[1][0] = trellis[0][0] + distance(vector,digit0,0) + self_trans[digit0][0]  # 计算每一列的第一个元素,只能从上一列的第一个元素得到
            cur_paths[0] = pre_paths[0][:]  # 用slice 新建一个list 并copy prepaths[0]的内容。copy list 方法里 slice最快
            cur_paths[0].append(0)  # 点trellis[1][0]的path 只能是 [0 0]

            #################### 计算trellis 的一列 ###############################
            for node_index in xrange(1, num_of_digit*5):  # state 1 -- 49
                state = node_index % 5 ####### node_index 0-4 表示第1个数字的5个状态，所以node_index % 5 表示当前处理的数字的第几个状态
                digit = node_index / 5 ####### node_index/5 表示处理到第几个数字
                digit = label[digit]   ####### 从label中读入当前处理的是数字几
                node_cost = distance(vector,digit, state)
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

        #print cur_paths[num_of_digit*5-1]
        ##################### cur_paths[49]即为分段的best path ################################
        it = iter(label)  #### 用迭代器访问label
        digit = it.next() #####从第1个label开始
        segment_info[digit][sequence_index][0] = 0
        for j in xrange(1, length):
            if j == length - 1:  ###　j为sequence的结尾，保存j为最后一个数字的结尾index
                if segment_info[digit][sequence_index][5] != j :
                    #print segment_info[digit][sequence_index][5]
                    segment_info[digit][sequence_index][5] = j
                    #print segment_info[digit][sequence_index][5]
                    changed = True
            if cur_paths[num_of_digit*5-1][j] != cur_paths[num_of_digit*5-1][j - 1]: #### 如果 j 是state的起始index
                state = cur_paths[num_of_digit*5-1][j]

                if state == 0:  ###　说明是下一个数字的开始
                    if segment_info[digit][sequence_index][5] != j - 1:   ### 将j-1保存为上一个数字的结尾index
                        #print segment_info[digit][sequence_index][5]
                        segment_info[digit][sequence_index][5] = j-1
                        #print segment_info[digit][sequence_index][5]
                        changed = True
                    digit = it.next()
                if segment_info[digit][sequence_index][state] != j:  ### 用segment info来记录state变化的地方
                    #print segment_info[digit][sequence_index][state]
                    segment_info[digit][sequence_index][state] = j
                    #print segment_info[digit][sequence_index][state]
                    changed = True
    #print segment_info
    return changed

def segmental_kmeans(initial_model):
    ################################# 初始化 load isolated words template ###############################
    initialize(initial_model)
    #################################  定义 pre_seg_info ##################################
    changed = True
    ite=0
    pre_Nj = 0

    while(changed):
        changed = True
        print("iteration ",ite)
        segment()
        update_parameters()
        if Nj == pre_Nj:
            changed = False
        pre_Nj = copy.deepcopy(Nj)
        ite = ite+1
    for integer in xrange(11):
        np.savetxt(str(integer)+"hmm_mean.txt",mean[integer])
        np.savetxt(str(integer) + "hmm_var.txt", var[integer])
        np.savetxt(str(integer) + "hmm_trans_p.txt", trans_p[integer])
        np.savetxt(str(integer) + "hmm_self_trans.txt", self_trans[integer])
        np.savetxt(str(integer) + "segment_info.txt", segment_info[integer])


if __name__ == '__main__':
    #global num_of_frames
    num_of_frames = []
    #global all_training_sequences
    all_training_sequences = []
    #global labels
    labels = []  ### labels[i]是一个list，对应第i条语音的label
    #global training_sequence_num
    training_sequence_num = 0
    sequence_num_digit = [0]*11   # 包含每个digit的sequence数量。sequence_num_digit[i]代表包含i的sequence数量

    f_label = open('../hwdata/smallTRAIN.transcripts', 'r')
    f_name = open('../hwdata/smallTRAIN.filelist', 'r')
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

    segmental_kmeans('../speech model/hmm continuous')
    del num_of_frames
    del all_training_sequences
    del labels