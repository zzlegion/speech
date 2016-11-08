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
    for i in xrange(10):  ###　数字ｉ
        ### segment_info不需要初始化, segment_info[0-4]为第0-4段的起始index，segment_info[5]为属于数字i的结束index
        matrix = np.arange(6 * training_sequence_num).reshape(training_sequence_num, 6)
        matrix = matrix % 6
        matrix.dtype = int
        segment_info.setdefault(i,matrix)

    global silence_seg_info
    silence_seg_info = {}
    for i in xrange(2):
        matrix = np.arange(3 * training_sequence_num,dtype = int).reshape(training_sequence_num,3)
        segment_info.setdefault(i,matrix)
    ####################### mean key: 数字i  value:数字i的5个state的mean，5*39的list ######################
    global mean
    mean = {}
    for i in xrange(10):  ###　数字ｉ
        ### 用训好的模型初始化
        matrix = np.loadtxt(initial_model+"/"+str(i)+"hmm_mean.txt")
        mean.setdefault(i, matrix)
    ####################### var key: 数字i  value:数字i的5个state的var，5*39的list ######################
    global var
    var = {}
    for i in xrange(10):  ###　数字ｉ
        ### 用训好的模型初始化
        matrix = np.loadtxt(initial_model+"/"+str(i)+"hmm_var.txt")
        var.setdefault(i, matrix)
    ####################### trans_p key: 数字i  value:数字i的transition p，1*5的list ######################
    global trans_p
    trans_p = {}
    for i in xrange(10):  ###　数字ｉ
        ### 用训好的模型初始化
        matrix = np.loadtxt(initial_model+"/"+str(i)+"hmm_trans_p.txt")
        trans_p.setdefault(i, matrix)
    ####################### self_trans key: 数字i  value:数字i的transition p，1*5的list ######################
    global self_trans
    self_trans = {}
    for i in xrange(10):  ###　数字ｉ
        ### 用训好的模型初始化
        matrix = np.loadtxt(initial_model+"/"+str(i)+"hmm_self_trans.txt")
        self_trans.setdefault(i, matrix)

def update_parameters():
    Nj = [[0 for col in xrange(5)] for row in xrange(10)] ### Nj[digit][i] 为数字digit的第i个segment的帧数
    ####################### number of vectors belonging to the jth segment. #######################
    for digit in xrange(10):
        for i in xrange(5):
            for k in xrange(training_sequence_num):
                if i < 4:
                    Nj[digit][i] += segment_info[digit][k][i + 1] - segment_info[digit][k][i]
                else:
                    Nj[digit][i] += segment_info[digit][k][5] - segment_info[digit][k][4]+1   ### segmentinfo[digit][k][5]为数字digit的第k个sequence的结束index
    print "Nj",Nj
    #######################   initialize means  ##################################################
    for digit in xrange(10):
        mean[digit].fill(0)

    for k in xrange(training_sequence_num):
        label = labels[k]
        for digit_index in xrange(10):
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

    for digit in xrange(10):
        for i in xrange(5):
            mean[digit][i] = mean[digit][i]*1.0/Nj[digit][i]
 #######################   initialize variances  ###################################
    for digit in xrange(10):
        var[digit].fill(0)

    for k in xrange(training_sequence_num):
        label = labels[k]
        for digit_index in xrange(10):
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

    for digit in xrange(10):
        for i in xrange(5):
            var[digit][i] = var[digit][i]*1.0/Nj[digit][i]

############################## compute transition scores ##########################
    for digit in xrange(10):
        for i in xrange(5):
            trans_p[digit][i] = training_sequence_num*1.0/Nj[digit][i]
            trans_p[digit][i] = - np.log(trans_p[digit][i])
    for digit in xrange(10):
        for i in xrange(5):
            self_trans[digit][i] = 1- training_sequence_num*1.0/Nj[digit][i]
            self_trans[digit][i] = -np.log(self_trans[digit][i])
    print trans_p
    print self_trans

def segment():
    trellis = np.zeros(50 * 2).reshape(2, 50) ###############  trellis y轴有5*10个state，10个数字， x轴有2列
    silence_trellis_start = np.zeros(3*2).reshape(2,3)  ####### 开头的silence
    silence_trellis_end = np.zeros(3*2).reshape(2,3) ######### 结尾的silence
    cur_paths = []  # 每个node都关联一条path，每条path用list表示，paths是5个path的集合
    pre_paths = []
    changed = False # 这次迭代是否改变了segment_info
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
        trellis[0][0] = distance(vector,label[0],0)  # trellis[0][0]为vector和state0的距离
        ####################### 计算trellis ########################################
        for index in xrange(1, length):
            vector = all_training_sequences[sequence_index][index]
            digit0 = label[0] ############### 第1个数字
            trellis[1][0] = trellis[0][0] + distance(vector,digit0,0) + self_trans[digit0][0]  # 计算每一列的第一个元素,只能从上一列的第一个元素得到
            cur_paths[0] = pre_paths[0][:]  # 用slice 新建一个list 并copy prepaths[0]的内容。copy list 方法里 slice最快
            cur_paths[0].append(0)  # 点trellis[1][0]的path 只能是 [0 0]

            #################### 计算trellis 的一列 ###############################
            for node_index in xrange(1, 50):  # state 1 -- 49
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

        #print cur_paths[49]
        ##################### cur_paths[49]即为分段的best path ################################
        it = iter(label)  #### 用迭代器访问label
        digit = it.next() #####从第1个label开始
        segment_info[digit][sequence_index][0] = 0
        for j in xrange(1, length):
            if j == length - 1:  ###　j为sequence的结尾，保存j为最后一个数字的结尾index
                if segment_info[digit][sequence_index][5] != j :
                    segment_info[digit][sequence_index][5] = j
                    changed = True
            if cur_paths[49][j] != cur_paths[49][j - 1]: #### 如果 j 是state的起始index
                state = cur_paths[49][j]

                if state == 0:  ###　说明是下一个数字的开始
                    if segment_info[digit][sequence_index][5] != j - 1:   ### 将j-1保存为上一个数字的结尾index
                        segment_info[digit][sequence_index][5] = j-1
                        changed = True
                    digit = it.next()
                if segment_info[digit][sequence_index][state] != j:  ### 用segment info来记录state变化的地方
                    segment_info[digit][sequence_index][state] = j
                    changed = True
    print segment_info
    return changed

def segmental_kmeans(initial_model):
    ################################# 初始化 load isolated words template ###############################
    initialize(initial_model)
    #################################  定义 pre_seg_info ##################################
    changed = True
    ite=0

    while(changed):
        print("iteration ",ite)
        changed = segment()
        update_parameters()
        ite = ite+1
    for integer in xrange(10):
        np.savetxt(str(integer)+"hmm_mean.txt",mean[integer])
        np.savetxt(str(integer) + "hmm_var.txt", var[integer])
        np.savetxt(str(integer) + "hmm_trans_p.txt", trans_p[integer])
        np.savetxt(str(integer) + "hmm_self_trans.txt", self_trans[integer])
        np.savetxt(str(integer) + "segment_info.txt", segment_info[integer])

def align(train_data,initial_model):
    global num_of_frames
    num_of_frames = []
    global all_training_sequences
    all_training_sequences = []
    global labels
    labels = []    ### labels[i]是一个list，对应第i条语音的label

    ### 俊优的train sequence
    for i in xrange(1, 7):
        if i == 1:
            labeltmp = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
        elif i ==2:
            labeltmp = [9,8,7,6,5,4,3,2,1,0]
        elif i==3:
            labeltmp = [1,2,3,4,5,6,7,8,9,0]
        elif i==4:
            labeltmp = [0,9,8,7,6,5,4,3,2,1,]
        elif i==5:
            labeltmp = [1,3,5,7,9,0,2,4,6,8]
        elif i == 6:
            labeltmp = [8,6,4,2,0,9,7,5,3,1]
        for index in train_data[0]:
            sequence = np.loadtxt(".\\new_record\\junyo_train\\junyo_"+str(i)+"_"+str(index)+".txt")
            num_of_frames.append(len(sequence))
            all_training_sequences.append(sequence)
            labels.append(labeltmp)

    ### 健炜的train sequence
    for i in xrange(1, 7):
        if i == 1:
            labeltmp = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
        elif i ==2:
            labeltmp = [9,8,7,6,5,4,3,2,1,0]
        elif i==3:
            labeltmp = [1,2,3,4,5,6,7,8,9,0]
        elif i==4:
            labeltmp = [0,9,8,7,6,5,4,3,2,1,]
        elif i==5:
            labeltmp = [1,3,5,7,9,0,2,4,6,8]
        elif i == 6:
            labeltmp = [8,6,4,2,0,9,7,5,3,1]
        for index in train_data[1]:
            sequence = np.loadtxt(".\\new_record\\jianwei_train\\jianwei_"+str(i)+"_"+str(index)+".txt")
            num_of_frames.append(len(sequence))
            all_training_sequences.append(sequence)
            labels.append(labeltmp)

    segmental_kmeans(initial_model)
    del num_of_frames
    del all_training_sequences
    del labels

if __name__ == '__main__':
    train_data = []
    train_data.append([1,2,3])  ## 俊优的用于训练的录音index
    train_data.append([]) ## 健炜的用于训练的录音index
    training_sequence_num = len(train_data[0]) * 6 + len(train_data[1])*6

    align(train_data,'hmm iso 11_1')

    #test(False,test_data) ## True 表示online test