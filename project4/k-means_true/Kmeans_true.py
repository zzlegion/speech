#coding: utf-8
import sys
import numpy as np
import profile
from time import time
import os
import mfcc
import record

def distance(vector,j):
    """"Cal the Gaussian distance between input vector and the jth mean vector"""
    res = 0.5 * np.sum(np.log(2 * np.pi * var[j]))+ 0.5* np.sum((vector-mean[j])**2*1.0/var[j])
    return res

def initialize():
    # average the sequence into 5 states
    global segment_info
    segment_info = np.arange(5 * training_sequence_num).reshape(training_sequence_num, 5)
    segment_info = segment_info % (5)
    segment_info.dtype = int
    for i in xrange(training_sequence_num):
        val = num_of_frames[i]*1.0/5
        segment_info[i] =segment_info[i]*val

def update_parameters():
    global mean
    mean = np.zeros(5 * 39).reshape(5, 39)
    global var
    var = np.zeros(5 * 39).reshape(5, 39)
    global trans_p
    trans_p = [0] * 4
    global self_trans
    self_trans = [0] * 5

    # number of vectors belonging to the jth segment.
    global Nj
    Nj = [0] * 5
    for i in xrange(5):
        for k in xrange(training_sequence_num):
            if i < 4:
                Nj[i] += segment_info[k][i + 1] - segment_info[k][i]
            else:
                Nj[i] += num_of_frames[k] - segment_info[k][4]

 #    #######################   initialize means  ###################################
    i=0
    for k in xrange(training_sequence_num):
        for i in xrange(num_of_frames[k]):
            if i >=segment_info[k][0] and i<segment_info[k][1]:
                mean[0] += all_training_sequences[k][i]
            elif i >=segment_info[k][1] and i<segment_info[k][2]:
                mean[1] +=all_training_sequences[k][i]
            elif i >=segment_info[k][2] and i<segment_info[k][3]:
                mean[2] +=all_training_sequences[k][i]
            elif i >=segment_info[k][3] and i<segment_info[k][4]:
                mean[3] +=all_training_sequences[k][i]
            elif i >=segment_info[k][4]:
                mean[4] +=all_training_sequences[k][i]

    for i in xrange(5):
        mean[i] = mean[i]*1.0/Nj[i]
 #######################   initialize variances  ###################################

    for k in xrange(training_sequence_num):
        for i in xrange(num_of_frames[k]):
            if i >=segment_info[k][0] and i<segment_info[k][1]:
                var[0] += (all_training_sequences[k][i]- mean[0]) ** 2
            elif i >=segment_info[k][1] and i<segment_info[k][2]:
                var[1] += (all_training_sequences[k][i] - mean[1]) ** 2
            elif i >=segment_info[k][2] and i<segment_info[k][3]:
                var[2] += (all_training_sequences[k][i] - mean[2]) ** 2
            elif i >=segment_info[k][3] and i<segment_info[k][4]:
                var[3] += (all_training_sequences[k][i] - mean[3]) ** 2
            elif i >=segment_info[k][4]:
                var[4] += (all_training_sequences[k][i] - mean[4]) ** 2

    for i in xrange(5):
        var[i] = var[i]*1.0/Nj[i]

############################## compte transition scores ##########################
    for i in xrange(4):
        trans_p[i] = training_sequence_num*1.0/Nj[i]
        trans_p[i] = - np.log(trans_p[i])

    for i in xrange(4):
        self_trans[i] = 1- training_sequence_num*1.0/Nj[i]
        self_trans[i] = -np.log(self_trans[i])
    self_trans[4] = 0

def segment():
    trellis = np.zeros(5*2).reshape(2,5)
    cur_paths = []  # 每个node都关联一条path，每条path用list表示，paths是5个path的集合
    pre_paths = []
    for sequence_index in xrange(training_sequence_num):
        ###################### 对第条训练录音操作 ################################
        length = num_of_frames[sequence_index]
        ####################### 初始化path #########################################
        pre_paths[:] = []
        pre_paths.append([0])
        pre_paths.append([1])
        pre_paths.append([2])
        pre_paths.append([3])
        pre_paths.append([4])

        cur_paths[:] = []
        cur_paths.append([])
        cur_paths.append([])
        cur_paths.append([])
        cur_paths.append([])
        cur_paths.append([])
        ####################### 初始化trellis ######################################
        trellis.fill(sys.maxint)
        vector = all_training_sequences[sequence_index][0]  # 第i个sequence的第一个mfcc向量,第0个mfcc向量为0*39
        trellis[0][0] = distance(vector,0) # trellis[0][0]为vector和state0的距离
        ####################### 计算trellis ########################################
        for index in xrange(1,length):
            vector = all_training_sequences[sequence_index][index]
            node_cost0 = distance(vector,0)
            trellis[1][0] = trellis[0][0] + node_cost0 + self_trans[0] # 计算每一列的第一个元素,只能从上一列的第一个元素得到
            cur_paths[0] = pre_paths[0][:] # 用slice 新建一个list 并copy prepaths[0]的内容。copy list 方法里 slice最快
            cur_paths[0].append(0) # 点trellis[1][0]的path 只能是 [0 0]

            for node_index in xrange(1,5): # state 1 2 3 4
                node_cost = distance(vector,node_index)
                stay_edge_cost = self_trans[node_index]
                move_edge_cost = trans_p[node_index-1]
                stay_cost = trellis[0][node_index] + node_cost + stay_edge_cost
                move_cost = trellis[0][node_index-1] + node_cost + move_edge_cost
                if stay_cost < move_cost:
                    trellis[1][node_index] = stay_cost
                    cur_paths[node_index] = pre_paths[node_index][:]
                    cur_paths[node_index].append(node_index) ## 把当前节点加入path中
                else:
                    trellis[1][node_index] = move_cost
                    cur_paths[node_index] = pre_paths[node_index-1][:] ## 复制上一节点的path
                    cur_paths[node_index].append(node_index) ## 把当前节点加入path中
            #################### 将 trellis[1][:] 复制到 trellis[0][:] ################
            trellis[0][:] = trellis[1][:]
            #print trellis[0]
            pre_paths[:] = cur_paths[:]
            #print pre_paths
        #print cur_paths[4]
        ##################### cur_paths[4]即为分段的best path ################################
        state = 1
        for j in xrange(1,length-1):
            if cur_paths[4][j]!=cur_paths[4][j-1]:
                segment_info[sequence_index][state] = j  ### 用segment info来记录state变化的地方
                state += 1
        print segment_info[sequence_index]

def kmeans(integer):
    ################################# 初始化 uniformly 分段 ###############################
    initialize()
    update_parameters()
    for i in xrange(4):       ################# 将transition probability 平均分配 ########
        trans_p[i] = 0.5
        self_trans[i] = 0.5
    #################################  定义 pre_seg_info ##################################
    pre_seg_info = np.arange(5 * training_sequence_num).reshape(training_sequence_num, 5)
    pre_seg_info = pre_seg_info % (5)
    pre_seg_info.dtype = int

    for i in xrange(training_sequence_num):
        for j in xrange(5):
            pre_seg_info[i][j] = segment_info[i][j]

    changed = True
    ite=0

    while(changed):
        print("iteration ",ite)
        changed = False
        segment()

        print(segment_info)

        for i in xrange(training_sequence_num):
            for j in xrange(5):
                if pre_seg_info[i][j] != segment_info[i][j]:
                    changed = True
                    break

        update_parameters()

        for i in xrange(training_sequence_num):
            for j in xrange(5):
                pre_seg_info[i][j] = segment_info[i][j]

        ite = ite+1

        ############################## compte transition scores ##########################
    trans_p.append(training_sequence_num * 1.0 / Nj[4])
    self_trans[4] = 1 - training_sequence_num * 1.0 / Nj[4]
    np.savetxt(str(integer)+"hmm_mean.txt",mean)
    np.savetxt(str(integer) + "hmm_var.txt", var)
    np.savetxt(str(integer) + "hmm_segment_info.txt", segment_info)
    np.savetxt(str(integer) + "hmm_trans_p.txt", trans_p)
    np.savetxt(str(integer) + "hmm_self_trans.txt", self_trans)

def load_hmm_model(integer):
    # global segment_info
    # segment_info = np.arange(5 * training_sequence_num).reshape(training_sequence_num, 5)
    # segment_info.dtype = int

    global mean
    mean = np.zeros(5 * 39).reshape(5, 39)
    global var
    var = np.zeros(5 * 39).reshape(5, 39)
    global trans_p
    trans_p = [0] * 4
    global self_trans
    self_trans = [0] * 5

    mean=np.loadtxt(str(integer)+"hmm_mean.txt")
    var=np.loadtxt(str(integer) + "hmm_var.txt")
    trans_p = np.loadtxt(str(integer) + "hmm_trans_p.txt")
    self_trans = np.loadtxt(str(integer) + "hmm_self_trans.txt")

def hmm(test_sequence,speak_number,name,isOnline):
    cost=[0 for col in xrange(10)]
    length = len(test_sequence)

    for i in xrange(10):
        ##################### load 第i个数字的hmm model ###########################
        load_hmm_model(i)
        ##################### 对 test_sequence 做 k means #########################
        trellis = np.zeros(5 * 2).reshape(2, 5)
        cur_paths = []  # 每个node都关联一条path，每条path用list表示，paths是5个path的集合
        pre_paths = []
        ####################### 初始化path #########################################
        pre_paths[:] = []
        pre_paths.append([0])
        pre_paths.append([1])
        pre_paths.append([2])
        pre_paths.append([3])
        pre_paths.append([4])

        cur_paths[:] = []
        cur_paths.append([])
        cur_paths.append([])
        cur_paths.append([])
        cur_paths.append([])
        cur_paths.append([])
        ####################### 初始化trellis ######################################
        trellis.fill(sys.maxint)
        vector = test_sequence[0]  # test_sequence的第0个mfcc向量
        trellis[0][0] = distance(vector, 0)  # trellis[0][0]为vector和state0的距离
        ####################### 计算trellis ########################################
        for index in xrange(1, length):
            vector = test_sequence[index]
            trellis[1][0] = trellis[0][0] + distance(vector,0) + self_trans[0]  # 计算每一列的第一个元素,只能从上一列的第一个元素得到
            cur_paths[0] = pre_paths[0][:]  # 用slice 新建一个list 并copy prepaths[0]的内容。copy list 方法里 slice最快
            cur_paths[0].append(0)  # 点trellis[1][0]的path 只能是 [0 0]

            for node_index in xrange(1, 5):  # state 1 2 3 4
                node_cost = distance(vector, node_index)
                stay_cost = trellis[0][node_index] + node_cost + self_trans[node_index]
                move_cost = trellis[0][node_index - 1] + node_cost + trans_p[node_index-1]
                if stay_cost < move_cost:
                    trellis[1][node_index] = stay_cost
                    cur_paths[node_index] = pre_paths[node_index][:]
                    cur_paths[node_index].append(node_index)  ## 把当前节点加入path中
                else:
                    trellis[1][node_index] = move_cost
                    cur_paths[node_index] = pre_paths[node_index - 1][:]  ## 复制上一节点的path
                    cur_paths[node_index].append(node_index)  ## 把当前节点加入path中
            #################### 将 trellis[1][:] 复制到 trellis[0][:] ################
            trellis[0][:] = trellis[1][:]
            pre_paths[:] = cur_paths[:]
        ##################### 保存test_sequence 对第i个模板的最小cost ################################
        cost[i] = trellis[1][4]

    #print(cost)
    mincost=cost[0]
    minindex=0
    for index,ele in enumerate(cost):
        if ele < mincost:
            minindex = index
            mincost = ele
    if isOnline:
        print "You are speaking ",minindex
        print "Cost is ",mincost
    else:
        if minindex == speak_number:
             print("Right!!  Min cost is ",mincost)
        else:
             print("Wrong.. Should be ",speak_number," But is ",minindex,mincost)
             print(name)

def align(train_data):
    for i in xrange(10):
        print "hmm model: ",i
        global num_of_frames
        num_of_frames = []
        global all_training_sequences
        all_training_sequences = []

        ### 俊优的train sequence
        for index in train_data[0]:
            sequence = np.loadtxt(".\\junyo_iso_11_1\\"+str(i)+"_"+str(index)+".txt")
            #sequence = sequence[1:]
            num_of_frames.append(len(sequence))
            all_training_sequences.append(sequence)

        ### 健炜的train sequence
        for index in train_data[1]:
            sequence = np.loadtxt(".\\jianwei_iso_11_1\\" + str(i) + "_" + str(index) + ".txt")
            #sequence = sequence[1:]
            num_of_frames.append(len(sequence))
            all_training_sequences.append(sequence)

        kmeans(i)
        del num_of_frames
        del all_training_sequences

def test(isOnline,test_data):
    if isOnline:
        test_sequence = mfcc.mfcc(record.record(),"fast_mode")
        hmm(test_sequence,3,"test0",True)
    else:
        for i in xrange(10):
            ### 俊优的test sequence
            for index in test_data[0]:
                name = ".\\txtDictionary\\junyo_" + str(i) + "_" + str(index) + ".txt"
                sequence = np.loadtxt(name)
                sequence = sequence[1:]
                hmm(sequence, i, name,False)

            ### 健炜的test sequence
            for index in test_data[1]:
                name = ".\\txtDictionary\\" + str(i) + "_" + str(index) + ".txt"
                sequence = np.loadtxt(name)
                sequence = sequence[1:]
                hmm(sequence, i, name,False)

if __name__ == '__main__':
    train_data = []
    train_data.append([0,1,2,3,4])  ## 俊优的用于训练的录音index
    train_data.append([0,1,2,3,4]) ## 健炜的用于训练的录音index
    training_sequence_num = len(train_data[0])+len(train_data[1])

    #test_data = []
    #test_data.append([])  ## 俊优的用于测试的录音index
    #test_data.append([3,5])  ## 健炜的用于测试的录音index

    align(train_data)
    #test(True,test_data) ## True 表示onl