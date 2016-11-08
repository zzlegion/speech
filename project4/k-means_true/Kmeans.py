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
    for i in range(training_sequence_num):
        val = num_of_frames[i]*1.0/5
        segment_info[i] =segment_info[i]*val

def update_parameters():
    global mean
    mean = np.zeros(5 * 39).reshape(5, 39)
    global var
    var = np.zeros(5 * 39).reshape(5, 39)
    global trans_p
    trans_p = [0] * 5
    global self_trans
    self_trans = [0] * 5

    # number of vectors belonging to the jth segment.
    Nj = [0] * 5
    for i in range(5):
        for k in range(training_sequence_num):
            if i < 4:
                Nj[i] += segment_info[k][i + 1] - segment_info[k][i]
            else:
                Nj[i] += num_of_frames[k] - segment_info[k][4]

 #    #######################   initialize means  ###################################
    i=0
    for k in range(training_sequence_num):
        for i in range(num_of_frames[k]):
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

    for i in range(5):
        mean[i] = mean[i]*1.0/Nj[i]
 #######################   initialize variances  ###################################

    for k in range(training_sequence_num):
        for i in range(num_of_frames[k]):
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

    for i in range(5):
        var[i] = var[i]*1.0/Nj[i]

############################## compte transition scores ##########################
    for i in range(5):
        trans_p[i] = training_sequence_num*1.0/Nj[i]
        trans_p[i] = - np.log(trans_p[i])

    for i in range(5):
        self_trans[i] = 1- training_sequence_num*1.0/Nj[i]
        self_trans[i] = -np.log(self_trans[i])

def segment():
    for i in range(training_sequence_num):
        pre_state = 0
        length = num_of_frames[i]
        for index in range(2,length):   ###since the first vector must in the first state, we start alignment from the second vector
            vector = all_training_sequences[i][index]
            if pre_state == 4:
                break
            else :
                stay_cost = distance(vector, pre_state) + self_trans[pre_state]
                move_cost = distance(vector, pre_state + 1) + trans_p[pre_state]
                if stay_cost > move_cost:
                    pre_state += 1
                    segment_info[i][pre_state] = index

def kmeans(integer):

    initialize()
    update_parameters()

    pre_seg_info = np.arange(5 * training_sequence_num).reshape(training_sequence_num, 5)
    pre_seg_info = pre_seg_info % (5)
    pre_seg_info.dtype = int

    for i in range(training_sequence_num):
        for j in range(5):
            pre_seg_info[i][j] = segment_info[i][j]

    changed = True
    ite=0

    while(changed):
        print("iteration ",ite)
        changed = False
        segment()

        #print(pre_seg_info)
        print(segment_info)

        for i in range(training_sequence_num):
            for j in range(5):
                if pre_seg_info[i][j] != segment_info[i][j]:
                    changed = True
                    break

        update_parameters()

        for i in range(training_sequence_num):
            for j in range(5):
                pre_seg_info[i][j] = segment_info[i][j]

        ite = ite+1

    # f = open(str(integer)+"template_hmm.txt", 'ab')
    # np.savetxt(f, mean)
    # np.savetxt(f,var)
    # np.savetxt(f,segment_info)
    # np.savetxt(f,trans_p)
    # np.savetxt(f,self_trans)
    # f.close()
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
    # segment_info = np.loadtxt(str(integer) + "hmm_segment_info.txt")
    trans_p = np.loadtxt(str(integer) + "hmm_trans_p.txt")
    self_trans = np.loadtxt(str(integer) + "hmm_self_trans.txt")

def hmm(test_sequence,speak_number,name):
    cost=[0 for col in range(10)]
    length = len(test_sequence)

    for i in range(10):
        #initialize the ith hmm model
        load_hmm_model(i)
        pre_state = 0
        for index in range(2,length):   ###since the first vector must in the first state, we start alignment from the second vector
            vector = test_sequence[index]
            stay_cost = distance(vector, pre_state) + self_trans[pre_state]
            if pre_state == 4:
                cost[i] += stay_cost
            else:
                # print "self_trans ", self_trans[pre_state]
                # print "trans", trans_p[pre_state]
                move_cost = distance(vector, pre_state + 1) + trans_p[pre_state]
                if stay_cost > move_cost:
                    pre_state += 1
                    cost[i] += move_cost
                else:
                    cost[i] += stay_cost

    #print(cost)
    mincost=cost[0]
    minindex=0
    for index,ele in enumerate(cost):
        if ele < mincost:
            minindex = index
            mincost = ele
    print "You are speaking ",minindex
    #print "Cost is ",mincost
    # if minindex == speak_number:
    #      print("Right!!  Min cost is ",mincost)
    # else:
    #      print("Wrong.. Should be ",speak_number," But is ",minindex,mincost)
    #      print(name)

def align():
    for i in range(10):
        print "hmm model: ",i
        zero_mfcc_sequence1 = np.loadtxt(".\\txtDictionary\\junyo_"+str(i)+"_1.txt")
        zero_mfcc_sequence2 = np.loadtxt(".\\txtDictionary\\junyo_"+str(i)+"_2.txt")
        zero_mfcc_sequence3 = np.loadtxt(".\\txtDictionary\\junyo_"+str(i)+"_3.txt")
    #    zero_mfcc_sequence1 = np.loadtxt(".\\txtDictionary\\junyo_" + str(i) + "_1.txt")
    #    zero_mfcc_sequence2 = np.loadtxt(".\\txtDictionary\\junyo_" + str(i) + "_2.txt")
    #    zero_mfcc_sequence3 = np.loadtxt(".\\txtDictionary\\junyo_" + str(i) + "_3.txt")
    #    zero_mfcc_sequence3 = np.loadtxt(".\\txtDictionary\\junyo_" + str(i) + "_3.txt")
        zero_mfcc_sequence4 = np.loadtxt(".\\txtDictionary\\junyo_"+str(i)+"_4.txt")
        zero_mfcc_sequence5 = np.loadtxt(".\\txtDictionary\\junyo_" + str(i) + "_5.txt")
    #    zero_mfcc_sequence6 = np.loadtxt(".\\txtDictionary\\" + str(i) + "_3.txt")
    #    zero_mfcc_sequence6 = np.loadtxt(".\\txtDictionaryNew\\junyo_" + str(i) + "_2_.txt")
    #    zero_mfcc_sequence7 = np.loadtxt(".\\txtDictionaryNew\\junyo_" + str(i) + "_5_.txt")
    #    zero_mfcc_sequence8 = np.loadtxt(".\\txtDictionaryNew\\" + str(i) + "_3_.txt")
    #    zero_mfcc_sequence9 = np.loadtxt(".\\txtDictionaryNew\\" + str(i) + "_4_.txt")
    #    zero_mfcc_sequence10 =np.loadtxt( ".\\txtDictionaryNew\\" + str(i) + "_5_.txt")

        global num_of_frames
        num_of_frames = (len(zero_mfcc_sequence1), len(zero_mfcc_sequence2),len(zero_mfcc_sequence3), len(zero_mfcc_sequence4),len(zero_mfcc_sequence5),
                         )#len(zero_mfcc_sequence7),len(zero_mfcc_sequence8),len(zero_mfcc_sequence9),len(zero_mfcc_sequence10))
        global all_training_sequences
        all_training_sequences = [zero_mfcc_sequence1,zero_mfcc_sequence2,zero_mfcc_sequence3,zero_mfcc_sequence4,zero_mfcc_sequence5,
                                 ]#zero_mfcc_sequence7,zero_mfcc_sequence8,zero_mfcc_sequence9,zero_mfcc_sequence10]

        kmeans(i)
        del num_of_frames
        del all_training_sequences

def test():
    test_sequence = mfcc.mfcc(record.record(),"fast_mode")
    hmm(test_sequence,3,"test0")
    # for i in range(10):
    #     name1=".\\txtDictionary\\"+str(i)+"_3.txt"
    #     name2 = ".\\txtDictionary\\" + str(i) + "_4.txt"
    #     name3 = ".\\txtDictionary\\" + str(i) + "_5.txt"
    #     name4=".\\txtDictionary\\junyo_" + str(i) + "_2.txt"
    #     name5=".\\txtDictionary\\junyo_" + str(i) + "_5.txt"
    #     # name6 = ".\\txtDictionaryNew\\junyo_" + str(i) + "_1_.txt"
    #     # name7 = ".\\txtDictionaryNew\\junyo_" + str(i) + "_2_.txt"
    #     # name8 = ".\\txtDictionaryNew\\junyo_" + str(i) + "_3_.txt"
    #     # name9 = ".\\txtDictionaryNew\\" + str(i) + "_4_.txt"
    #     # name10 = ".\\txtDictionaryNew\\" + str(i) + "_5_.txt"
    #     test_sequence1 = np.loadtxt(name1)
    #     test_sequence2 = np.loadtxt(name2)
    #     test_sequence3 = np.loadtxt(name3)
    #     test_sequence4 = np.loadtxt(name4)
    #     test_sequence5 = np.loadtxt(name5)
    #     # test_sequence6 = np.loadtxt(name6)
    #     # test_sequence7 = np.loadtxt(name7)
    #     # test_sequence8 = np.loadtxt(name8)
    #     # test_sequence9 = np.loadtxt(name9)
    #     # test_sequence10 = np.loadtxt(name10)
    #
    #     hmm(test_sequence1,i,name1)
    #     hmm(test_sequence2, i,name2)
    #     hmm(test_sequence3, i,name3)
    #     hmm(test_sequence4, i, name4)
    #     hmm(test_sequence5, i, name5)
    #     # hmm(test_sequence6, i, name6)
    #     # hmm(test_sequence7, i, name7)
    #     # hmm(test_sequence8, i, name8)
    #     # hmm(test_sequence9, i, name9)
    #     # hmm(test_sequence10, i, name10)

def alignWSQ():
    for i in range(1,3):
        zero_mfcc_sequence1 = np.loadtxt(".\\txtDictionaryWSQ\\wsq_" + str(i) + "_0_.txt")
        zero_mfcc_sequence2 = np.loadtxt(".\\txtDictionaryWSQ\\wsq_" + str(i) + "_1_.txt")
        zero_mfcc_sequence3 = np.loadtxt(".\\txtDictionaryWSQ\\wsq_" + str(i) + "_2_.txt")
        zero_mfcc_sequence4 = np.loadtxt(".\\txtDictionaryWSQ\\wsq_" + str(i) + "_3_.txt")

        global num_of_frames
        num_of_frames = (
        len(zero_mfcc_sequence1), len(zero_mfcc_sequence2), len(zero_mfcc_sequence3), len(zero_mfcc_sequence4)
        )
        #                         len(zero_mfcc_sequence6),len(zero_mfcc_sequence7),len(zero_mfcc_sequence8),len(zero_mfcc_sequence9),len(zero_mfcc_sequence10))
        global all_training_sequences
        all_training_sequences = [zero_mfcc_sequence1, zero_mfcc_sequence2, zero_mfcc_sequence3, zero_mfcc_sequence4
        ]
        #                                 zero_mfcc_sequence6,zero_mfcc_sequence7,zero_mfcc_sequence8,zero_mfcc_sequence9,zero_mfcc_sequence10]

        kmeans(i)
        del num_of_frames
        del all_training_sequences

def testWSQ():
    for i in range(1,3):
        name1=".\\txtDictionaryWSQ\\wsq_"+str(i)+"_4_.txt"
        test_sequence1 = np.loadtxt(name1)
        hmm(test_sequence1,i,name1)

if __name__ == '__main__':

    segment_num = 5
    training_sequence_num = 5 ## in our cases, it should be five

    #align()
    test()