import mfcc

train_list = open("../hwdata/TRAIN.filelist")
while 1:
    line = train_list.readline()
    if not line:
        break
    wav = op