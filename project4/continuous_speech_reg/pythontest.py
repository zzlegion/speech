import readwave
import mfcc
import numpy as np

for digit in range(0, 10):
    for index in range(0, 5):
        file_name = "./junyo_iso_11_1/" + str(digit) + "_" + str(index) + ".wav"
        speech = mfcc.mfcc(readwave.read_wave(file_name), "fm")
        np.savetxt(str(digit)+"_"+str(index)+".txt", speech)
