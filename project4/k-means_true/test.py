import numpy as np
import record
import time
import copy
import profile


if __name__ == '__main__':

    Nj = [[0 for col in xrange(5)] for row in xrange(10)]
    Nj2 = copy.deepcopy(Nj)
    Nj2[0][2] = 3
    print Nj == Nj2