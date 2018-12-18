from __future__ import print_function

import numpy as np
from csh import CSH
from csvec import CSVec
import time

class Timer:
    def __init__(self, name):
        print(name + "...")
    def __enter__(self):
        self.startTime = time.clock()
    def __exit__(self, et, ev, tb):
        print("Time taken: {} sec".format(time.clock() - self.startTime))

print("running a toy example - sketch size 5x1000, vector of dimension 10^6 to compress ")
with Timer("initializing"):
    # d is the dimension of the vector
    csVec = CSVec(c=1000, r=5, d=1000000)
    vec = np.ones(1000000)
    vec[42] = 10000
    vec[43] = 10000
    vec[44] = 10000

with Timer("one update"):
    csVec += vec
with Timer("finding heavy..."):
    print(csVec.findHH(5000))

with Timer("accumulate vec"):
    csVec += vec
    print("HHs", csVec.findHH(5000))

with Timer("accumulate csVec"):
    csVec += csVec
    print("HHs", csVec.findHH(5000))

with Timer("zeroing"):
    csVec.zero()
    csVec += vec
    print("HHs", csVec.findHH(5000))
