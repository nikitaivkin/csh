from __future__ import print_function

import numpy as np
from csh import CSH
from csvec import CSVec
import time


print("running a toy example - sketch size 5x1000, vector of dimension 10^6 to compress ")
t1 = time.clock()
print ("initializing ...")
csv = CSVec(c=1000, r=5, d=1000000) # d is the dimension of the vector
vec = np.ones(1000000)
vec[42] = 10000
vec[43] = 10000
vec[44] = 10000
print(time.clock() - t1 , 'sec')
t1 = time.clock()

print(("one update..."))
csv.updateVec(vec)
print(time.clock() - t1, 'sec')
t1 = time.clock()
print(("finding heavy..."))
print(csv.findHH(5000))
print(time.clock() - t1 , 'sec')

csv.updateVec(vec)
print("HHs", csv.findHH(5000))
