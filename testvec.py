import numpy as np
from csh import CSH
from csvec import CS_VEC

print("running a toy example - sketch size 3x20 with 5 levels in hierarchy, dictionary size 10**6")
c = 200
r=3
d=100001 # dimension of the vector

csv = CS_VEC(c=200, r=3, d=100001) 
vec = np.arange(1000001)
vec[:]= 1
vec[42] = 10000
csv.updateVec(vec) 
print csv.findHH(5000)

