import math
import numpy as np
import copy
import torch
from csvec import *
import time

d = 50 * 10**6; c = 10**6; r = 10

n = 5
# dummy vectore with one heavy guy
g = [torch.randint(0,1000, (d,), dtype=torch.int64, device="cuda").float() for i in range(n)]

csv1 = CSVec( d, c, r, precomputeHashes=True)
csv2 = CSVec( d, c, r)

# make an update
t1 = time.clock()
for i in range(n):
  csv1.accumulateVec(g[i])

t2 = time.clock()
print ("precomputed and stored on host: ",(t2- t1)/n, " seconds per iteration")

t1 = time.clock()
for i in range(n):
  csv2.accumulateVec(g[i])

t2 = time.clock()
print ("compted on the fly: ",(t2- t1)/n, " seconds per iteration")
