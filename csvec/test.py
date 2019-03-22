import numpy as np
from csh import CSH

print("running a toy example - sketch size 3x20 with 5 levels in hierarchy, dictionary size 10**6")
c = 200
r=3
d=5
n=10**6  

print("sketch csh1 fed with 10^4 updates with item 43 appearing 500 times and everybody else once")
csh1 = CSH(c, r, d, n) 
for i in range(10**4):
    csh1.update(i,1)
    if i%20 ==0:
        csh1.update(43,1)
        csh1.update(44,1)
        csh1.update(45,1)
        csh1.update(46,1)
        csh1.update(47,1)
        csh1.update(48,1)
        csh1.update(49,1)
        csh1.update(50,1)

print("sketch csh2 fed with 10^4 updates with item 42 appearing 1000 times and everybody else once")
csh2 = CSH(c, r, d, n) 
for i in range(10000):
    csh2.update(i,1)
    if i%10 == 0:
        csh2.update(42,1)
# print out l2 norms for both streams    
print("l2 norm for stream1: ", csh1.l2estimate(), "; l2 norm for stream2: ", csh2.l2estimate())
csh2.merge(csh1)
print("l2 norm from merged sketches", csh2.l2estimate())
hhs = csh2.findHH(thr= 0.2 * csh2.l2estimate(), prefix=0, h=4)
print("coordinates with values > 0.2 l2 norm: ", hhs)
