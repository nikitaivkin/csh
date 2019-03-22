from csvec import CSVec
import torch

c = CSVec(d=10**4, c=1000, r=100, k=10)
a = torch.ones(10**4).float().cuda()
for i in range(240, 250):
    a[i] = 100000
c += a

#rec = c.unSketch()
hhs, rec = c._findHHK(10)
#hhs = rec.nonzero()
print(hhs)
print(rec)
