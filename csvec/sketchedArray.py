import numpy as np
from .csh import CSH

class SketchedArray:
    def __init__(self, ary=None, d=5, c=800, r=120):
        self.d = d
        self.c = c
        self.r = r
        self.csh = CSH(self.c, self.r, self.d, n=0)
        if ary is not None:
            self.aryLen = len(ary)
            self._sketch(ary)

    def __add__(self, other):
        assert(self.d == other.d)
        assert(self.c == other.c)
        assert(self.r == other.r)
        # this last assert probably not actually necessary,
        # but needed to maintain the array abstraction
        assert(self.aryLen == other.aryLen)
        result = CSH(self.c, self.r, self.d, n=0)
        result.merge(self.csh)
        result.merge(other.csh)
        resultSA = SketchedArray(d=self.d, c=self.c, r=self.r)
        resultSA.aryLen = self.aryLen
        resultSA.csh = result
        return resultSA

    def _sketch(self, ary):
        for i in range(len(ary)):
            self.csh.update(i, ary[i])

    def unsketch(self, l2Frac):
        assert(self.csh is not None)
        hhs = self.csh.findHH(thr=l2Frac * self.csh.l2estimate(),
                              prefix=0, h=4)
        hhs = np.array(hhs)
        ary = np.zeros(self.aryLen)
        # remove HHs that are outside the bounds of the array
        # since those are clearly false positives
        hhs = np.where((hhs[:,0] < self.aryLen)[:,np.newaxis], hhs, np.zeros(hhs.shape))
        ary[hhs[:,0].astype(np.int)] = hhs[:,1]
        return ary


