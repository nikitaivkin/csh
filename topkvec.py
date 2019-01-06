import numpy as np
import copy

class SumOfTopKVec:
    def __init__(self, k, d):
        self.k = k
        self.d = d
        self.vec = np.zeros(d)

    def zero(self):
        self.vec = np.zeros(self.vec.shape)

    def __deepcopy__(self, memodict={}):
        newTopKVec = SumOfTopKVec(k=self.k, d=self.d)
        newTopKVec.vec = copy.deepcopy(self.vec)
        return newTopKVec

    def __iadd__(self, other):
        # note that the sum may have more than k non-zero elements
        if isinstance(other, SumOfTopKVec):
            # if adding a topkvec, add the entire vectors together
            assert(self.vec.shape == other.vec.shape)
            self.vec += other.vec
        elif isinstance(other, np.ndarray):
            # if adding an array, add only the top k
            assert(self.vec.shape == other.shape)
            topk = copy.deepcopy(other)
            cutoff = np.percentile(topk**2, 100*(1-self.k/topk.size))
            topk[topk**2 <= cutoff] = 0
            self.vec += topk
        else:
            raise ValueError("Can't add this to a SumOfTopKVec: {}".format(other))
        return self

    def __add__(self, other):
        new = copy.deepcopy(self)
        new += other
        return new

    def __del__(self):
        #print("DELETING TOP K!!!")
        pass

    def unSketch(self, thr):
        # ignore thr
        return self.vec

    def l2estimate(self):
        return np.linalg.norm(self.vec)

class TopKOfSumVec:
    def __init__(self, k, d):
        self.k = k
        self.d = d
        self.vec = np.zeros(d)

    def zero(self):
        self.vec = np.zeros(self.vec.shape)

    def __deepcopy__(self, memodict={}):
        newTopKVec = TopKOfSumVec(k=self.k, d=self.d)
        newTopKVec.vec = copy.deepcopy(self.vec)
        return newTopKVec

    def __iadd__(self, other):
        # note that the sum may have more than k non-zero elements
        if isinstance(other, TopKOfSumVec):
            # if adding a topkvec, add the entire vectors together
            assert(self.vec.shape == other.vec.shape)
            self.vec += other.vec
        elif isinstance(other, np.ndarray):
            # if adding an array, add only the top k
            assert(self.vec.shape == other.shape)
            self.vec += other
        else:
            raise ValueError("Can't add this to a TopKVec: {}".format(other))
        return self

    def __add__(self, other):
        new = copy.deepcopy(self)
        new += other
        return new

    def __del__(self):
        #print("DELETING TOP K!!!")
        pass

    def unSketch(self, thr):
        # ignore thr
        ret = copy.deepcopy(self.vec)
        cutoff = np.percentile(self.vec**2, 100*(1-self.k/self.vec.size))
        ret[ret**2 <= cutoff] = 0
        return ret

    def l2estimate(self):
        return np.linalg.norm(self.vec)
