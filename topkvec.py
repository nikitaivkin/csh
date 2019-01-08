import numpy as np
import copy

class TopKVec:
    def __init__(self, d, k=None, epsilon=None):
        # exactly one of k and epsilon must be None
        assert((k is None) != (epsilon is None))
        self.k = k
        self.epsilon = epsilon
        self.d = d
        self.vec = np.zeros(d)

    def zero(self):
        self.vec = np.zeros(self.vec.shape)

    def __deepcopy__(self, memodict={}):
        newTopKVec = self.__class__(d=self.d, k=self.k,
                                    epsilon=self.epsilon)
        newTopKVec.vec = copy.deepcopy(self.vec)
        return newTopKVec

    def __add__(self, other):
        new = copy.deepcopy(self)
        new += other
        return new

    def l2estimate(self):
        return np.linalg.norm(self.vec)


class SumOfTopKVec(TopKVec):
    def __init__(self, d, k=None, epsilon=None):
        super().__init__(d, k, epsilon)

    def __iadd__(self, other):
        # note that the sum may have more than k non-zero elements
        if isinstance(other, SumOfTopKVec):
            # if adding a topkvec, add the entire vectors together
            assert(self.vec.shape == other.vec.shape)
            self.vec += other.vec
        elif isinstance(other, np.ndarray):
            # if adding an array, add only the largest elements
            assert(self.vec.shape == other.shape)
            topk = copy.deepcopy(other)
            # cutoff determined by whichever of k and epsilon is given
            if self.k is not None:
                cutoff = np.percentile(topk**2, 100*(1-self.k/topk.size))
            else:
                assert(self.epsilon is not None)
                cutoff = (self.epsilon * self.l2estimate())**2
            topk[topk**2 <= cutoff] = 0
            self.vec += topk
        else:
            raise ValueError("Can't add this to a SumOfTopKVec: {}".format(other))
        return self

    def unSketch(self):
        return self.vec

class TopKOfSumVec(TopKVec):
    def __init__(self, d, k=None, epsilon=None):
        super().__init__(d, k, epsilon)

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

    def unSketch(self):
        ret = copy.deepcopy(self.vec)
        if self.k is not None:
            percentile = 100*(1-self.k/self.vec.size)
            cutoff = np.percentile(self.vec**2, percentile)
        else:
            assert(self.epsilon is not None)
            cutoff = (self.epsilon * self.l2estimate())**2
        ret[ret**2 <= cutoff] = 0
        return ret
