import math
import numpy as np
import copy
import pdb

from countSketch.csvec import CSVec
import torch
device="cuda"

LARGEPRIME = 2**61-1

class SlowCSVec(object):
    """ Simple Count Sketched Vector """
    def __init__(self, d, c, r, epsilon, k=None, doInitialize=True):
        self.r = r  # num of rows
        self.c = c  # num of columns
        self.d = d  # vector dimensionality
        self.epsilon = epsilon # threshold for unsketching
        self.k = k

        if not doInitialize:
            return

        # initialize the sketch
        self.table = np.zeros((r, c))

        # initialize hashing functions for each row:
        # 2 random numbers for bucket hashes + 4 random numbers for
        # sign hashes
        # maintain existing random state so we don't mess with
        # the main module trying to set the random seed but still
        # get reproducible hashes for the same value of r
        """
        rand_state = np.random.get_state()
        np.random.seed(42)
        self.hashes = np.random.randint(0, LARGEPRIME, (r, 6)).astype(int)
        np.random.set_state(rand_state)
        """
        rand_state = torch.random.get_rng_state()
        torch.random.manual_seed(42)
        self.hashes = torch.randint(0, LARGEPRIME, (r, 6),
                                    dtype=torch.int64, device=device)
        self.hashes = self.hashes.cpu().numpy()
        torch.random.set_rng_state(rand_state)


        tokens = np.arange(self.d).reshape((1, self.d))

        # computing sign hashes (4 wise independence)
        h1 = self.hashes[:,2:3]
        h2 = self.hashes[:,3:4]
        h3 = self.hashes[:,4:5]
        h4 = self.hashes[:,5:6]
        self.signs = (((h1 * tokens + h2) * tokens + h3) * tokens + h4)
        self.signs = self.signs % LARGEPRIME % 2 * 2 - 1

        # computing bucket hashes  (2-wise independence)
        h1 = self.hashes[:,0:1]
        h2 = self.hashes[:,1:2]
        self.buckets = (h1 * tokens + h2) % LARGEPRIME % self.c

        #computing bucket-coordinate mapping
        self.bc = []
        for r in range(self.r):
            self.bc.append([])
            for c in range(self.c):
                self.bc[-1].append(np.nonzero(self.buckets[r,:] == c)[0])

    def zero(self):
        self.table = np.zeros(self.table.shape)

    def __deepcopy__(self, memodict={}):
        # don't initialize new CSVec, since that will calculate bc,
        # which is slow, even though we can just copy it over
        # directly without recomputing it
        newCSVec = SlowCSVec(d=self.d, c=self.c, r=self.r,
                         epsilon=self.epsilon, k=self.k, doInitialize=False)
        newCSVec.table   = copy.deepcopy(self.table)
        newCSVec.hashes  = copy.deepcopy(self.hashes)
        newCSVec.signs   = copy.deepcopy(self.signs)
        newCSVec.buckets = copy.deepcopy(self.buckets)
        newCSVec.bc      = copy.deepcopy(self.bc)
        return newCSVec

    def __add__(self, other):
        # a bit roundabout in order to avoid initializing a new CSVec
        returnCSVec = copy.deepcopy(self)
        returnCSVec += other
        return returnCSVec

    def __iadd__(self, other):
        if isinstance(other, SlowCSVec):
            self.accumulateCSVec(other)
        elif isinstance(other, np.ndarray):
            self.accumulateVec(other)
        else:
            raise ValueError("Can't add this to a CSVec: {}".format(other))
        return self

    def accumulateVec(self, vec):
        #pdb.set_trace()
        # updating the sketch
        assert(len(vec.shape) == 1 and vec.size == self.d)
        for r in range(self.r):
            for c in range(self.c):
                #print(vec[self.bc[r][c]].shape,
                #      self.signs[r, self.bc[r][c]].shape)
                self.table[r,c] += np.sum(vec[self.bc[r][c]]
                                          * self.signs[r, self.bc[r][c]])

    def accumulateCSVec(self, csVec):
        # merges csh sketch into self
        assert(self.d == csVec.d)
        assert(self.c == csVec.c)
        assert(self.r == csVec.r)
        self.table += csVec.table

    def findHH(self, thr):
        # next 5 lines ensure that we compute the median
        # only for those who is heavy
        tablefiltered = 1 * (self.table > thr) - 1 * (self.table < -thr)
        est = np.zeros(self.d)
        for r in range(self.r):
            est += tablefiltered[r,self.buckets[r,:]] * self.signs[r,:]
        est = (  1 * (est >=  math.ceil(self.r/2.))
               - 1 * (est <= -math.ceil(self.r/2.)))

        # HHs- heavy coordinates
        HHs = np.nonzero(est)[0]

        # estimating frequency for heavy coordinates
        est = []
        for r in range(self.r):
            est.append(self.table[r,self.buckets[r,HHs]]
                       * self.signs[r,HHs])

        return HHs, np.median(np.array(est), 0)

    def unSketch(self):
        #pdb.set_trace()
        hhs = list(self.findHH(thr=0.001))#self.epsilon * self.l2estimate())
        hhs[0] = np.argsort(hhs[1]**2)[-self.k:]
        hhs[1] = hhs[1][hhs[0]]
        unSketched = np.zeros(self.d)
        unSketched[hhs[0]] = hhs[1]
        return unSketched

    def l2estimate(self):
        # l2 norm esimation from the sketch
        return np.sqrt(np.median(np.sum(self.table**2, 1)))


"""
if __name__ == "__main__": 
    np.random.seed(0)
    torch.random.manual_seed(0)

    D = 100000
    k=100
    fastC = CSVec(d=D, c=10000, r=5, k=k)
    slowC = SlowCSVec(d=D, c=10000, r=5, epsilon=0.0001)

    # make sure that the sketch (csvec.table) is equivalent
    # for the two methods of sketching
    vec = np.random.randn(D)
    fastC += torch.from_numpy(vec).float().cuda()
    slowC += vec

    print(np.abs(fastC.table.cpu().numpy() - slowC.table).sum())

    # make sure that fastC can recover topk heavy hitters
    trueHHs = np.random.randint(D, size=k)
    vec[trueHHs] *= 100
    fastC.zero()
    fastC += torch.from_numpy(vec).float().cuda()
    HHs = np.argsort(vec**2)[-k:]
    print(np.sort(trueHHs), np.sort(HHs))
    trueTopkVec = np.zeros(D)
    trueTopkVec[trueHHs] = vec[trueHHs]

    print(list(trueTopkVec[trueHHs]))
    print(list(fastC.unSketch().cpu().numpy()[trueHHs]))
    print(np.abs(trueTopkVec - fastC.unSketch().cpu().numpy()).sum())
"""
