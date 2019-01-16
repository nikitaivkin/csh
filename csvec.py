import math
import numpy as np
import copy
import torch

LARGEPRIME = 2**61-1
device = 'cuda' if torch.cuda.is_available() else 'cpu'

class CSVec(object):
    """ Simple Count Sketched Vector """
    def __init__(self, d, c, r, k=None, epsilon=None, doInitialize=True):
        self.r = r # num of rows
        self.c = c # num of columns
        # need int() here b/c annoying np returning np.int64...
        self.d = int(d) # vector dimensionality
        self.k = k # threshold for unsketching
        self.epsilon = epsilon # threshold for unsketching

        if not doInitialize:
            return

        # initialize the sketch
        self.table = torch.zeros((r, c), device=device)

        # initialize hashing functions for each row:
        # 2 random numbers for bucket hashes + 4 random numbers for
        # sign hashes
        # maintain existing random state so we don't mess with
        # the main module trying to set the random seed but still
        # get reproducible hashes for the same value of r
        rand_state = torch.random.get_rng_state()
        torch.random.manual_seed(42)
        self.hashes = torch.randint(0, LARGEPRIME, (r, 6),
                                    dtype=torch.int64, device=device)
        torch.random.set_rng_state(rand_state)

        tokens = torch.arange(self.d, dtype=torch.int64, device=device)
        tokens = tokens.reshape((1, self.d))

        # computing sign hashes (4 wise independence)
        h1 = self.hashes[:,2:3]
        h2 = self.hashes[:,3:4]
        h3 = self.hashes[:,4:5]
        h4 = self.hashes[:,5:6]
        self.signs = h1 * tokens
        self.signs.add_(h2).mul_(tokens).add_(h3).mul_(tokens).add_(h4)
        #self.signs = (((h1 * tokens + h2) * tokens + h3) * tokens + h4)
        self.signs = (self.signs % LARGEPRIME % 2).float().mul_(2).add_(-1)

        # computing bucket hashes  (2-wise independence)
        h1 = self.hashes[:,0:1]
        h2 = self.hashes[:,1:2]
        self.buckets = (h1 * tokens + h2) % LARGEPRIME % self.c

    def zero(self):
        self.table.zero_()

    def __deepcopy__(self, memodict={}):
        # don't initialize new CSVec, since that will calculate bc,
        # which is slow, even though we can just copy it over
        # directly without recomputing it
        newCSVec = CSVec(d=self.d, c=self.c, r=self.r,
                         k=self.k, epsilon=self.epsilon,
                         doInitialize=False)
        newCSVec.table   = copy.deepcopy(self.table)
        newCSVec.hashes  = copy.deepcopy(self.hashes)
        newCSVec.signs   = copy.deepcopy(self.signs)
        newCSVec.buckets = copy.deepcopy(self.buckets)
        # CORRECTNESS CHECK
        #newCSVec.bc      = copy.deepcopy(self.bc)
        return newCSVec

    def __add__(self, other):
        # a bit roundabout in order to avoid initializing a new CSVec
        returnCSVec = copy.deepcopy(self)
        returnCSVec += other
        return returnCSVec

    def __iadd__(self, other):
        if isinstance(other, CSVec):
            self.accumulateCSVec(other)
        elif isinstance(other, torch.Tensor):
            self.accumulateVec(other)
        else:
            raise ValueError("Can't add this to a CSVec: {}".format(other))
        return self

    def accumulateVec(self, vec):
        # updating the sketch
        assert(len(vec.size()) == 1 and vec.size()[0] == self.d)
        for r in range(self.r):
            self.table[r,:] += torch.bincount(input=self.buckets[r,:],
                                           weights=self.signs[r,:] * vec,
                                           minlength=self.c)

    def accumulateCSVec(self, csVec):
        # merges csh sketch into self
        assert(self.d == csVec.d)
        assert(self.c == csVec.c)
        assert(self.r == csVec.r)
        self.table += csVec.table

    def _findHHK(self, k):
        assert(k is not None)
        vals = self._findValues(torch.arange(self.d, device=device))
        HHs = torch.sort(vals**2)[1][-k:]
        return HHs, vals[HHs]

        # below is a potentially faster (but broken ha) version
        # of the code above. Leaving it here for now in case it
        # makes any speed difference later, but I kinda doubt it...
        # also it isn't even ported to pytorch

        # set a conservative threshold to quickly rule out
        # most possible heavy hitters
        thr = 0.1 / np.sqrt(self.c)
        mediumHitters = np.array([])
        while True:
            mediumHitters, mediumHittersVals = self._findHHThr(thr)
            if mediumHitters.size >= k:
                break

            # if mediumHitters.size < k even with this threshold,
            # use an even lower threshold next iteration
            # and warn the user
            print("THRESHOLD TOO HIGH!!")
            thr *= 0.1

        # get HHs from the medium hitters
        percentile = 100 * (1 - k / mediumHitters.size)
        cutoff = np.percentile(mediumHittersVals**2, percentile)
        HHs = np.where(mediumHittersVals**2 >= cutoff)[0]

        # in case there are multiple values of mediumHittersVals
        # exactly equal to cutoff, now take the topk by sorting
        if len(HHs) != k:
            assert(len(HHs) > k)
            HHs = np.argsort(mediumHittersVals[HHs]**2)[:k]
        HHValues = mediumHittersVals[HHs]

        return HHs, HHValues

    def _findHHThr(self, thr):
        assert(thr is not None)
        # to figure out which items are heavy hitters, check whether
        # self.table exceeds thr (in magnitude) in at least r/2 of
        # the rows. These elements are exactly those for which the median
        # exceeds thr, but computing the median is expensive, so only
        # calculate it after we identify which ones are heavy
        tablefiltered = (  (self.table >  thr).float()
                         - (self.table < -thr).float())
        est = torch.zeros(self.d, device=device) #XXX .float()?
        for r in range(self.r):
            est += tablefiltered[r,self.buckets[r,:]] * self.signs[r,:]
        est = (  (est >=  math.ceil(self.r/2.)).float()
               - (est <= -math.ceil(self.r/2.)).float())

        # HHs - heavy coordinates
        HHs = torch.nonzero(est)
        return HHs, self._findValues(HHs)

    def _findValues(self, coords):
        # estimating frequency of input coordinates
        vals = torch.zeros(self.r, coords.size()[0], device=device)
        for r in range(self.r):
            vals[r] = (self.table[r, self.buckets[r, coords]]
                       * self.signs[r, coords])

        # take the median over rows in the sketch
        return vals.median(dim=0)[0]

    def findHHs(self, k=None, thr=None):
        assert((k is None) != (thr is None))
        if k is not None:
            return self._findHHK(k)
        else:
            return self._findHHThr(thr)

    def unSketch(self):
        # either self.epsilon or self.k might be specified
        # (but not both). Act accordingly
        if self.epsilon is None:
            thr = None
        else:
            thr = self.epsilon * self.l2estimate()

        hhs = self.findHHs(k=self.k, thr=thr)

        if self.k is not None:
            assert(len(hhs[1]) == self.k)
        if self.epsilon is not None:
            assert((hhs[1] < thr).sum() == 0)

        # the unsketched vector is 0 everywhere except for HH
        # coordinates, which are set to the HH values
        unSketched = torch.zeros(self.d, device=device)
        unSketched[hhs[0]] = hhs[1]
        return unSketched

    def l2estimate(self):
        # l2 norm esimation from the sketch
        return np.sqrt(torch.median(torch.sum(self.table**2,1)).item())

