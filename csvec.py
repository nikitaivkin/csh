import math
import numpy as np
LARGEPRIME = 2**61-1

class CSVec(object):
    """ Simple Count Sketched Vector """
    def __init__(self, c, r, d):
        self.r = r  # num of rows
        self.c = c  # num of columns
        self.d = d  # vector dimensionality

        # initialize the sketch
        self.table = np.zeros((r, c))

        # initialize hashing functions for each row:
        # 2 random numbers for bucket hashes + 4 random numbers for
        # sign hashes
        # maintain existing random state so we don't mess with
        # the main module trying to set the random seed but still
        # get reproducible hashes for the same value of r
        rand_state = np.random.get_state()
        np.random.seed(42)
        self.hashes = np.random.randint(0, LARGEPRIME, (r, 6)).astype(int)
        np.random.set_state(rand_state)

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

    def __add__(self, other):
        msg = "Don't do this -- it takes too long to create a new " + \
              "CSVec. Just CSVec.zero() this one if needed and then " + \
              "use +="
        raise NotImplementedError(msg)

    def __iadd__(self, other):
        if isinstance(other, CSVec):
            self.accumulateCSVec(other)
        elif isinstance(other, np.ndarray):
            self.accumulateVec(other)
        else:
            raise ValueError("Can't add this to a CSVec: {}".format(other))
        return self

    def accumulateVec(self, vec):
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

    def l2estimate(self):
        # l2 norm esimation from the sketch
        return np.sqrt(np.median(np.sum(self.tables[0]**2, 1)))


#    def vec2bs(self, vec):
#        vec.shape = 1, len(vec)
#        # computing bucket hashes  (2-wise independence)
#        buckets = (self.hashes[:,0:1]*vec + self.hashes[:,1:2])%LARGEPRIME%self.c
#        # computing sign hashes (4 wise independence)
#        signs = (((self.hashes[:,2:3]*vec + self.hashes[:,3:4])*vec + self.hashes[:,4:5])*vec + self.hashes[:,5:6])%LARGEPRIME%2 * 2 - 1
#        vec.shape =  vec.shape[1]
#        return buckets, signs
#
#    def updateVec(self, vec):
#        # computing all hashes \\ can potentially precompute and store
#        buckets, signs = self.vec2bs(np.arange(self.d))
#        # updating the sketch
#        print self.table
#        for r in range(self.r):
#            self.table[r,buckets[r,:]] += vec * signs[r,:]
#        print self.table

#    def evalFreq(self):
#        # computing hashes
#        buckets, signs = self.vec2bs(np.arange(self.d))
#        # returning estimation of frequency for item
#        estimates = [self.table[r,buckets[r,:]] * signs[r,:] for r in range(self.r)]
#        print self.table#[r,buckets[r,:]]
#        print estimates
#        #return np.median(estimates,0)
#

