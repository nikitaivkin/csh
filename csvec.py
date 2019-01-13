import math
import numpy as np
import copy

LARGEPRIME = 2**61-1

class CSVec(object):
    """ Simple Count Sketched Vector """
    def __init__(self, d, c, r, epsilon, doInitialize=True):
        self.r = r  # num of rows
        self.c = c  # num of columns
        self.d = d  # vector dimensionality
        self.epsilon = epsilon # threshold for unsketching

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

        # computing bucket-coordinate mapping
        """
        # CORRECTNESS CHECK
        self.bc = []
        for r in range(self.r):
            self.bc.append([])
            for c in range(self.c):
                self.bc[-1].append(np.nonzero(self.buckets[r,:] == c)[0])
        """

    def zero(self):
        self.table = np.zeros(self.table.shape)

    def __deepcopy__(self, memodict={}):
        # don't initialize new CSVec, since that will calculate bc,
        # which is slow, even though we can just copy it over
        # directly without recomputing it
        newCSVec = CSVec(d=self.d, c=self.c, r=self.r,
                         epsilon=self.epsilon, doInitialize=False)
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
        elif isinstance(other, np.ndarray):
            self.accumulateVec(other)
        else:
            raise ValueError("Can't add this to a CSVec: {}".format(other))
        return self

    #@profile
    def accumulateVec(self, vec):
        # updating the sketch
        assert(len(vec.shape) == 1 and vec.size == self.d)

        # CORRECTNESS CHECK
        #table2 = copy.deepcopy(self.table)

        for r in range(self.r):
            # this is incorrect, since numpy only does one addition even
            # if there are multiple copies of the same index in buckets[r,:]
            #self.table[r, self.buckets[r,:]] += self.signs[r,:] * vec

            # this works but is slower than a python for loop
            #np.add.at(self.table[r,:], self.buckets[r,:], self.signs[r,:] * vec)

            # this works and is about 4x faster than the python loop below
            self.table[r,:] += np.bincount(self.buckets[r,:],
                                           weights=self.signs[r,:] * vec,
                                           minlength=self.c)

            """
            # CORRECTNESS CHECK (this is the original code)
            for c in range(self.c):
                tmp1 = vec[self.bc[r][c]]
                tmp2 = self.signs[r, self.bc[r][c]]
                tmp = tmp1 * tmp2
                tmp = np.sum(tmp)
                table2[r,c] += tmp
                #^^ is an expanded version of: self.table[r,c] += np.sum(self.signs[r, self.bc[r][c]] * vec[self.bc[r][c]])
            """

        # CORRECTNESS CHECK (show some arbitrary elements of self.table)
        #print("fast", self.table[3, :4])
        #print("slow", table2[3, :4])

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
        hhs = self.findHH(self.epsilon * self.l2estimate())
        unSketched = np.zeros(self.d)
        unSketched[hhs[0]] = hhs[1]
        return unSketched

    def l2estimate(self):
        # l2 norm esimation from the sketch
        return np.sqrt(np.median(np.sum(self.table**2, 1)))


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

