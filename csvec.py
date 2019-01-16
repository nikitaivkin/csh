import math
import numpy as np
import copy
import torch

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
        self.hashes = torch.from_numpy(self.hashes).cuda()
        np.random.set_state(rand_state)

        tokens = np.arange(self.d).reshape((1, self.d))
        tokens = torch.from_numpy(tokens).cuda()

        # computing sign hashes (4 wise independence)
        h1 = self.hashes[:,2:3]
        h2 = self.hashes[:,3:4]
        h3 = self.hashes[:,4:5]
        h4 = self.hashes[:,5:6]
        self.signs = (((h1 * tokens + h2) * tokens + h3) * tokens + h4)
        self.signs = (self.signs % LARGEPRIME % 2 * 2 - 1).float()

        # computing bucket hashes  (2-wise independence)
        h1 = self.hashes[:,0:1]
        h2 = self.hashes[:,1:2]
        self.buckets = (h1 * tokens + h2) % LARGEPRIME % self.c

        self.table = torch.from_numpy(self.table).float().cuda()

    def zero(self):
        self.table[] = np.zeros(self.table.shape)

#    # Nikita:  __deepcopy__ is the only used in __add__, 
#    #         therefore redundunt, we can have just __iadd__,
    # def __deepcopy__(self, memodict={}):
    #     # don't initialize new CSVec, since that will calculate bc,
    #     # which is slow, even though we can just copy it over
    #     # directly without recomputing it
    #     newCSVec = CSVec(d=self.d, c=self.c, r=self.r,
    #                      epsilon=self.epsilon, doInitialize=False)
    #     newCSVec.table   = copy.deepcopy(self.table)
    #     newCSVec.hashes  = copy.deepcopy(self.hashes)
    #     newCSVec.signs   = copy.deepcopy(self.signs)
    #     newCSVec.buckets = copy.deepcopy(self.buckets)
    #     # CORRECTNESS CHECK
    #     #newCSVec.bc      = copy.deepcopy(self.bc)
    #     return newCSVec
#    # Nikita: redundunt, we can have just __iadd__ 
    # def __add__(self, other):
    #     # a bit roundabout in order to avoid initializing a new CSVec
    #     returnCSVec = copy.deepcopy(self)
    #     returnCSVec += other
    #     return returnCSVec

    def __iadd__(self, other):
        if isinstance(other, ptCSVec):
            self.accumulateCSVec(other)
        elif isinstance(other, torch.Tensor):
            self.accumulateVec(other)
        else:
            raise ValueError("Can't add this to a CSVec: {}".format(other))
        return self

    #@profile
    def accumulateVec(self, vec):
        # updating the sketch
        # Nikita: this assert need to be fixed
        # assert(len(vec.shape) == 1 and vec.size == self.d)
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

    def findHH(self, thr):
        # next 5 lines ensure that we compute the median
        # only for those who is heavy
        tablefiltered = (self.table > thr).float() - (self.table < -thr).float()
        est = torch.zeros(self.d, device=torch.device("cuda")).float()
        for r in range(self.r):
            est += tablefiltered[r,self.buckets[r,:]] * self.signs[r,:]
        est = (  1 * (est >=  math.ceil(self.r/2.)).float()
               - 1 * (est <= -math.ceil(self.r/2.)).float())
        
        # HHs- heavy coordinates
        HHs = torch.nonzero(est)[0]

        # estimating frequency for heavy coordinates
        est = []
        for r in range(self.r):
            est.append((self.table[r,self.buckets[r,HHs]]
                        *self.signs[r,HHs]).cpu().numpy())
            
        return HHs.cpu().numpy(), np.median(np.array(est), 0)

    def unSketch(self):
        hhs = self.findHH(self.epsilon * self.l2estimate())
        print(hhs)
        unSketched = np.zeros(self.d)
        unSketched[hhs[0]] = hhs[1]
        return unSketched

    def l2estimate(self):
        # l2 norm esimation from the sketch
        return np.sqrt(torch.median(torch.sum(self.table**2,1)).item())


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

