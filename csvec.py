import math
import numpy as np
LARGEPRIME = 2**61-1 
HIERARCHYBASE = 10

class CS_VEC(object):
    """ Simple Hieratchical Count Sketch """
    def __init__(self, c, r, d):
        np.random.seed(42) 
        self.r = r  # num of rows       
        self.c = c  # num of columns
        self.d = d  # vector dimensionality 
        # initialize the sketch 
        self.table = np.zeros((r, c)) 
        # initialize hashing functions for each row: 
        # 2 random numbers for bucket hashes + 4 random numbers for sign hashes
        self.hashes = np.random.randint(0, LARGEPRIME, (r, 6)).astype(int) 
    
        vec = np.arange(self.d).reshape(( 1, self.d))
        # computing sign hashes (4 wise independence) 
        self.signs = (((self.hashes[:,2:3]*vec + self.hashes[:,3:4])*vec + self.hashes[:,4:5])*vec + self.hashes[:,5:6])%LARGEPRIME%2 * 2 - 1 
        # computing bucket hashes  (2-wise independence)
        self.buckets = (self.hashes[:,0:1]*vec + self.hashes[:,1:2])%LARGEPRIME%self.c
        #computing bucket-coordinate mapping
        self.bc = []
        for r in range(self.r):
            self.bc.append([])
            for c in range(self.c):
                self.bc[-1].append(np.nonzero(self.buckets[r,:]== c)[0])
     
     
    def updateVec(self, vec):
        # updating the sketch 
        for r in range(self.r):
            for c in range(self.c):
        #        print vec[self.bc[r][c]].shape, self.signs[r, self.bc[r][c]].shape 
                self.table[r,c] += np.sum(vec[self.bc[r][c]] * self.signs[r, self.bc[r][c]]) 

    def findHH(self, thr):
        # returning estimation of frequency for item 
        tablefiltered = 1 * (self.table > thr) - 1*(self.table < - thr)
        est = np.zeros(self.d)       
        for r in range(self.r):
            est += tablefiltered[r,self.buckets[r,:]]*  self.signs[r,:]
        est =  1 * (est >= math.ceil(self.r/2.)) - 1*(est <= - math.ceil(self.r/2.))
        HHs = np.nonzero(est)[0]  
        est = [] 
        for r in range(self.r):
            est.append(self.table[r,self.buckets[r,HHs]]*  self.signs[r,HHs]) 
        return HHs, np.median(np.array(est),0)

    def l2estimate(self):
        # l2 norm esimation from the sketch
        return np.sqrt(np.median(np.sum(self.tables[0]**2, 1)))

    def merge(self, csh):
        # merges csh sketch into self
        self.table += csh.table

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

