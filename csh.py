import numpy as np
LARGEPRIME = 2**61-1 
HIERARCHYBASE = 10

class CSH(object):
    """ Simple Hieratchical Count Sketch """
    def __init__(self, c, r, d, n):
        np.random.seed(42) 
        self.r = r  # num of rows       
        self.c = c  # num of columns
        self.d = d  # depth of the sketch (hierarchy)
        self.n = n  # dictionary size 
        # initialize sketches for all d levels of hierarchy 
        self.tables = [np.zeros((r, c)) for _ in range(d)]
        # initialize hashing functions for each row for all d levels of hierarchy
        # 2 random numbers for bucket hashes + 4 random numbers for sign hashes
        self.hashes = [np.random.randint(0, LARGEPRIME, (r, 6)).astype(int)
                       for _ in range(d)]

    def item2bs(self, item, H): 
        # computing bucket hashes  (2 wise independence)
        buckets = (H[:,0]*item + H[:,1])%LARGEPRIME%self.c
        # computing sign hashes (4 wise independence) 
        signs = (((H[:,2]*item + H[:,3])*item + H[:,4])*item + H[:,5])%LARGEPRIME%2 * 2 - 1 
        return (list(range(self.r)), tuple(buckets)), signs

    def update(self, item, value):
        # updating all levels of hierarchy  
        for h in range(self.d):
            itemH = item // (HIERARCHYBASE**h)
            # computing all hashes 
            buckets, signs = self.item2bs(itemH, self.hashes[h]) 
            # updating sketch 
            self.tables[h][buckets] += signs * value 
    
    def evalFreq(self, item, h):
        # computing hashes 
        buckets, signs = self.item2bs(item, self.hashes[h]) 
        # returning estimation of frequency for item 
        return np.median(self.tables[h][buckets] *signs) 
    
    def merge(self, csh):
        # merges csh sketch into self
        for h in range(self.d):
            self.tables[h] += csh.tables[h]

    def findHH(self, thr, prefix, h):
        # recursive search of items with frequency large than thr
        # s.t. item/(HIERARCHYBASE^h) = prefix 
        allHH = []
        for item in prefix*HIERARCHYBASE + np.arange(HIERARCHYBASE): 
            freq = self.evalFreq(item, h)
            if freq > thr: 
                allHH += self.findHH(thr, item, h - 1) if h > 0 else [(item,freq)]
        return allHH 

    def l2estimate(self):
        # l2 norm esimation from the sketch
        return np.sqrt(np.median(np.sum(self.tables[0]**2, 1)))
