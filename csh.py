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
        self.tables = [np.zeros((r,c)) for _ in xrange(d)]
        # initialize hashing functions for each row for all d levels of hierarchy
        # 2 random numbers for bucket hashes + 4 random numbers for sign hashes
        self.hashes = [np.random.randint(0, LARGEPRIME, (r,6)) for _ in xrange(d)]

    def item2bs(self, item, H): 
        # computing bucket hashes  (2 wise independence)
        buckets = (H[:,0]*item + H[:,1])%LARGEPRIME%self.c
        # computing sign hashes (4 wise independence) 
        signs = (((H[:,2]*item + H[:,3])*item + H[:,4])*item + H[:,5])%LARGEPRIME%2 * 2 - 1 
        return [xrange(self.r), buckets], signs

    def update(self, item, value):
        # updating all levels of hierarchy  
        for h in xrange(self.d):
            itemH = item/(HIERARCHYBASE**h) 
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
        for h in xrange(self.d):
            self.tables[h] += csh.tables[h]

    def findHH(self, thr, prefix, h):
        # recursive search of items with frequency large than thr
        # s.t. item/(HIERARCHYBASE^h) = prefix 
        allHH =[] 
        for item in prefix*HIERARCHYBASE + np.arange(HIERARCHYBASE): 
            freq = self.evalFreq(item, h)
            if freq > thr: 
                allHH += self.findHH(thr, item, h - 1) if h>0 else [(item,freq)]
        return allHH 

    def l2estimate(self):
        # l2 norm esimation from the sketch
        return np.sqrt(np.median(np.sum(self.tables[0]**2, 1)))

if __name__=="__main__":
    print "running a toy example - sketch size 3x20 with 5 levels in hierarchy, dictionary size 10**6" 
    c= 20; r=3; d=5; n=10**6  
   
    print "sketch csh1 fed with 10^4 updates with item 43 appearing 500 times and everybody else once"
    csh1 = CSH(c, r, d, n) 
    for i in range(10**4):
        csh1.update(i,1)
        if i%20 ==0:  
            csh1.update(43,1) 
    
    print "sketch csh2 fed with 10^4 updates with item 42 appearing 1000 times and everybody else once"
    csh2 = CSH(c, r, d, n) 
    for i in range(10**4):
        csh2.update(i,1)
        if i%10 ==0:  
            csh2.update(42,1)
    # print out l2 norms for both streams    
    print "l2 norm for stream1: ", csh1.l2estimate(), "; l2 norm for stream2: ", csh2.l2estimate()
    csh2.merge(csh1)
    print "l2 norm from merged sketches", csh2.l2estimate()
    hhs = csh2.findHH(thr= 0.2 * csh2.l2estimate(), prefix=0, h=4)
    print "coordinates with values > 0.2 l2 norm: ", hhs 
