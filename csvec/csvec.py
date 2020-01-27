import math
import numpy as np
import copy
import torch

LARGEPRIME = 2**61-1

cache = {}

#import line_profiler
#import atexit
#profile = line_profiler.LineProfiler()
#atexit.register(profile.print_stats)

class CSVec(object):
    """ Count Sketch of a vector

    Treating a vector as a stream of tokens with associated weights,
    this class computes the count sketch of an input vector, and
    supports operations on the resulting sketch.

    public methods: zero, unSketch, l2estimate, __add__, __iadd__
    """

    def __init__(self, d, c, r, doInitialize=True, device=None,
                 numBlocks=1):
        """ Constductor for CSVec

        Args:
            d: the cardinality of the skteched vector
            c: the number of columns (buckets) in the sketch
            r: the number of rows in the sketch
            doInitialize: if False, you are responsible for setting
                self.table, self.signs, self.buckets, self.blockSigns,
                and self.blockOffsets
            device: which device to use (cuda or cpu). If None, chooses
                cuda if available, else cpu
            numBlocks: mechanism to reduce memory consumption. A value
                of 1 leads to a normal sketch. Higher values reduce
                peak memory consumption proportionally but decrease
                randomness of the hashes
        Note:
            Since sketching a vector always requires the hash functions
            to be evaluated for all of 0..d-1, we precompute the
            hash values in the constructor. However, this takes d*r
            memory, which is sometimes too big. We therefore only
            compute hashes of 0..(d/numBlocks - 1), and we let the
            hash of all other tokens be the hash of that token modulo
            d/numBlocks. In order to recover some of the lost randomness,
            we add a random number to each "block" (self.blockOffsets)
            and multiply each block by a random sign (self.blockSigns)
        """

        # save random quantities in a module-level variable so we can
        # reuse them if someone else makes a sketch with the same d, c, r
        global cache

        self.r = r # num of rows
        self.c = c # num of columns
        # need int() here b/c annoying np returning np.int64...
        self.d = int(d) # vector dimensionality

        # reduce memory consumption of signs & buckets by constraining
        # them to be repetitions of a single block
        self.numBlocks = numBlocks

        # choose the device automatically if none was given
        if device is None:
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
        else:
            if (not isinstance(device, torch.device) and
                    not ("cuda" in device or device == "cpu")):
                msg = "Expected a valid device, got {}"
                raise ValueError(msg.format(device))

        self.device = device

        # this flag indicates that the caller plans to set up
        # self.signs, self.buckets, self.blockSigns, and self.blockOffsets
        # itself (e.g. self.deepcopy does this)
        if not doInitialize:
            return

        # initialize the sketch to all zeros
        self.table = torch.zeros((r, c), device=self.device)

        # if we already have these, don't do the same computation
        # again (wasting memory storing the same data several times)
        cacheKey = (d, c, r, numBlocks, device)
        if cacheKey in cache:
            self.signs = cache[cacheKey]["signs"]
            self.buckets = cache[cacheKey]["buckets"]
            if self.numBlocks > 1:
                self.blockSigns = cache[cacheKey]["blockSigns"]
                self.blockOffsets = cache[cacheKey]["blockOffsets"]
            return

        # initialize hashing functions for each row:
        # 2 random numbers for bucket hashes + 4 random numbers for
        # sign hashes
        # maintain existing random state so we don't mess with
        # the main module trying to set the random seed but still
        # get reproducible hashes for the same value of r

        # do all these computations on the CPU, since pytorch
        # is incapable of in-place mod, and without that, this
        # computation uses up too much GPU RAM
        rand_state = torch.random.get_rng_state()
        torch.random.manual_seed(42)
        hashes = torch.randint(0, LARGEPRIME, (r, 6),
                               dtype=torch.int64, device="cpu")

        # compute random blockOffsets and blockSigns
        if self.numBlocks > 1:
            nTokens = self.d // numBlocks
            if self.d % numBlocks != 0:
                # so that we only need numBlocks repetitions
                nTokens += 1
            self.blockSigns = torch.randint(0, 2, size=(self.numBlocks,),
                                            device=self.device) * 2 - 1
            self.blockOffsets = torch.randint(0, self.c,
                                              size=(self.numBlocks,),
                                              device=self.device)
        else:
            assert(numBlocks == 1)
            nTokens = self.d

        torch.random.set_rng_state(rand_state)

        # tokens are the indices of the vector entries
        tokens = torch.arange(nTokens, dtype=torch.int64, device="cpu")
        tokens = tokens.reshape((1, nTokens))

        # computing sign hashes (4 wise independence)
        h1 = hashes[:,2:3]
        h2 = hashes[:,3:4]
        h3 = hashes[:,4:5]
        h4 = hashes[:,5:6]
        self.signs = (((h1 * tokens + h2) * tokens + h3) * tokens + h4)
        self.signs = ((self.signs % LARGEPRIME % 2) * 2 - 1).float()

        # only move to device now, since this computation takes too
        # much memory if done on the GPU, and it can't be done
        # in-place because pytorch (1.0.1) has no in-place modulo
        # function that works on large numbers
        self.signs = self.signs.to(self.device)

        # computing bucket hashes (2-wise independence)
        h1 = hashes[:,0:1]
        h2 = hashes[:,1:2]
        self.buckets = ((h1 * tokens) + h2) % LARGEPRIME % self.c

        # only move to device now. See comment above.
        # can't cast this to int, unfortunately, since we index with
        # this below, and pytorch only lets us index with long
        # tensors
        self.buckets = self.buckets.to(self.device)

        cache[cacheKey] = {"signs": self.signs,
                           "buckets": self.buckets}
        if numBlocks > 1:
            cache[cacheKey].update({"blockSigns": self.blockSigns,
                                    "blockOffsets": self.blockOffsets})

    def zero(self):
        """ Set all the entries of the sketch to zero """
        self.table.zero_()

    def cpu_(self):
        self.device = "cpu"
        self.table = self.table.cpu()

    def cuda_(self, device="cuda"):
        self.device = device
        self.table = self.table.cuda()

    def half_(self):
        self.table = self.table.half()

    def float_(self):
        self.table = self.table.float()

    def __deepcopy__(self, memodict={}):
        # don't initialize new CSVec, since that will calculate bc,
        # which is slow, even though we can just copy it over
        # directly without recomputing it
        newCSVec = CSVec(d=self.d, c=self.c, r=self.r,
                         doInitialize=False, device=self.device,
                         numBlocks=self.numBlocks)
        newCSVec.table = copy.deepcopy(self.table)
        global cache
        cachedVals = cache[(self.d, self.c, self.r, self.numBlocks, self.device)]
        newCSVec.signs = cachedVals["signs"]
        newCSVec.buckets = cachedVals["buckets"]
        if self.numBlocks > 1:
            newCSVec.blockSigns = cachedVals["blockSigns"]
            newCSVec.blockOffsets = cachedVals["blockOffsets"]
        return newCSVec

    def __imul__(self, other):
        if isinstance(other, int) or isinstance(other, float):
            self.table = self.table.mul_(other)
        else:
            raise ValueError(f"Can't multiply a CSVec by {other}")
        return self

    def __truediv__(self, other):
        if isinstance(other, int) or isinstance(other, float):
            self.table = self.table.div_(other)
        else:
            raise ValueError(f"Can't divide a CSVec by {other}")
        return self

    def __add__(self, other):
        """ Returns the sum of self with other

        Args:
            other: a CSVec with identical values of d, c, and r
        """
        # a bit roundabout in order to avoid initializing a new CSVec
        returnCSVec = copy.deepcopy(self)
        returnCSVec += other
        return returnCSVec

    def __iadd__(self, other):
        """ Accumulates another sketch

        Args:
            other: a CSVec with identical values of d, c, r, device, numBlocks
        """
        if isinstance(other, CSVec):
            # merges csh sketch into self
            assert(self.d == other.d)
            assert(self.c == other.c)
            assert(self.r == other.r)
            assert(self.device == other.device)
            assert(self.numBlocks == other.numBlocks)
            self.table += other.table
        else:
            raise ValueError("Can't add this to a CSVec: {}".format(other))
        return self

    def accumulateTable(self, table):
        """ Adds a CSVec.table to self

        Args:
            table: the table to be added

        """
        if table.size() != self.table.size():
            msg = "Passed in table has size {}, expecting {}"
            raise ValueError(msg.format(table.size(), self.table.size()))

        self.table += table

    def accumulateVec(self, vec):
        """ Sketches a vector and adds the result to self

        Args:
            vec: the vector to be sketched
        """
        assert(len(vec.size()) == 1 and vec.size()[0] == self.d)

        # the vector is sketched to each row independently
        for r in range(self.r):
            buckets = self.buckets[r,:].to(self.device)
            signs = self.signs[r,:].to(self.device)
            # the main computation here is the bincount below, but
            # there's lots of index accounitng leading up to it due
            # to numBlocks being potentially > 1
            for blockId in range(self.numBlocks):
                start = blockId * buckets.size()[0]
                end = (blockId + 1) * buckets.size()[0]
                end = min(end, self.d)
                offsetBuckets = buckets[:end-start].clone()
                offsetSigns = signs[:end-start].clone()
                if self.numBlocks > 1:
                    offsetBuckets += self.blockOffsets[blockId]
                    offsetBuckets %= self.c
                    offsetSigns *= self.blockSigns[blockId]
                # bincount computes the sum of all values in the vector
                # that correspond to each bucket
                self.table[r,:] += torch.bincount(
                                    input=offsetBuckets,
                                    weights=offsetSigns * vec[start:end],
                                    minlength=self.c
                                   )

    def _findHHK(self, k):
        assert(k is not None)
        #tokens = torch.arange(self.d, device=self.device)
        #vals = self._findValues(tokens)
        vals = self._findAllValues()

        # sort is faster than torch.topk...
        #HHs = torch.sort(vals**2)[1][-k:]

        # topk on cuda returns what looks like uninitialized memory if
        # vals has nan values in it
        # saving to a zero-initialized output array instead of using the
        # output of topk appears to solve this problem
        outVals = torch.zeros(k, device=vals.device)
        HHs = torch.zeros(k, device=vals.device).long()
        torch.topk(vals**2, k, sorted=False, out=(outVals, HHs))
        return HHs, vals[HHs]

    def _findHHThr(self, thr):
        assert(thr is not None)
        vals = self._findAllValues()
        HHs = vals.abs() >= thr
        return HHs, vals[HHs]

        """ this is a potentially faster way to compute the same thing,
        but it doesn't play nicely with numBlocks > 1, so for now I'm
        just using the slower code above

        # to figure out which items are heavy hitters, check whether
        # self.table exceeds thr (in magnitude) in at least r/2 of
        # the rows. These elements are exactly those for which the median
        # exceeds thr, but computing the median is expensive, so only
        # calculate it after we identify which ones are heavy
        tablefiltered = (  (self.table >  thr).float()
                         - (self.table < -thr).float())
        est = torch.zeros(self.d, device=self.device)
        for r in range(self.r):
            est += tablefiltered[r, self.buckets[r,:]] * self.signs[r, :]
        est = (  (est >=  math.ceil(self.r/2.)).float()
               - (est <= -math.ceil(self.r/2.)).float())

        # HHs - heavy coordinates
        HHs = torch.nonzero(est)
        return HHs, self._findValues(HHs)
        """

    def _findValues(self, coords):
        # estimating frequency of input coordinates
        assert(self.numBlocks == 1)
        d = coords.size()[0]
        vals = torch.zeros(self.r, self.d, device=self.device)
        for r in range(self.r):
            vals[r] = (self.table[r, self.buckets[r, coords]]
                       * self.signs[r, coords])
        return vals.median(dim=0)[0]

    def _findAllValues(self):
        if self.numBlocks == 1:
            vals = torch.zeros(self.r, self.d, device=self.device)
            for r in range(self.r):
                vals[r] = (self.table[r, self.buckets[r,:]]
                           * self.signs[r,:])
            return vals.median(dim=0)[0]
        else:
            medians = torch.zeros(self.d, device=self.device)
            for blockId in range(self.numBlocks):
                start = blockId * self.buckets.size()[1]
                end = (blockId + 1) * self.buckets.size()[1]
                end = min(end, self.d)
                vals = torch.zeros(self.r, end-start, device=self.device)
                for r in range(self.r):
                    buckets = self.buckets[r, :end-start]
                    signs = self.signs[r, :end-start]
                    offsetBuckets = buckets + self.blockOffsets[blockId]
                    offsetBuckets %= self.c
                    offsetSigns = signs * self.blockSigns[blockId]
                    vals[r] = (self.table[r, offsetBuckets]
                                * offsetSigns)
                medians[start:end] = vals.median(dim=0)[0]
            return medians

    def _findHHs(self, k=None, thr=None):
        assert((k is None) != (thr is None))
        if k is not None:
            return self._findHHK(k)
        else:
            return self._findHHThr(thr)

    def unSketch(self, k=None, epsilon=None):
        """ Performs heavy-hitter recovery on the sketch

        Args:
            k: if not None, the number of heavy hitters to recover
            epsilon: if not None, the approximation error in the recovery.
                The returned heavy hitters are estimated to be greater
                than epsilon * self.l2estimate()

        Returns:
            A vector containing the heavy hitters, with zero everywhere
            else

        Note:
            exactly one of k and epsilon must be non-None
        """

        # either epsilon or k might be specified
        # (but not both). Act accordingly
        if epsilon is None:
            thr = None
        else:
            thr = epsilon * self.l2estimate()

        hhs = self._findHHs(k=k, thr=thr)

        if k is not None:
            assert(len(hhs[1]) == k)
        if epsilon is not None:
            assert((hhs[1] < thr).sum() == 0)

        # the unsketched vector is 0 everywhere except for HH
        # coordinates, which are set to the HH values
        unSketched = torch.zeros(self.d, device=self.device)
        unSketched[hhs[0]] = hhs[1]
        return unSketched

    def l2estimate(self):
        """ Return an estimate of the L2 norm of the sketch """
        # l2 norm esimation from the sketch
        return np.sqrt(torch.median(torch.sum(self.table**2,1)).item())

    @classmethod
    def median(cls, csvecs):
        # make sure all CSVecs match
        d = csvecs[0].d
        c = csvecs[0].c
        r = csvecs[0].r
        device = csvecs[0].device
        numBlocks = csvecs[0].numBlocks
        for csvec in csvecs:
            assert(csvec.d == d)
            assert(csvec.c == c)
            assert(csvec.r == r)
            assert(csvec.device == device)
            assert(csvec.numBlocks == numBlocks)

        tables = [csvec.table for csvec in csvecs]
        med = torch.median(torch.stack(tables), dim=0)[0]
        returnCSVec = copy.deepcopy(csvecs[0])
        returnCSVec.table = med
        return returnCSVec
