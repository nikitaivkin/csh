import unittest
import csvec
from csvec import CSVec
import torch

class Base:
    # use Base class to hide CSVecTestCase from the unittest runner
    # we only want the subclasses to actually be run

    class CSVecTestCase(unittest.TestCase):
        def testRandomness(self):
            # make sure two sketches get the same hashes and signs
            d = 100
            c = 20
            r = 5
            a = CSVec(d, c, r, **self.csvecArgs)
            b = CSVec(d, c, r, **self.csvecArgs)
            self.assertTrue(torch.allclose(a.signs, b.signs))
            self.assertTrue(torch.allclose(a.buckets, b.buckets))
            self.assertTrue(torch.allclose(a.signs, b.signs))

            if self.numBlocks > 1:
                self.assertTrue(torch.allclose(a.blockOffsets,
                                               b.blockOffsets))
                self.assertTrue(torch.allclose(a.blockSigns,
                                               b.blockSigns))

        def testInit(self):
            # make sure the table starts out zeroed
            d = 100
            c = 20
            r = 5
            a = CSVec(d, c, r, **self.csvecArgs)
            zeros = torch.zeros(r, c).to(self.device)
            self.assertTrue(torch.allclose(a.table, zeros))

        def testSketchVec(self):
            # sketch a vector with all zeros except a single 1
            # then the table should be zeros everywhere except a single
            # 1 in each row
            d = 100
            c = 1
            r = 5
            a = CSVec(d=d, c=c, r=r, **self.csvecArgs)
            vec = torch.zeros(d).to(self.device)
            vec[0] = 1
            a.accumulateVec(vec)
            # make sure the sketch only has one nonzero entry per row
            for i in range(r):
                with self.subTest(row=i):
                    self.assertEqual(a.table[i,:].nonzero().numel(), 1)

            # make sure each row sums to +-1
            summed = a.table.abs().sum(dim=1).view(-1)
            ones = torch.ones(r).to(self.device)
            self.assertTrue(torch.allclose(summed, ones))

        def testZeroSketch(self):
            d = 100
            c = 20
            r = 5
            a = CSVec(d, c, r, **self.csvecArgs)
            vec = torch.rand(d).to(self.device)
            a.accumulateVec(vec)

            zeros = torch.zeros((r, c)).to(self.device)
            self.assertFalse(torch.allclose(a.table, zeros))

            a.zero()
            self.assertTrue(torch.allclose(a.table, zeros))

        def testUnsketch(self):
            # make sure heavy hitter recovery works correctly

            # use a gigantic sketch so there's no chance of collision
            d = 5
            c = 10000
            r = 20
            a = CSVec(d, c, r, **self.csvecArgs)
            vec = torch.rand(d).to(self.device)

            a.accumulateVec(vec)

            with self.subTest(method="topk"):
                recovered = a.unSketch(k=d)
                self.assertTrue(torch.allclose(recovered, vec))

            with self.subTest(method="epsilon"):
                thr = vec.abs().min() * 0.9
                recovered = a.unSketch(epsilon=thr / vec.norm())
                self.assertTrue(torch.allclose(recovered, vec))

        def testSketchSum(self):
            d = 5
            c = 10000
            r = 20

            summed = CSVec(d, c, r, **self.csvecArgs)
            for i in range(d):
                vec = torch.zeros(d).to(self.device)
                vec[i] = 1
                sketch = CSVec(d, c, r, **self.csvecArgs)
                sketch.accumulateVec(vec)
                summed += sketch

            recovered = summed.unSketch(k=d)
            trueSum = torch.ones(d).to(self.device)
            self.assertTrue(torch.allclose(recovered, trueSum))

        def testL2(self):
            d = 5
            c = 10000
            r = 20

            vec = torch.randn(d).to(self.device)
            a = CSVec(d, c, r, **self.csvecArgs)
            a.accumulateVec(vec)

            tol = 0.0001
            self.assertTrue((a.l2estimate() - vec.norm()).abs() < tol)

        def testMedian(self):
            d = 5
            c = 10000
            r = 20

            csvecs = [CSVec(d, c, r, **self.csvecArgs) for _ in range(3)]
            for i, csvec in enumerate(csvecs):
                vec = torch.arange(d).float().to(self.device) + i
                csvec.accumulateVec(vec)
            median = CSVec.median(csvecs)
            recovered = median.unSketch(k=d)
            trueMedian = torch.arange(d).float().to(self.device) + 1
            self.assertTrue(torch.allclose(recovered, trueMedian))

class TestCaseCPU1(Base.CSVecTestCase):
    def setUp(self):
        # hack to reset csvec's global cache between tests
        csvec.cache = {}

        self.device = "cpu"
        self.numBlocks = 1

        self.csvecArgs = {"numBlocks": self.numBlocks,
                          "device": self.device}

class TestCaseCPU2(Base.CSVecTestCase):
    def setUp(self):
        csvec.cache = {}

        self.device = "cpu"
        self.numBlocks = 2

        self.csvecArgs = {"numBlocks": self.numBlocks,
                          "device": self.device}

@unittest.skipIf(not torch.cuda.is_available(), "CUDA not available")
class TestCaseCUDA2(Base.CSVecTestCase):
    def setUp(self):
        csvec.cache = {}

        self.device = "cuda"
        self.numBlocks = 2

        self.csvecArgs = {"numBlocks": self.numBlocks,
                          "device": self.device}
