from csh import SketchedArray
import numpy as np
import time

startTime = time.time()

ary = np.arange(300000)
sketchedAry = SketchedArray(ary)
print("sketch time", time.time() - startTime)

startTime = time.time()
unsketched = sketchedAry.unsketch(1e-6)
print("unsketch time", time.time() - startTime)

print(list(ary[-300:]))
print(list(unsketched[-300:]))
print("rmse", np.sqrt(np.mean(np.square(ary - unsketched))))
