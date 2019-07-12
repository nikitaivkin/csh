# CSVec: Count Sketch Vector

## Installation
Dependencies: `pytorch` and `numpy`. Tested with `torch==1.0.1` and `numpy==1.15.3`, but this should work with a wide range of versions.

`git clone` the repository to your local machine, move to the directory containing `setup.py`, then run
```
pip install -e .
```
to install this package.

## Description

This package contains one main class, `CSVec`, which computes the Count Sketch of input vectors, and can extract heavy hitters from a Count Sketch.

Link to the Count Sketch paper -> http://www.mathcs.emory.edu/~cheung/Courses/584-StreamDB/Syllabus/papers/Frequency-count/FrequentStream.pdf
