#! /usr/src/python3
import time

import numpy as np

from GaussSeidel import GaussSeidel
from Gradient import Gradient
from IterativeMethod import IterativeMethod
from Jacobi import Jacobi


def main():
    filename = "spa2.mtx"
    a, n, _ = IterativeMethod.read_matrix(filename)
    b = np.ones(n)
    j = Jacobi(a, b, 0.0001)
    s = GaussSeidel(a, b, 0.0001)
    g = Gradient(a, b, 0.0001)
    t0 = time.time()
    print(j.solve()[0])
    t1 = time.time()
    print('tempo jacobi: ' + str(t1 - t0))
    t0 = time.time()
    print(s.solve()[0])
    t1 = time.time()
    print('tempo gauss-seidel: ' + str(t1 - t0))
    t0 = time.time()
    print(g.solve()[0])
    t1 = time.time()
    print('tempo gradiente: ' + str(t1 - t0))


if __name__ == '__main__':
    main()
