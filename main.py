#! /usr/src/python3
import time

import numpy as np

from GaussSeidel import GaussSeidel
from Gradient import Gradient
from Jacobi import Jacobi


def main():
    filename = "spa1.mtx"
    b = np.ones(1000)
    j = Jacobi(filename, b, 0.0001)
    s = GaussSeidel(filename, b, 0.0001)
    g = Gradient(filename, b, 0.0001)
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
