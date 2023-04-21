#! /usr/src/python3
import time

import numpy as np

from ConjugateGradient import ConjugateGradient
from GaussSeidel import GaussSeidel
from Gradient import Gradient
from IterativeMethod import IterativeMethod
from Jacobi import Jacobi


def main():
    filename = "spa1.mtx"
    a, n, _ = IterativeMethod.read_matrix(filename)
    b = np.ones(n)
    j = Jacobi(a, b, 0.0001)
    s = GaussSeidel(a, b, 0.0001)
    g = Gradient(a, b, 0.0001)
    c = ConjugateGradient(filename, b, 0.0001)
    print('\n--------------------JACOBI--------------------\n')
    t0 = time.time()
    print('   Soluzione: ' + str(j.solve()[0]))
    t1 = time.time()
    print('   Tempo: ' + str("%.5f" % (t1 - t0)))
    print('\n-----------------GAUSSSEIDEL------------------\n')
    t0 = time.time()
    print('   Soluzione: ' + str(s.solve()[0]))
    t1 = time.time()
    print('   Tempo: ' + str("%.5f" % (t1 - t0)))
    print('\n------------------GRADIENTE-------------------\n')
    t0 = time.time()
    print('   Soluzione: ' + str(g.solve()[0]))
    t1 = time.time()
    print('   Tempo: ' + str("%.5f" % (t1 - t0)))
    print('\n-------------GRADIENTE CONIUGATO--------------\n')
    t0 = time.time()
    print('   Soluzione: ' + str(c.solve()[0]))
    t1 = time.time()
    print('   Tempo: ' + str("%.5f" % (t1 - t0)))


if __name__ == '__main__':
    main()
