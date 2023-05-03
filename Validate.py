#! /usr/src/python3
import time

import numpy as np
from scipy.linalg import blas

from ConjugateGradient import ConjugateGradient
from GaussSeidel import GaussSeidel
from Gradient import Gradient
from IterativeMethod import IterativeMethod
from Jacobi import Jacobi


class Validate:

    @staticmethod
    def test():
        names = ["spa1.mtx", "spa2.mtx", "vem1.mtx", "vem2.mtx"]
        tolerances = [1e-4, 1e-6, 1e-8, 1e-10]
        for filename in names:
            print('\n--------------------' + filename + '----------------------\n')
            a, n, _ = IterativeMethod.read_matrix(filename)
            x = np.ones(n)
            b = a @ x
            for tol in tolerances:
                print("\tTOLLERANZA: " + str(tol))
                Validate.validate(a, b, x, tol)

    @staticmethod
    def validate(a, b, x, tol):
        methods = (Jacobi, GaussSeidel, Gradient, ConjugateGradient)
        method_names = ('JACOBI', 'GAUSS-SEIDEL', 'GRADIENT', 'CONJUGATE GRADIENT')
        for method, name in zip(methods, method_names):
            m = method(a, b, tol)
            print('\n\t--------------------' + name + '--------------------\n')
            t0 = time.time()
            xh, k = m.solve()
            t1 = time.time()
            print('\t   Errore relativo: ' + str(Validate.rel_err(xh, x)))
            print('\t   Iterazioni: ' + str(k))
            print('\t   Tempo: ' + str("%.5f" % (t1 - t0)) + '\n')

    @staticmethod
    def validate_method(method, x):
        t0 = time.time()
        xh, k = method.solve()
        t1 = time.time()
        err = Validate.rel_err(xh, x)
        tf = (t1 - t0)
        return err, k, tf

    @staticmethod
    def rel_err(xt, x):
        a = blas.dnrm2(xt - x)
        b = blas.dnrm2(x)
        return a / b
    