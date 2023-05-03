#! /usr/src/python3
import time

import numpy as np
from scipy.linalg import blas
from tqdm import tqdm
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
        print("Generating report.txt")
        bar = tqdm(range(0, 16), ncols=100, desc="Progress")
        report = open("report.txt", "w")
        for filename in names:
            report.write('\n--------------------' + filename + '----------------------\n')
            a, n, _ = IterativeMethod.read_matrix(filename)
            x = np.ones(n)
            b = a @ x
            for tol in tolerances:
                report.write("\tTOLLERANZA: " + str(tol) + "\n")
                out = Validate.validate(a, b, x, tol)
                report.write(out)
                bar.update()
        report.close()

    @staticmethod
    def validate(a, b, x, tol):
        out = ""
        methods = (Jacobi, GaussSeidel, Gradient, ConjugateGradient)
        method_names = ('JACOBI', 'GAUSS-SEIDEL', 'GRADIENT', 'CONJUGATE GRADIENT')
        for method, name in zip(methods, method_names):
            m = method(a, b, tol)
            out += '\n\t--------------------' + name + '--------------------\n' + '\n'
            t0 = time.time()
            xh, k = m.solve()
            t1 = time.time()
            out += '\t   Errore relativo: ' + str(Validate.rel_err(xh, x)) + '\n'
            out += '\t   Iterazioni: ' + str(k) + '\n'
            out += '\t   Tempo: ' + str("%.5f" % (t1 - t0)) + '\n' + '\n'
        return out

    @staticmethod
    def validate_method(method, a, b, tol, x):
        m = None
        if method == "Jacobi":
            m = Jacobi(a, b, tol)
        elif method == "GaussSeidel":
            m = GaussSeidel(a, b, tol)
        elif method == "Gradient":
            m = Gradient(a, b, tol)
        elif method == "Conjugate":
            m = ConjugateGradient(a, b, tol)
        else:
            print("error")
        t0 = time.time()
        xh, k = m.solve()
        t1 = time.time()
        err = Validate.rel_err(xh, x)
        tf = (t1 - t0)
        return err, k, tf

    @staticmethod
    def rel_err(xt, x):
        a = blas.dnrm2(xt - x)
        b = blas.dnrm2(x)
        return a / b

