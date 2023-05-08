#! /usr/src/python3
import time

import numpy as np
from scipy.linalg import blas
from tqdm import tqdm
from methods.ConjugateGradient import ConjugateGradient
from methods.GaussSeidel import GaussSeidel
from methods.Gradient import Gradient
from methods.IterativeMethod import IterativeMethod
from methods.Jacobi import Jacobi


class Validate:

    @staticmethod
    def test():
        names = ["spa1.mtx", "spa2.mtx", "vem1.mtx", "vem2.mtx"]
        tolerances = [1e-4, 1e-6, 1e-8, 1e-10]
        print("Generating report.txt...")
        bar = tqdm(range(0, 16), ncols=100, desc="Progress")
        report = open("../../report.txt", "w")
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
        bar.close()
        print("Generated report.txt.")

    @staticmethod
    def validate(a, b, x, tol):
        out = ""
        method_names = ('Jacobi', 'GaussSeidel', 'Gradient', 'Conjugate')
        for name in method_names:
            out += '\n\t--------------------' + name + '--------------------\n' + '\n'
            err, k, tf = Validate.validate_method(name, a, b, x, tol)
            out += '\t   Errore relativo: ' + str(err) + '\n'
            out += '\t   Iterazioni: ' + str(k) + '\n'
            out += '\t   Tempo: ' + str("%.5f" % tf) + '\n' + '\n'
        return out

    @staticmethod
    def test_filename(matrix_file, tol):
        a, n, _ = IterativeMethod.read_matrix(matrix_file)
        x = np.ones(n)
        b = a @ x
        filename = "report_" + matrix_file.split(".")[0] + ".txt"
        print("Generating " + filename + "...")
        f = open(filename, "w")
        out = Validate.validate(a, b, x, tol)
        f.write(out)
        print("Generated " + filename + ".")
        f.close()

    @staticmethod
    def validate_method(method, a, b, x, tol):
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
            raise Exception("Method not found")
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

