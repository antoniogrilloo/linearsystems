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
from tabulate import tabulate


class Validate:

    @staticmethod
    def test():
        names = ["spa1.mtx", "spa2.mtx", "vem1.mtx", "vem2.mtx"]
        tolerances = [1e-4, 1e-6, 1e-8, 1e-10]
        print("Generating report.txt")
        bar = tqdm(range(0, 16), ncols=100, desc="Progress")
        report = open("report.txt", "w")
        data = open("data.csv", "w")
        for filename in names:
            report.write('\n======================|   ' + filename + '   |======================\n')
            a, n, _ = IterativeMethod.read_matrix(filename)
            x = np.ones(n)
            b = a @ x
            for tol in tolerances:
                report.write("\n" + "\t\t\t\t\twith Tolerance: " + str(tol) + "\n")
                table, row = Validate.validate(a, b, x, tol)
                report.write(tabulate(table, headers="firstrow", tablefmt="simple_grid", numalign="center") + "\n")
                data.write(row)
                bar.update()
        data.close()
        report.close()
        bar.close()
        print("Generated report.txt.")

    @staticmethod
    def validate(a, b, x, tol):
        out = ""
        row = ""
        method_names = ('Jacobi', 'GaussSeidel', 'Gradient', 'Conjugate')
        table = [["Method", " Relative error", "Iterations", "Time"]]
        for name in method_names:
            err, k, tf = Validate.validate_method(name, a, b, x, tol)
            table.append([name, str(err), str(k), str("%.5f" % tf)])
            row = row + "\n" + str(err) + ";" + str(k) + ";" + str("%.5f" % tf)
        return table, row

    @staticmethod
    def test_filename(matrix_file, tol):
        a, n, _ = IterativeMethod.read_matrix(matrix_file)
        x = np.ones(n)
        b = a @ x
        filename = "report_" + matrix_file.split(".")[0] + ".txt"
        print("Generating " + filename + "...")
        f = open(filename, "w")
        out, _ = Validate.validate(a, b, x, tol)
        f.write('\n======================|   ' + matrix_file + '   |======================\n')
        f.write("\n" + "\t\t\t\t\twith Tolerance: " + str(tol) + "\n")
        f.write(tabulate(out, headers="firstrow", tablefmt="simple_grid", numalign="center"))
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

