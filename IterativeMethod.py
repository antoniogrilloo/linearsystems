#! /usr/bin/python3
import math
from abc import ABC, abstractmethod
import numpy as np
from scipy.linalg import blas
import scipy.sparse as sparse


class IterativeMethod(ABC):
    MAX_ITER = 30000

    def __init__(self, a, b, tol):
        if isinstance(a, str):
            self.a, self.n, _ = IterativeMethod.read_matrix(a)
        else:
            self.a = a  # sparse matrix
            self.n = self.a.shape[0]  # size of matrix
        self.b = b  # the constant term
        self.tol = tol  # tolerance
        self.r = np.empty_like(self.n)  # residue
        self.bn = blas.dnrm2(self.b)  # norm of the constant term
        self.x0 = np.empty_like(self.n)  # vector of the unknowns

    def solve(self):
        self.init_x()
        k = 0  # Iterations number
        while not self.stopping_criterion():
            self.update()
            k = k + 1
            if k > self.MAX_ITER:
                raise Exception("Iterations exceeded")
        print(k)
        return self.x0

    @abstractmethod
    def init_x(self):
        raise NotImplementedError

    def stopping_criterion(self):
        self.r = self.b - (self.a @ self.x0)
        rn = blas.dnrm2(self.r)
        return (rn / self.bn) < self.tol

    @abstractmethod
    def update(self):
        raise NotImplementedError

    @staticmethod
    def read_matrix(filename):
        data = np.genfromtxt(filename, comments='%')
        m, n = int(data[0, 0]), int(data[0, 1])
        data = data[1:]
        rows = data[:, 0] - 1
        cols = data[:, 1] - 1
        vals = data[:, 2]
        a = sparse.coo_matrix((vals, (rows, cols)), shape=(m, n)).tocsr()
        return a, m, n

    @staticmethod
    def is_strictly_diagonally_dominant(a):
        a = abs(a)
        d = a.diagonal()
        s = a.sum(axis=1).transpose() - d
        if (abs(d - s) / d < 1e-15).all():
            d = s
        if (d < s).all():
            return False
        return True
