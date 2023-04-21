#! /usr/bin/python3
import numpy as np
import scipy.sparse as sparse
from IterativeMethod import IterativeMethod


class Jacobi(IterativeMethod):

    def __init__(self, a, b, tol):
        super().__init__(a, b, tol)
        self.invP = None

    def init_x(self):
        if self.is_strictly_diagonally_dominant(self.a):
            raise Exception("Matrix not strictly diagonally dominant")
        self.x0 = np.zeros(self.n)
        self.invP = sparse.diags(np.reciprocal(self.a.diagonal())).tocsr()

    def update(self):
        self.x0 = self.x0 + self.invP @ self.r
