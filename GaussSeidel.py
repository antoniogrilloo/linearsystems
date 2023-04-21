#! /usr/bin/python3
import numpy as np

from IterativeMethod import IterativeMethod
from scipy.sparse import tril


class GaussSeidel(IterativeMethod):

    def __init__(self, a, b, tol):
        super().__init__(a, b, tol)
        self.p = None

    def init_x(self):
        if not self.is_strictly_diagonally_dominant(self.a):
            raise Exception("Matrix not strictly diagonally dominant")
        self.p = tril(self.a).tocsr()
        self.x0 = np.zeros(self.n)

    def update(self):
        self.x0 = self.x0 + self.forward_substitution(self.p)

    def forward_substitution(self, l):
        x = np.zeros(self.n)
        x = (self.r - l.dot(x)) / l.diagonal()
        return x