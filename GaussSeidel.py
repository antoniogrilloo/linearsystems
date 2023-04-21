#! /usr/bin/python3
import numpy as np

from IterativeMethod import IterativeMethod
from scipy.sparse import tril


class GaussSeidel(IterativeMethod):

    def __init__(self, a, b, tol):
        super().__init__(a, b, tol)
        self.p = None

    def init_x(self):
        self.p = tril(self.a).tocsr()
        self.x0 = np.ones(self.n)

    def update(self):
        self.x0 = self.x0 + self.forward_substitution(self.p)

    def forward_substitution(self, l):
        x = np.zeros(self.n)
        for i in range(self.n):
            dot = l[i, ].dot(x)
            x[i] = (self.r[i] - dot) / l[i, i]
        return x