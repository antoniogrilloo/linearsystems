#! /bin/usr/python3
from IterativeMethod import IterativeMethod
import numpy as np


class Gradient(IterativeMethod):

    def __init__(self, a, b, tol):
        super().__init__(a, b, tol)

    def init_x(self):
        self.x0 = np.zeros(self.n)

    def update(self):
        y = self.a @ self.r
        rt = (self.r.transpose())
        a = rt @ self.r
        b = rt @ y
        alpha = a / b
        self.x0 = self.x0 + (alpha * self.r)
