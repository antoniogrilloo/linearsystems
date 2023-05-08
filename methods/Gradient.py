#! /bin/usr/python3
from methods.IterativeMethod import IterativeMethod


class Gradient(IterativeMethod):

    def __init__(self, a, b, tol):
        super().__init__(a, b, tol)

    def init_x(self):
        super().init_x()

    def update(self):
        y = self.a @ self.r
        rt = (self.r.transpose())
        a = rt @ self.r
        b = rt @ y
        alpha = a / b
        self.x0 = self.x0 + (alpha * self.r)
