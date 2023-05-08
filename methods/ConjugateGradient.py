from methods.IterativeMethod import IterativeMethod


class ConjugateGradient(IterativeMethod):

    def __init__(self, a, b, tol):
        super().__init__(a, b, tol)
        self.d0 = None

    def init_x(self):
        super().init_x()
        self.d0 = self.r = self.b - (self.a @ self.x0)

    def update(self):
        y = self.a @ self.d0
        alpha = (self.d0 @ self.r) / (self.d0 @ y)
        self.x0 = self.x0 + alpha * self.d0
        self.r = self.b - self.a @ self.x0
        w = self.a @ self.r
        beta = (self.d0 @ w) / (self.d0 @ y)
        self.d0 = self.r - beta * self.d0
