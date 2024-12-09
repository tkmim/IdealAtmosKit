#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Loranz 96 model

class Lorenz96:
    


"""

import numpy as np
from numba import jit, float64, int64
from numba.experimental import jitclass

spec = [
    ("F", float64),
    ("dt", float64),
]


@jitclass(spec)
class Lorenz96:
    def __init__(self, dt, F=8.0):
        self.F = F
        self.dt = dt
        # print("Lorenz96: Model initialisation", f"F = {F:f}", flush=True)

    def RungeKutta(self, X, steps):
        """
        Runge-Kutta 4th order method
        """

        dt = self.dt

        for _ in range(steps):
            k_1 = self.LorenzEquation(X)
            k_2 = self.LorenzEquation(X + k_1 * dt / 2)
            k_3 = self.LorenzEquation(X + k_2 * dt / 2)
            k_4 = self.LorenzEquation(X + k_3 * dt)
            X_tend = (k_1 + 2 * k_2 + 2 * k_3 + k_4) / 6
            X = X + X_tend * dt

        return X

    def LorenzEquation(self, X):
        """
        dX_i/dt = X_{i-1} * (X_{i+1} - X_{i-2}) - X_{i} + F
        """

        N = X.shape[0]
        _X = np.zeros_like(X)

        for i in range(N - 1):
            _X[i] = (X[i + 1] - X[i - 2]) * X[i - 1] - X[i] + self.F

        _X[-1] = (X[0] - X[-3]) * X[-2] - X[-1] + self.F

        return _X
