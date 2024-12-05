#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Wuersch and Craig (2014) shallow water model

class swm1d:
    


"""

import numpy as np
from numba import jit, float64, int64
from numba.experimental import jitclass

spec = [
    ("F", float64),
    ("H0", float64),
    ("Hc", float64),
    ("Hr", float64),
    ("alpha", float64),
    ("beta", float64),
    ("Ku", float64),
    ("Kh", float64),
    ("Kr", float64),
    ("g", float64),
    ("phic", float64),
    ("c", float64),
    ("dt", float64),
    ("dx", float64),
]


@jitclass(spec)
class swm1d:
    """A one-dimensional shallow water model with convective rain.

    This class implements a one-dimensional shallow water model with convective rain based on the work of Würsch and Craig (2014).
    The model uses a leapfrog scheme with a Robert–Asselin–Williams (RAW) filter to reduce computational mode.

    Attributes
    ----------
    F : float
        forcing term
    H0 : float
        reference height
    Hc : float
        critical height for convective rain
    Hr : float
        rain height threshold
    alpha : float
        rain decay coefficient
    beta : float
        wind convergence coefficient
    Ku : float
        diffusion coefficient for wind
    Kh : float
        diffusion coefficient for height
    Kr : float
        diffusion coefficient for rain
    g : float
        gravitational acceleration
    phic : float
        critical geopotential height
    c : float
        speed of gravity waves
    dt : float
        time step for integration
    dx : float
        grid spacing

    Methods
    -------
    __init__(dx=500.0, dt=5.0)
        Initializes the model with given grid spacing and time step.
    RAW(uwind_0, height_0, rain_0, steps, noise=False)
        Runs the model for a given number of steps with optional noise.
    Equation(uwind, height, rain, verbose=0)
        Computes the tendencies of the model variables.
    make_noise(N, length=10)
        Generates a noise field for the wind.
    """

    def __init__(self, dx=500.0, dt=5.0):
        """__init__

        Parameters
        ----------
        dx : float, optional
            grid spacing , by default 500.0 [m]
        dt : float, optional
            integration time step, by default 5.0 [s]
        """

        # set constsnts
        self.F = 0.0  # forcing term
        self.H0 = 90.0
        self.Hc = 90.02
        self.Hr = 90.4
        self.alpha = 2.5e-4  # 1.4e-4
        self.beta = 1 / 10  # 1/ 300
        self.Ku = 2000 * 10  # 21000: WC2014
        self.Kh = 6000  # 6000: Tempest et al. (2022), 25000: WC2014
        self.Kr = 10  # 200 Diffusion of rain
        self.g = 9.80665
        self.phic = 899.77  # 899.77
        self.c = np.sqrt(self.g * self.H0)

        # model configuration
        self.dt = dt
        self.dx = dx

    def RAW(self, uwind_0, height_0, rain_0, steps, noise=False):
        # prepare for leapfrog
        uwind = np.zeros((3, uwind_0.shape[0]))
        height = np.zeros((3, height_0.shape[0]))
        rain = np.zeros((3, rain_0.shape[0]))

        # initialisation time step
        uwind[0, :] = uwind_0
        height[0, :] = height_0
        rain[0, :] = rain_0

        duwind, dheight, drain = self.Equation(uwind_0, height_0, rain_0)
        uwind[1, :], height[1, :], rain[1, :] = [
            var + dvar * self.dt for var, dvar in zip((uwind_0, height_0, rain_0), (duwind, dheight, drain))
        ]

        for _ in range(steps):
            uwind, height, rain = self._RAW(uwind, height, rain, noise)

            if np.sum(np.isnan(height)) > 0:
                print(_)
                break

        return uwind[0], height[0], rain[0]

    def _RAW(self, uwind, height, rain, noise=False):
        """Robert–Asselin–Williams (RAW)-like filter with the leapfrog scheme

        Parameters
        ----------
        uwind: float, 1D ndarray
            zonal wind (m/s)
        height: float, 1D ndarray
            height (m)
        rain: float, 1D ndarray
            rain rate
        noise: bool, optional
            add noise to the wind field, by default False

        Returns
        -------
        uwind, height, rain: float, 1D ndarray
            updated variables

        """

        dt = self.dt
        filter_parameter = 0.7  # RAW filter reduces computational mode.
        alpha_filt = 0.53  # Want just above 0.5 to be conditionally stable.

        if noise is True:
            # adding noise
            N = uwind[0].size
            np.random.seed(42)
            noise = self.make_noise(N)
            # select a point to be add noise
            noise_on = np.zeros((N * 2))
            ratio = 1 / 75
            num_points = int(N * ratio * np.random.normal(1, 0.05))  # Total number of points to sample
            indices = np.zeros(num_points).astype(np.int64)
            for _ in range(num_points):
                indices[_] = np.random.randint(0, N)

            for ii, shift in enumerate(indices):
                noise_on[shift : N + shift] += noise
            uwind[1] = uwind[1] + noise_on[:N] + noise_on[N:]

        # execute 1D update equation
        duwind, dheight, drain = self.Equation(uwind[0], height[0], rain[0])
        uwind[2], height[2], rain[2] = [
            var + dvar * 2 * dt for var, dvar in zip((uwind[0], height[0], rain[0]), (duwind, dheight, drain))
        ]

        # RAW filter like update. Accounts for the growing computational mode.
        d = filter_parameter * 0.5 * (uwind[0] - 2.0 * uwind[1] + uwind[2])
        uwind[0] = uwind[1] + d * alpha_filt
        uwind[1] = uwind[2] + d * (alpha_filt - 1)
        d = filter_parameter * 0.5 * (height[0] - 2.0 * height[1] + height[2])
        height[0] = height[1] + d * alpha_filt
        height[1] = height[2] + d * (alpha_filt - 1)
        d = filter_parameter * 0.5 * (rain[0] - 2.0 * rain[1] + rain[2])
        rain[0] = rain[1] + d * alpha_filt
        rain[1] = rain[2] + d * (alpha_filt - 1)

        # upper/lower limit
        rain = np.where(rain < 0.0, 0.0, rain)  # no negative rain allowed

        return uwind, height, rain

    def Equation(self, uwind, height, rain, verbose=0):
        """One-dimenstional shallow water model with convective rain

        Parameters
        ----------
        uwind : float, 1D ndarray
            zonal wind (m/s)
        height : float, 1D ndarray
            height (m)
        rain : float, 1D ndarray
            rain amount

        Returns
        -------
        _uwind, _height, _rain : float, 1D ndarray
            tendency of the input variables

        Theory
        ------
        du/dt = -u * ∆u/∆x - ∆(phi + c^2*rain)/∆x + Ku * ∆^2u/∆x^2 + F
        dh/dt = -∆(uh)/∆x + Ku * ∆^2h/∆x^2
        dr/dt = -u * ∆u/∆x + Kr * ∆^2r/∆x^2 - alpha * r - beta * ∆u/∆x
        phi = g * (H + h)

        References
        ----------
        Würsch, M. and Craig, G. C.: A simple dynamical model of cumulus convection for data assimilation research,
        Meteorol. Z., 23, 483–490, https://doi.org/10.1127/0941-2948/2014/0492, 2014.


        """

        assert uwind.size == height.size

        N = uwind.size
        H = np.zeros((N))  # orography

        # set phi from height
        phi = np.where(height > self.Hc, self.phic, self.g * height)

        # FTCS with a staggered grid
        #
        # Staggered Grid:
        #
        #   ---+---  ---+---
        #      |        |    * (nx, ny)  h points at grid centres
        # h-1 u0  h0   u1    * (nx, ny)  u points on vertical edges  (u[0] and u[nx+1] are identical)
        #      |        |
        #   ---+---  ---+---
        #

        # conversion between staggerd and full grids
        # useful for a slicing method, but not used here in this implementation
        # since using np.roll() is more readable and decently fast
        def half2full(var):
            # remap staggered uwind onto full grid
            ret = np.zeros(var.size)
            ret[:-1] = (var[0:-1] + var[1:]) * 0.5
            ret[-1] = (var[-1] + var[0]) * 0.5
            return ret

        def full2half(var):
            # remap staggered height onto half grid
            ret = np.zeros(var.size)
            ret[0] = (var[-1] + var[0]) * 0.5
            ret[1:] = (var[:-1] + var[1:]) * 0.5
            return ret

        # Note: advection terms are in flux form
        _uwind = (
            -1 * (np.roll(uwind, -1) ** 2 - np.roll(uwind, 1) ** 2) / (4 * self.dx)
            - (np.roll(phi, 0) - np.roll(phi, 1) + self.c**2 * (np.roll(rain, 0) - np.roll(rain, 1))) / self.dx
            + self.Ku * (np.roll(uwind, -1) - 2 * np.roll(uwind, 0) + np.roll(uwind, 1)) / self.dx**2
            + self.F
        )

        _height = (
            -1 * height * (np.roll(uwind, -1) - np.roll(uwind, 0)) / (2 * self.dx)
            + self.Kh * (np.roll(height, -1) - 2 * np.roll(height, 0) + np.roll(height, 1)) / self.dx**2
        )

        # crt 1: height above rain level
        hcrt = height + _height > self.Hr
        # crt 2: wind convergence
        wcon = np.roll(uwind, -1) - np.roll(uwind, 0) < 0

        _rain_a = (
            -1 * rain * (np.roll(uwind, -1) - np.roll(uwind, 0)) / self.dx
            + self.Kr * (np.roll(rain, -1) - 2 * np.roll(rain, 0) + np.roll(rain, 1)) / self.dx**2
            - self.alpha * rain
        )
        _rain_b = -self.beta * (np.roll(uwind, -1) - np.roll(uwind, 0)) / self.dx

        _rain = np.where(hcrt * wcon, _rain_a + _rain_b, _rain_a)  #

        # boundary conditions
        boundary_mode = "periodoc"
        if boundary_mode == "constant":
            # Boundary condition: constant
            _uwind[0], _uwind[-1] = 0.0, 0.0
            _rain[0], _rain[-1] = 0.0, 0.0
            _height[0], _height[-1] = 0.0, 0.0

        else:
            # Boundary condition: periodic (default)
            # do nothing here
            ...

        return _uwind, _height, _rain

    def make_noise(self, N, length=10):
        """Preparation of the noise field

        Parameters
        ----------
        length : int, optional
            _description_, by default 10 (5 km)

        Returns
        -------
        zsum : ndarray[length]
            noise field
        """

        mu = float(N / 2)  # Center of the noise
        d = np.arange(N)
        l = float(length * 0.5 - 1)  # 4: l (half width -1) of the noise field
        amp = float(8.95e-3)  # Amplitude of the added noise field (in m/s)  normal 0.005
        #  ub = 0.005 l = 2000
        z = (1.0 / (l * np.sqrt(2.0 * np.pi))) * np.exp(-0.5 * ((d - mu) / l) ** 2)

        dz = np.roll(z, -1) - np.roll(z, 0)
        zsum = amp * dz / max(dz)  # Normalize the noise to 1 and multiply with the amplitude

        return zsum
