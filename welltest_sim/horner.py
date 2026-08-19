"""
horner.py
---------
Classic Horner buildup analysis:

    Pws = P* - m * log10[(tp + dt) / dt]

Fit a straight line through the points identified as infinite-acting
radial flow (IARF); the slope m gives permeability, the intercept at
Horner ratio = 1 gives P* (an estimate of the initial pressure Pi for a
well with no boundaries yet felt), and the standard Horner skin equation
gives the skin factor.
"""

import numpy as np


def horner_ratio(tp: float, dt: np.ndarray) -> np.ndarray:
    dt = np.asarray(dt, dtype=float)
    return (tp + dt) / dt


def horner_regression(tp: float, dt: np.ndarray, Pws: np.ndarray, mask: np.ndarray):
    """
    Linear regression of Pws vs log10(Horner ratio) over the points where
    mask is True. Returns (slope, intercept) such that
        Pws = intercept + slope * log10(horner_ratio)
    (slope is negative, since Pws falls as Horner ratio grows).
    """
    x = np.log10(horner_ratio(tp, dt[mask]))
    y = Pws[mask]
    slope, intercept = np.polyfit(x, y, 1)
    return slope, intercept


def estimate_k(slope: float, q: float, mu: float, B: float, h: float):
    """Permeability [md] from the Horner slope. m is psi per log cycle."""
    m = abs(slope)
    k = 162.6 * q * mu * B / (m * h)
    return k, m


def estimate_pstar(intercept: float) -> float:
    """P* = value of the fitted line at Horner ratio = 1 (log10 = 0)."""
    return intercept


def estimate_skin(m: float, slope: float, intercept: float, tp: float,
                   Pwf_tp: float, k: float, phi: float, mu: float,
                   ct: float, rw: float) -> float:
    """
    Standard Horner skin equation:

        S = 1.151 * [ (P_1hr - Pwf(dt=0)) / m
                       - log10(k / (phi*mu*ct*rw^2)) + 3.23 ]

    P_1hr is read from the FITTED straight line at dt = 1 hr (not the raw
    data), which is the standard, more robust convention.
    """
    x_1hr = np.log10((tp + 1.0) / 1.0)
    p_1hr = intercept + slope * x_1hr

    S = 1.151 * (
        (p_1hr - Pwf_tp) / m
        - np.log10(k / (phi * mu * ct * rw ** 2))
        + 3.23
    )
    return S
