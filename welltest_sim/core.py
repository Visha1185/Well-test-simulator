"""
core.py
-------
Analytical simulation of a pressure buildup test for a well with wellbore
storage and skin, producing at constant rate in an infinite-acting,
homogeneous reservoir.

Method
------
1. The dimensionless wellbore pressure for constant-rate drawdown with
   wellbore storage (C_D) and skin (S) has a known closed form in Laplace
   space (Agarwal, Al-Hussainy & Ramey, 1970):

        pD_bar(s)  = K0(sqrt(s)) / (s^1.5 * K1(sqrt(s)))          (no C, S)

        pwD_bar(s) = [s*pD_bar(s) + S] /
                     ( s * [1 + C_D*s*(s*pD_bar(s) + S)] )

   where K0, K1 are modified Bessel functions of the second kind.

2. pwD_bar(s) is numerically inverted to real (dimensionless) time using
   the Stehfest algorithm -> pwD(tD).

3. Because the underlying diffusivity equation + wellbore storage ODE is
   linear, a shut-in (buildup) response can be built by superposition of
   two constant-rate drawdown solutions (rate +q starting at t=0, rate -q
   starting at t=tp), both carrying the SAME C_D and S:

        dP(dt) = Pi - Pws(dt)
                = (141.2 q mu B / k h) * [ pwD(tpD + dtD) - pwD(dtD) ]

This reproduces the classic diagnostic shape: unit-slope wellbore storage
at early time, a transition (with possible hump for S>0), then a flat
derivative during infinite-acting radial flow (IARF).
"""

from __future__ import annotations
import numpy as np
import mpmath as mp

mp.mp.dps = 30  # decimal digits of precision for the Laplace inversion


# ----------------------------------------------------------------------
# Dimensionless variable conversions (field units -> dimensionless)
# ----------------------------------------------------------------------

def tD_from_t(t_hr: float, k: float, phi: float, mu: float, ct: float, rw: float) -> float:
    """Dimensionless time from real time in hours."""
    return 0.0002637 * k * t_hr / (phi * mu * ct * rw ** 2)


def CD_from_C(C: float, phi: float, ct: float, h: float, rw: float) -> float:
    """Dimensionless wellbore storage coefficient."""
    return 0.8936 * C / (phi * ct * h * rw ** 2)


def dP_from_pD(pD: float, q: float, mu: float, B: float, k: float, h: float) -> float:
    """Convert dimensionless pressure to real pressure drop [psi]."""
    return 141.2 * q * mu * B / (k * h) * pD


# ----------------------------------------------------------------------
# Stehfest numerical Laplace inversion
# ----------------------------------------------------------------------

def stehfest_coefficients(N: int):
    """Stehfest weights V_i, i=1..N (N must be even)."""
    if N % 2 != 0:
        raise ValueError("Stehfest N must be even")
    V = []
    for i in range(1, N + 1):
        total = mp.mpf(0)
        kmin = int(mp.floor((i + 1) / 2))
        kmax = int(min(i, N // 2))
        for k in range(kmin, kmax + 1):
            term = (mp.mpf(k) ** (N // 2) * mp.factorial(2 * k)) / (
                mp.factorial(N // 2 - k) * mp.factorial(k) *
                mp.factorial(k - 1) * mp.factorial(i - k) * mp.factorial(2 * k - i)
            )
            total += term
        sign = -1 if (i + N // 2) % 2 else 1
        V.append(sign * total)
    return V


def stehfest_invert(f_bar, t: float, V, N: int) -> float:
    """Invert a Laplace-space function f_bar(s) at real time t."""
    ln2 = mp.log(2)
    t_mp = mp.mpf(t)
    total = mp.mpf(0)
    for i in range(1, N + 1):
        s = i * ln2 / t_mp
        total += V[i - 1] * f_bar(s)
    return float((ln2 / t_mp) * total)


# ----------------------------------------------------------------------
# Laplace-space solution
# ----------------------------------------------------------------------

def _pD_bar(s):
    sq = mp.sqrt(s)
    return mp.besselk(0, sq) / (s ** mp.mpf('1.5') * mp.besselk(1, sq))


def _pwD_bar(s, CD, S):
    pD = _pD_bar(s)
    num = s * pD + S
    den = s * (1 + CD * s * num)
    return num / den


def pwD_of_tD(tD: float, CD: float, S: float, V, N: int) -> float:
    """Dimensionless wellbore pressure for constant-rate drawdown at time tD."""
    if tD <= 0:
        return 0.0
    return stehfest_invert(lambda s: _pwD_bar(s, CD, S), tD, V, N)


# ----------------------------------------------------------------------
# Full buildup simulation
# ----------------------------------------------------------------------

def simulate_buildup(params, dt_hours: np.ndarray, N: int = 12,
                      noise_std: float = 0.0, seed: int | None = None):
    """
    Simulate a pressure buildup test.

    Parameters
    ----------
    params    : ReservoirParams
    dt_hours  : array of shut-in times [hr] at which to compute Pws
    N         : Stehfest parameter (even integer, 10-16 typical)
    noise_std : std-dev of optional Gaussian noise added to Pws [psi]
    seed      : RNG seed for reproducible noise

    Returns
    -------
    dict with:
        dt        : shut-in time array [hr]
        Pws       : simulated shut-in pressure [psi]
        Pwf_tp    : flowing pressure at the instant of shut-in [psi]
        CD, S, tpD: dimensionless parameters used
    """
    V = stehfest_coefficients(N)

    CD = CD_from_C(params.C, params.phi, params.ct, params.h, params.rw)
    tpD = tD_from_t(params.tp, params.k, params.phi, params.mu, params.ct, params.rw)

    # flowing pressure right before shut-in (needed later for skin calc)
    pD_tp = pwD_of_tD(tpD, CD, params.S, V, N)
    dP_tp = dP_from_pD(pD_tp, params.q, params.mu, params.B, params.k, params.h)
    Pwf_tp = params.Pi - dP_tp

    Pws = np.zeros_like(dt_hours, dtype=float)
    for j, dt in enumerate(dt_hours):
        dtD = tD_from_t(dt, params.k, params.phi, params.mu, params.ct, params.rw)
        pD_sum = pwD_of_tD(tpD + dtD, CD, params.S, V, N)
        pD_dt = pwD_of_tD(dtD, CD, params.S, V, N)
        dP = dP_from_pD(pD_sum - pD_dt, params.q, params.mu, params.B, params.k, params.h)
        Pws[j] = params.Pi - dP

    if noise_std > 0:
        rng = np.random.default_rng(seed)
        Pws = Pws + rng.normal(0, noise_std, size=Pws.shape)

    return {
        "dt": dt_hours,
        "Pws": Pws,
        "Pwf_tp": Pwf_tp,
        "CD": CD,
        "S": params.S,
        "tpD": tpD,
    }
