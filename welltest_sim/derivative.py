"""
derivative.py
-------------
Agarwal equivalent time and the Bourdet pressure derivative, the two
ingredients of the log-log diagnostic plot used in well test analysis.
"""

import numpy as np


def agarwal_equivalent_time(tp: float, dt: np.ndarray) -> np.ndarray:
    """
    Converts buildup shut-in time into an equivalent drawdown time so that
    a buildup test can be interpreted with drawdown-type-curve logic.

        dte = tp * dt / (tp + dt)
    """
    dt = np.asarray(dt, dtype=float)
    return tp * dt / (tp + dt)


def bourdet_derivative(dte: np.ndarray, dP: np.ndarray, L: float = 0.0) -> np.ndarray:
    """
    Bourdet pressure derivative w.r.t. ln(equivalent time).

    L = 0 (default): raw three-point centered stencil using the immediate
        neighboring data points:

            p'_i = [ (dp1/dx1)*dx2 + (dp2/dx2)*dx1 ] / (dx1 + dx2)

        Exact and sharp on clean (noise-free) synthetic data, but very
        sensitive to noise since it differentiates adjacent points.

    L > 0: Bourdet's "L-algorithm" smoothing, the industry-standard method
        (used in commercial PTA software) for noisy data. Instead of the
        immediate neighbors, the left/right points are taken a distance L
        log10-cycles away in time (interpolated if no data point falls
        exactly there), which averages out point-to-point noise while
        still resolving genuine regime changes. Typical L = 0.1-0.3.
    """
    dte = np.asarray(dte, dtype=float)
    dP = np.asarray(dP, dtype=float)
    n = len(dte)
    deriv = np.full(n, np.nan)

    if L <= 0:
        x = np.log(dte)
        for i in range(1, n - 1):
            dx1 = x[i] - x[i - 1]
            dx2 = x[i + 1] - x[i]
            dp1 = dP[i] - dP[i - 1]
            dp2 = dP[i + 1] - dP[i]
            if dx1 <= 0 or dx2 <= 0:
                continue
            deriv[i] = (dp1 / dx1 * dx2 + dp2 / dx2 * dx1) / (dx1 + dx2)
        return deriv

    # --- L-algorithm (log10-cycle smoothing) ---
    log_t = np.log10(dte)
    log_t_min, log_t_max = log_t.min(), log_t.max()

    for i in range(n):
        xl_target = log_t[i] - L
        xr_target = log_t[i] + L

        if xl_target < log_t_min or xr_target > log_t_max:
            continue  # not enough data on one side to honor the L window

        # linear interpolation of dP vs log10(t) to get values at the
        # exact +/- L points (falls back to real data if it lands exactly
        # on an existing point)
        p_left = np.interp(xl_target, log_t, dP)
        p_right = np.interp(xr_target, log_t, dP)

        x1 = np.log(10 ** xl_target)
        x2 = np.log(dte[i])
        x3 = np.log(10 ** xr_target)
        dx1 = x2 - x1
        dx2 = x3 - x2
        if dx1 <= 0 or dx2 <= 0:
            continue
        dp1 = dP[i] - p_left
        dp2 = p_right - dP[i]
        deriv[i] = (dp1 / dx1 * dx2 + dp2 / dx2 * dx1) / (dx1 + dx2)

    return deriv
