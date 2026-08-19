"""
regime.py
---------
Classifies each point of the log-log diagnostic plot into a flow regime
based on the LOCAL SLOPE of the derivative curve (log-log):

    slope ~ +1     -> wellbore storage (unit slope straight line)
    slope ~  0     -> infinite-acting radial flow (flat derivative)
    otherwise      -> transition

Then finds the longest contiguous "radial flow" window in the middle/late
part of the test, which is the window used for the Horner regression.
"""

import numpy as np


def classify_points(dte: np.ndarray, deriv: np.ndarray,
                     wbs_slope_thresh: float = 0.85,
                     iarf_slope_thresh: float = 0.08) -> np.ndarray:
    """Return an array of regime labels, one per data point."""
    dte = np.asarray(dte, dtype=float)
    deriv = np.asarray(deriv, dtype=float)
    n = len(dte)
    labels = np.array(["unknown"] * n, dtype=object)

    logt = np.log10(dte)
    with np.errstate(invalid="ignore", divide="ignore"):
        logd = np.log10(np.abs(deriv))

    slopes = np.full(n, np.nan)
    for i in range(1, n - 1):
        if np.isnan(logd[i - 1]) or np.isnan(logd[i + 1]):
            continue
        slopes[i] = (logd[i + 1] - logd[i - 1]) / (logt[i + 1] - logt[i - 1])

    # Light smoothing (3-point centered mean, ignoring NaNs) so an isolated
    # zero-crossing at the TOP of a transition hump doesn't get mistaken
    # for a stabilized radial-flow point.
    smoothed = slopes.copy()
    for i in range(1, n - 1):
        window = slopes[i - 1:i + 2]
        valid = window[~np.isnan(window)]
        if len(valid) > 0:
            smoothed[i] = np.mean(valid)
    slopes = smoothed

    for i in range(n):
        s = slopes[i]
        if np.isnan(s):
            continue
        if s >= wbs_slope_thresh:
            labels[i] = "wellbore storage"
        elif abs(s) <= iarf_slope_thresh:
            labels[i] = "radial flow (IARF)"
        else:
            labels[i] = "transition"

    labels = _suppress_short_runs(labels, "radial flow (IARF)", min_run=3)
    return labels


def _suppress_short_runs(labels: np.ndarray, target: str, min_run: int) -> np.ndarray:
    """Relabel isolated short runs of `target` (shorter than min_run) back
    to 'transition', so a single zero-slope point at a hump peak doesn't
    get painted as a stabilized regime."""
    labels = labels.copy()
    n = len(labels)
    i = 0
    while i < n:
        if labels[i] == target:
            j = i
            while j < n and labels[j] == target:
                j += 1
            if (j - i) < min_run:
                labels[i:j] = "transition"
            i = j
        else:
            i += 1
    return labels


def find_iarf_window(labels: np.ndarray, min_points: int = 5):
    """
    Find the longest contiguous run of 'radial flow (IARF)' points.
    Returns a boolean mask (True = part of the chosen IARF window), or an
    all-False mask if no run of at least `min_points` is found.
    """
    n = len(labels)
    best_start, best_len = -1, 0
    cur_start, cur_len = -1, 0

    for i in range(n):
        if labels[i] == "radial flow (IARF)":
            if cur_len == 0:
                cur_start = i
            cur_len += 1
            if cur_len > best_len:
                best_len = cur_len
                best_start = cur_start
        else:
            cur_len = 0

    mask = np.zeros(n, dtype=bool)
    if best_len >= min_points:
        mask[best_start: best_start + best_len] = True
    return mask
