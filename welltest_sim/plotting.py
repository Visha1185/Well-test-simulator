"""
plotting.py
-----------
Log-log diagnostic plot (dP and Bourdet derivative vs equivalent time,
shaded by flow regime) and the Horner semi-log plot with the fitted IARF
line.
"""

import numpy as np
import matplotlib.pyplot as plt

REGIME_COLORS = {
    "wellbore storage": "#f4a261",
    "transition": "#e9c46a",
    "radial flow (IARF)": "#2a9d8f",
    "unknown": "#cccccc",
}


def plot_loglog(dte, dP, deriv, labels, iarf_mask, savepath=None):
    fig, ax = plt.subplots(figsize=(7.5, 6))

    ax.loglog(dte, dP, 'o', color='#264653', ms=4, label=r'$\Delta P$')
    ax.loglog(dte, deriv, 's', color='#e76f51', ms=4, label=r"Bourdet derivative $P'$")

    # shade background by regime
    labels = np.asarray(labels, dtype=object)
    for regime, color in REGIME_COLORS.items():
        idx = np.where(labels == regime)[0]
        if len(idx) == 0:
            continue
        ax.scatter(dte[idx], deriv[idx], color=color, s=45, zorder=5,
                   label=f'derivative: {regime}', edgecolor='k', linewidth=0.3)

    if iarf_mask.any():
        ax.loglog(dte[iarf_mask], deriv[iarf_mask], 'D', color='black',
                  ms=7, mfc='none', mew=1.5, label='IARF window (used for Horner fit)')

    ax.set_xlabel(r'Equivalent time, $\Delta t_e$ (hr)')
    ax.set_ylabel(r'$\Delta P$, $P\prime$ (psi)')
    ax.set_title('Log-Log Diagnostic Plot')
    ax.grid(True, which='both', ls=':', alpha=0.6)
    ax.legend(fontsize=8, loc='best')
    fig.tight_layout()

    if savepath:
        fig.savefig(savepath, dpi=150)
    return fig


def plot_horner(tp, dt, Pws, iarf_mask, slope, intercept, savepath=None):
    from .horner import horner_ratio
    hr = horner_ratio(tp, dt)
    x = np.log10(hr)

    fig, ax = plt.subplots(figsize=(7.5, 6))
    ax.plot(x, Pws, 'o', color='#264653', ms=4, label='Simulated Pws data')
    ax.plot(x[iarf_mask], Pws[iarf_mask], 'D', mfc='none', mec='black',
            ms=8, mew=1.5, label='IARF points used in fit')

    xx = np.linspace(0, x.max(), 50)
    ax.plot(xx, intercept + slope * xx, '--', color='#e76f51',
            label='Horner straight-line fit')

    ax.invert_xaxis()  # Horner ratio decreases -> to the right conventionally
    ax.set_xlabel(r'$\log_{10}[(t_p+\Delta t)/\Delta t]$')
    ax.set_ylabel(r'$P_{ws}$ (psi)')
    ax.set_title('Horner Plot')
    ax.grid(True, ls=':', alpha=0.6)
    ax.legend(fontsize=8, loc='best')
    fig.tight_layout()

    if savepath:
        fig.savefig(savepath, dpi=150)
    return fig
