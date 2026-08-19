"""
demo.py
-------
End-to-end well test simulator demo:

    Reservoir inputs
        -> Pressure transient simulation (wellbore storage + skin, IARF)
        -> Synthetic buildup data
        -> Log-log + Bourdet derivative diagnostic plot
        -> Automatic flow-regime identification
        -> Horner analysis on the identified IARF window
        -> Estimate k, Pi (P*), skin
        -> Compare estimated vs. input (true) values

Run:
    python demo.py
"""

import numpy as np
import pandas as pd

from welltest_sim import (
    ReservoirParams, simulate_buildup,
    agarwal_equivalent_time, bourdet_derivative,
    classify_points, find_iarf_window,
    horner_regression, estimate_k, estimate_pstar, estimate_skin,
)
from welltest_sim.plotting import plot_loglog, plot_horner


def run(params: ReservoirParams, noise_std: float = 0.0, seed: int | None = 42,
        outdir: str = ".", L: float = 0.0):

    # ------------------------------------------------------------------
    # 1) Build a log-spaced shut-in time schedule and simulate the test
    # ------------------------------------------------------------------
    dt_max = min(params.tp * 3.0, 3000.0)
    dt = np.logspace(np.log10(0.001), np.log10(dt_max), 65)
    result = simulate_buildup(params, dt, N=12, noise_std=noise_std, seed=seed)
    Pws = result["Pws"]
    Pwf_tp = result["Pwf_tp"]

    # Conventional buildup delta-P for the diagnostic plot:
    #   dP(dt) = Pws(dt) - Pwf(dt=0)
    dP = Pws - Pwf_tp

    # ------------------------------------------------------------------
    # 2) Equivalent time + Bourdet derivative
    # ------------------------------------------------------------------
    dte = agarwal_equivalent_time(params.tp, dt)
    deriv = bourdet_derivative(dte, dP, L=L)

    # ------------------------------------------------------------------
    # 3) Flow regime identification
    # ------------------------------------------------------------------
    labels = classify_points(dte, deriv)
    iarf_mask = find_iarf_window(labels, min_points=5)

    if not iarf_mask.any():
        raise RuntimeError(
            "No stabilized infinite-acting-radial-flow (IARF) window was found.\n"
            "This is a genuine well-test-design issue, not just a numerical bug: "
            "with this combination of skin, wellbore storage and producing time,\n"
            "the derivative has not flattened out within the simulated shut-in "
            "period. In real testing this means the well was not tested long\n"
            "enough. Try increasing `tp` (longer flow period before shut-in) "
            "and/or lowering `C` (wellbore storage), then re-run."
        )

    # ------------------------------------------------------------------
    # 4) Horner analysis -> k, P*, skin
    # ------------------------------------------------------------------
    slope, intercept = horner_regression(params.tp, dt, Pws, iarf_mask)
    k_est, m = estimate_k(slope, params.q, params.mu, params.B, params.h)
    pstar_est = estimate_pstar(intercept)
    S_est = estimate_skin(m, slope, intercept, params.tp, Pwf_tp,
                           k_est, params.phi, params.mu, params.ct, params.rw)

    # ------------------------------------------------------------------
    # 5) Compare estimated vs. true input
    # ------------------------------------------------------------------
    rows = [
        ("Permeability, k (md)", params.k, k_est),
        ("Initial pressure, Pi / P* (psi)", params.Pi, pstar_est),
        ("Skin factor, S", params.S, S_est),
    ]
    report = pd.DataFrame(rows, columns=["Parameter", "True (input)", "Estimated"])
    report["Error (%)"] = 100 * (report["Estimated"] - report["True (input)"]) / report["True (input)"]

    # ------------------------------------------------------------------
    # 6) Plots
    # ------------------------------------------------------------------
    fig1 = plot_loglog(dte, dP, deriv, labels, iarf_mask,
                        savepath=f"{outdir}/loglog_diagnostic.png")
    fig2 = plot_horner(params.tp, dt, Pws, iarf_mask, slope, intercept,
                        savepath=f"{outdir}/horner_plot.png")

    return {
        "dt": dt, "dte": dte, "Pws": Pws, "dP": dP, "deriv": deriv,
        "labels": labels, "iarf_mask": iarf_mask,
        "slope": slope, "intercept": intercept, "m": m,
        "k_est": k_est, "pstar_est": pstar_est, "S_est": S_est,
        "Pwf_tp": Pwf_tp, "report": report,
        "fig_loglog": fig1, "fig_horner": fig2,
    }


if __name__ == "__main__":
    # ---- EDIT THESE INPUTS TO DEFINE YOUR SYNTHETIC RESERVOIR ----
    params = ReservoirParams(
        k=50.0,        # md
        h=40.0,        # ft
        phi=0.20,      # fraction
        mu=1.0,        # cp
        ct=1.5e-5,     # 1/psi
        rw=0.30,       # ft
        q=500.0,       # STB/D
        B=1.2,         # rb/STB
        Pi=4500.0,     # psi
        C=0.01,        # bbl/psi
        S=5.0,         # dimensionless skin
        tp=100.0,      # hr producing time before shut-in
    )

    print("Input reservoir / test parameters:")
    print(" ", params.summary())
    print()

    out = run(params, noise_std=0.0, outdir=".")

    print("Regime breakdown (point counts):")
    labels, counts = np.unique(out["labels"], return_counts=True)
    for lab, c in zip(labels, counts):
        print(f"  {lab:22s}: {c} points")
    print()

    print("Horner slope m = %.3f psi/cycle" % out["m"])
    print()
    print("=== Estimated vs. True Parameters ===")
    print(out["report"].to_string(index=False, float_format=lambda v: f"{v:,.3f}"))
    print()
    print("Plots saved: loglog_diagnostic.png, horner_plot.png")
