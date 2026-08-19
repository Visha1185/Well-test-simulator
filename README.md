# Well Test Simulator — Pressure Transient Analysis (PTA)

A small engineering + programming project that closes the loop on well test
interpretation:

```
Reservoir inputs (k, h, phi, mu, ct, rw, q, B, Pi, C, S, tp)
        |
Pressure transient simulation  (analytical Laplace-space solution,
                                 Stehfest numerical inversion,
                                 wellbore storage + skin, infinite reservoir)
        |
Synthetic buildup data  Pws(dt)
        |
Log-log diagnostic plot: delta-P and Bourdet derivative vs. equivalent time
        |
Automatic flow-regime identification (wellbore storage / transition / IARF)
        |
Horner analysis on the identified radial-flow window
        |
Estimate k, Pi (P\*), skin S
        |
Compare estimated vs. true input values  <-- accuracy check
```

the simulator solves the actual diffusivity equation (in Laplace space) for a well with wellbore storage and skin,
so the generated data has the real physical signature analysts are trained to
recognize: a unit-slope wellbore-storage line at early time, a transition
(hump for `S > 0`, trough for `S < 0`), and a flat derivative once the reservoir
response becomes infinite-acting radial flow (IARF).

## Project structure

```
well\_test\_simulator/
├── welltest\_sim/
│   ├── reservoir.py     # ReservoirParams input dataclass
│   ├── core.py          # Laplace-space solution + Stehfest inversion + simulate\_buildup()
│   ├── derivative.py     # Agarwal equivalent time + Bourdet derivative (raw \& L-smoothed)
│   ├── regime.py          # Flow-regime classification from local log-log slopes
│   ├── horner.py           # Horner regression -> k, P\*, skin
│   └── plotting.py          # Log-log diagnostic plot + Horner plot
├── demo.py                    # Run the full workflow from the command line
├── well\_test\_simulator.ipynb  # Same workflow, interactive, with explanations
└── README.md
```

## Quick start

```bash
pip install numpy pandas matplotlib mpmath
python demo.py
```

This prints the input parameters, a breakdown of how many points were classified
into each flow regime, the fitted Horner slope, and a table comparing estimated vs.
true `k`, `Pi`, and skin — then saves `loglog\_diagnostic.png` and `horner\_plot.png`.

To explore interactively (it has thephysics explained inline, plus a noisy-data example):

```bash
jupyter notebook well\_test\_simulator.ipynb
```

## The physics (field units)

Wellbore pressure in Laplace space, constant-rate production, wellbore storage
`C`, skin `S`, infinite-acting homogeneous reservoir (Agarwal, Al-Hussainy \& Ramey,
1970):

```
pD\_bar(s)  = K0(sqrt(s)) / (s^1.5 \* K1(sqrt(s)))
pwD\_bar(s) = \[s\*pD\_bar(s) + S] / ( s \* \[1 + CD\*s\*(s\*pD\_bar(s) + S)] )
```

Inverted to real (dimensionless) time via the **Stehfest algorithm**. The buildup
response is obtained by superposition of a `+q` drawdown from `t=0` and a `-q`
drawdown from `t=tp` (valid because the system is linear):

```
Pi - Pws(dt) = 141.2\*q\*mu\*B/(k\*h) \* \[ pwD(tpD + dtD) - pwD(dtD) ]
```

**Diagnostic (log-log) plot**: delta-P and the Bourdet derivative are plotted
against Agarwal equivalent time `dte = tp\*dt/(tp+dt)`, which maps the buildup onto
an equivalent drawdown so standard type-curve logic applies.

**Horner analysis**: on the points classified as IARF, fit

```
Pws = P\* - m \* log10\[(tp+dt)/dt]
```

then:

* `k = 162.6\*q\*mu\*B / (m\*h)`
* `P\* ≈ Pi` (intercept at Horner ratio = 1)
* skin from the standard Horner equation using `P\_1hr` read off the fitted line

## Typical accuracy

On clean (noise-free) synthetic data, the default example (`k=50 md`, `S=5`,
`Pi=4500 psi`) recovers:

|Parameter|True|Estimated|Error|
|-|-|-|-|
|k (md)|50.0|\~49.0|\~2%|
|Pi / P\* (psi)|4500.0|\~4500.3|\~0.01%|
|Skin|5.0|\~4.75|\~5%|

The small residual error is real and expected — it comes from where you draw the
IARF window and normal regression scatter, exactly like real interpretation.
`Pi` is recovered almost exactly because this is a truly infinite-acting reservoir
with no boundaries (no aquifer/fault effects) — a good sanity check that the
simulator and analysis are self-consistent.

**Skin near zero** will show a large *relative* error even when the *absolute*
error is small (dividing by a small true value) — check absolute error too.

## Adding noise / testing robustness

`simulate\_buildup(params, dt, noise\_std=0.5, seed=42)` adds Gaussian noise (in psi)
to the simulated gauge data. For noisy data, switch the derivative to Bourdet's
**L-algorithm** smoothing (`bourdet\_derivative(dte, dP, L=0.15)` to `L=0.3`,
in log10 cycles) — the same technique used in commercial PTA software — instead of
the raw adjacent-point differentiation, which is too noise-sensitive on its own.

## When the analysis fails to find an IARF window

This is usually a real well-test-design issue rather than a bug: with strong
negative skin and large wellbore storage, the derivative can take a long time to
flatten out. If `demo.run(...)` raises `RuntimeError`, try a longer `tp` (flow
period before shut-in) and/or a smaller `C`. This mirrors a genuine lesson from
real test design: you have to flow (and monitor) long enough to actually reach
radial flow.

## Ideas for extending this project

* Multi-rate superposition (variable-rate flow history, not just one flow period)
* Bounded reservoir (adds a second stabilization level or a unit-slope pseudo-steady
state at very late time) or a single sealing fault (slope doubles)
* Monte Carlo sweep over noise levels / test duration to characterize how estimation
error degrades — turns the "accuracy check" into a proper statistical study

