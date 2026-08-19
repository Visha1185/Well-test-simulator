"""
reservoir.py
------------
Defines the input parameters for the synthetic well test.

All units are FIELD (oilfield) units, the standard for well test equations:
    k    : permeability                [md]
    h    : net pay thickness           [ft]
    phi  : porosity                    [fraction, e.g. 0.20]
    mu   : fluid viscosity             [cp]
    ct   : total system compressibility[1/psi]
    rw   : wellbore radius             [ft]
    q    : flow rate before shut-in    [STB/day]
    B    : formation volume factor     [rb/STB]
    Pi   : initial reservoir pressure  [psi]
    C    : wellbore storage coefficient[bbl/psi]
    S    : skin factor                 [dimensionless]
    tp   : producing time before shut-in [hours]
"""

from dataclasses import dataclass


@dataclass
class ReservoirParams:
    k: float      # permeability, md
    h: float      # net pay thickness, ft
    phi: float    # porosity, fraction
    mu: float     # viscosity, cp
    ct: float     # total compressibility, 1/psi
    rw: float     # wellbore radius, ft
    q: float      # rate before shut-in, STB/D
    B: float      # formation volume factor, rb/STB
    Pi: float     # initial reservoir pressure, psi
    C: float      # wellbore storage coefficient, bbl/psi
    S: float      # skin factor, dimensionless
    tp: float     # producing time before shut-in, hours

    def summary(self) -> str:
        return (
            f"k={self.k} md, h={self.h} ft, phi={self.phi}, mu={self.mu} cp, "
            f"ct={self.ct:.2e} 1/psi, rw={self.rw} ft, q={self.q} STB/D, "
            f"B={self.B} rb/STB, Pi={self.Pi} psi, C={self.C} bbl/psi, "
            f"S={self.S}, tp={self.tp} hr"
        )
