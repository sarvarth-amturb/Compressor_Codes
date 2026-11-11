#!/usr/bin/env python3
"""
Axial preload and 5x5 stiffness matrix for a double-row (or duplex pair of)
angular contact ball bearings.

Model: Gunduz, Dreyer, Singh, "Effect of bearing preloads on the modal
characteristics of a shaft-bearing assembly", Mechanical Systems and Signal
Processing 31 (2012) 176-195, Eqs. (2)-(13) and (25).

Units:        N, mm, rad  (stiffness in N/mm, N/rad, N*mm/rad)
Coordinates:  x, y radial; z along the shaft.
              Row 0 is the left row (z = -dz), row 1 the right row (z = +dz).
Vectors:      q = [delta_x, delta_y, delta_z, beta_x, beta_y]
              f = [F_x, F_y, F_z, M_x, M_y]

Validation status:
  - The numerical kxx matches the paper's closed-form Eq. (13) to 4 digits.
  - It does NOT reproduce the paper's Table 2 exactly (back-to-back kxx is
    -11% at 0.5 kN and +34% at 5 kN with zero radial clearance). The paper
    does not state the radial clearance used for that table.
  - Static model (zero speed): no centrifugal or gyroscopic ball loads.

Usage:
  python bearing_preload.py          validation run against the paper
  python bearing_preload.py --ask    enter your own bearing
"""

import argparse
import math
from dataclasses import dataclass

import numpy as np

# mu^i in Eq. (2): sign of each row's axial position
ROW_SIDE = (-1.0, +1.0)

# nu^i in Eq. (8): direction each row's contact line leans
ARRANGEMENTS = {
    "back-to-back": (+1.0, -1.0),  # Eq. (8a)
    "face-to-face": (-1.0, +1.0),  # Eq. (8b)
    "tandem":       (+1.0, +1.0),  # Eq. (8c)
}

# Finite-difference steps for [delta_x, delta_y, delta_z, beta_x, beta_y]
FD_STEPS = (1e-7, 1e-7, 1e-7, 1e-8, 1e-8)  # mm, mm, mm, rad, rad


@dataclass
class Bearing:
    Z: int                  # balls per row
    Kn: float               # Hertz load-deflection constant, N/mm^n
    R: float                # pitch radius, mm
    A0: float               # unloaded distance between inner and outer
                            #   raceway groove curvature centers, mm
    dz: float               # axial distance from pair center to one row, mm
    alpha0_deg: float       # unloaded contact angle, deg
    arrangement: str = "back-to-back"
    n: float = 1.5          # load-deflection exponent (1.5 for balls)
    rL: float = 0.0         # radial clearance, mm (negative = interference)

    def __post_init__(self):
        if self.arrangement not in ARRANGEMENTS:
            raise ValueError(f"arrangement must be one of {list(ARRANGEMENTS)}")
        self.Z = int(self.Z)
        self.alpha0 = math.radians(self.alpha0_deg)
        self.lean = ARRANGEMENTS[self.arrangement]
        # ball angular positions, measured from +x
        self.psi = 2.0 * math.pi * np.arange(self.Z) / self.Z

    # ---- Eqs. (2)-(11): displacement -> load --------------------------------

    def loads(self, q, dz0):
        """Load vector f for displacement vector q and displacement preload
        dz0 (mm; one value for both rows, or a (left, right) pair)."""
        dx, dy, dzm, bx, by = q
        dz0_rows = (dz0, dz0) if np.isscalar(dz0) else dz0
        cos_psi, sin_psi = np.cos(self.psi), np.sin(self.psi)

        f = np.zeros(5)
        for side, lean, preload in zip(ROW_SIDE, self.lean, dz0_rows):
            # Eq. (2): radial deflection at each ball
            d_r = ((dx + side * self.dz * by) * cos_psi
                   + (dy - side * self.dz * bx) * sin_psi
                   - self.rL)
            # Eq. (3): axial deflection at each ball
            d_z = dzm + self.R * (bx * sin_psi - by * cos_psi)
            # Eqs. (4), (5): loaded distances between curvature centers
            dr_star = d_r + self.A0 * math.cos(self.alpha0)
            dz_star = d_z + lean * (self.A0 * math.sin(self.alpha0) + preload)
            # Eq. (6)
            A = np.hypot(dr_star, dz_star)
            # Eq. (7): elastic deformation (zero when the ball is unloaded)
            delta = np.maximum(A - self.A0, 0.0)
            # Eq. (9): Hertz contact load
            Q = self.Kn * delta ** self.n
            # Eq. (10): loaded contact angle (sign carries the row lean)
            cos_a, sin_a = dr_star / A, dz_star / A
            # Eq. (11)
            arm = self.R * sin_a - side * self.dz * cos_a
            f += [np.sum(Q * cos_a * cos_psi),
                  np.sum(Q * cos_a * sin_psi),
                  np.sum(Q * sin_a),
                  np.sum(Q * arm * sin_psi),
                  np.sum(-Q * arm * cos_psi)]
        return f

    # ---- Preload force <-> preload displacement -----------------------------

    def row_axial_force(self, dz0):
        """Axial force (N) carried by one row at q = 0."""
        dr_star = self.A0 * math.cos(self.alpha0) - self.rL
        dz_star = self.A0 * math.sin(self.alpha0) + dz0
        A = math.hypot(dr_star, dz_star)
        delta = max(A - self.A0, 0.0)
        return self.Z * self.Kn * delta ** self.n * dz_star / A

    def dz0_eq25(self, Fz0):
        """Eq. (25): closed form, contact angle held at alpha0."""
        return (Fz0 / (self.Z * self.Kn
                       * math.sin(self.alpha0) ** (1.0 + self.n))) ** (1.0 / self.n)

    def dz0_exact(self, Fz0):
        """Displacement preload (mm) that gives row force Fz0 (N), with the
        contact angle free to change. Solved by bisection."""
        lo, hi = 0.0, max(10.0 * self.dz0_eq25(Fz0), 1e-6)
        while self.row_axial_force(hi) < Fz0:
            hi *= 2.0
        for _ in range(100):
            mid = 0.5 * (lo + hi)
            if self.row_axial_force(mid) < Fz0:
                lo = mid
            else:
                hi = mid
        return 0.5 * (lo + hi)

    # ---- Eq. (12): stiffness matrix -----------------------------------------

    def stiffness(self, dz0, q0=None):
        """Kb = df/dq about operating point q0, by central differences."""
        q0 = np.zeros(5) if q0 is None else np.asarray(q0, dtype=float)
        K = np.zeros((5, 5))
        for k, step in enumerate(FD_STEPS):
            dq = np.zeros(5)
            dq[k] = step
            K[:, k] = (self.loads(q0 + dq, dz0)
                       - self.loads(q0 - dq, dz0)) / (2.0 * step)
        return K

    def kxx_eq13(self, dz0):
        """Closed-form kxx at q = 0 from Eq. (13); cross-check on stiffness()."""
        dr_star = self.A0 * math.cos(self.alpha0) - self.rL
        dz_star = self.A0 * math.sin(self.alpha0) + dz0
        A = math.hypot(dr_star, dz_star)
        delta = A - self.A0
        if delta <= 0.0:
            return 0.0
        per_ball = (delta ** self.n
                    * (self.n * A * dr_star ** 2 / delta + dz_star ** 2) / A ** 3)
        rows = 2
        return self.Kn * rows * per_ball * np.sum(np.cos(self.psi) ** 2)

    # ---- Kn from catalog data -----------------------------------------------

    def calibrate_Kn(self, Fz0, kzz_target):
        """Set Kn so the pair's kzz at preload Fz0 (N) equals kzz_target
        (N/mm), e.g. a catalog axial rigidity. Returns the new Kn."""
        lo, hi = 1e2, 1e8
        for _ in range(100):
            self.Kn = math.sqrt(lo * hi)
            if self.stiffness(self.dz0_exact(Fz0))[2, 2] < kzz_target:
                lo = self.Kn
            else:
                hi = self.Kn
        self.Kn = math.sqrt(lo * hi)
        return self.Kn


def natural_frequencies_hz(K, mass_kg, inertia_kgm2):
    """Undamped modes (Hz) of a rigid shaft on the bearing (paper Section 5).
    K is in N, mm, rad; inertia is the transverse moment about the pair center."""
    # K[i, j] -> SI: force rows stay in N, moment rows N*mm -> N*m,
    # translation columns per mm -> per m, rotation columns stay per rad.
    row_scale = np.array([1.0, 1.0, 1.0, 1e-3, 1e-3])
    col_scale = np.array([1e3, 1e3, 1e3, 1.0, 1.0])
    K_si = K * row_scale[:, None] * col_scale[None, :]
    M = np.diag([mass_kg] * 3 + [inertia_kgm2] * 2)
    omega_sq = np.sort(np.linalg.eigvals(np.linalg.solve(M, K_si)).real)
    return np.sqrt(np.maximum(omega_sq, 0.0)) / (2.0 * math.pi)


def print_report(bearing, Fz0):
    """Print preload displacement and main stiffness terms; return Kb."""
    dz0 = bearing.dz0_exact(Fz0)
    K = bearing.stiffness(dz0)
    print(f"  arrangement           : {bearing.arrangement}")
    print(f"  preload force per row : {Fz0:.1f} N")
    print(f"  preload displacement  : {dz0 * 1e3:.3f} um exact, "
          f"{bearing.dz0_eq25(Fz0) * 1e3:.3f} um by Eq. 25")
    print(f"  kxx = kyy             : {K[0, 0] / 1e3:8.1f} kN/mm "
          f"(Eq. 13: {bearing.kxx_eq13(dz0) / 1e3:.1f})")
    print(f"  kzz                   : {K[2, 2] / 1e3:8.1f} kN/mm")
    print(f"  k_thetax = k_thetay   : {K[3, 3] / 1e6:8.1f} MN*mm/rad")
    print(f"  k_x,thetay            : {K[0, 4] / 1e6:8.2f} MN/rad")
    print(f"  k_y,thetax            : {K[1, 3] / 1e6:8.2f} MN/rad")
    return K


def validate():
    """Run the paper's Table 1 bearing at the Table 2 preloads."""
    print("Gunduz et al. Table 1 bearing, zero radial clearance\n")
    for arrangement in ARRANGEMENTS:
        for Fz0 in (500.0, 5000.0):
            bearing = Bearing(Z=13, Kn=414000.0, R=31.75, A0=0.65, dz=18.20,
                              alpha0_deg=35.0, arrangement=arrangement)
            K = print_report(bearing, Fz0)
            modes = natural_frequencies_hz(K, mass_kg=5.0, inertia_kgm2=0.015)
            print(f"  rigid-shaft modes     : {np.round(modes).astype(int)} Hz\n")


def ask_float(prompt, default=None):
    suffix = f" [{default}]" if default is not None else ""
    while True:
        text = input(f"{prompt}{suffix}: ").strip()
        if not text and default is not None:
            return float(default)
        try:
            return float(text)
        except ValueError:
            print("  enter a number")


def ask():
    """Interactive run for a user-supplied bearing."""
    print("Bearing data in N, mm, deg. Press Enter to accept a default.\n")
    Z = ask_float("Balls per row")
    R = ask_float("Pitch radius = pitch diameter / 2 (mm)")
    alpha0_deg = ask_float("Unloaded contact angle (deg)")
    D = ask_float("Ball diameter (mm)")
    fi = ask_float("Inner raceway conformity (groove radius / ball diameter)", 0.52)
    fo = ask_float("Outer raceway conformity (groove radius / ball diameter)", 0.52)
    A0 = (fi + fo - 1.0) * D
    print(f"  A0 = (fi + fo - 1) * D = {A0:.4f} mm")
    dz = ask_float("Axial distance from pair center to one ball row (mm)")
    arrangement = input("Arrangement (back-to-back / face-to-face / tandem) "
                        "[back-to-back]: ").strip() or "back-to-back"
    rL = ask_float("Radial clearance (mm, negative = interference)", 0.0)

    bearing = Bearing(Z=Z, Kn=1.0, R=R, A0=A0, dz=dz, alpha0_deg=alpha0_deg,
                      arrangement=arrangement, rL=rL)

    text = input("Hertz constant Kn (N/mm^1.5), or Enter to calibrate "
                 "from catalog axial rigidity: ").strip()
    if text:
        bearing.Kn = float(text)
    else:
        F_cat = ask_float("Catalog preload (N)")
        k_cat = ask_float("Catalog axial rigidity of the pair at that preload (N/um)")
        print(f"  calibrated Kn = {bearing.calibrate_Kn(F_cat, k_cat * 1e3):.0f} N/mm^1.5")

    np.set_printoptions(precision=4, linewidth=140)
    while True:
        text = input("\nAxial preload force (N), or Enter to quit: ").strip()
        if not text:
            break
        K = print_report(bearing, float(text))
        print("  full Kb (N, mm, rad):")
        print(K)


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--ask", action="store_true",
                        help="enter your own bearing instead of the validation run")
    args = parser.parse_args()
    ask() if args.ask else validate()


if __name__ == "__main__":
    main()
