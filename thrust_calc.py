import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

IN = 0.0254
KAPPA, R = 1.4, 287.058
CP = KAPPA * R / (KAPPA - 1)
D_TIP = 10.998935 * IN
D_IND_TIP = 5.808501 * IN
D_HUB_IN = 2.019929 * IN
B2 = 0.438432 * IN
BLOCKAGE2 = 0.95
D_SEAL = 2.0 * IN
D_BACKDISK = D_TIP
K_SWIRL = 0.45
N_DES = 35816.390991
MDOT_DES = 2.68
POWER_DES = 508.952633e3
P01 = 101352.928
T01 = 288.15
P_AMB = 101325.0
P2S_DES = 240e3

SPEEDS = [35800, 40000]
OVERRIDES = {40000: dict(mdot=None, power=None, p2s=None)}

SEAL_SWEEP_IN = [1.25, 1.5, 2.0, 2.5, 3.0, 4.0]
K_SWEEP = [0.35, 0.45, 0.55]
N_RADIAL = 401

def ring_force(r, p):
    f = p * 2 * np.pi * r
    return float(np.sum(0.5 * (f[1:] + f[:-1]) * np.diff(r)))

def cumtrapz(f, x):
    return np.concatenate([[0.0], np.cumsum(0.5 * (f[1:] + f[:-1]) * np.diff(x))])


def inlet_state(mdot):
    A1 = np.pi / 4 * (D_IND_TIP ** 2 - D_HUB_IN ** 2)
    C1 = 150.0
    for _ in range(100):
        T1 = T01 - C1 ** 2 / (2 * CP)
        p1 = P01 * (T1 / T01) ** (KAPPA / (KAPPA - 1))
        C1_new = mdot / (p1 / (R * T1) * A1)
        if abs(C1_new - C1) < 1e-6:
            C1 = C1_new
            break
        C1 = C1_new
    T1 = T01 - C1 ** 2 / (2 * CP)
    p1 = P01 * (T1 / T01) ** (KAPPA / (KAPPA - 1))
    return dict(C1=C1, T1=T1, p1=p1, mdot=mdot)

def exit_state(omega, mdot, dh, p2s=None, n_exp=None):
    U2 = omega * D_TIP / 2
    wt2 = U2 - dh / U2
    A2 = np.pi * D_TIP * B2 * BLOCKAGE2
    cr2 = 100.0
    for _ in range(200):
        W2 = np.hypot(cr2, wt2)
        T2s = T01 + (U2 ** 2 - W2 ** 2) / (2 * CP)
        p = p2s if p2s is not None else P01 * (T2s / T01) ** n_exp
        cr2_new = mdot / (p / (R * T2s) * A2)
        if abs(cr2_new - cr2) < 1e-8:
            cr2 = cr2_new
            break
        cr2 = cr2_new
    return dict(W2=np.hypot(cr2, wt2), T2s=T2s, p2s=p)

def hub_profile(omega, inl, ex):
    r_h1, r2 = D_HUB_IN / 2, D_TIP / 2
    r = np.linspace(r_h1, r2, N_RADIAL)
    s = (r - r_h1) / (r2 - r_h1)
    W_h1 = np.hypot(inl["C1"], omega * r_h1)
    W = W_h1 + (ex["W2"] - W_h1) * s
    T = inl["T1"] + ((omega * r) ** 2 - (omega * r_h1) ** 2 - W ** 2 + W_h1 ** 2) / (2 * CP)
    m = np.log(ex["p2s"] / inl["p1"]) / np.log(T[-1] / inl["T1"])
    return r, inl["p1"] * (T / inl["T1"]) ** m

def cavity_profile(omega, p_tip, T_cav, K, d_seal):
    r = np.linspace(d_seal / 2, D_BACKDISK / 2, N_RADIAL)
    c = cumtrapz((K * omega) ** 2 * r / (R * T_cav), r)
    return r, p_tip * np.exp(c - c[-1])

def thrust(omega, inl, ex, K=K_SWIRL, d_seal=D_SEAL):
    rf, pf = hub_profile(omega, inl, ex)
    rb, pb = cavity_profile(omega, ex["p2s"], ex["T2s"], K, d_seal)
    F = {
        "Nose [N]": -(inl["p1"] - P_AMB) * np.pi * (D_HUB_IN / 2) ** 2,
        "Hub [N]": -ring_force(rf, pf - P_AMB),
        "Inlet momentum [N]": -inl["mdot"] * inl["C1"],
        "Back face [N]": ring_force(rb, pb - P_AMB),
    }
    F["NET [N] (+ toward inlet)"] = sum(F.values())
    return F, (rf, pf), (rb, pb)

def main():
    dh_des = POWER_DES / MDOT_DES
    ex_des = exit_state(N_DES * np.pi / 30, MDOT_DES, dh_des, p2s=P2S_DES)
    n_exp = np.log(P2S_DES / P01) / np.log(ex_des["T2s"] / T01)

    rows, sens, profiles = [], [], {}
    for N in SPEEDS:
        ratio = N / N_DES
        ov = OVERRIDES.get(N, {})
        mdot = ov.get("mdot") or MDOT_DES * ratio
        dh = ov["power"] / mdot if ov.get("power") else dh_des * ratio ** 2
        omega = N * np.pi / 30

        inl = inlet_state(mdot)
        ex = exit_state(omega, mdot, dh, p2s=ov.get("p2s"), n_exp=n_exp)
        F, front, back = thrust(omega, inl, ex)
        profiles[N] = (front, back)
        rows.append({"rpm": N, "mdot [kg/s]": mdot, "p2s [kPaA]": ex["p2s"] / 1e3, **F})

        for ds in SEAL_SWEEP_IN:
            for K in K_SWEEP:
                Fs, _, _ = thrust(omega, inl, ex, K=K, d_seal=ds * IN)
                sens.append({"rpm": N, "seal [in]": ds, "K": K,
                             "F_net": Fs["NET [N] (+ toward inlet)"]})

    sens_df = pd.DataFrame(sens)
    print(pd.DataFrame(rows).set_index("rpm").T.round(1).to_string())
    print()
    print(sens_df.pivot_table(index="seal [in]", columns=["rpm", "K"], values="F_net").round(0).to_string())

    fig, ax = plt.subplots(1, 2, figsize=(15, 6))
    for N, ls in zip(SPEEDS, ["-", "--"]):
        (rf, pf), (rb, pb) = profiles[N]
        ax[0].plot(rf / IN, pf / 1e3, ls, color="tab:blue", label=f"Hub {N}")
        ax[0].plot(rb / IN, pb / 1e3, ls, color="tab:orange", label=f"Back {N}")
        for K, col in zip(K_SWEEP, ["tab:green", "tab:blue", "tab:red"]):
            d = sens_df[(sens_df.rpm == N) & (sens_df.K == K)]
            ax[1].plot(d["seal [in]"], d.F_net, ls, marker="o", color=col, label=f"{N}, K={K}")
    ax[0].set(xlabel="Radius [in]", ylabel="Static pressure [kPaA]")
    ax[1].set(xlabel="Seal diameter [in]", ylabel="Net thrust [N]")
    for a in ax:
        a.grid(alpha=0.3)
        a.legend()
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()