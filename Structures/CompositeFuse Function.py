import numpy as np
import matplotlib.pyplot as plt

import numpy as np


def analyze_composite_tube(
    # --- Material properties ---
    E1,                # psi, fiber-direction modulus
    E2,                # psi, transverse modulus
    G12,               # psi, shear modulus
    v12,               # major Poisson's ratio

    # --- Strength properties ---
    Xt,                # psi, fiber tensile strength
    Xc,                # psi, fiber compressive strength
    Yt,                # psi, transverse tensile strength
    Yc,                # psi, transverse compressive strength
    S12,               # psi, in-plane shear strength

    # --- Max strain criterion ---
    eps_max,           # allowable strain magnitude (applied to ε1 and ε2)

    # --- Applied loads ---
    Shear,             # lbs, transverse shear
    Moment,            # lb-in, bending moment

    # --- Layup ---
    ply_angles,        # list of ply angles in degrees, e.g. [0, 45, 0]
    t_ply,             # inches, single ply thickness

    # --- Geometry ---
    D,                 # inches, outer diameter
    half_section=0,    # 0 = full tube, 1 = half tube (hatch cutout)

    verbose=True
):
    """
    Analyze a composite circular tube (fuselage) under bending + shear.

    Parameters
    ----------
    half_section : int
        0 → full circular tube
        1 → half tube (open section, e.g. hatch cutout on bottom half)

    Returns
    -------
    dict:
        'safety_factor'      : minimum SF across all plies
        'critical_ply'       : 1-based ply index
        'critical_angle'     : ply angle at critical ply
        'critical_criterion' : 'Tsai-Wu' or 'Max Strain'
        'tsai_wu_sf'         : Tsai-Wu SF (min across plies)
        'max_strain_sf'      : Max Strain SF (min across plies)
        'per_ply'            : list of per-ply result dicts
        'NM_at_failure'      : {'N': ..., 'M': ...} scaled to failure load
    """

    if half_section not in (0, 1):
        raise ValueError("half_section must be 0 (full) or 1 (half).")

    # ── derived material constants ─────────────────────────────────────────
    v21   = v12 * E2 / E1
    delta = 1.0 - v12 * v21

    Q_mat = np.array([
        [E1/delta,      v12*E2/delta, 0.0],
        [v12*E2/delta,  E2/delta,     0.0],
        [0.0,           0.0,          G12 ]
    ])

    # ── geometry & section properties ─────────────────────────────────────
    n_plies = len(ply_angles)
    thick   = n_plies * t_ply
    R       = D / 2
    r       = R - thick

    if half_section == 1:
        A_area = np.pi / 2 * (R**2 - r**2)
        ybar   = (4 / (3 * np.pi)) * (R**3 - r**3) / (R**2 - r**2)
        I_base = np.pi / 64 * (D**4 - (D - thick*2)**4)
        Inertia = I_base - A_area * ybar**2
        Q_area  = (R**3 - r**3) / 3
    else:  # full tube
        Inertia = np.pi / 64 * (D**4 - (D - thick*2)**4)
        Q_area  = 2 / 3 * (R**3 - r**3)

    # Stress resultants
    q        = Shear * Q_area / Inertia
    Nxy      = q                                    # shear flow
    sigma_m  = Moment * (D / 2) / 2 / Inertia
    Nx       = -sigma_m * thick                     # bending (compression)

    NM_vec = np.array([Nx, 0, Nxy, 0, 0, 0])

    # ── CLT: build ABD ─────────────────────────────────────────────────────
    z = [-thick/2 + k * t_ply for k in range(n_plies + 1)]

    A_lam = np.zeros((3, 3))
    B_lam = np.zeros((3, 3))
    D_lam = np.zeros((3, 3))

    for k, theta in enumerate(ply_angles):
        Qbar  = _transform_Qbar(Q_mat, theta)
        dz    = z[k+1] - z[k]
        A_lam += Qbar * dz
        B_lam += 0.5   * Qbar * (z[k+1]**2 - z[k]**2)
        D_lam += (1/3) * Qbar * (z[k+1]**3 - z[k]**3)

    ABD           = np.block([[A_lam, B_lam], [B_lam, D_lam]])
    strain_curv   = np.linalg.inv(ABD) @ NM_vec
    eps0          = strain_curv[:3]
    kappa         = strain_curv[3:]

    # ── per-ply analysis ───────────────────────────────────────────────────
    per_ply = []

    for k, theta in enumerate(ply_angles):
        z_mid      = (z[k] + z[k+1]) / 2
        eps_glob   = eps0 + z_mid * kappa
        eps_mat    = _global_to_material_strain(eps_glob, theta)
        sigma_mat  = Q_mat @ eps_mat

        s1, s2, t12        = sigma_mat
        eps1, eps2, gamma12 = eps_mat

        # Tsai-Wu SF
        tw_sf = _tsai_wu_sf(s1, s2, t12, Xt, Xc, Yt, Yc, S12)

        # Max Strain SF (ε1 and ε2 checked against eps_max)
        ms_sf, ms_comp = _max_strain_sf(eps1, eps2, eps_mat, eps_max)

        governing_sf   = min(tw_sf, ms_sf)
        governing_crit = "Tsai-Wu" if tw_sf <= ms_sf else "Max Strain"

        per_ply.append({
            "ply"            : k + 1,
            "angle_deg"      : theta,
            "eps_material"   : eps_mat,
            "sigma_material" : sigma_mat,
            "tsai_wu_sf"     : tw_sf,
            "max_strain_sf"  : ms_sf,
            "max_strain_comp": ms_comp,
            "sf"             : governing_sf,
            "criterion"      : governing_crit,
        })

    # ── first-ply failure ──────────────────────────────────────────────────
    critical     = min(per_ply, key=lambda p: p["sf"])
    overall_sf   = critical["sf"]
    tw_sf_global = min(p["tsai_wu_sf"]   for p in per_ply)
    ms_sf_global = min(p["max_strain_sf"] for p in per_ply)

    NM_fail = {
        "N": overall_sf * NM_vec[:3],
        "M": overall_sf * NM_vec[3:],
    }

    # -- results ---------------------------------------------------------------

    return {
        "safety_factor"      : overall_sf,
        "critical_ply"       : critical["ply"],
        "critical_angle"     : critical["angle_deg"],
        "critical_criterion" : critical["criterion"],
        "critical_component" : critical.get("max_strain_comp"),
        "tsai_wu_sf"         : tw_sf_global,
        "max_strain_sf"      : ms_sf_global,
        "per_ply"            : per_ply,
        "NM_at_failure"      : NM_fail,
    }


# -- helpers ────────────────────────────────────────────────────────────────────

def _transform_Qbar(Q, theta_deg):
    th = np.radians(theta_deg)
    m, n = np.cos(th), np.sin(th)
    Q11, Q12, Q22, Q66 = Q[0,0], Q[0,1], Q[1,1], Q[2,2]
    return np.array([
        [Q11*m**4 + 2*(Q12+2*Q66)*m**2*n**2 + Q22*n**4,
         (Q11+Q22-4*Q66)*m**2*n**2 + Q12*(m**4+n**4),
         (Q11-Q12-2*Q66)*m**3*n + (Q12-Q22+2*Q66)*m*n**3],
        [(Q11+Q22-4*Q66)*m**2*n**2 + Q12*(m**4+n**4),
         Q11*n**4 + 2*(Q12+2*Q66)*m**2*n**2 + Q22*m**4,
         (Q11-Q12-2*Q66)*m*n**3 + (Q12-Q22+2*Q66)*m**3*n],
        [(Q11-Q12-2*Q66)*m**3*n + (Q12-Q22+2*Q66)*m*n**3,
         (Q11-Q12-2*Q66)*m*n**3 + (Q12-Q22+2*Q66)*m**3*n,
         (Q11+Q22-2*Q12-2*Q66)*m**2*n**2 + Q66*(m**4+n**4)]
    ])


def _global_to_material_strain(eps_global, theta_deg):
    th = np.radians(theta_deg)
    m, n = np.cos(th), np.sin(th)
    T = np.array([
        [ m**2,   n**2,   m*n],
        [ n**2,   m**2,  -m*n],
        [-2*m*n,  2*m*n,  m**2 - n**2]
    ])
    return T @ eps_global


def _tsai_wu_sf(s1, s2, t12, Xt, Xc, Yt, Yc, S):
    F1  = 1/Xt - 1/Xc
    F11 = 1/(Xt*Xc)
    F2  = 1/Yt - 1/Yc
    F22 = 1/(Yt*Yc)
    F66 = 1/S**2
    F12 = -1/(2*np.sqrt(Xt*Xc*Yt*Yc))

    A = F1*s1 + F2*s2
    B = F11*s1**2 + F22*s2**2 + F66*t12**2 + 2*F12*s1*s2

    roots = np.roots([B, A, -1])
    roots = roots[np.isreal(roots)].real
    roots = roots[roots > 0]
    return float(np.min(roots)) if len(roots) > 0 else np.inf


def _max_strain_sf(eps1, eps2, eps_mat, eps_max):
    """
    Check ε1 and ε2 against eps_max.
    Returns (SF, component_name).
    """
    checks = [
        (abs(eps1), "ε1 (fiber)"),
        (abs(eps2), "ε2 (transverse)"),
    ]
    best_sf   = np.inf
    best_comp = "None"
    for actual, name in checks:
        if actual > 1e-14:
            sf = eps_max / actual
            if sf < best_sf:
                best_sf   = sf
                best_comp = name
    return best_sf, best_comp


# ── example usage ──────────────────────────────────────────────────────────────
if __name__ == "__main__":

    # Full tube
    results_full = analyze_composite_tube(
        E1=20e6, E2=20e6, G12=0.6e6, v12=0.3,
        Xt=200e3, Xc=150e3, Yt=200e3, Yc=150e3, S12=14e3,
        eps_max=0.01,
        Shear=8, Moment=0,
        ply_angles=[0, 45, 0],
        t_ply=0.02,
        D=0.3,
        half_section=0,    # 0 = full tube
    )

    # Half tube (hatch cutout)
    results_half = analyze_composite_tube(
        E1=20e6, E2=20e6, G12=0.6e6, v12=0.3,
        Xt=200e3, Xc=150e3, Yt=200e3, Yc=150e3, S12=14e3,
        eps_max=0.01,
        Shear=8, Moment=0,
        ply_angles=[0, 45, 0],
        t_ply=0.02,
        D=0.3,
        half_section=1,    # 1 = half tube
    )