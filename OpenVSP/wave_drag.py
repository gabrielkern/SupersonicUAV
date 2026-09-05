import numpy as np
from matplotlib import pyplot as plt

def le_sweep_to_te_sweep(le_sweep_deg: float, span: float, root_chord: float, tip_chord: float) -> float:
    """Convert a panel's leading-edge sweep to its trailing-edge sweep.

    Swept-trapezoid relation: the LE and TE sweep of a panel differ only by
    how much the chord shrinks across the span, i.e.
    tan(TE) = tan(LE) - (root_chord - tip_chord) / span.
    """
    le_sweep_rad = np.deg2rad(le_sweep_deg)
    root_tip_le_x_diff = span * np.tan(le_sweep_rad)
    te_sweep_rad = np.arctan( (root_tip_le_x_diff + tip_chord - root_chord) / span)
    return np.rad2deg(te_sweep_rad)

def le_sweep_to_midchord_sweep(le_sweep_deg: float, span: float, root_chord: float, tip_chord: float):
    """Convert's the wing's leading edge sweep to midchord sweep."""
    le_sweep_rad = np.deg2rad(le_sweep_deg)
    root_tip_le_x_diff = span * np.tan(le_sweep_rad)
    midchord_sweep_rad = np.arctan( (root_tip_le_x_diff + (tip_chord/2) - (root_chord/2)) / span)
    return np.rad2deg(midchord_sweep_rad)

def le_sweep_to_quarter_chord_sweep(le_sweep_deg: float, span: float, root_chord: float, tip_chord: float):
    """Converts the wing's leading edge sweep to an approximate quarter-chord sweep."""
    le_sweep_rad = np.deg2rad(le_sweep_deg)
    root_tip_le_x_diff = span * np.tan(le_sweep_rad)
    quarter_chord_sweep_rad = np.arctan( (root_tip_le_x_diff + (tip_chord / 4) - (root_chord / 4) ) / span )
    return np.rad2deg(quarter_chord_sweep_rad)

# The extended Korn equation below (M_DD as a function of sweep) is only
# validated against conventional swept transport wings: Mason, "Configuration
# Aerodynamics" (VT AOE 4124 notes), Sec. 7.5.2, checks it against 747-100
# (~34 deg half-chord sweep) and 777 (~28 deg) flight-test drag rise -- both
# well under 40 deg. The cos(sweep)^-1/-2/-3 terms extrapolate very
# aggressively past that range, so a highly-swept, thin, low-aspect-ratio wing
# (e.g. a supersonic UAV wing swept 50-60 deg) pushes the raw equation's
# output past M_DD = 1, which is not physically possible: any lifting section
# of finite thickness reaches local sonic flow -- and therefore drag
# divergence -- below M = 1. When that happens M_crit ends up above the
# design Mach number and the drag-rise term silently evaluates to zero,
# which is the mechanism behind wave drag being severely underestimated for
# this class of wing. Sweep is therefore capped at the validated envelope and
# M_DD is capped just under 1 as a physical backstop.
MAX_VALIDATED_SWEEP_DEG = 40.0
M_DD_CEILING = 0.99

def transonic_wave_drag(Mach, CL, t_c, eff_sweep, technology_factor=0.87):
    """Korn/Lock transonic wave-drag estimate.

    eff_sweep should be the wing's half-chord (midchord) sweep -- see
    le_sweep_to_midchord_sweep() -- matching how the extended Korn equation
    was validated (Mason Sec. 7.5.2, citing Grassmeyer's strip method, which
    uses half-chord sweep per strip). See the module-level comment above for
    why sweep and M_DD are both capped before the drag-rise term is applied.

    Accepts scalars or array-likes for any of the inputs (broadcast via
    numpy); returns a Python float for scalar input, else an ndarray.
    """
    Mach = np.asarray(Mach, dtype=float)
    CL = np.asarray(CL, dtype=float)
    t_c = np.asarray(t_c, dtype=float)
    eff_sweep = np.asarray(eff_sweep, dtype=float)

    # Turn sweep angle from degrees to radians, capped to the validated range
    sweep_rad = np.deg2rad(np.minimum(eff_sweep, MAX_VALIDATED_SWEEP_DEG))

    # Korn equation
    M_DD = (technology_factor/np.cos(sweep_rad)) - (t_c/((np.cos(sweep_rad)**2))) - (CL/(10*(np.cos(sweep_rad)**3)))
    M_DD = np.minimum(M_DD, M_DD_CEILING)

    # Solve for critical mach number from this divergence number
    M_crit = M_DD - (0.1/80)**(1/3)

    # Final drag (Lock's empirical drag-rise shape), zero below divergence
    CD_wave = np.where(Mach > M_crit, 20*(Mach - M_crit)**4, 0.0)

    return float(CD_wave) if CD_wave.ndim == 0 else CD_wave

def wave_drag_sensitivity():
    """Sensitivity analysis for various paraemters in Korn eqn."""

    # SET BASELINES
    SWEEP = 60
    LIFT_COEFF = 0.25
    THICKNESS = 0.1
    TECHNOLOGY_FACTOR = 0.87
    MACH = 0.85  # comfortably past M_DD for this baseline so %-change is defined (nonzero base)

    varied_parameters = {
        'Delta': SWEEP,
        'CL': LIFT_COEFF,
        'T': THICKNESS,
        'TF': TECHNOLOGY_FACTOR,
        'M': MACH
    }
    # Run base case
    temp_params = varied_parameters.copy()
    total_base = transonic_wave_drag(temp_params['M'],temp_params['CL'],temp_params['T'],
                                     temp_params['Delta'],temp_params['TF'])

    sweep = np.linspace(-0.05,0.05,51)

    all_results = {}
    for idx, (items, values) in enumerate(varied_parameters.items()):
        prc_list = {}
        for position in sweep:
            temp_params = varied_parameters.copy()
            temp_params[items] = values + (values * position)
            prc_list[position] = transonic_wave_drag(temp_params['M'],temp_params['CL'],temp_params['T'],
                                     temp_params['Delta'],temp_params['TF'])
        all_results[items] = prc_list

    for items, values in all_results.items():
        final_parm_list = []
        for position, scores in values.items():
            final_parm_list.append((scores-total_base)/total_base*100)
        plt.plot(sweep*100, final_parm_list, label=items, linewidth=2, marker='o', markersize=4)

    plt.xlabel('Parameter Variation (%)', fontsize=12)
    plt.ylabel('Total Score Change (%)', fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=11)
    plt.axhline(y=0, color='black', linestyle='--', alpha=0.7)
    plt.axvline(x=0, color='black', linestyle='--', alpha=0.7)

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    le_sweeps = np.linspace(0,75,100)
    span = 1.45
    root_chord = 1.45
    tip_chord = 0.16
    mach = 0.99
    CL = 0.15
    t_c = 0.04
    technology_factor = 0.87

    wave_drags = np.zeros_like(le_sweeps)

    for index,sweep in enumerate(le_sweeps):
        midchord_sweep = le_sweep_to_midchord_sweep(sweep, span, root_chord, tip_chord)
        drag = transonic_wave_drag(mach, CL, t_c, midchord_sweep, technology_factor)
        wave_drags[index] = drag

    plt.figure(figsize=(10, 6))
    plt.plot(le_sweeps, wave_drags)
    plt.xlabel('Sweep Value at Midchord')
    plt.ylabel('Drag Coefficient Value')
    plt.grid(True, alpha=0.3)
    plt.title('Drag versus Sweep')
    plt.show()

    # wave_drag_sensitivity()