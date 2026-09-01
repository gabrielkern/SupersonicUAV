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

def transonic_wave_drag(Mach, CL, t_c, eff_sweep, technology_factor=0.87) -> float:
    """Based on the work by Korn. Korn equation. Yes that's his real name."""
    
    # Turn sweep angle from degrees to radians
    sweep_rad = np.deg2rad(eff_sweep)

    # Korn equation
    M_DD = (technology_factor/np.cos(sweep_rad)) - (t_c/((np.cos(sweep_rad)**2))) - (CL/(10*(np.cos(sweep_rad)**3)))

    # Solve for critical mach number from this divergence number
    M_crit = M_DD - (0.1/80)**(1/3)

    # Final drag
    if isinstance(M_DD,float):
        CD_wave = 20*(Mach - M_crit)**4
    else:
        CD_wave = [20*(Mach - M)**4 if Mach > M else 0 for M in M_crit]

    return CD_wave

def wave_drag_sensitivity():
    """Sensitivity analysis for various paraemters in Korn eqn."""

    # SET BASELINES
    SWEEP = 60
    LIFT_COEFF = 0.25
    THICKNESS = 0.1
    TECHNOLOGY_FACTOR = 0.87
    MACH = 0.8

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
    le_sweep = 60
    span = 1.45
    root_chord = 1.45
    tip_chord = 0.16
    mach = 0.99
    CL = [0.15]
    t_c = 0.04
    technology_factor = 0.85

    wave_drag_sensitivity()