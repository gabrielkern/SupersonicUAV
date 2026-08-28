import numpy as np

def transonic_wave_drag(Mach, CL, t_c, eff_sweep, technology_factor=0.87) -> float:
    """Based on the work by Korn. Korn equation. Yes that's his real name."""
    
    # Turn sweep angle from degrees to radians
    sweep_rad = np.deg2rad(eff_sweep)

    # Korn equation
    M_DD = (technology_factor/np.cos(sweep_rad)) - (t_c/((np.cos(sweep_rad)**2))) - (CL/(10*(np.cos(sweep_rad)**3)))

    # Solve for critical mach number from this divergence number
    M_crit = M_DD - (0.1/80)**(1/3)

    # Final drag
    CD_wave = [20*(Mach - M)**4 if Mach > M else 0 for M in M_crit]

    return CD_wave

if __name__ == "__main__":
    mach = 0.9
    CL = [0.02]
    t_c = 0.08
    eff_sweep = 45
    technology_factor = 0.87
    print(transonic_wave_drag(mach,CL,t_c,eff_sweep,technology_factor))