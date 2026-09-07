import argparse
import numpy as np
import sys
from typing import Union
from pathlib import Path
from matplotlib import pyplot as plt
from scipy.interpolate import CubicSpline

sys.path.append(str(Path(__file__).parent.parent))
from tools.atmos import get_atmosphere

# Constants
GAMMA = 1.4
R = 1716 # ft*lbf/slug/R or s2/ft2/R

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

def wislicenus_wave_drag(mach, CL, t_c, eff_sweep, altitude, technology_factor=0.87):
    """
    Using Wislicenus's modified-Lock transonic wave-drag model (Eq. 10-11, 18-19):

    https://www.mdpi.com/2226-4310/9/4/192
    Wislicenus, J.; Daidzic, N.E. "Estimation of Transport-Category Jet Airplane
    Maximum Range and Airspeed in the Presence of Transonic Wave Drag."
    Aerospace 2022, 9, 192.
    """

    mach = np.asarray(mach, dtype=float)
    CL = np.asarray(CL, dtype=float)
    t_c = np.asarray(t_c, dtype=float)
    eff_sweep = np.asarray(eff_sweep, dtype=float)

    sweep_rad = np.deg2rad(eff_sweep)

    _, temp_alt = get_atmosphere(altitude)
    _, temp_sl = get_atmosphere(0)
    # ambiance's Atmosphere always returns arrays, even for scalar altitude
    # input (see variable_plane_analysis.py's identical squeeze) -- this
    # function assumes one altitude per call, so reduce to plain floats.
    temp_alt = float(np.ravel(temp_alt)[0])
    temp_sl = float(np.ravel(temp_sl)[0])

    # Define predefined empirical parameters, using same notation as Wislicenus
    dCD_dM = 0.1
    M_DD0 = (technology_factor/np.cos(sweep_rad)) - (t_c/((np.cos(sweep_rad))**2))
    a_SL = np.sqrt(R * GAMMA * temp_sl)
    f = 5e-3
    z = 20
    m = 4
    kappa = 0.1
    theta = temp_alt / temp_sl

    # Define velocity
    v = mach * np.sqrt(R * GAMMA * temp_alt)

    # Now construct lowercase coefficients
    a = 1 / (a_SL * np.sqrt(theta))
    b = M_DD0 - ((dCD_dM / m / z)**(1 / (m-1)))
    c = f
    d = np.sqrt(CL) * v
    e = kappa / ((np.cos(sweep_rad))**3)

    # Now we construct the capital letter coefficients (Eq. 19). B's sign
    # and F's missing term are corrected from the raw OCR text -- see the
    # docstring note above for how/why.
    A = a**4
    B = -4 * (a**3) * b
    C = 2 * (a**2) * (2*a*c*d + b**2) + (4*a**2*b**2)
    D = 4 * a**2 * (a*e*d**2 - b*c*d) - (4*a*b)*(2*a*c*d + b**2)
    E = (2*a**2*d**2*(c**2-2*b*e)) - (8*a*b*((a*e*d**2 - b*c*d))) + ((2*a*c*d + b**2)**2)
    F = (4*a**2*c*e*d**3) - ((4*a*b*d**2)*(c**2-2*b*e)) + (4*(2*a*c*d+b**2)*(a*e*d**2 - b*c*d))
    G = (2*a**2*e**2*d**4) - (8*a*b*c*d**3*e) + (2*d**2*(2*a*c*d+b**2)*(c**2-2*b*e)) + 4*((a*e*d**2 - b*c*d)**2)
    H = (-4*a*b*e**2*d**4) + (4*c*e*d**3*(2*a*c*d+b**2)) + (4*d**2*(c**2-2*b*e)*(a*e*d**2 - b*c*d))
    I = 2*e**2*d**4*(2*a*c*d+b**2) + 8*c*e*d**3*(a*e*d**2 - b*c*d) + d**4*(c**2-2*b*e)**2
    J = 4*e**2*d**4*(a*e*d**2 - b*c*d) + 4*c*e*d**5*(c**2-2*b*e)
    K = 2*e**2*d**6*(c**2-2*b*e) + 4*c**2*e**2*d**6
    M_ = 4*c*d**7*e**3
    N_ = d**8*e**4

    # Eq. (18)
    laurent_sum = (A*v**4 + B*v**3 + C*v**2 + D*v + E
                   + F/v + G/v**2 + H/v**3 + I/v**4
                   + J/v**5 + K/v**6 + M_/v**7 + N_/v**8)
    CD_wave_raw = z * laurent_sum

    # Domain gate matching Eq. (11) and Eq. (12)
    M_CR = -e*CL + b # MDD0 in 11 and 12 cancel
    CD_wave = np.where(mach > M_CR, CD_wave_raw, 0.0)

    return float(CD_wave) if CD_wave.ndim == 0 else CD_wave

def grassmeyer_strip_wave_drag(mach, strip_cl, strip_t_c, strip_half_chord_sweep, strip_area_fraction, technology_factor=0.87):
    """
    Joel Grassmeyer's spanwise-strip wave-drag buildup (Mason, "Configuration
    Aerodynamics" (VT AOE 4124 notes), Sec. 7.5.2, Eq. 7-9): sums the swept
    Korn/Lock wave-drag coefficient (transonic_wave_drag) of each spanwise
    strip, weighted by that strip's fraction of the total wing reference
    area. Validated by Mason against Boeing 747-100 flight-test drag-rise
    data across a range of Mach numbers and lift coefficients -- see
    wiki/sources/mason-korn-equation-transonic-airfoil-technology-factor.md
    and wiki/concepts/full-configuration-transonic-wave-drag-estimation.md
    in this repository.

    strip_cl, strip_t_c, strip_half_chord_sweep, and strip_area_fraction
    must all be 1D array-likes of the same length (one entry per strip);
    strip_area_fraction should sum to 1.0 across all strips (each strip's
    own planform area divided by the total wing reference area).
    strip_half_chord_sweep must be in degrees, half-chord (midchord) sweep
    -- see le_sweep_to_midchord_sweep() -- matching transonic_wave_drag's
    own eff_sweep convention. mach and technology_factor are scalars shared
    across all strips.

    Scope: like transonic_wave_drag itself, this reproduces only the WING's
    own thickness/lift-driven wave drag -- no fuselage term, no wing/body
    interference term. That is a real, documented limitation of this
    method specifically (not an oversight here): see
    wiki/concepts/full-configuration-transonic-wave-drag-estimation.md for
    why interference wave drag needs the combined-geometry area-rule method
    (suave_corrected_wave_drag() / variable_plane_wave_drag.py) instead.
    """
    strip_cl = np.asarray(strip_cl, dtype=float)
    strip_t_c = np.asarray(strip_t_c, dtype=float)
    strip_half_chord_sweep = np.asarray(strip_half_chord_sweep, dtype=float)
    strip_area_fraction = np.asarray(strip_area_fraction, dtype=float)

    if not (strip_cl.shape == strip_t_c.shape == strip_half_chord_sweep.shape == strip_area_fraction.shape):
        raise ValueError(
            "strip_cl, strip_t_c, strip_half_chord_sweep, and strip_area_fraction "
            "must all be the same shape (one entry per strip); got shapes "
            f"{strip_cl.shape}, {strip_t_c.shape}, {strip_half_chord_sweep.shape}, "
            f"{strip_area_fraction.shape}."
        )

    strip_cdwave = np.atleast_1d(np.asarray(
        transonic_wave_drag(mach, strip_cl, strip_t_c, strip_half_chord_sweep, technology_factor),
        dtype=float,
    ))

    return float(np.sum(strip_cdwave * strip_area_fraction))

def grassmeyer_strip_wave_drag_from_planform(mach, CL, t_c, root_chord, tip_chord, span, le_sweep_deg,
                                              technology_factor=0.87, num_strips=10):
    """
    Convenience wrapper around grassmeyer_strip_wave_drag() for a single
    straight-tapered trapezoidal wing panel -- i.e. exactly the planform
    TopSpeedSim.py's __main__ configuration already describes via
    root_chord/tip_chord/b_ref(span)/le_sweep. Discretizes the panel into
    num_strips spanwise strips of equal span, using linear taper for local
    chord (and hence strip area) between root_chord and tip_chord.

    CL and t_c are applied uniformly across all strips, since this codebase
    has no spanwise-varying lift or thickness distribution to draw on (its
    aero comes from a whole-wing VSPAERO CL, and t_c is a single scalar in
    every config dict in this repo) -- only local chord (and, for a
    non-linear/multi-panel planform, local sweep) can meaningfully vary
    strip-to-strip here.

    Note: for a SINGLE straight-tapered panel, the leading-edge, half-chord,
    and trailing-edge lines are all straight from root to tip, so half-chord
    sweep is mathematically identical at every spanwise station -- every
    strip this function builds will have the same sweep, and (combined with
    the uniform CL/t_c above) the same per-strip wave-drag coefficient, so
    the area-weighted sum here is expected to come out equal to a single
    whole-wing transonic_wave_drag() call at this sweep/CL/t_c. That's the
    correct, expected result for this simple wing model, not a bug -- this
    function still computes sweep per-strip (rather than once) so it
    generalizes correctly if it's ever pointed at strips carved out of a
    multi-panel wing (e.g. the inboard/outboard panels already defined in
    sizingEstimation.py's build_config()), where sweep genuinely does vary
    strip-to-strip.
    """
    root_chord = float(root_chord)
    tip_chord = float(tip_chord)
    span = float(span)

    y_edges = np.linspace(0.0, span, num_strips + 1)
    chord_edges = root_chord + (tip_chord - root_chord) * (y_edges / span)

    strip_span = np.diff(y_edges)
    strip_area = 0.5 * (chord_edges[:-1] + chord_edges[1:]) * strip_span
    total_area = 0.5 * (root_chord + tip_chord) * span
    strip_area_fraction = strip_area / total_area

    strip_half_chord_sweep = np.full(
        num_strips,
        le_sweep_to_midchord_sweep(le_sweep_deg, span, root_chord, tip_chord),
        dtype=float,
    )
    strip_cl = np.full(num_strips, CL, dtype=float)
    strip_t_c = np.full(num_strips, t_c, dtype=float)

    return grassmeyer_strip_wave_drag(mach, strip_cl, strip_t_c, strip_half_chord_sweep,
                                       strip_area_fraction, technology_factor)

def suave_corrected_wave_drag(config:dict, CL:Union[float, list], vspfile:str,
                               begin_drag_rise_mach=0.87, end_drag_rise_mach=1.2,
                               peak_mach=1.04, transonic_drag_multiplier=1.25,
                               num_slices=20, num_rots=10):
    """
    SUAVE-style bridge between the subsonic Korn/Lock drag-divergence
    correlation (transonic_wave_drag) and OpenVSP's supersonic-area-rule
    Wave Drag tool (variable_plane_wave_drag.py -- a separate module, since
    it drives the actual OpenVSP interface), spanning the true transonic
    gap where *neither* theory is valid (the Mach angle underlying the
    area-rule method is undefined for M<=1).

    end_drag_rise_mach/peak_mach/transonic_drag_multiplier match SUAVE's own
    documented defaults (SUAVE/Analyses/Aerodynamics/Supersonic_Zero.py:
    end_drag_rise_mach=1.2, peak_mach=1.04, transonic_drag_multiplier=1.25
    -- themselves calibrated against Concorde-class data (Yoshida, K.,
    "Supersonic drag reduction technology in the scaled supersonic
    experimental airplane project by JAXA," Progress in Aerospace Sciences,
    45(4-5), 2009)). transonic_drag_multiplier corrects for linear
    supersonic theory, evaluated at the low edge of its own valid range,
    systematically underestimating the true M~1 drag peak. begin_drag_rise_mach
    is deliberately *not* SUAVE's own default of 0.95: SUAVE derives its
    value from the Korn equation's technology_factor/cos(sweep) term, which
    this module's own MAX_VALIDATED_SWEEP_DEG=40 comment (above) documents
    as unreliable for this UAV's much more highly swept wing. 0.87 is
    instead a deliberately conservative flat empirical baseline, chosen
    independent of that term, that this project may tune further downward
    as sweep-specific data becomes available. See
    wiki/concepts/full-configuration-transonic-wave-drag-estimation.md and
    wiki/sources/suave-transonic-wave-drag-bridging-implementation.md in
    this repository for the full derivation and citations.

    Cubic-spline fairing (SUAVE's own older/simpler module uses linear
    interpolation between anchors instead; its current default module uses
    a parabolic/cubic-spline blend for a continuous slope, which is what
    this function's CubicSpline call reproduces):
        M <= begin_drag_rise_mach:                    subsonic correlation
        begin_drag_rise_mach < M <= end_drag_rise_mach: cubic spline through
                                                         (M_CR, peak_mach, end_drag_rise_mach)
                                                         anchors
        M > end_drag_rise_mach:                        fresh OpenVSP call at M

    mach must be a scalar: OpenVSP's WaveDrag analysis (run through
    variable_plane_wave_drag) is evaluated one Mach number at a time, same
    as variable_plane_analysis.py/variable_plane_parasitic.py elsewhere in
    this directory. CL/t_c/eff_sweep may be scalars or arrays (broadcast
    through transonic_wave_drag for the subsonic anchor only -- the
    supersonic/OpenVSP anchor is a whole-geometry volume-wave-drag result
    and does not depend on CL/t_c/eff_sweep at all, unlike the subsonic
    correlation).

    wing_area/model_unit/vspfile mirror the config['wing_area'],
    config['model_unit'], and vspfile arguments already used to call
    variable_plane_analysis.main()/variable_plane_parasitic.main() elsewhere
    in this codebase (e.g. sizingEstimation.py's evaluate_configuration()).

    NOTE: variable_plane_wave_drag is imported lazily, inside this function,
    rather than at module level. That module imports the OpenVSP Python
    bindings as soon as it's imported, which would otherwise make every
    other function in this file (transonic_wave_drag, wislicenus_wave_drag,
    the Grassmeyer functions above -- none of which need OpenVSP) fail to
    import wherever OpenVSP isn't installed/configured on PYTHONPATH.
    """
    mach = config['mach_start']
    wing_area = config['wing_area']
    model_unit = config['model_unit']
    eff_sweep = config['effective_sweep']
    # atleast_1d: a scalar CL (per the CL:Union[float, list] signature, and
    # what the __main__ demo below passes) would otherwise produce a 0-d
    # M_CR array, which the enumerate() loop below can't iterate over.
    CL = np.atleast_1d(np.asarray(CL, dtype=float))
    config = {'wing_area': wing_area, 'model_unit': model_unit}

    # Subsonic anchor: Calculate MCR
    dCD_dM = 0.1
    m = 4
    z = 20
    kappa = 0.1
    M_DD = begin_drag_rise_mach - (kappa * CL / (np.cos(np.deg2rad(eff_sweep)))**3)
    M_CR = M_DD - (dCD_dM / m / z)**(1/(m-1))
    cd_subsonic = 0 # Assume no wavedrag at critical mach

    def run_openvsp(query_mach):
        # Imported here (not at module level, and not at the top of this
        # function) so that a query entirely below the transonic gap -- the
        # common case for a "how does drag build up subsonically" sweep --
        # never touches the OpenVSP Python bindings at all.
        import variable_plane_wave_drag
        cd = variable_plane_wave_drag.main(config, mach=query_mach, filename=vspfile,
                                            num_slices=num_slices, num_rots=num_rots)
        if cd is None:
            raise RuntimeError(
                f"OpenVSP WaveDrag analysis failed at Mach {query_mach} -- "
                "see printed OpenVSP error output above."
            )
        return float(cd)

    # Supersonic anchor: OpenVSP's Mach-angle/Eminton-Lord area-rule result,
    # which already captures wing/fuselage interference natively (see
    # full-configuration-transonic-wave-drag-estimation.md) -- unlike
    # cd_subsonic, this is a whole-geometry result and does not depend on
    # CL/t_c/eff_sweep directly.
    cd_supersonic_end = run_openvsp(end_drag_rise_mach)

    # Peak-drag estimate at peak_mach: SUAVE's own recipe is simply the
    # supersonic anchor scaled up by transonic_drag_multiplier.
    cd_peak = cd_supersonic_end * transonic_drag_multiplier

    constant_mach_wave_drags = np.zeros_like(M_CR)
    for index,mcr in enumerate(M_CR):
        mach_nums = np.array([mcr, peak_mach, end_drag_rise_mach])
        cd_nums = np.array([cd_subsonic, cd_peak, cd_supersonic_end])

        interp_cd = CubicSpline(mach_nums,cd_nums)

        if mach <= begin_drag_rise_mach:
            constant_mach_wave_drags[index] = cd_subsonic
        elif (mach <= end_drag_rise_mach):
            constant_mach_wave_drags[index] = interp_cd(mach)
        else:
            # Don't extrapolate upwards, get directly from VSP
            constant_mach_wave_drags[index] = run_openvsp(mach)

    return constant_mach_wave_drags

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

# TODO: Add in the wave drag from all the wing surfaces in the shape and do some testing on these equations.

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Wave drag model exploration/comparison plots")
    parser.add_argument('--run-all', action='store_true',
                         help='Run every wave-drag-vs-Mach mode (Korn/Lock, Wislicenus, Grassmeyer '
                              'strip, SUAVE-corrected) at a fixed 60 deg LE sweep across Mach '
                              '0.5-1.5, and plot them together as one multi-panel figure, instead '
                              'of the default single Korn/Lock vs. Mach plot.')
    args = parser.parse_args()

    # Shared baseline geometry/flight-condition parameters, common to every mode.
    # Fixed at a high (60 deg) LE sweep and swept across the full Mach 0.5-1.5
    # range, to see how each method handles a high-sweep wing through the
    # transonic regime -- note this pushes well past MAX_VALIDATED_SWEEP_DEG
    # (40 deg, see the module-level comment above transonic_wave_drag), which
    # is deliberate here (that's exactly the extrapolation regime being probed).
    span = 30.0
    root_chord = 26.25952
    tip_chord = 4.46961
    le_sweep_deg = 60.0
    midchord_sweep = le_sweep_to_midchord_sweep(le_sweep_deg, span, root_chord, tip_chord)
    mach_range = np.linspace(0.75, 1.2, 21)
    CL = 0.15
    t_c = 0.04
    technology_factor = 0.87
    altitude = 0.0  # sea level; only used by the Wislicenus mode

    def korn_sweep():
        """Korn/Lock (transonic_wave_drag) wave drag vs. Mach, at fixed 60 deg LE sweep/CL/t_c."""
        wave_drags = np.zeros_like(mach_range)
        for index, m in enumerate(mach_range):
            wave_drags[index] = transonic_wave_drag(m, CL, t_c, midchord_sweep, technology_factor)
        return wave_drags

    def wislicenus_sweep():
        """Wislicenus wave drag vs. Mach, at the same fixed 60 deg LE sweep/CL/t_c/altitude."""
        wave_drags = np.zeros_like(mach_range)
        for index, m in enumerate(mach_range):
            wave_drags[index] = wislicenus_wave_drag(m, CL, t_c, midchord_sweep, altitude, technology_factor)
        return wave_drags

    def grassmeyer_sweep():
        """Grassmeyer spanwise-strip wave drag vs. Mach, for the same 60 deg LE sweep trapezoidal planform."""
        wave_drags = np.zeros_like(mach_range)
        for index, m in enumerate(mach_range):
            wave_drags[index] = grassmeyer_strip_wave_drag_from_planform(
                m, CL, t_c, root_chord, tip_chord, span, le_sweep_deg, technology_factor, num_strips=10)
        return wave_drags

    def suave_sweep():
        """SUAVE-corrected (suave_corrected_wave_drag) wave drag vs. Mach.

        suave_corrected_wave_drag() does not take t_c/technology_factor (see
        its current docstring/signature) -- its subsonic-anchor term does use
        CL/effective_sweep, and its supersonic side is a whole-geometry
        OpenVSP result, so the 60 deg LE sweep used for the other three modes
        is passed through here (as midchord_sweep, same convention as the
        other three) for a side-by-side comparison, but only affects this
        curve through that one CL-dependent term, not through t_c or
        technology_factor.

        Every Mach point above M_CR triggers a real OpenVSP WaveDrag analysis
        run against the actual Mach1UAV_V2 vehicle file used elsewhere in this
        codebase (TopSpeedSim.py, sizingEstimation.py) -- so this mode is much
        slower than the other three (pure-Python) modes, and requires a working
        OpenVSP install/PYTHONPATH. Returns None if that's not available, so
        the caller can render a clear "unavailable" panel instead of crashing
        the whole --run-all plot.
        """
        vspfile = "/Users/gabrielkern/Documents/hypersonics/supersonicUAV/OpenVSP/OpenVSPConceptualDesign/Mach1UAV_V2.vsp3"
        wing_area = 3.201  # ft^2, matches TopSpeedSim.py's Mach1UAV_V2 config
        model_unit = 'in'  # matches that .vsp3 model's own native unit

        try:
            return np.array([
                suave_corrected_wave_drag(
                    {'mach_start': m, 'wing_area': wing_area, 'model_unit': model_unit,
                     'effective_sweep': midchord_sweep},
                    CL, vspfile)
                for m in mach_range
            ])
        except (ImportError, RuntimeError) as exc:
            print(f"[WARNING] Skipping SUAVE-corrected mode in --run-all: {exc}")
            return None

    if args.run_all:
        sweep_modes = [
            ('Korn/Lock (transonic_wave_drag)', korn_sweep()),
            ('Wislicenus (wislicenus_wave_drag)', wislicenus_sweep()),
            ('Grassmeyer Strip (from planform)', grassmeyer_sweep()),
        ]
        suave_drags = suave_sweep()

        fig, axes = plt.subplots(1, len(sweep_modes) + 1, figsize=(6 * (len(sweep_modes) + 1), 6))
        for ax, (title, wave_drags) in zip(axes, sweep_modes):
            ax.plot(mach_range, wave_drags)
            ax.set_xlabel('Mach')
            ax.set_ylabel('Drag Coefficient Value')
            ax.set_title(title)
            ax.grid(True, alpha=0.3)

        ax_suave = axes[-1]
        if suave_drags is not None:
            ax_suave.plot(mach_range, suave_drags)
            ax_suave.set_xlabel('Mach')
            ax_suave.set_ylabel('Drag Coefficient Value')
            ax_suave.grid(True, alpha=0.3)
        else:
            ax_suave.text(0.5, 0.5, 'Unavailable:\nOpenVSP not installed/configured\n(see console warning)',
                          ha='center', va='center', transform=ax_suave.transAxes,
                          fontsize=10, color='firebrick')
            ax_suave.set_xticks([])
            ax_suave.set_yticks([])
        ax_suave.set_title('SUAVE-Corrected (suave_corrected_wave_drag)')

        fig.suptitle(f'Wave Drag vs. Mach, all modes (LE sweep={le_sweep_deg} deg, CL={CL}, t/c={t_c})', fontsize=14)
        fig.tight_layout()
        plt.show()
    else:
        wave_drags = korn_sweep()

        plt.figure(figsize=(10, 6))
        plt.plot(mach_range, wave_drags)
        plt.xlabel('Mach')
        plt.ylabel('Drag Coefficient Value')
        plt.grid(True, alpha=0.3)
        plt.title(f'Drag versus Mach (LE sweep={le_sweep_deg} deg)')
        plt.show()

    # wave_drag_sensitivity()