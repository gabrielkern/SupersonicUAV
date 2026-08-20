import sys
import os
from contextlib import contextmanager
import numpy as np
from scipy.interpolate import interp1d
from scipy.stats import qmc

print(os.getcwd())

import create_base_Mach1UAV, variable_plane_analysis, variable_plane_parasitic, wave_drag


@contextmanager
def suppress_output():
    """Context manager to suppress stdout and stderr output."""
    # Save original file descriptors
    original_stdout_fd = sys.stdout.fileno()
    original_stderr_fd = sys.stderr.fileno()

    # Save original streams
    original_stdout = sys.stdout
    original_stderr = sys.stderr

    # Open null devices
    devnull = os.open(os.devnull, os.O_WRONLY)

    try:
        # Flush streams
        sys.stdout.flush()
        sys.stderr.flush()

        # Duplicate file descriptors
        saved_stdout_fd = os.dup(original_stdout_fd)
        saved_stderr_fd = os.dup(original_stderr_fd)

        # Redirect to devnull
        os.dup2(devnull, original_stdout_fd)
        os.dup2(devnull, original_stderr_fd)

        # Reassign streams
        sys.stdout = os.fdopen(saved_stdout_fd, 'w')
        sys.stderr = os.fdopen(saved_stderr_fd, 'w')

        yield

    finally:
        # Flush temporary streams
        sys.stdout.flush()
        sys.stderr.flush()

        # Restore original file descriptors BEFORE closing temporary streams
        os.dup2(saved_stdout_fd, original_stdout_fd)
        os.dup2(saved_stderr_fd, original_stderr_fd)

        # Now close temporary streams and saved fds
        sys.stdout.close()
        sys.stderr.close()
        # os.close(saved_stdout_fd)
        # os.close(saved_stderr_fd)

        # Restore original stream objects
        sys.stdout = original_stdout
        sys.stderr = original_stderr

        # Close devnull
        os.close(devnull)

def MAC(chords, spans, sweeps):
    """Assumes all trapezoidal sections defined by chords and spans.

    sweeps: trailing-edge sweep angle (degrees) for each section, same
    length as spans. Used to locate the leading edge in space so the
    quarter-MAC point can be placed relative to the wing root LE.

    Returns a dict:
      mac: mean aerodynamic chord length
      mac_span_location: spanwise station (from root) where local chord
        first drops below mac
      mac_section: index of the section containing that station
      quarter_mac_le_distance: distance from the wing root leading edge
        (the absolute LE of the wing) to the quarter-chord point of the MAC
    """
    if len(chords) - 1 != len(spans) or len(spans) != len(sweeps):
        return -1

    station_points = 10000

    total_span = sum(spans)

    span_stations = np.linspace(0,total_span,station_points)
    chord_stations = np.zeros_like(span_stations)
    le_stations = np.zeros_like(span_stations)

    span_segment = total_span / (station_points - 1)

    slopes = [(chords[i+1]-chords[i])/spans[i] for i in range(len(spans))]
    le_slopes = [np.tan(np.radians(sweeps[i])) - slopes[i] for i in range(len(spans))]

    mac_sum = 0
    area_sum = 0

    span_count = 0
    segment_start = 0.0
    le_segment_start = 0.0
    traversed_span = spans[0]
    for index, y in enumerate(span_stations):
        if y > traversed_span:
            le_segment_start += le_slopes[span_count] * spans[span_count]
            segment_start = traversed_span
            span_count += 1
            traversed_span += spans[span_count]
        chord_y = slopes[span_count]*(y - segment_start) + chords[span_count]
        mac_sum += np.square(chord_y) * total_span / (station_points)
        chord_stations[index] = chord_y
        le_stations[index] = le_slopes[span_count]*(y - segment_start) + le_segment_start
        if y == 0:
            continue
        else:
            chord_prev = slopes[span_count]*(span_stations[index-1] - segment_start) + chords[span_count]
            area_sum += ( (chord_y + chord_prev) / 2 ) * span_segment

    mac = mac_sum / area_sum

    # Spanwise location of the MAC: outermost station where local chord
    # is still >= mac (chord assumed monotonically non-increasing outboard).
    mac_station_index = 0
    for index, c in enumerate(chord_stations):
        if c >= mac:
            mac_station_index = index
        else:
            break

    mac_span_location = span_stations[mac_station_index]
    cumulative_span = np.cumsum(spans)
    mac_section = int(np.searchsorted(cumulative_span, mac_span_location))

    quarter_mac_le_distance = le_stations[mac_station_index] + 0.25 * mac

    return {
        'mac': mac,
        'mac_span_location': mac_span_location,
        'mac_section': mac_section,
        'quarter_mac_le_distance': quarter_mac_le_distance,
    }

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def calculate_total_weight(cargo_units: int) -> float:
    """Calculate total aircraft empty weight based on cargo capacity."""

    return 3.87 + (0.105 * cargo_units) + (0.0231 * (cargo_units ** 2)) + \
           (1.74E-03 * (cargo_units ** 3))

def create_lift_drag_mapper(aero_results: dict, parasitic_drag: float, wave_drag: float):
    """Create CL -> total CD interpolation function."""
    total_drag = [cd + parasitic_drag + wave_drag[index] for index,cd in enumerate(aero_results['CD'])]
    return interp1d(aero_results['CL'], total_drag,
                   kind='cubic', fill_value='extrapolate')

def print_compact_csv(result: dict):
    """Print single-line CSV output for quiet mode."""

    # 2 input parameters in fixed order
    params = [
        result['wing_area'], result['mach_start']
    ]

    # Print with consistent precision
    cl_vals = np.linspace(0,result['max_cl'],10)
    for cl in cl_vals:
        csv_row = params + [cl] + [result['aero'](cl)] + [cl/result['aero'](cl)]
        print(','.join(str(v) if isinstance(v, int) else f'{v:.6g}' for v in csv_row))


# ============================================================================
# MAIN EVALUATION FUNCTION
# ============================================================================

def evaluate_configuration(config: dict) -> dict:
    """
    Evaluate a single aircraft configuration.

    Args:
        config: Configuration dictionary

    Returns:
        results
    """
    try:
        # 1. Calculate derived parameters
        total_weight = calculate_total_weight(1)

        # Context manager for suppressing output
        context = suppress_output()

        with context:
            # 2. Create OpenVSP geometry
            create_base_Mach1UAV.main(config)
            
            # 3. Run aerodynamic analysis
            aero_results = variable_plane_analysis.main(config)
            max_cl = max(aero_results['CL'])

            # 4. Calculate parasitic drag
            parasitic_drag = variable_plane_parasitic.main(config)

        d_wave = wave_drag.transonic_wave_drag(config['mach_start'], aero_results['CL'], config['thickness'], config['effective_sweep'], config['technology_factor'])

        # 5. Build lift-drag mapper
        lift_drag_mapper = create_lift_drag_mapper(aero_results, parasitic_drag, d_wave)

        # 9. Build result dictionary
        return {
            **config,  # All input parameters
            'aero': lift_drag_mapper,
            'weight': total_weight,
            'max_cl': max_cl
        }

    except Exception as e:
        # On failure, return config with zero flight rewards but preserve gm score
        raise


# ============================================================================
# CLI ENTRY POINT
# ============================================================================

if __name__ == '__main__':
    
    # Test with single manual configuration
    print("="*60)
    print("CONFIGURATIONS")
    print("="*60)

    # Making plane in VSP
    # planform_sweep = [1,2,3,4,5,6,7,8,9,10]
    # mach_sweep = [0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0,1.1,1.2]
    planform_sweep = [10]
    mach_sweep = [0.1]
    alpha_start = -5
    alpha_end = 20
    alpha_points = 26

    # print(MAC([1.45145,0.80636,.16127],[.23103,.49438],[20,20]))

    for planform_area in planform_sweep:
        for mach in mach_sweep:

            # Geom def items
            AR_INBOARD = 0.20465
            AREA_INBOARD = 0.26081 * planform_area
            TAPER_INBOARD = 0.55555
            TE_SWEEP_INBOARD = 20

            AR_OUTBOARD = 1.02183
            AREA_OUTBOARD = 0.23919 * planform_area
            TAPER_OUTBOARD = 0.2
            TE_SWEEP_OUTBOARD = 20

            inboard_span = np.sqrt(AR_INBOARD*AREA_INBOARD)
            root_chord = (2*AREA_INBOARD)/((1+TAPER_INBOARD)*inboard_span)
            middle_chord = root_chord * TAPER_INBOARD
            tip_chord = middle_chord * TAPER_OUTBOARD
            outboard_span = np.sqrt(AR_OUTBOARD*AREA_OUTBOARD)
            body_length = 1.55 * root_chord
            x_wing = 0.22222 * body_length
            body_diameter = 0.1 * body_length

            config = {}
            config['ar_inboard'] = AR_INBOARD
            config['area_inboard'] = AREA_INBOARD
            config['taper_inboard'] = TAPER_INBOARD
            config['te_sweep_inboard'] = TE_SWEEP_INBOARD
            config['ar_outboard'] = AR_OUTBOARD
            config['area_outboard'] = AREA_OUTBOARD
            config['taper_outboard'] = TAPER_OUTBOARD
            config['te_sweep_outboard'] = TE_SWEEP_OUTBOARD
            config['inboard_span'] = inboard_span
            config['outboard_span'] = outboard_span
            config['root_chord'] = root_chord
            config['middle_chord'] = middle_chord
            config['tip_chord'] = tip_chord
            config['wing_area'] =  planform_area
            config['taper_ratio'] = tip_chord / root_chord
            config['wing_span'] = inboard_span + outboard_span
            config['body_length'] = body_length
            config['diameter'] = body_diameter
            mac_result = MAC([root_chord, middle_chord, tip_chord], [inboard_span, outboard_span],
                            [TE_SWEEP_INBOARD, TE_SWEEP_OUTBOARD])
            config['MAC'] = mac_result['mac']
            config['MAC_span_location'] = mac_result['mac_span_location']
            config['MAC_section'] = mac_result['mac_section']
            config['quarter_MAC'] = mac_result['quarter_mac_le_distance']
            config['mach_start'] = mach
            config['mach_end'] = mach
            config['mach_points'] = 1
            config['alpha_start'] = alpha_start
            config['alpha_end'] = alpha_end
            config['alpha_points'] = alpha_points
            config['x_rel'] = x_wing + mac_result['quarter_mac_le_distance']
            config['v_ref'] = mach * 1116
            config['x_wing'] = x_wing

            config['thickness'] = 0.04
            te_sweep = 20 # Degrees at trailing edge
            config['effective_sweep'] = np.atan(outboard_span / ((middle_chord/2) + (outboard_span*np.tan(te_sweep * np.pi / 180)) - (root_chord/2)))
            config['technology_factor'] = 0.87 # For Korn eq

            # Run full evaluation
            print("\n" + "-"*60)
            print("RUNNING FULL EVALUATION:")
            print("-"*60)

            result = evaluate_configuration(config)

            print_compact_csv(result)

