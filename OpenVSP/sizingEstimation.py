import sys
import os
import csv
from contextlib import contextmanager
from collections import defaultdict
import numpy as np
import matplotlib.pyplot as plt
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


def write_results_csv(all_results: list, filepath: str):
    """Write per-alpha aerodynamic coefficients for every evaluated
    configuration to a CSV file. One row per (planform area, mach, alpha)
    point; all lift/drag values are dimensionless coefficients."""
    with open(filepath, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([
            'mach', 'planform_area', 'angle_of_attack', 'lift',
            'induced_drag', 'parasitic_drag', 'wave_drag', 'total_drag',
            'lift_over_drag'
        ])
        for result in all_results:
            mach = result['mach_start']
            planform_area = result['wing_area']
            for i in range(len(result['alpha'])):
                writer.writerow([
                    mach, planform_area, result['alpha'][i], result['CL'][i],
                    result['CD_induced'][i], result['CD_parasitic'],
                    result['CD_wave'][i], result['CD_total'][i],
                    result['L_over_D'][i]
                ])


def plot_sweep_results(all_results: list, show: bool = True, save_path: str = None):
    """Windowed 2x2 subplot summarizing the planform/mach sweep:
      1. Lift vs drag (CL vs CD_total) - one line per (planform, mach) pair
      2. Drag vs mach at each combo's best-L/D angle of attack - one line
         per planform area
      3. Drag breakdown (induced/parasitic/wave/total) vs angle of attack
         for the first evaluated (planform, mach) case, as a representative
         example of how the components combine
      4. L/D vs mach at each combo's best-L/D angle of attack - one line
         per planform area
    """
    fig, ((ax_lift_drag, ax_drag_mach), (ax_breakdown, ax_ld_mach)) = plt.subplots(2, 2, figsize=(12, 9))

    # 1. Lift vs drag, one line per (planform, mach) pair
    for result in all_results:
        label = f"S={result['wing_area']:.3g} ft², M={result['mach_start']:.2g}"
        ax_lift_drag.plot(result['CD_total'], result['CL'], marker='o', markersize=2, label=label)
    ax_lift_drag.set_xlabel('Total drag coefficient, $C_D$')
    ax_lift_drag.set_ylabel('Lift coefficient, $C_L$')
    ax_lift_drag.set_title('Lift vs Drag')
    if len(all_results) <= 12:
        ax_lift_drag.legend(fontsize=7)

    # 2 & 4. Drag and L/D at the best-L/D angle of attack, vs mach,
    # one line per planform area
    by_planform = defaultdict(list)
    for result in all_results:
        by_planform[result['wing_area']].append(result)

    for planform_area, results in sorted(by_planform.items()):
        results_sorted = sorted(results, key=lambda r: r['mach_start'])
        machs = [r['mach_start'] for r in results_sorted]
        best_idx = [int(np.argmax(r['L_over_D'])) for r in results_sorted]
        drag_at_best = [r['CD_total'][i] for r, i in zip(results_sorted, best_idx)]
        ld_at_best = [r['L_over_D'][i] for r, i in zip(results_sorted, best_idx)]

        label = f"S={planform_area:.3g} ft²"
        ax_drag_mach.plot(machs, drag_at_best, marker='o', label=label)
        ax_ld_mach.plot(machs, ld_at_best, marker='o', label=label)

    ax_drag_mach.set_xlabel('Mach')
    ax_drag_mach.set_ylabel('$C_D$ at best L/D')
    ax_drag_mach.set_title('Drag vs Mach (at best L/D angle of attack)')
    ax_drag_mach.legend(fontsize=7)

    ax_ld_mach.set_xlabel('Mach')
    ax_ld_mach.set_ylabel('Max L/D')
    ax_ld_mach.set_title('L/D vs Mach (at best L/D angle of attack)')
    ax_ld_mach.legend(fontsize=7)

    # 3. Drag breakdown vs angle of attack, representative case
    rep = all_results[8]
    ax_breakdown.plot(rep['alpha'], rep['CD_induced'], label='Induced')
    ax_breakdown.plot(rep['alpha'], np.full_like(rep['alpha'], rep['CD_parasitic']), label='Parasitic')
    ax_breakdown.plot(rep['alpha'], rep['CD_wave'], label='Wave')
    ax_breakdown.plot(rep['alpha'], rep['CD_total'], label='Total', linewidth=2, color='black')
    ax_breakdown.set_xlabel('Angle of attack (deg)')
    ax_breakdown.set_ylabel('Drag coefficient, $C_D$')
    ax_breakdown.set_title(f"Drag Breakdown vs AoA (S={rep['wing_area']:.3g} ft², M={rep['mach_start']:.2g})")
    ax_breakdown.legend(fontsize=8)

    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150)
    if show:
        plt.show()
    return fig


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

        # 6. Per-alpha coefficient breakdown (all dimensionless), used for
        # the CSV export and diagnostic plots.
        cl_arr = np.array(aero_results['CL'])
        cd_induced = np.array(aero_results['CD'])
        cd_wave = np.array(d_wave)
        cd_total = cd_induced + parasitic_drag + cd_wave
        l_over_d = np.divide(cl_arr, cd_total, out=np.zeros_like(cl_arr), where=cd_total != 0)

        # 9. Build result dictionary
        return {
            **config,  # All input parameters
            'aero': lift_drag_mapper,
            'weight': total_weight,
            'max_cl': max_cl,
            'alpha': np.array(aero_results['alpha']),
            'CL': cl_arr,
            'CD_induced': cd_induced,
            'CD_parasitic': parasitic_drag,
            'CD_wave': cd_wave,
            'CD_total': cd_total,
            'L_over_D': l_over_d,
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
    planform_sweep = [1,2,3,4,5,6,7,8,9,10]
    mach_sweep = [0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,0.95,1.05,1.1,1.2]
    alpha_start = -5
    alpha_end = 20
    alpha_points = 26

    # Where the per-alpha sweep results are written (same directory as this script)
    CSV_OUTPUT_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'sizing_sweep_results.csv')

    # print(MAC([1.45145,0.80636,.16127],[.23103,.49438],[20,20]))

    all_results = []

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
            all_results.append(result)

    write_results_csv(all_results, CSV_OUTPUT_PATH)
    print(f"\nWrote {sum(len(r['alpha']) for r in all_results)} rows to {CSV_OUTPUT_PATH}")
    plot_sweep_results(all_results)

