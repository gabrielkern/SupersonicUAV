"""
Lap Simulator for Supersonic UAV
"""

import os
from functools import lru_cache
import argparse

import numpy as np
from typing import Dict
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from ambiance import Atmosphere
from scipy.interpolate import interp1d

import sizingEstimation

# Constants
GAMMA = 1.4
R = 1716 # ft*lbf/slug/R or s2/ft2/R
g = 32.174

class LowThrustException(Exception):
    pass

def get_atmosphere(altitude):
    """Get the density at a specific altitude, specified in ft."""
    altitude_meters = altitude * 0.3048
    density = Atmosphere(altitude_meters).density
    temperature = Atmosphere(altitude_meters).temperature
    density_imperial = density * 0.0685218 / 35.3147 # Convert kg/m3 to slugs/ft3
    temperature_imperial = temperature * 1.8
    return density_imperial, temperature_imperial

def thrust_model(constant_thrust): # Can implement more complex thrust schemes in the future
    mach_adjusted_thrust = lambda mach: constant_thrust + (constant_thrust * mach * 0.3703703704)
    altitude_multiplier = lambda altitude: 1.02 - 2.66E-05*altitude + 1.74E-10*altitude**2
    def thrust_interp(altitude, mach):
        thrust = altitude_multiplier(altitude) * mach_adjusted_thrust(mach)
        return thrust
    return thrust_interp

def sfc_model(constant_sfc):
    def sfc_interp(altitude, mach):
        return constant_sfc
    return sfc_interp

def update_fuel_mass(curr_fuel_mass, sfc, dt, thrust):
    """Calculates the fuel lost in lbm."""
    dt_hours = dt / 3600 # Convert to hours
    mass_used = sfc * dt_hours * thrust # Mass in lbm
    new_fuel_mass = curr_fuel_mass - mass_used if mass_used > 0 else curr_fuel_mass
    return new_fuel_mass

@lru_cache(maxsize=None)
def _load_lift_drag_csv(csv_path):
    """Load a lift/drag sweep CSV into regular grids over mach, planform_area, and angle_of_attack."""
    data = np.genfromtxt(csv_path, delimiter=',', names=True)

    mach_grid = np.unique(data['mach'])
    area_grid = np.unique(data['planform_area'])
    aoa_grid = np.unique(data['angle_of_attack'])

    lift_3d = np.empty((len(mach_grid), len(area_grid), len(aoa_grid)))
    drag_3d = np.empty_like(lift_3d)

    mach_idx = np.searchsorted(mach_grid, data['mach'])
    area_idx = np.searchsorted(area_grid, data['planform_area'])
    aoa_idx = np.searchsorted(aoa_grid, data['angle_of_attack'])

    lift_3d[mach_idx, area_idx, aoa_idx] = data['lift']
    drag_3d[mach_idx, area_idx, aoa_idx] = data['total_drag']

    return mach_grid, area_grid, aoa_grid, lift_3d, drag_3d

@lru_cache(maxsize=None)
def build_lift_drag_interp(csv_path, wing_area):
    """
    Build a CL/mach -> CD lookup from a lift/drag sweep CSV, area-matched to wing_area.

    Interpolates linearly across planform_area to synthesize a CL(mach, AoA) / CD(mach, AoA)
    table at the exact wing_area, builds one CL->CD interpolant per mach in the sweep, then
    linearly interpolates between the two bracketing mach interpolants at query time.
    """
    mach_grid, area_grid, aoa_grid, lift_3d, drag_3d = _load_lift_drag_csv(csv_path)

    if len(area_grid) < 2:
        lift_2d = lift_3d[:, 0, :]  # single area in sweep — squeeze out area axis
        drag_2d = drag_3d[:, 0, :]
    else:
        area_interp_lift = interp1d(area_grid, lift_3d, axis=1, bounds_error=False, fill_value="extrapolate")
        area_interp_drag = interp1d(area_grid, drag_3d, axis=1, bounds_error=False, fill_value="extrapolate")
        lift_2d = area_interp_lift(wing_area)  # shape: (len(mach_grid), len(aoa_grid))
        drag_2d = area_interp_drag(wing_area)

    per_mach_interps = []
    cl_min_per_mach = np.empty(len(lift_2d))
    cl_max_per_mach = np.empty_like(cl_min_per_mach)
    for k, (cl_row, cd_row) in enumerate(zip(lift_2d, drag_2d)):
        order = np.argsort(cl_row)
        cl_sorted = cl_row[order]
        per_mach_interps.append(
            interp1d(cl_sorted, cd_row[order], bounds_error=False, fill_value="extrapolate")
        )
        cl_min_per_mach[k] = cl_sorted[0]
        cl_max_per_mach[k] = cl_sorted[-1]

    def drag_lookup(mach, cl):
        # mach/cl may arrive as length-1 arrays (ambiance's Atmosphere always
        # returns arrays, even for scalar altitude input), so normalize to scalars.
        mach = float(np.ravel(mach)[0])
        cl = float(np.ravel(cl)[0])

        # Determine mach out-of-range flags and bracketing station indices
        mach_low  = mach < mach_grid[0]
        mach_high = mach > mach_grid[-1]

        if mach_low:
            station_indices = [0, 1]
        elif mach_high:
            station_indices = [-2, -1]
        else:
            i = int(np.searchsorted(mach_grid, mach) - 1)
            i = max(0, min(i, len(mach_grid) - 2))  # guard against fp edge cases at boundary
            station_indices = [i, i + 1]

        # Detect CL out-of-range against the relevant mach stations
        cl_low  = cl < cl_min_per_mach[station_indices].min()
        cl_high = cl > cl_max_per_mach[station_indices].max()

        if mach_low or mach_high or cl_low or cl_high:
            print(f"[WARNING] Extrapolation required with mach {mach} and CL {cl}")

        # Compute CD
        if mach_low:
            cd0   = float(per_mach_interps[0](cl))
            cd1   = float(per_mach_interps[1](cl))
            slope = (cd1 - cd0) / (mach_grid[1] - mach_grid[0])
            return cd0 + slope * (mach - mach_grid[0])

        if mach_high:
            cd_n   = float(per_mach_interps[-1](cl))
            cd_nm1 = float(per_mach_interps[-2](cl))
            slope  = (cd_n - cd_nm1) / (mach_grid[-1] - mach_grid[-2])
            return cd_n + slope * (mach - mach_grid[-1])

        # Interior: linear interpolation between bracketing stations
        t   = (mach - mach_grid[i]) / (mach_grid[i + 1] - mach_grid[i])
        cd0 = float(per_mach_interps[i](cl))
        cd1 = float(per_mach_interps[i + 1](cl))
        return float((1 - t) * cd0 + t * cd1)

    return drag_lookup

def find_thrust_limited_speed(*, thrust_interp, lift_drag_interp, altitude, wing_area, weight=None, top_speed=1500):
    """Find max speed where thrust = drag using thrust curve. Returns tuple of speed (ft/s) and mach"""

    if weight == None:
        if isinstance(lift_drag_interp, str): # This means its a path
            drag_lookup = build_lift_drag_interp(lift_drag_interp, wing_area)
        else:
            drag_lookup = lambda mach, cl: lift_drag_interp(cl)

        for speed in range(1, top_speed):  # range in ft/s
            weight = weight_from_wing_area(wing_area)
            rho,temp = get_atmosphere(altitude)
            q = 0.5 * rho * speed**2
            lift_coefficient = weight / q / wing_area
            sos = np.sqrt(GAMMA * R * temp) # speed of sound
            mach = speed / sos
            drag_coefficient = drag_lookup(mach, lift_coefficient)
            thrust = thrust_interp(altitude, mach) # Call the callable object
            drag = drag_coefficient * q * wing_area
            if thrust <= drag:
                return max(speed - 1, 1), max((speed-1)/sos,1/sos)  # ensure minimum speed of 1 ft/s
        return -1, -1  # ft/s, fallback
    else:
        if isinstance(lift_drag_interp, str):
            drag_lookup = build_lift_drag_interp(lift_drag_interp, wing_area)
        else:
            drag_lookup = lambda mach, cl: lift_drag_interp(cl)

        for speed in range(1, top_speed):  # range in ft/s
            rho,temp = get_atmosphere(altitude)
            q = 0.5 * rho * speed**2
            lift_coefficient = weight / q / wing_area
            sos = np.sqrt(GAMMA * R * temp) # speed of sound
            mach = speed / sos
            drag_coefficient = drag_lookup(mach, lift_coefficient)
            thrust = thrust_interp(altitude, mach) # Call the callable object
            drag = drag_coefficient * q * wing_area
            if thrust <= drag:
                return max(speed - 1, 1), max((speed-1)/sos,1/sos)  # ensure minimum speed of 1 ft/s
        return -1, -1  # ft/s, fallback
    
def weight_from_wing_area(wing_area):
    """Return weight estimation based on wing area. Empirical model."""
    return 6.28 * wing_area

def engine_weight_from_thrust(constant_thrust):
    """Return an estimation of weight from the constant thrust of the turbojet. Lbf thrust in, lbm weight out."""
    return 0.101 * constant_thrust - 0.878

def sfc_from_thrust(constant_thrust):
    """Return an estimation of specific fuel consuption in lbm/lbf/hr from constant thrust in lbf."""
    return -0.00159*constant_thrust + 1.65

def climb(state: dict, config: dict):
    """
    Function that calculates the climb of the plane.
    Occurs only once per simulation, at takeoff.
    """
    EW = config['structural_weight'] + config['engine_weight']
    S = config['wing_area']
    dt = config['dt']
    cruise_alt = config['cruise_altitude']
    theta = np.radians(config['theta'])
    CL_stall = config['CL_stall']
    drag_lookup = config['drag_lookup']
    thrust_interp = config['thrust_interp']
    sfc_interp = config['sfc_interp']

    print(EW)

    i = state['i']

    while (state['position'][i, 1] <= cruise_alt) and (state['fuel'][i] >= config['landing_fuel_frac']*config['fuel_capacity']):

        # Define altitude-dependent variables
        altitude = state['position'][i,1]
        rho, T = get_atmosphere(altitude)
        sos = np.sqrt(GAMMA * R * T)

        # Fuel dependent weight and mass
        W = EW + state['fuel'][i]
        m = W / g

        v = state['velocity'][i] # if state['velocity'][i] <= constants['velocity_max'] else constants['velocity_max']
        mach = v / sos

        # if v < constants['stall_speed']:
        #     raise LowThrustException

        q = 0.5 * rho * v**2

        CL_Climb = (W * np.cos(theta)) / (q * S)
        if CL_Climb > CL_stall:
            CL_Climb = CL_stall

        CD_Climb = drag_lookup(np.atleast_1d(mach),np.atleast_1d(CL_Climb))

        drag = CD_Climb * q * S

        # Get thrust from interpolation model
        thrust = thrust_interp(altitude, mach)

        # Get fuel consumption from thrust
        sfc = sfc_interp(altitude, mach)

        new_acceleration = ((thrust) - (drag) - (W * np.sin(theta))) / m
        new_velocity = v + new_acceleration * dt
        new_position = np.add(state['position'][i], np.transpose([v * dt * np.cos(theta), v * dt * np.sin(theta)]))

        state['velocity'] = np.append(state['velocity'], new_velocity)
        state['position'] = np.vstack((state['position'], new_position))
        state['acceleration'] = np.append(state['acceleration'], new_acceleration)
        state['fuel'] = np.append(state['fuel'], update_fuel_mass(state['fuel'][i], sfc, dt, thrust))
        state['time'] = np.append(state['time'], state['time'][i] + dt)
        state['thrust'] = np.append(state['thrust'], thrust)
        state['CL'] = np.append(state['CL'], CL_Climb)
        state['CD'] = np.append(state['CD'], CD_Climb)
        state['lift'] = np.append(state['lift'], W * np.cos(theta))
        state['drag'] = np.append(state['drag'], drag)
        state['F_long'] = np.append(state['F_long'], new_acceleration * m)
        state['temp'] = np.append(state['temp'], T)
        state['rho'] = np.append(state['rho'], rho)
        state['mach'] = np.append(state['mach'], mach)
        i += 1
        state['i'] = i

        # print("-"*60)
        # print(f"Thrust: {thrust}")
        # print(f"Drag: {drag}")
        # print(f"CD: {CD_Climb}")
        # print(f"Weight: {W}")
        # print(f"Theta: {theta}")
        # print(f"Mass: {m}")
        # print(f"Acceleration: {new_acceleration}")
        # print(f"Velocity: {new_velocity}")
        # input()


def straight(state: dict, config: dict):
    """
    The function to calculate the straightaways, uses the assumption of constant altitude.
    """
    EW = config['structural_weight'] + config['engine_weight']
    S = config['wing_area']
    dt = config['dt']
    CL_stall = config['CL_stall']
    drag_lookup = config['drag_lookup']
    thrust_interp = config['thrust_interp']
    sfc_interp = config['sfc_interp']

    i = state['i']

    while state['fuel'][i] >= config['landing_fuel_frac']*config['fuel_capacity']:
        
        # Define altitude-dependent variables
        altitude = state['position'][i,1]
        rho, T = get_atmosphere(altitude)
        sos = np.sqrt(GAMMA * R * T)

        # Fuel dependent weight and mass
        W = EW + state['fuel'][i]
        m = W / g
        
        v = state['velocity'][i] # if state['velocity'][i] <= constants['velocity_max'] else constants['velocity_max']
        mach = v / sos

        # if v < constants['stall_speed']:
        #     raise LowThrustException

        q = 0.5 * rho * v**2

        CL_Straight = W / q / S
        if CL_Straight > CL_stall:
            CL_Straight = CL_stall
        CD_Straight = drag_lookup(mach, CL_Straight)

        drag = CD_Straight * q * S

        # Get thrust from interpolation model
        thrust = thrust_interp(altitude, mach)

        # Get fuel consumption from thrust
        sfc = sfc_interp(altitude, mach)

        new_acceleration = ((thrust) - (drag)) / m
        new_velocity = v + new_acceleration * dt
        new_position = np.add(state['position'][i], [v * dt, 0.0])

        state['velocity'] = np.append(state['velocity'], new_velocity)
        state['position'] = np.vstack((state['position'], new_position))
        state['acceleration'] = np.append(state['acceleration'], new_acceleration)
        state['fuel'] = np.append(state['fuel'], update_fuel_mass(state['fuel'][i], sfc, dt, thrust))
        state['time'] = np.append(state['time'], state['time'][i] + dt)
        state['thrust'] = np.append(state['thrust'], thrust)
        state['CL'] = np.append(state['CL'], CL_Straight)
        state['CD'] = np.append(state['CD'], CD_Straight)
        state['lift'] = np.append(state['lift'], W)
        state['drag'] = np.append(state['drag'], drag)
        state['F_long'] = np.append(state['F_long'], new_acceleration * m)
        state['temp'] = np.append(state['temp'], T)
        state['rho'] = np.append(state['rho'], rho)
        state['mach'] = np.append(state['mach'], mach)
        i += 1
        state['i'] = i

def execute_lap_sim(constants: Dict):
    """
    Executes the lap simulator.

    For M3 (mission=3), uses empty weight only (no cargo).

    Args:
        constants: Dictionary with aircraft parameters and lift_drag_mapper
        mission: Mission number (3 for banner towing)
        debug: Enable debug output

    Returns:
        Number of laps completed
    """
    try:
        # Set max velocity
        constants['velocity_max'] = find_thrust_limited_speed(
            diameter=constants['propeller_diameter'],
            pitch=constants['propeller_pitch'],
            kv=constants['motor_kv'],
            battery_cell_count=constants['battery_cells'],
            rho=constants['rho'],
            S=constants['S'],
            Cd_estimate=constants['CD_p']
        )

        # Set up initial state
        initial_velocity = constants['stall_speed'] * 1.05
        state = {
            'velocity': np.array([initial_velocity]),
            'position': np.array([[0.0, 0.0]]),
            'acceleration': np.array([0.0]),
            'battery_charge': np.array([constants['battery_capacity']]),
            'time': np.array([0.0]),
            'turn_angle': np.array([0.0]),
            'thrust': np.array([0.0]),
            'CL': np.array([0.0]),
            'CD': np.array([0.0]),
            'lift': np.array([0.0]),
            'drag': np.array([0.0]),
            'F_long': np.array([0.0]),
            'F_lat': np.array([0.0]),
            'i': 0
        }

        # Climb to altitude
        climb(state, constants)

        lap_counter = 0

        # Lap loop: stop if battery below 30% or time exceeds 5 minutes
        while (state['battery_charge'][-1] > constants['battery_capacity'] * 0.3) and (state['time'][-1] < 300):
            straight(state, constants, 500, mission, debug=debug)
            turn(state, constants, 180, mission, debug=debug)
            straight(state, constants, 500, mission, debug=debug)
            turn(state, constants, 360, mission, debug=debug)
            straight(state, constants, 500, mission, debug=debug)
            turn(state, constants, 180, mission, debug=debug)
            straight(state, constants, 500, mission, debug=debug)
            lap_counter += 1

        print(f"Time: {state['time'][-1]}")
        return lap_counter

    except LowThrustException:
        raise
    except Exception as e:
        raise
    
def generate_max_speed_plot(altitude_range, wing_area_range, thrust, lift_drag_interp, weights = None):
    """
    Generate a 2D contour plot of thrust-limited max speed vs. altitude and wing area.

    altitude_range / wing_area_range: (start, stop, num) tuples passed to np.linspace.
    thrust: constant thrust in lbf, held fixed across the sweep.
    lift_drag_interp: CSV filepath (str) or a CL->CD callable (e.g. interp1d).
    """
    if weights == None:
        altitudes = np.linspace(*altitude_range)
        wing_areas = np.linspace(*wing_area_range)
        thrust_interp = thrust_model(thrust) # call with (altitude,mach)

        max_speeds = np.empty((len(altitudes), len(wing_areas)))
        machs_speeds = np.empty((len(altitudes), len(wing_areas)))
        for j, wing_area in enumerate(wing_areas):
            for i, altitude in enumerate(altitudes):
                max_speeds[i, j], machs_speeds[i, j] = find_thrust_limited_speed(thrust_interp=thrust_interp, lift_drag_interp=lift_drag_interp, altitude=altitude, wing_area=wing_area)

        wing_area_grid, altitude_grid = np.meshgrid(wing_areas, altitudes)

        fig, ax = plt.subplots()
        contour = ax.contourf(wing_area_grid, altitude_grid, machs_speeds, levels=20, cmap="viridis")
        fig.colorbar(contour, ax=ax, label="Max Speed (mach)")
        ax.set_xlabel("Wing Area (ft^2)")
        ax.set_ylabel("Altitude (ft)")
        ax.set_title(f"Thrust-Limited Max Speed vs. Wing Area and Altitude (Thrust = {thrust} lbf)")
        plt.show()

        return max_speeds, machs_speeds
    else:
        altitude = altitude_range
        wing_areas = np.linspace(*wing_area_range)
        weights = np.linspace(*weights)
        thrust_interp = thrust_model(thrust)

        max_speeds = np.empty((len(weights), len(wing_areas)))
        machs_speeds = np.empty((len(weights), len(wing_areas)))
        for j, wing_area in enumerate(wing_areas):
            for i, weight in enumerate(weights):
                max_speeds[i, j], machs_speeds[i, j] = find_thrust_limited_speed(thrust_interp=thrust_interp, lift_drag_interp=lift_drag_interp, altitude=altitude, wing_area=wing_area, weight=weight)

        wing_area_grid, weight_grid = np.meshgrid(wing_areas, weights)

        fig, ax = plt.subplots()
        contour = ax.contourf(wing_area_grid, weight_grid, machs_speeds, levels=20, cmap="viridis")
        fig.colorbar(contour, ax=ax, label="Max Speed (mach)")
        ax.set_xlabel("Wing Area (ft^2)")
        ax.set_ylabel("Weight (lbs)")
        ax.set_title(f"Thrust-Limited Max Speed vs. Wing Area and Weight (Thrust = {thrust} lbf)")
        plt.show()

        return max_speeds, machs_speeds
    
def plot_results(state: dict, config: dict):
    """Takes the results from simulation and plots values for various states over the flight time."""
    
    # Comprehensive plotting of all flight parameters
    fig = plt.figure(figsize=[15, 8])
    gs = GridSpec(3, 3, figure=fig, hspace=0.4, wspace=0.3)
    fig.suptitle(f"Top Speed Lap Simulation Results", fontsize=18)

    # Row 1: Velocities, Battery, Position
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.plot(state['time'], state['mach'], 'b-', linewidth=2, label='Absolute Velocity')
    ax1.set_xlabel("Time (s)"); ax1.set_ylabel("Mach")
    ax1.set_title("Velocity vs Time"); ax1.grid(True); ax1.legend()

    ax2 = fig.add_subplot(gs[0, 1])
    ax2.plot(state['time'], state['fuel'], 'g-', linewidth=2, label="Fuel remaining")
    ax2.plot(state['time'], (np.ones_like(state['time']) * config['landing_fuel_frac'] * config['fuel_capacity']), 'r.', linewidth=2, label="Landing fuel minimum")
    ax2.set_xlabel("Time (s)"); ax2.set_ylabel("Fuel Mass (lbm)")
    ax2.set_title("Fuel Mass vs Time"); ax2.grid(True); ax2.legend()

    ax3 = fig.add_subplot(gs[0, 2])
    ax3.plot(state['time'], state['position'][:,0], 'b-', linewidth=2, label='Traveled Distance')
    ax3.plot(state['time'], state['position'][:,1], 'r-', linewidth=2, label='Altitude')
    ax3.set_xlabel("Time (s)"); ax3.set_ylabel("Position (ft)")
    ax3.set_title("Position vs Time"); ax3.grid(True); ax3.legend()

    # Row 2: Accelerations, Thrust
    ax4 = fig.add_subplot(gs[1, 0])
    ax4.plot(state['time'], state['acceleration'], 'b-', linewidth=2)
    ax4.set_xlabel("Time (s)"); ax4.set_ylabel("Acceleration (ft/s²)")
    ax4.set_title("Acceleration vs Time"); ax4.grid(True); ax4.legend()

    ax5 = fig.add_subplot(gs[1, 1])
    ax5.plot(state['time'], state['thrust'], 'orange', linewidth=2)
    ax5.set_xlabel("Time (s)"); ax5.set_ylabel("Thrust (lbs)")
    ax5.set_title("Thrust vs Time"); ax5.grid(True)

    # Row 3: Aerodynamic coefficients with L/D, Combined Forces, Longitudinal/Lateral Forces
    ax6 = fig.add_subplot(gs[1, 2])
    ax6.plot(state['CD'], state['CL'], 'c-', linewidth=2, label='Drag Polar')
    ax6.set_xlabel("Drag Coefficient"); ax6.set_ylabel("Lift Coefficient")
    ax6.set_title("CL vs CD"); ax6.grid(True); ax6.legend()

    # Calculate L/D ratio, avoiding division by zero
    ax7 = fig.add_subplot(gs[2, 0])
    ld_ratio = np.divide(state['CL'], state['CD'], out=np.zeros_like(state['CL']), where=state['CD']!=0)
    ax7.plot(state['time'], ld_ratio, 'purple', linewidth=2, label='L/D')
    ax7.set_xlabel("Time (s)"); ax7.set_ylabel("L/D Ratio")
    ax7.set_title("L/D vs Time"); ax7.grid(True); ax7.legend()

    ax8 = fig.add_subplot(gs[2, 1])
    ax8.plot(state['time'], state['lift'], 'g-', linewidth=2, label='Lift')
    ax8.plot(state['time'], state['drag'], 'brown', linewidth=2, label='Drag')
    ax8.set_xlabel("Time (s)"); ax8.set_ylabel("Force (lbs)")
    ax8.set_title("Lift and Drag vs Time"); ax8.grid(True); ax8.legend()

    ax9 = fig.add_subplot(gs[2, 2])
    ax9.plot(state['time'], state['F_long'], 'k-', linewidth=2)
    ax9.set_xlabel("Time (s)"); ax9.set_ylabel("Force (lbs)")
    ax9.set_title("Longitudinal Force vs Time"); ax9.grid(True); ax9.legend()

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Mach1 UAV top speed sim")
    parser.add_argument('--max-speed-plot', action='store_true',
                         help='Restrict analysis to a simple top speed check as function of wing area, weight, and altitude.')
    args = parser.parse_args()

    if args.max_speed_plot:
        altitudes = (0,10000,11)
        altitude = 0
        wing_areas = (1,10,10)
        weights = (1, 51, 6)
        thrust = 250  # lbf, user-specified constant thrust
        lift_drag_csv = os.path.join(os.path.dirname(__file__), "Mach1Sizing", "sizing_sweep_results.csv")
        generate_max_speed_plot(altitudes, wing_areas, thrust, lift_drag_csv) # For altitude vs wing area with weight tied to wing area
        # generate_max_speed_plot(altitude, wing_areas, thrust, lift_drag_csv, weights) # For wing area vs weight at set altitude
    else:
        print("Running the full top speed simulator.")

        CONFIG_NAME = "Mach1UAV_V2"

        print(f"Vehicle identifier: {CONFIG_NAME}")

        altitude_range = (0,10000,2) # ft
        mach_range = (0.1,0.9,2)
        alpha_range = (-5,20,13)
        constant_thrust = 95 # lbs
        climb_angle = 25 # deg
        cruise_altitude = 1000
        wing_area = 3.201 # ft^2
        max_cl = 0.70 # based on airfoil
        wing_thickness = 0.04
        root_chord = 2.1882933333 # feet
        tip_chord = 0.3724675 # feet
        b_ref = 2.5 # feet
        c_ref = 1.4949791667 # feet
        cg_distance_x = 3.33333 # inches
        le_sweep = 60 # degrees
        technology_factor = 0.87 # For 6 series
        fuel_frac_empty = 1.333 # Ratio of structural weight to fuel weight
        landing_fuel_frac = 0.25 # % fuel where landing is required and simulation cannot continue

        model_unit = 'in' # unit of the model, 'in', 'ft', or 'm' rn

        vspfile = "/Users/gabrielkern/Documents/hypersonics/supersonicUAV/OpenVSP/OpenVSPConceptualDesign/Mach1UAV_V2.vsp3"

        # print(f"Input: {wing_area}")
        # print(f"Weight: {weight_from_wing_area(wing_area)}")
        # print(f"Fuel Weight: {weight_from_wing_area(wing_area) * fuel_frac_empty}")

        config = {
            'wing_area': wing_area,
            'thickness': wing_thickness,
            'wing_span': b_ref,
            'MAC': c_ref,
            'x_rel': cg_distance_x,
            'alpha_start': alpha_range[0],
            'alpha_end': alpha_range[1],
            'alpha_points': alpha_range[2],
            'effective_sweep': sizingEstimation.le_sweep_to_quarter_chord_sweep(le_sweep, b_ref, root_chord, tip_chord),
            'technology_factor': technology_factor,
            'structural_weight': weight_from_wing_area(wing_area),
            'g': g,
            'dt': 0.01,
            'fuel_capacity': weight_from_wing_area(wing_area) * fuel_frac_empty, # In lbs
            'constant_thrust': constant_thrust,
            'engine_weight': engine_weight_from_thrust(constant_thrust),
            'cruise_altitude': cruise_altitude,
            'theta': climb_angle,
            'CL_stall': max_cl,
            'landing_fuel_frac': landing_fuel_frac,
            'constant_sfc': sfc_from_thrust(constant_thrust),
            'model_unit': model_unit
        }

        lift_drag_csv = os.path.join(os.path.dirname(vspfile), f"{CONFIG_NAME}.csv")

        rerun_flag = ""
        while not rerun_flag:
            if os.path.isfile(lift_drag_csv):
                rerun_flag = input(f"Existing CSV found for vehicle with the name {CONFIG_NAME}.\nPlease type Y to use this or N to re-generate the csv. ")
            else:
                print("File not found. Generating now.")
                break
            if rerun_flag.strip().upper() == "Y" or rerun_flag.strip().upper() == "N":
                break
            else:
                rerun_flag = ""
        
        if not os.path.isfile(lift_drag_csv) or rerun_flag == "N":
            sizingEstimation.generate_csv_from_file(vspfile=vspfile, csvoutput=lift_drag_csv, altitude_range=altitude_range, mach_range=mach_range, config=config)
        
        config['drag_lookup'] = build_lift_drag_interp(lift_drag_csv, wing_area)
        config['thrust_interp'] = thrust_model(constant_thrust)
        config['sfc_interp'] = sfc_model(config['constant_sfc'])

        # Build the starting state for the sim. Start at base velocity and base mach
        rho_start, t_start = get_atmosphere(altitude_range[0]) # Comes back in rankine
        sos_start = np.sqrt(GAMMA * R * t_start)
        initial_velocity = mach_range[0] * sos_start
        state = {
            'velocity': np.array([initial_velocity]),
            'position': np.array([[0.0, 0.0]]),
            'acceleration': np.array([0.0]),
            'fuel': np.array([config['fuel_capacity']]),
            'time': np.array([0.0]),
            'thrust': np.array([0.0]),
            'CL': np.array([0.0]),
            'CD': np.array([0.0]),
            'lift': np.array([0.0]),
            'drag': np.array([0.0]),
            'F_long': np.array([0.0]),
            'temp' : np.array([t_start]),
            'rho' : np.array([rho_start]),
            'mach': np.array([mach_range[0]]),
            'i': 0
        }

        climb(state=state,config=config)

        straight(state=state,config=config)

        plot_results(state=state,config=config)

        print(f"Final Time: {state['time'][-1]}")