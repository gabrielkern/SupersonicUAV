"""
Lap Simulator for Banner Optimizer

Point-mass flight dynamics simulator for M3 banner towing mission.
"""

import os
from functools import lru_cache
import argparse

import numpy as np
from typing import Dict
import matplotlib.pyplot as plt
from ambiance import Atmosphere
from scipy.interpolate import interp1d

import sizingEstimation

# Constants
GAMMA = 1.4
R = 1716 # ft*lbf/slug/R or s2/ft2/R

# Static parameters
LAP_ALTITUDE = 200  # ft
CLIMB_ANGLE = 20  # deg
GRAVITY = 32.174  # ft/s^2
RHO = 0.0023769  # slugs/ft^3 at sea level
DT = 0.05  # seconds
PROPULSIVE_EFFICIENCY = 0.75

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

    area_interp_lift = interp1d(area_grid, lift_3d, axis=1, bounds_error=False, fill_value="extrapolate")
    area_interp_drag = interp1d(area_grid, drag_3d, axis=1, bounds_error=False, fill_value="extrapolate")
    lift_2d = area_interp_lift(wing_area)  # shape: (len(mach_grid), len(aoa_grid))
    drag_2d = area_interp_drag(wing_area)

    per_mach_interps = []
    for cl_row, cd_row in zip(lift_2d, drag_2d):
        order = np.argsort(cl_row)
        per_mach_interps.append(
            interp1d(cl_row[order], cd_row[order], bounds_error=False, fill_value="extrapolate")
        )

    def drag_lookup(mach, cl):
        # mach/cl may arrive as length-1 arrays (ambiance's Atmosphere always
        # returns arrays, even for scalar altitude input), so normalize to scalars.
        mach = float(np.ravel(mach)[0])
        cl = float(np.ravel(cl)[0])

        # Check bounds for violation of mach
        if mach <= mach_grid[0]:
            return float(per_mach_interps[0](cl))
        if mach >= mach_grid[-1]:
            return float(per_mach_interps[-1](cl))
        
        # Get python index of location of machs in grid
        i = int(np.searchsorted(mach_grid, mach) - 1)
        m0, m1 = mach_grid[i], mach_grid[i + 1]

        # Linear interpolation
        t = (mach - m0) / (m1 - m0)
        cd0 = per_mach_interps[i](cl)
        cd1 = per_mach_interps[i + 1](cl)
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
    lift_drag_mapper = config['drag_lookup']

    i = state['i']

    ## TODO: Get the loop initialized and the fuel burn function written

    while state['position'][i, 1] <= cruise_alt and state['time'][-1] < 300:
        v = state['velocity'][i] if state['velocity'][i] <= constants['velocity_max'] else constants['velocity_max']

        if v < constants['stall_speed']:
            raise LowThrustException

        q = 0.5 * rho * v**2

        CL_Climb = (W * np.cos(theta)) / (q * S)
        if CL_Climb > CL_stall:
            CL_Climb = CL_stall
        CD_i_p = lift_drag_mapper(CL_Climb)

        drag = CD_i_p * q * S
        CD_Climb = drag / (q * S)

        thrust, current = apc_data.get_propeller_performance(
            diameter=diameter,
            pitch=pitch,
            motor_kv=kv,
            battery_cell_count=battery_cell_count,
            airspeed_mph=v / 1.46667
        )

        new_acceleration = ((thrust) - (drag) - (W * np.sin(theta))) / m
        new_velocity = v + new_acceleration * dt
        new_position = np.add(state['position'][i], [v * dt * np.cos(theta), v * dt * np.sin(theta)])

        state['velocity'] = np.append(state['velocity'], new_velocity)
        state['position'] = np.vstack((state['position'], new_position))
        state['acceleration'] = np.append(state['acceleration'], new_acceleration)
        state['battery_charge'] = np.append(state['battery_charge'], update_charge(state['battery_charge'][i], current, dt))
        state['time'] = np.append(state['time'], state['time'][i] + dt)
        state['turn_angle'] = np.append(state['turn_angle'], 0)
        state['thrust'] = np.append(state['thrust'], thrust)
        state['Cl'] = np.append(state['Cl'], CL_Climb)
        state['Cd'] = np.append(state['Cd'], CD_Climb)
        state['lift'] = np.append(state['lift'], W * np.cos(theta))
        state['drag'] = np.append(state['drag'], drag)
        state['F_long'] = np.append(state['F_long'], new_acceleration * m)
        state['F_lat'] = np.append(state['F_lat'], 0)
        i += 1
        state['i'] = i

    print(state['battery_charge'][-1])


def straight(state: dict, constants: dict, distance_needed: float, mission: int, debug: bool = False):
    """
    The function to calculate the straightaways, uses the assumption of constant altitude.
    """
    W = constants['W']
    m = constants['m']
    S = constants['S']
    rho = constants['rho']
    dt = constants['dt']
    CL_stall = constants['CL_stall']
    diameter = constants['propeller_diameter']
    pitch = constants['propeller_pitch']
    kv = constants['motor_kv']
    battery_cell_count = constants['battery_cells']
    lift_drag_mapper = constants['lift_drag_mapper']

    i = state['i']
    distance_traveled = 0

    while (distance_needed >= distance_traveled) and state['time'][-1] < 300:
        v = state['velocity'][i] if state['velocity'][i] <= constants['velocity_max'] else constants['velocity_max']

        if v < constants['stall_speed']:
            raise LowThrustException

        q = 0.5 * rho * v**2

        CL_Straight = W / q / S
        if CL_Straight > CL_stall:
            CL_Straight = CL_stall
        CD_i_p = lift_drag_mapper(CL_Straight)

        D_i_p = CD_i_p * q * S
        D_b = get_banner_drag(length=constants['banner_length'], velocity=v) if mission == 3 else 0
        drag = D_i_p + D_b
        CD_Straight = drag / (q * S)

        thrust, current = apc_data.get_propeller_performance(
            diameter=diameter,
            pitch=pitch,
            motor_kv=kv,
            battery_cell_count=battery_cell_count,
            airspeed_mph=v / 1.46667
        )

        new_acceleration = ((thrust) - (drag)) / m
        new_velocity = v + new_acceleration * dt
        new_position = np.add(state['position'][i], [v * dt, 0.0])

        state['velocity'] = np.append(state['velocity'], new_velocity)
        state['position'] = np.vstack((state['position'], new_position))
        state['acceleration'] = np.append(state['acceleration'], new_acceleration)
        state['battery_charge'] = np.append(state['battery_charge'], update_charge(state['battery_charge'][i], current, dt))
        state['time'] = np.append(state['time'], state['time'][i] + dt)
        state['turn_angle'] = np.append(state['turn_angle'], 0)
        state['thrust'] = np.append(state['thrust'], thrust)
        state['Cl'] = np.append(state['Cl'], CL_Straight)
        state['Cd'] = np.append(state['Cd'], CD_Straight)
        state['lift'] = np.append(state['lift'], W)
        state['drag'] = np.append(state['drag'], drag)
        state['F_long'] = np.append(state['F_long'], new_acceleration * m)
        state['F_lat'] = np.append(state['F_lat'], 0)
        distance_traveled = distance_traveled + (v * dt)
        i += 1
        state['i'] = i

    print(state['battery_charge'][-1])


def turn(state: dict, constants: dict, turn_needed: float, mission: int, debug: bool = False):
    """
    Calculate through a turn with constant altitude.
    """
    W = constants['W']
    m = constants['m']
    S = constants['S']
    rho = constants['rho']
    dt = constants['dt']
    CL_Turn = constants['CL_turn']
    diameter = constants['propeller_diameter']
    pitch = constants['propeller_pitch']
    kv = constants['motor_kv']
    battery_cell_count = constants['battery_cells']
    lift_drag_mapper = constants['lift_drag_mapper']

    i = state['i']
    turn_traveled = 0

    while (turn_needed >= turn_traveled) and state['time'][-1] < 300:
        v = state['velocity'][i] if state['velocity'][i] <= constants['velocity_max'] else constants['velocity_max']

        if v < constants['stall_speed']:
            raise LowThrustException

        q = 0.5 * rho * v**2

        CD_i_p = lift_drag_mapper(CL_Turn)
        D_i_p = CD_i_p * q * S
        D_b = get_banner_drag(length=constants['banner_length'], velocity=v) if mission == 3 else 0
        drag = D_i_p + D_b
        CD_Turn = drag / (q * S)
        lift = CL_Turn * q * S

        if lift > W:
            F_lat = np.sqrt(lift**2 - W**2)
            omega = F_lat / (m * v)
        else:
            F_lat = 0
            omega = 0

        thrust, current = apc_data.get_propeller_performance(
            diameter=diameter,
            pitch=pitch,
            motor_kv=kv,
            battery_cell_count=battery_cell_count,
            airspeed_mph=v / 1.46667
        )

        new_acceleration = ((thrust) - (drag)) / m
        new_velocity = v + new_acceleration * dt
        new_position = np.add(state['position'][i], [v * dt, 0.0])
        turn_traveled = turn_traveled + np.degrees(omega * dt)

        state['velocity'] = np.append(state['velocity'], new_velocity)
        state['position'] = np.vstack((state['position'], new_position))
        state['acceleration'] = np.append(state['acceleration'], new_acceleration)
        state['battery_charge'] = np.append(state['battery_charge'], update_charge(state['battery_charge'][i], current, dt))
        state['time'] = np.append(state['time'], state['time'][i] + dt)
        state['turn_angle'] = np.append(state['turn_angle'], np.degrees(omega * dt))
        state['thrust'] = np.append(state['thrust'], thrust)
        state['Cl'] = np.append(state['Cl'], CL_Turn)
        state['Cd'] = np.append(state['Cd'], CD_Turn)
        state['lift'] = np.append(state['lift'], lift)
        state['drag'] = np.append(state['drag'], drag)
        state['F_long'] = np.append(state['F_long'], thrust - drag)
        state['F_lat'] = np.append(state['F_lat'], F_lat)
        i += 1
        state['i'] = i
    
    print(state['battery_charge'][-1])


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
            'Cl': np.array([0.0]),
            'Cd': np.array([0.0]),
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
        return 0
    except Exception as e:
        if debug:
            print(f"Error in lap simulation: {e}")
            import traceback
            traceback.print_exc()
        return 0
    
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

        CONFIG_NAME = "Test_Vehicle"

        print(f"Vehicle identifier: {CONFIG_NAME}")

        altitude_range = (0,10000,1) # ft
        mach_range = (0.01,0.99,11)
        alpha_range = (-5,20,26)
        constant_thrust = 50 # lbs
        climb_angle = 15 # deg
        cruise_altitude = 1000
        wing_area = 1 # ft^2
        max_cl = 0.75 # based on airfoil
        wing_thickness = 0.04
        root_chord = 1.4
        tip_chord = 0.1
        b_ref = 1.4
        c_ref = 0.87
        cg_distance_x = 1.3
        le_sweep = 60
        technology_factor = 0.87
        fuel_frac_empty = 1.333 # Ratio of structural weight to fuel weight
        landing_fuel_frac = 0.25 # % fuel where landing is required and simulation cannot continue

        vspfile = "/Users/gabrielkern/Documents/hypersonics/supersonicUAV/OpenVSP/Mach1Sizing/Mach1_Sizing.vsp3"

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
            'g': 32.174,
            'dt': 0.01,
            'fuel_capacity': weight_from_wing_area(wing_area) * fuel_frac_empty, # In lbs
            'constant_thrust': constant_thrust,
            'engine_weight': engine_weight_from_thrust(constant_thrust),
            'cruise_altitude': cruise_altitude,
            'theta': climb_angle,
            'CL_stall': max_cl,
            'landing_fuel_frac': landing_fuel_frac,
            'constant_sfc': sfc_from_thrust(constant_thrust)
        }

        lift_drag_csv = os.path.join(os.path.dirname(__file__), "TopSpeedSimResults", f"{CONFIG_NAME}.csv")

        if os.path.isfile(lift_drag_csv):
            rerun_flag = input(f"Existing CSV found for vehicle with the name {CONFIG_NAME}.\nPlease type Y to use this, else press any other key to re-generate the csv.")
        
        if not os.path.isfile(lift_drag_csv) or rerun_flag == "Y":
            sizingEstimation.generate_csv_from_file(vspfile=vspfile, csvoutput=lift_drag_csv, altitude_range=altitude_range, mach_range=mach_range, config=config)
        
        config['drag_lookup'] = build_lift_drag_interp(lift_drag_csv, wing_area)

        # Build the starting state for the sim. Start at base velocity and base mach
        _, t_start = get_atmosphere(altitude_range[0]) # Comes back in rankine
        sos_start = np.sqrt(GAMMA * R * t_start)
        initial_velocity = mach_range[0] * sos_start
        state = {
            'velocity': np.array([initial_velocity]),
            'position': np.array([[0.0, 0.0]]),
            'acceleration': np.array([0.0]),
            'fuel': np.array([config['fuel_capacity']]),
            'time': np.array([0.0]),
            'thrust': np.array([0.0]),
            'Cl': np.array([0.0]),
            'Cd': np.array([0.0]),
            'lift': np.array([0.0]),
            'drag': np.array([0.0]),
            'F_long': np.array([0.0]),
            'i': 0
        }

        climb(state=state,config=config)
