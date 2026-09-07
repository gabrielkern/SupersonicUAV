"""
Lap Simulator for Supersonic UAV
"""

import os
from functools import lru_cache
import argparse
import math
import numpy as np
from typing import Dict
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from ambiance import Atmosphere
from scipy.interpolate import interp1d
from scipy import optimize

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
    # ambiance's Atmosphere always returns arrays, even for scalar altitude input.
    return float(np.ravel(density_imperial)[0]), float(np.ravel(temperature_imperial)[0])

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
    """Load a lift/drag sweep CSV into regular grids over mach, planform_area, altitude, and angle_of_attack."""
    data = np.genfromtxt(csv_path, delimiter=',', names=True)

    mach_grid = np.unique(data['mach'])
    area_grid = np.unique(data['planform_area'])
    aoa_grid = np.unique(data['angle_of_attack'])

    if 'altitude' in data.dtype.names:
        altitude_grid = np.unique(data['altitude'])
        altitude_idx = np.searchsorted(altitude_grid, data['altitude'])
    else:
        # Older sweeps (e.g. Mach1Sizing/sizing_sweep_results.csv) have no altitude column --
        # treat the whole file as one implicit altitude station.
        altitude_grid = np.array([0.0])
        altitude_idx = np.zeros(len(data), dtype=int)

    expected_rows = len(mach_grid) * len(area_grid) * len(altitude_grid) * len(aoa_grid)
    if len(data) != expected_rows:
        raise ValueError(
            f"{csv_path}: expected a dense sweep of {expected_rows} rows "
            f"({len(mach_grid)} mach x {len(area_grid)} area x {len(altitude_grid)} altitude x "
            f"{len(aoa_grid)} AoA) but found {len(data)} rows -- the sweep is incomplete."
        )

    lift_4d = np.empty((len(mach_grid), len(area_grid), len(altitude_grid), len(aoa_grid)))
    drag_4d = np.empty_like(lift_4d)

    mach_idx = np.searchsorted(mach_grid, data['mach'])
    area_idx = np.searchsorted(area_grid, data['planform_area'])
    aoa_idx = np.searchsorted(aoa_grid, data['angle_of_attack'])

    lift_4d[mach_idx, area_idx, altitude_idx, aoa_idx] = data['lift']
    drag_4d[mach_idx, area_idx, altitude_idx, aoa_idx] = data['total_drag']

    return mach_grid, area_grid, altitude_grid, aoa_grid, lift_4d, drag_4d

@lru_cache(maxsize=None)
def _lift_drag_grids(csv_path, wing_area):
    """
    Load a lift/drag sweep CSV and area-match it to wing_area.

    Returns (mach_grid, altitude_grid, aoa_grid, lift_3d, drag_3d), where lift_3d/drag_3d have
    shape (len(mach_grid), len(altitude_grid), len(aoa_grid)) -- the CL(mach, altitude, AoA)/
    CD(mach, altitude, AoA) table interpolated linearly across planform_area down to the exact
    wing_area.
    """
    mach_grid, area_grid, altitude_grid, aoa_grid, lift_4d, drag_4d = _load_lift_drag_csv(csv_path)

    if len(area_grid) < 2:
        lift_3d = lift_4d[:, 0, :, :]  # single area in sweep — squeeze out area axis
        drag_3d = drag_4d[:, 0, :, :]
    else:
        area_interp_lift = interp1d(area_grid, lift_4d, axis=1, bounds_error=False, fill_value="extrapolate")
        area_interp_drag = interp1d(area_grid, drag_4d, axis=1, bounds_error=False, fill_value="extrapolate")
        lift_3d = area_interp_lift(wing_area)  # shape: (len(mach_grid), len(altitude_grid), len(aoa_grid))
        drag_3d = area_interp_drag(wing_area)

    return mach_grid, altitude_grid, aoa_grid, lift_3d, drag_3d

def _bracket(grid, value):
    """Return (station_indices, low, high) bracketing value in grid (a 1D array of >=2 points)."""
    low = value < grid[0]
    high = value > grid[-1]

    if low:
        station_indices = [0, 1]
    elif high:
        station_indices = [-2, -1]
    else:
        i = int(np.searchsorted(grid, value) - 1)
        i = max(0, min(i, len(grid) - 2))  # guard against fp edge cases at boundary
        station_indices = [i, i + 1]

    return station_indices, low, high

def _blend(grid, station_indices, low, high, v0, v1, value):
    """Blend two station values (v0 at station_indices[0], v1 at station_indices[1]) at value,
    linearly extrapolating past the grid's edge stations when low/high."""
    i0, i1 = station_indices

    if low:
        slope = (v1 - v0) / (grid[1] - grid[0])
        return v0 + slope * (value - grid[0])

    if high:
        slope = (v1 - v0) / (grid[-1] - grid[-2])
        return v1 + slope * (value - grid[-1])

    t = (value - grid[i0]) / (grid[i1] - grid[i0])
    return (1 - t) * v0 + t * v1

@lru_cache(maxsize=None)
def build_lift_drag_interp(csv_path, wing_area):
    """
    Build an altitude/mach/CL -> CD lookup from a lift/drag sweep CSV, area-matched to wing_area.

    Builds one CL->CD interpolant per (mach, altitude) station in the sweep, then bilinearly
    interpolates across the two bracketing mach stations and two bracketing altitude stations
    at query time (falling back to mach-only interpolation if the sweep has a single altitude).
    """
    mach_grid, altitude_grid, aoa_grid, lift_3d, drag_3d = _lift_drag_grids(csv_path, wing_area)
    n_mach, n_altitude = len(mach_grid), len(altitude_grid)
    has_altitude = n_altitude >= 2

    per_station_interps = [[None] * n_altitude for _ in range(n_mach)]
    cl_min = np.empty((n_mach, n_altitude))
    cl_max = np.empty((n_mach, n_altitude))
    for mi in range(n_mach):
        for ai in range(n_altitude):
            cl_row = lift_3d[mi, ai, :]
            cd_row = drag_3d[mi, ai, :]
            order = np.argsort(cl_row)
            cl_sorted = cl_row[order]
            per_station_interps[mi][ai] = interp1d(cl_sorted, cd_row[order], bounds_error=False, fill_value="extrapolate")
            cl_min[mi, ai] = cl_sorted[0]
            cl_max[mi, ai] = cl_sorted[-1]

    def drag_lookup(altitude, mach, cl):
        altitude = float(np.ravel(altitude)[0])
        mach = float(np.ravel(mach)[0])
        cl = float(np.ravel(cl)[0])

        m_idx, m_low, m_high = _bracket(mach_grid, mach)
        if has_altitude:
            a_idx, a_low, a_high = _bracket(altitude_grid, altitude)
            relevant_cl_min = cl_min[m_idx][:, a_idx].min()
            relevant_cl_max = cl_max[m_idx][:, a_idx].max()
        else:
            a_low = a_high = False
            relevant_cl_min = cl_min[m_idx, 0].min()
            relevant_cl_max = cl_max[m_idx, 0].max()

        if m_low or m_high or a_low or a_high or cl < relevant_cl_min or cl > relevant_cl_max:
            print(f"[WARNING] Extrapolation required with altitude {altitude}, mach {mach}, and CL {cl}")

        def cd_at_mach(mi):
            if not has_altitude:
                return float(per_station_interps[mi][0](cl))
            v0 = float(per_station_interps[mi][a_idx[0]](cl))
            v1 = float(per_station_interps[mi][a_idx[1]](cl))
            return _blend(altitude_grid, a_idx, a_low, a_high, v0, v1, altitude)

        cd0 = cd_at_mach(m_idx[0])
        cd1 = cd_at_mach(m_idx[1])
        return float(_blend(mach_grid, m_idx, m_low, m_high, cd0, cd1, mach))

    return drag_lookup

@lru_cache(maxsize=None)
def build_alpha_interp(csv_path, wing_area):
    """
    Build an altitude/mach/CL -> angle of attack lookup from a lift/drag sweep CSV, area-matched
    to wing_area. Same structure as build_lift_drag_interp, but interpolates the angle of attack
    that produced each CL instead of the resulting CD.
    """
    mach_grid, altitude_grid, aoa_grid, lift_3d, drag_3d = _lift_drag_grids(csv_path, wing_area)
    n_mach, n_altitude = len(mach_grid), len(altitude_grid)
    has_altitude = n_altitude >= 2

    per_station_interps = [[None] * n_altitude for _ in range(n_mach)]
    cl_min = np.empty((n_mach, n_altitude))
    cl_max = np.empty((n_mach, n_altitude))
    for mi in range(n_mach):
        for ai in range(n_altitude):
            cl_row = lift_3d[mi, ai, :]
            order = np.argsort(cl_row)
            cl_sorted = cl_row[order]
            per_station_interps[mi][ai] = interp1d(cl_sorted, aoa_grid[order], bounds_error=False, fill_value="extrapolate")
            cl_min[mi, ai] = cl_sorted[0]
            cl_max[mi, ai] = cl_sorted[-1]

    def alpha_lookup(altitude, mach, cl):
        altitude = float(np.ravel(altitude)[0])
        mach = float(np.ravel(mach)[0])
        cl = float(np.ravel(cl)[0])

        m_idx, m_low, m_high = _bracket(mach_grid, mach)
        if has_altitude:
            a_idx, a_low, a_high = _bracket(altitude_grid, altitude)
            relevant_cl_min = cl_min[m_idx][:, a_idx].min()
            relevant_cl_max = cl_max[m_idx][:, a_idx].max()
        else:
            a_low = a_high = False
            relevant_cl_min = cl_min[m_idx, 0].min()
            relevant_cl_max = cl_max[m_idx, 0].max()

        if m_low or m_high or a_low or a_high or cl < relevant_cl_min or cl > relevant_cl_max:
            print(f"[WARNING] Extrapolation required with altitude {altitude}, mach {mach}, and CL {cl}")

        def alpha_at_mach(mi):
            if not has_altitude:
                return float(per_station_interps[mi][0](cl))
            v0 = float(per_station_interps[mi][a_idx[0]](cl))
            v1 = float(per_station_interps[mi][a_idx[1]](cl))
            return _blend(altitude_grid, a_idx, a_low, a_high, v0, v1, altitude)

        a0 = alpha_at_mach(m_idx[0])
        a1 = alpha_at_mach(m_idx[1])
        return float(_blend(mach_grid, m_idx, m_low, m_high, a0, a1, mach))

    return alpha_lookup

@lru_cache(maxsize=None)
def build_lift_interp(csv_path, wing_area):
    """
    Build an altitude/mach/alpha -> lift coefficient lookup from a lift/drag sweep CSV,
    area-matched to wing_area. Forward evaluation of CL(alpha) -- the inverse direction of
    build_alpha_interp. Since angle_of_attack is already a regular, sorted sweep axis (unlike
    CL, which isn't monotonic across mach/altitude), no per-station sorting is needed here.
    """
    mach_grid, altitude_grid, aoa_grid, lift_3d, drag_3d = _lift_drag_grids(csv_path, wing_area)
    n_mach, n_altitude = len(mach_grid), len(altitude_grid)
    has_altitude = n_altitude >= 2

    per_station_interps = [[None] * n_altitude for _ in range(n_mach)]
    for mi in range(n_mach):
        for ai in range(n_altitude):
            cl_row = lift_3d[mi, ai, :]
            per_station_interps[mi][ai] = interp1d(aoa_grid, cl_row, bounds_error=False, fill_value="extrapolate")

    def lift_lookup(altitude, mach, alpha):
        altitude = float(np.ravel(altitude)[0])
        mach = float(np.ravel(mach)[0])
        alpha = float(np.ravel(alpha)[0])

        m_idx, m_low, m_high = _bracket(mach_grid, mach)
        if has_altitude:
            a_idx, a_low, a_high = _bracket(altitude_grid, altitude)
        else:
            a_low = a_high = False

        if m_low or m_high or a_low or a_high or alpha < aoa_grid[0] or alpha > aoa_grid[-1]:
            print(f"[WARNING] Extrapolation required with altitude {altitude}, mach {mach}, and alpha {alpha}")

        def cl_at_mach(mi):
            if not has_altitude:
                return float(per_station_interps[mi][0](alpha))
            v0 = float(per_station_interps[mi][a_idx[0]](alpha))
            v1 = float(per_station_interps[mi][a_idx[1]](alpha))
            return _blend(altitude_grid, a_idx, a_low, a_high, v0, v1, altitude)

        cl0 = cl_at_mach(m_idx[0])
        cl1 = cl_at_mach(m_idx[1])
        return float(_blend(mach_grid, m_idx, m_low, m_high, cl0, cl1, mach))

    return lift_lookup

@lru_cache(maxsize=None)
def build_cl_max_interp(csv_path, wing_area):
    """
    Build an altitude/mach -> CL_max lookup from a lift/drag sweep CSV, area-matched to wing_area.

    CL_max at each (mach, altitude) station is the largest CL actually achieved across the
    angle-of-attack sweep there.
    """
    mach_grid, altitude_grid, aoa_grid, lift_3d, drag_3d = _lift_drag_grids(csv_path, wing_area)
    n_altitude = len(altitude_grid)
    has_altitude = n_altitude >= 2

    cl_max_per_station = lift_3d.max(axis=2)  # shape: (n_mach, n_altitude)

    def cl_max_lookup(altitude, mach):
        altitude = float(np.ravel(altitude)[0])
        mach = float(np.ravel(mach)[0])

        m_idx, m_low, m_high = _bracket(mach_grid, mach)
        if has_altitude:
            a_idx, a_low, a_high = _bracket(altitude_grid, altitude)
        else:
            a_low = a_high = False

        if m_low or m_high or a_low or a_high:
            print(f"[WARNING] Extrapolating CL_max lookup at altitude {altitude}, mach {mach}")

        def cl_max_at_mach(mi):
            if not has_altitude:
                return cl_max_per_station[mi, 0]
            v0, v1 = cl_max_per_station[mi, a_idx[0]], cl_max_per_station[mi, a_idx[1]]
            return _blend(altitude_grid, a_idx, a_low, a_high, v0, v1, altitude)

        v0 = cl_max_at_mach(m_idx[0])
        v1 = cl_max_at_mach(m_idx[1])
        return float(_blend(mach_grid, m_idx, m_low, m_high, v0, v1, mach))

    return cl_max_lookup

def find_thrust_limited_speed(*, thrust_interp, lift_drag_interp, altitude, wing_area, weight=None, top_speed=1500):
    """Find max speed where thrust = drag using thrust curve. Returns tuple of speed (ft/s) and mach"""

    if weight == None:
        if isinstance(lift_drag_interp, str): # This means its a path
            drag_lookup = build_lift_drag_interp(lift_drag_interp, wing_area)
        else:
            drag_lookup = lambda altitude, mach, cl: lift_drag_interp(cl)

        for speed in range(1, top_speed):  # range in ft/s
            weight = weight_from_wing_area(wing_area)
            rho,temp = get_atmosphere(altitude)
            q = 0.5 * rho * speed**2
            lift_coefficient = weight / q / wing_area
            sos = np.sqrt(GAMMA * R * temp) # speed of sound
            mach = speed / sos
            drag_coefficient = drag_lookup(altitude, mach, lift_coefficient)
            thrust = thrust_interp(altitude, mach) # Call the callable object
            drag = drag_coefficient * q * wing_area
            if thrust <= drag:
                return max(speed - 1, 1), max((speed-1)/sos,1/sos)  # ensure minimum speed of 1 ft/s
        return -1, -1  # ft/s, fallback
    else:
        if isinstance(lift_drag_interp, str):
            drag_lookup = build_lift_drag_interp(lift_drag_interp, wing_area)
        else:
            drag_lookup = lambda altitude, mach, cl: lift_drag_interp(cl)

        for speed in range(1, top_speed):  # range in ft/s
            rho,temp = get_atmosphere(altitude)
            q = 0.5 * rho * speed**2
            lift_coefficient = weight / q / wing_area
            sos = np.sqrt(GAMMA * R * temp) # speed of sound
            mach = speed / sos
            drag_coefficient = drag_lookup(altitude, mach, lift_coefficient)
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

def takeoff(state: dict, config: dict):
    """
    Function that calculates the ground-roll/takeoff of the plane.
    Flies level (gamma=0), accelerating, until there is enough speed to generate the CL needed
    to climb at the commanded climb angle theta -- i.e. until CL_climb_required < CL_max(mach).
    Occurs only once per simulation, before climb.
    """
    EW = config['structural_weight'] + config['engine_weight']
    S = config['wing_area']
    dt = config['dt']
    theta = np.deg2rad(config['theta'])
    drag_lookup = config['drag_lookup']
    alpha_lookup = config['alpha_lookup']
    lift_lookup = config['lift_lookup']
    cl_max_lookup = config['cl_max_lookup']
    thrust_interp = config['thrust_interp']
    sfc_interp = config['sfc_interp']

    i = state['i']

    while True:
        # Define altitude-dependent variables
        altitude = state['position'][i, 1]
        rho, T = get_atmosphere(altitude)
        sos = np.sqrt(GAMMA * R * T)

        # Fuel dependent weight and mass
        W = EW + state['fuel'][i]
        m = W / g

        v = np.linalg.norm(state['velocity'][i])
        mach = v / sos
        q = 0.5 * rho * v**2

        # Get thrust from interpolation model
        thrust = thrust_interp(altitude, mach)

        alpha = 0 # Takeoff assumption
        CL_Takeoff = lift_lookup(altitude, mach, alpha)
        CD_Takeoff = drag_lookup(altitude, mach, CL_Takeoff)

        lift = CL_Takeoff * q * S
        drag = CD_Takeoff * q * S

        CL_max = cl_max_lookup(altitude, mach)
        alpha_max = np.deg2rad(alpha_lookup(altitude, mach, CL_max)) # Convert to radians
        CL_climb_required = ( (W*np.cos(theta)) - (thrust*np.sin(alpha_max)) ) / q / S # This makes acceleration vertical to theta zero
        if (CL_climb_required < CL_max):# and not math.isnan(np.arcsin((thrust-drag)/W)) and (np.arcsin((thrust-drag)/W) >= theta):
            break  # enough speed built up to support the commanded climb

        if state['fuel'][i] < config['landing_fuel_frac']*config['fuel_capacity']:
            break

        # Get fuel consumption from thrust
        sfc = sfc_interp(altitude, mach)

        new_acceleration = (thrust - drag) / m
        new_v = v + new_acceleration * dt
        new_velocity = [new_v, 0.0]
        new_position = np.add(state['position'][i], [v * dt, 0.0])

        state['velocity'] = np.vstack((state['velocity'], new_velocity))
        state['position'] = np.vstack((state['position'], new_position))
        state['acceleration'] = np.vstack((state['acceleration'], [new_acceleration, 0.0]))
        state['fuel'] = np.append(state['fuel'], update_fuel_mass(state['fuel'][i], sfc, dt, thrust))
        state['time'] = np.append(state['time'], state['time'][i] + dt)
        state['thrust'] = np.append(state['thrust'], thrust)
        state['alpha'] = np.append(state['alpha'], alpha)
        state['CL'] = np.append(state['CL'], CL_Takeoff)
        state['CD'] = np.append(state['CD'], CD_Takeoff)
        state['lift'] = np.append(state['lift'], lift)
        state['drag'] = np.append(state['drag'], drag)
        state['F_long'] = np.append(state['F_long'], new_acceleration * m)
        state['temp'] = np.append(state['temp'], T)
        state['rho'] = np.append(state['rho'], rho)
        state['mach'] = np.append(state['mach'], mach)
        i += 1
        state['i'] = i

def climb(state: dict, config: dict):
    """
    Function that calculates the climb of the plane.
    Occurs only once per simulation, after takeoff.
    """
    EW = config['structural_weight'] + config['engine_weight']
    S = config['wing_area']
    dt = config['dt']
    cruise_alt = config['cruise_altitude']
    theta = np.deg2rad(config['theta'])
    drag_lookup = config['drag_lookup']
    alpha_lookup = config['alpha_lookup']
    lift_lookup = config['lift_lookup']
    cl_max_lookup = config['cl_max_lookup']
    thrust_interp = config['thrust_interp']
    sfc_interp = config['sfc_interp']

    i = state['i']

    while (state['position'][i, 1] <= cruise_alt) and (state['fuel'][i] >= config['landing_fuel_frac']*config['fuel_capacity']):

        # Define altitude-dependent variables
        altitude = state['position'][i,1]
        rho, T = get_atmosphere(altitude)
        sos = np.sqrt(GAMMA * R * T)

        # Fuel dependent weight and mass
        W = EW + state['fuel'][i]
        m = W / g

        v = np.linalg.norm(state['velocity'][i])
        mach = v / sos

        q = 0.5 * rho * v**2

        CL_max = cl_max_lookup(altitude, mach)
        alpha_min = 0
        alpha_max = alpha_lookup(altitude, mach, CL_max)

        # Get thrust from interpolation model
        thrust = thrust_interp(altitude, mach)

        # Implicitly solve for both CL Climb and alpha simultaneously (mind blown)
        CL_solve = lambda alpha: lift_lookup(altitude, mach, alpha)
        alpha_func = lambda alpha: ( (W*np.cos(theta)) - (thrust*np.sin(np.deg2rad(alpha))) - (CL_solve(alpha)*q*S) )
        alpha_climb = optimize.brentq(alpha_func, alpha_min, alpha_max)
        CL_Climb = CL_solve(alpha_climb)
        if CL_Climb > CL_max:
            CL_Climb = CL_max
            alpha_climb = alpha_lookup(altitude, mach, CL_Climb)

        CD_Climb = drag_lookup(altitude, mach, CL_Climb)

        lift = CL_Climb * q * S
        drag = CD_Climb * q * S

        # Get fuel consumption from thrust
        sfc = sfc_interp(altitude, mach)

        new_acceleration = ((thrust*np.cos(np.deg2rad(alpha_climb))) - (drag) - (W * np.sin(theta))) / m
        new_v = v + new_acceleration * dt
        new_velocity = [new_v * np.cos(theta), new_v * np.sin(theta)]
        new_position = np.add(state['position'][i], [v * dt * np.cos(theta), v * dt * np.sin(theta)])

        state['velocity'] = np.vstack((state['velocity'], new_velocity))
        state['position'] = np.vstack((state['position'], new_position))
        state['acceleration'] = np.vstack((state['acceleration'], [new_acceleration * np.cos(theta), new_acceleration * np.sin(theta)]))
        state['fuel'] = np.append(state['fuel'], update_fuel_mass(state['fuel'][i], sfc, dt, thrust))
        state['time'] = np.append(state['time'], state['time'][i] + dt)
        state['thrust'] = np.append(state['thrust'], thrust)
        state['alpha'] = np.append(state['alpha'], alpha_climb) # In degrees
        state['CL'] = np.append(state['CL'], CL_Climb)
        state['CD'] = np.append(state['CD'], CD_Climb)
        state['lift'] = np.append(state['lift'], lift)
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
    drag_lookup = config['drag_lookup']
    alpha_lookup = config['alpha_lookup']
    cl_max_lookup = config['cl_max_lookup']
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

        v = np.linalg.norm(state['velocity'][i])
        mach = v / sos

        q = 0.5 * rho * v**2

        CL_max = cl_max_lookup(altitude, mach)
        CL_Straight = W / q / S
        if CL_Straight > CL_max:
            CL_Straight = CL_max

        CD_Straight = drag_lookup(altitude, mach, CL_Straight)
        alpha = alpha_lookup(altitude, mach, CL_Straight)

        lift = CL_Straight * q * S
        drag = CD_Straight * q * S

        # Get thrust from interpolation model
        thrust = thrust_interp(altitude, mach)

        # Get fuel consumption from thrust
        sfc = sfc_interp(altitude, mach)

        new_acceleration = ((thrust) - (drag)) / m
        new_v = v + new_acceleration * dt
        new_velocity = [new_v, 0.0]
        new_position = np.add(state['position'][i], [v * dt, 0.0])

        state['velocity'] = np.vstack((state['velocity'], new_velocity))
        state['position'] = np.vstack((state['position'], new_position))
        state['acceleration'] = np.vstack((state['acceleration'], [new_acceleration, 0.0]))
        state['fuel'] = np.append(state['fuel'], update_fuel_mass(state['fuel'][i], sfc, dt, thrust))
        state['time'] = np.append(state['time'], state['time'][i] + dt)
        state['thrust'] = np.append(state['thrust'], thrust)
        state['alpha'] = np.append(state['alpha'], alpha)
        state['CL'] = np.append(state['CL'], CL_Straight)
        state['CD'] = np.append(state['CD'], CD_Straight)
        state['lift'] = np.append(state['lift'], lift)
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
    
def plot_drag_simple(altitude, mach, CL, drag_lookup, points=100, min_mach = 0.1, max_mach = 1.2, min_CL = 0.0, max_CL = 0.7):
    """Simple plot function which plots the drag versus lift at a specified mach and drag versus mach at a specified lift, at a fixed altitude."""
    plot_machs = np.linspace(min_mach, max_mach, points)
    plot_CLs = np.linspace(min_CL, max_CL, points)
    mach_sweep = np.zeros_like(plot_machs)
    CL_sweep = np.zeros_like(plot_CLs)
    for index,_ in enumerate(range(points)):
        mach_sweep[index] = drag_lookup(altitude, plot_machs[index], CL)
        CL_sweep[index] = drag_lookup(altitude, mach, plot_CLs[index])

    # Comprehensive plotting of all flight parameters
    fig = plt.figure(figsize=[15, 8])
    gs = GridSpec(1, 2, figure=fig, hspace=0.4, wspace=0.3)
    fig.suptitle(f"Quicklooks Drag Polar (Altitude = {altitude} ft)", fontsize=18)

    # Row 1: Velocities, Battery, Position
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.plot(plot_machs, mach_sweep, 'b-', linewidth=2, label=f'CL = {CL}')
    ax1.set_xlabel("Mach"); ax1.set_ylabel("CD")
    ax1.set_title("CD vs Mach"); ax1.grid(True); ax1.legend()

    ax2 = fig.add_subplot(gs[0, 1])
    ax2.plot(plot_CLs, CL_sweep, 'r-', linewidth=2, label=f'Mach = {mach}')
    ax2.set_xlabel("CL"); ax2.set_ylabel("CD")
    ax2.set_title("Drag Polar"); ax2.grid(True); ax2.legend()

    plt.tight_layout()
    plt.show()
    
def plot_results(state: dict, config: dict):
    """Takes the results from simulation and plots values for various states over the flight time."""
    
    # Comprehensive plotting of all flight parameters
    true_airspeed = np.linalg.norm(state['velocity'], axis=1)

    fig = plt.figure(figsize=[15, 8])
    gs = GridSpec(3, 4, figure=fig, hspace=0.4, wspace=0.3)
    fig.suptitle(f"Top Speed Lap Simulation Results", fontsize=18)

    # Row 1: Velocities, Fuel, Position
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.plot(state['time'], state['mach'], 'b-', linewidth=2, label='Mach')
    ax1.set_xlabel("Time (s)"); ax1.set_ylabel("Mach")
    ax1.set_title("Mach vs Time"); ax1.grid(True); ax1.legend()

    ax1b = fig.add_subplot(gs[0, 1])
    ax1b.plot(state['time'], true_airspeed, 'b-', linewidth=2, label='True Airspeed')
    ax1b.set_xlabel("Time (s)"); ax1b.set_ylabel("Speed (ft/s)")
    ax1b.set_title("True Airspeed vs Time"); ax1b.grid(True); ax1b.legend()

    ax2 = fig.add_subplot(gs[0, 2])
    ax2.plot(state['time'], state['fuel'], 'g-', linewidth=2, label="Fuel remaining")
    ax2.plot(state['time'], (np.ones_like(state['time']) * config['landing_fuel_frac'] * config['fuel_capacity']), 'r.', linewidth=2, label="Landing fuel minimum")
    ax2.set_xlabel("Time (s)"); ax2.set_ylabel("Fuel Mass (lbm)")
    ax2.set_title("Fuel Mass vs Time"); ax2.grid(True); ax2.legend()

    ax3 = fig.add_subplot(gs[0, 3])
    ax3.plot(state['time'], state['position'][:,0], 'b-', linewidth=2, label='Traveled Distance')
    ax3.plot(state['time'], state['position'][:,1], 'r-', linewidth=2, label='Altitude')
    ax3.set_xlabel("Time (s)"); ax3.set_ylabel("Position (ft)")
    ax3.set_title("Position vs Time"); ax3.grid(True); ax3.legend()

    # Row 2: Accelerations, Thrust, Angle of Attack
    ax4 = fig.add_subplot(gs[1, 0])
    ax4.plot(state['time'], state['acceleration'][:,0], 'b-', linewidth=2, label='Forward (ax)')
    ax4.plot(state['time'], state['acceleration'][:,1], 'r-', linewidth=2, label='Vertical (ay)')
    ax4.set_xlabel("Time (s)"); ax4.set_ylabel("Acceleration (ft/s²)")
    ax4.set_title("Acceleration vs Time"); ax4.grid(True); ax4.legend()

    ax5 = fig.add_subplot(gs[1, 1])
    ax5.plot(state['time'], state['thrust'], 'orange', linewidth=2)
    ax5.set_xlabel("Time (s)"); ax5.set_ylabel("Thrust (lbs)")
    ax5.set_title("Thrust vs Time"); ax5.grid(True)

    ax5b = fig.add_subplot(gs[1, 2])
    ax5b.plot(state['time'], state['alpha'], 'm-', linewidth=2, label='Alpha')
    ax5b.set_xlabel("Time (s)"); ax5b.set_ylabel("Angle of Attack (deg)")
    ax5b.set_title("Angle of Attack vs Time"); ax5b.grid(True); ax5b.legend()

    # Row 2/3: Aerodynamic coefficients with L/D, Combined Forces, Longitudinal/Lateral Forces
    ax6 = fig.add_subplot(gs[1, 3])
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
    ax9.set_title("Longitudinal Force vs Time"); ax9.grid(True)

    print("Close plot to get final statistics.")
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

        altitude_range = (0,10000,11) # ft
        mach_range = (0.01,1.2,20)
        alpha_range = (-5,20,13)
        constant_thrust = 95 # lbs
        climb_angle = 15 # deg
        cruise_altitude = 2000
        wing_area = 3.201 # ft^2
        wing_thickness = 0.04
        root_chord = 2.1882933333 # feet
        tip_chord = 0.3724675 # feet
        b_ref = 2.5 # feet
        c_ref = 1.4949791667 # feet
        cg_distance_x = 3.33333 # inches
        le_sweep = 60 # degrees
        technology_factor = 0.87 # For 6 series
        fuel_frac_empty = 0.5 # Ratio of structural weight to fuel weight
        landing_fuel_frac = 0.10 # % fuel where landing is required and simulation cannot continue

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
            confirm = input(f"Are you sure you would like to regenerate? This will delete existing results under the same name. Type YES to confirm.")
            if confirm.upper() == "YES":
                sizingEstimation.generate_csv_from_file(vspfile=vspfile, csvoutput=lift_drag_csv, altitude_range=altitude_range, mach_range=mach_range, config=config)
        
        config['drag_lookup'] = build_lift_drag_interp(lift_drag_csv, wing_area)
        config['alpha_lookup'] = build_alpha_interp(lift_drag_csv, wing_area)
        config['lift_lookup'] = build_lift_interp(lift_drag_csv, wing_area)
        config['cl_max_lookup'] = build_cl_max_interp(lift_drag_csv, wing_area)
        config['thrust_interp'] = thrust_model(constant_thrust)
        config['sfc_interp'] = sfc_model(config['constant_sfc'])

        quicklooks_answer = input("Would you like a quicklooks of the drag being used? Type Y to see graphs and any other button to skip.\n")
        if quicklooks_answer == "Y":
            quicklooks_altitude = input("Type the altitude (ft) to hold constant in both graphs: \n")
            mach = input("Type the mach to be held constant in CD vs CL graph: \n")
            CL = input("Type the CL to be held constant in CD vs Mach graph: \n")
            plot_drag_simple(quicklooks_altitude,mach,CL,config['drag_lookup'])
        else:
            print("Running...")

        # Build the starting state for the sim. Start at base velocity and base mach
        rho_start, t_start = get_atmosphere(altitude_range[0]) # Comes back in rankine
        sos_start = np.sqrt(GAMMA * R * t_start)
        initial_velocity = mach_range[0] * sos_start
        state = {
            'velocity': np.array([[initial_velocity, 0.0]]),
            'position': np.array([[0.0, 0.0]]),
            'acceleration': np.array([[0.0, 0.0]]),
            'fuel': np.array([config['fuel_capacity']]),
            'time': np.array([0.0]),
            'thrust': np.array([0.0]),
            'alpha': np.array([0.0]),
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

        print(f"Taking off at time {state['time'][-1]}...")
        takeoff(state=state,config=config)

        print(f"Beginning climb at time {state['time'][-1]}...")
        climb(state=state,config=config)

        print(f"Accelerating in straight line at time {state['time'][-1]}...")
        straight(state=state,config=config)

        print(f"Plotting results...")
        plot_results(state=state,config=config)

        print('_'*60)
        print("FINAL STATS:")
        maxxius_macchius = int(np.argmax(state['mach']))
        print(f"Top speed achieved: Mach {state['mach'][maxxius_macchius]} at {state['position'][maxxius_macchius, 1]} feet altitude.")
        print(f"Final thrust: {state['thrust'][-1]}.")
        print(f"Final drag: {state['drag'][-1]}.")
        print(f"Final CL: {state['CL'][-1]}.")
        print(f"Final Time: {state['time'][-1]}.")
        print('_'*60)