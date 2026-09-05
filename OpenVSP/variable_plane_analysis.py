#!/usr/bin/env python3
"""
VSPAERO Analysis Runner for OpenVSP
====================================

This script loads a .vsp3 file (from create_base_Mach1UAV.py)
and runs VSPAERO analysis to compute aerodynamic coefficients over an angle of attack range
"""

import sys
import os
import numpy as np
from typing import Dict
from scipy.interpolate import interp1d
from pathlib import Path
from ambiance import Atmosphere

from TopSpeedSim import get_atmosphere, GAMMA, R

# Add OpenVSP Python path
sys.path.append('/Users/gabrielkern/Documents/Python/AgenticDesigner/OpenVSP-3.45.4-MacOS/python/openvsp')

import openvsp as vsp

# Pa*s -> slug/(ft*s) (equivalently lbf*s/ft^2)
PASCAL_SECOND_TO_SLUG_PER_FT_S = 0.0208854342

# ==============================================================================
# MAIN ANALYSIS FUNCTIONS
# ==============================================================================

def initialize_vsp(filename):
    """Initialize OpenVSP and load the geometry file."""
    print("=" * 60)
    print("Initializing VSPAERO Analysis")
    print("=" * 60)

    vsp.VSPCheckSetup()  # Initialize VSP system
    vsp.VSPRenew()  # Clear existing model

    version = vsp.GetVSPVersion()  # Get VSP version string
    print(f"OpenVSP Version: {version}")

    print(f"\nLoading file: {filename}")
    vsp.ReadVSPFile(filename)  # Load .vsp3 geometry file
    vsp.Update()  # Update all geometry
    print("File loaded successfully")

    return vsp.ErrorMgrSingleton.getInstance()  # Return error manager for checking


def run_compgeom_analysis(vsp_aero_data: Dict):
    """Run CompGeom analysis to generate DegenGeom file required by VSPAERO."""
    print("\n" + "=" * 60)
    print("Running CompGeom Analysis")
    print("=" * 60)

    vsp.SetAnalysisInputDefaults(vsp_aero_data['geom_analysis'])  # Load default parameters

    # Set geometry sets: GeomSet for thick surfaces, ThinGeomSet for VLM lifting surfaces
    vsp.SetIntAnalysisInput(vsp_aero_data['geom_analysis'], "ThinGeomSet", [vsp_aero_data['thin_geom_set']], 0)

    vsp.SetIntAnalysisInput(vsp_aero_data['geom_analysis'], "Symmetry", [vsp_aero_data['symmetry']], 0)  # No symmetry plane
    vsp.SetIntAnalysisInput(vsp_aero_data['geom_analysis'], "CullFracFlag", [1], 0)  # Enable culling

    print("\nCompGeom Analysis Inputs:")
    vsp.PrintAnalysisInputs(vsp_aero_data['geom_analysis'])  # Display configured inputs

    print("\nExecuting CompGeom...")
    compgeom_resid = vsp.ExecAnalysis(vsp_aero_data['geom_analysis'])  # Execute analysis
    print("CompGeom COMPLETE")

    vsp.PrintResults(compgeom_resid)  # Display results

    return compgeom_resid


def setup_vspaero_inputs(vsp_aero_data: Dict):
    """Configure VSPAERO analysis inputs: reference values, flight conditions, wake, and stability."""
    print("\n" + "=" * 60)
    print(f"Setting up {vsp_aero_data['vsp_analysis']} Analysis")
    print("=" * 60)

    vsp.SetAnalysisInputDefaults(vsp_aero_data['vsp_analysis'])  # Load default parameters

    input_names = vsp.GetAnalysisInputNames(vsp_aero_data['vsp_analysis'])
    print("Available inputs for VSPAEROSweep:")
    for name in input_names:
        print(f"  - {name}")

    # Set case parameters
    vsp.SetIntAnalysisInput(vsp_aero_data['vsp_analysis'], "ThinGeomSet", [vsp_aero_data['thin_geom_set']], 0)
    vsp.SetIntAnalysisInput(vsp_aero_data['vsp_analysis'], "NCPU", [8], 0)  # VLM method
    vsp.SetIntAnalysisInput(vsp_aero_data['vsp_analysis'], "Symmetry", [vsp_aero_data['symmetry']], 0)  # Symmetry plane

    # Set center of gravity location [ft]
    # vsp.SetIntAnalysisInput(ANALYSIS_TYPE, "UseCGModeFlag", [1], 0)  # Use CG mode
    # vsp.SetIntAnalysisInput(ANALYSIS_TYPE, "CGGeomSet", [vsp.SET_ALL], 0)  # Use all geometry for CG

    vsp.SetDoubleAnalysisInput(vsp_aero_data['vsp_analysis'], "Xcg", [vsp_aero_data['x_rel']], 0)
    vsp.SetDoubleAnalysisInput(vsp_aero_data['vsp_analysis'], "Ycg", [0.0], 0)
    vsp.SetDoubleAnalysisInput(vsp_aero_data['vsp_analysis'], "Zcg", [0.0], 0)

    # Set flight conditions
    vsp.SetDoubleAnalysisInput(vsp_aero_data['vsp_analysis'], "MachStart", [vsp_aero_data['mach_start']], 0)  # Freestream Mach number
    vsp.SetIntAnalysisInput(vsp_aero_data['vsp_analysis'], "MachNpts", [vsp_aero_data['mach_points']], 0)
    vsp.SetDoubleAnalysisInput(vsp_aero_data['vsp_analysis'], "ReCref", [vsp_aero_data['re_start']], 0)  # Reynolds number
    vsp.SetIntAnalysisInput(vsp_aero_data['vsp_analysis'], "ReCrefNpts", [vsp_aero_data['re_points']], 0)
    vsp.SetDoubleAnalysisInput(vsp_aero_data['vsp_analysis'], "AlphaStart", [vsp_aero_data['alpha_start']], 0)  # Angle of attack [deg]
    vsp.SetIntAnalysisInput(vsp_aero_data['vsp_analysis'], "AlphaNpts", [vsp_aero_data['alpha_points']], 0)
    vsp.SetDoubleAnalysisInput(vsp_aero_data['vsp_analysis'], "AlphaEnd", [vsp_aero_data['alpha_end']], 0) 
    vsp.SetDoubleAnalysisInput(vsp_aero_data['vsp_analysis'], "BetaStart", [vsp_aero_data['beta_start']], 0)  # Sideslip angle [deg]
    vsp.SetIntAnalysisInput(vsp_aero_data['vsp_analysis'], "BetaNpts", [vsp_aero_data['beta_points']], 0)

    # Set wake and symmetry parameters
    vsp.SetIntAnalysisInput(vsp_aero_data['vsp_analysis'], "FixedWakeFlag", [0], 0)
    vsp.SetIntAnalysisInput(vsp_aero_data['vsp_analysis'], "WakeNumIter", [vsp_aero_data['wake_iter']], 0)  # Wake iterations
    vsp.SetIntAnalysisInput(vsp_aero_data['vsp_analysis'], "NumWakeNodes", [vsp_aero_data['wake_nodes']], 0)  # Wake nodes
    vsp.SetDoubleAnalysisInput(vsp_aero_data['vsp_analysis'], "WakeRelax", [vsp_aero_data['wake_relax']], 0)

    # Set reference values
    vsp.SetIntAnalysisInput(vsp_aero_data['vsp_analysis'], "RefFlag", [vsp_aero_data['ref_flag']], 0)  # Reference computation method
    vsp.SetDoubleAnalysisInput(vsp_aero_data['vsp_analysis'], "bref", [vsp_aero_data['b_ref']], 0)  # Reference span [ft]
    vsp.SetDoubleAnalysisInput(vsp_aero_data['vsp_analysis'], "Sref", [vsp_aero_data['s_ref']], 0)  # Reference area [ft²]
    vsp.SetDoubleAnalysisInput(vsp_aero_data['vsp_analysis'], "cref", [vsp_aero_data['c_ref']], 0)  # Reference MAC [ft]
    vsp.SetIntAnalysisInput(vsp_aero_data['vsp_analysis'], "MACFlag", [1], 0)  # Use MAC
    vsp.SetIntAnalysisInput(vsp_aero_data['vsp_analysis'], "ScurveFlag", [0], 0)  # No S-curve correction needed -  no blending

    # Advanced:

    # Other
    vsp.SetIntAnalysisInput(vsp_aero_data['vsp_analysis'], "StallModel", [vsp_aero_data['stall_model_flag']], 0)  # Use stall model
    vsp.SetDoubleAnalysisInput(vsp_aero_data['vsp_analysis'], "CLMax2D", [vsp_aero_data['cl_max']], 0)  # 2D CLmax
    vsp.SetDoubleAnalysisInput(vsp_aero_data['vsp_analysis'], "Clo2D", [vsp_aero_data['cl0']], 0)  # 2D CL at 0 deg

    # DIAGNOSTIC: Verify stall parameters were actually set
    print("\n" + "=" * 60)
    print("🔍 STALL MODEL VERIFICATION")
    print("=" * 60)
    print(f"StallModel flag set to: {vsp_aero_data['stall_model_flag']}")
    print(f"CLMax2D set to: {vsp_aero_data['cl_max']}")
    print(f"Clo2D set to: {vsp_aero_data['cl0']}")
    print("=" * 60)

    # Advanced flow conditions
    vsp.SetDoubleAnalysisInput(vsp_aero_data['vsp_analysis'], "Vinf", [vsp_aero_data['v_ref']], 0)  # Freestream velocity [ft/s]
    vsp.SetDoubleAnalysisInput(vsp_aero_data['vsp_analysis'], "Vref", [vsp_aero_data['v_ref']], 0)  # Reference velocity [ft/s]
    vsp.SetDoubleAnalysisInput(vsp_aero_data['vsp_analysis'], "Machref", [vsp_aero_data['mach_start']], 0)  # Reference Mach number
    vsp.SetDoubleAnalysisInput(vsp_aero_data['vsp_analysis'], "Rho", [vsp_aero_data['rho']], 0)  # Air density [slugs/ft³]

    # Stability type
    vsp.SetIntAnalysisInput(vsp_aero_data['vsp_analysis'], "UnsteadyType", [vsp_aero_data['stability_type']], 0)

    vsp.Update()  # Update all parameters

    print("\nVSPAERO Analysis Inputs:")
    vsp.PrintAnalysisInputs(vsp_aero_data['vsp_analysis'])  # Display all configured inputs


def run_vspaero_analysis(vsp_aero_data: Dict):
    """Execute VSPAERO analysis and retrieve results."""
    print("\n" + "=" * 60)
    print("Running VSPAERO Analysis")
    print("=" * 60)

    print("\nExecuting VSPAERO...")
    results_id = vsp.ExecAnalysis(vsp_aero_data['vsp_analysis'])  # Run VSPAERO solver
    print("VSPAERO Analysis COMPLETE")

    print("\nResults Summary:")
    vsp.PrintResults(results_id)  # Display results summary

    return results_id


def extract_stability_data(results_id, stability_type=1):
    """
    Extract stability data from VSPAERO results using vsp.GetDoubleResults().
    Returns data in the same nested dict format as process_stability_files.py

    Args:
        results_id: VSPAERO results ID
        stability_type: 1=static, 2=p (roll), 3=q (pitch), 4=r (yaw)
    """
    print("\n" + "=" * 60)
    print(f"Extracting Stability Data (Type {stability_type})")
    print("=" * 60)

    # Get the results vector - static analysis results are in ResultsVec[-2]
    results_vec = vsp.GetStringResults(results_id, "ResultsVec")
    print(f"All ResultsVec ({len(results_vec)} items): {results_vec}")

    data = {
        'base_aero': {},
        'derivatives': {},
        'dynamic_derivatives': {}
    }

    # STABILITY_TYPE = 1: Static analysis - extract base aero and static derivatives
    if stability_type == 1:

        # Base aero extraction
        data['base_aero']['CFx'] = vsp.GetDoubleResults(results_vec[-2], "Base_Aero_CFx")[0]
        data['base_aero']['CFz'] = vsp.GetDoubleResults(results_vec[-2], "Base_Aero_CFz")[0]
        data['base_aero']['SM'] = vsp.GetDoubleResults(results_vec[-2], "SM")[0]

        # Static derivative extraction
        stab_coefs = [
            "CFx", "CFy", "CFz", "CMx", "CMy", "CMz"
        ]

        varied_parm = [
            "U", "Alpha", "Beta", "p", "q", "r"
        ]

        for coef in stab_coefs:
            for parm in varied_parm:
                result_name = f"{coef}_{parm}"
                data['derivatives'][result_name] = vsp.GetDoubleResults(results_vec[-2], result_name)[0]

    # STABILITY_TYPE = 3
    else:
        # Map stability type to derivative key (FUTURE USE POTENTIALLY)
        type_map = {
            2: '_p',           # Roll rate
            3: '_q+alpha_dot', # Pitch rate
            4: '_r-beta_dot'   # Yaw rate
        }
        
        deriv_key = type_map.get(stability_type)

        # Dynamic derivative extraction
        dyn_stab_coefs = [
            "CFx", "CFy", "CFz", "CMx", "CMy", "CMz"
        ]

        for coef in dyn_stab_coefs:
            result_name = f"{coef}{deriv_key}"
            data['dynamic_derivatives'][result_name] = vsp.GetDoubleResults(results_vec[-2], result_name)[0]

    print("\n" + "=" * 60)
    print("Stability Data Extraction Complete")
    print("=" * 60)

    print(data)

    return data

def check_errors(error_mgr):
    """Check for OpenVSP errors and display them."""
    num_errors = error_mgr.GetNumTotalErrors()
    if num_errors > 0:
        print("\n" + "=" * 60)
        print(f"OpenVSP Errors ({num_errors}):")
        print("=" * 60)
        while error_mgr.GetNumTotalErrors() > 0:
            err = error_mgr.PopLastError()  # Get and remove last error
            print(f"  {err.m_ErrorString}")
        return True
    return False


# ==============================================================================
# MAIN EXECUTION
# ==============================================================================

def main(config, filename=None):
    """
    Run complete VSPAERO analysis to get base aero data
    """
    # Suppress output if not verbose
    old_stdout = sys.stdout
    old_stderr = sys.stderr

    # Store all data
    vsp_aero_data = {}

    # Config
    if filename == None:
        filename = "Mach1_Sizing.vsp3"
    vsp_analysis = "VSPAEROSweep"
    geom_analysis = "VSPAEROComputeGeometry"

    vsp_aero_data['filename'] = filename
    vsp_aero_data['vsp_analysis'] = vsp_analysis
    vsp_aero_data['geom_analysis'] = geom_analysis

    # Ref parameters
    if config['model_unit'] == 'in':
        vsp_aero_data['x_rel'] = config['x_rel'] * 12
        vsp_aero_data['b_ref'] = config['wing_span'] * 12
        vsp_aero_data['s_ref'] = config['wing_area'] * 12 * 12
        vsp_aero_data['c_ref'] = config['MAC'] * 12
    elif config['model_unit'] == 'ft':
        vsp_aero_data['x_rel'] = config['x_rel']
        vsp_aero_data['b_ref'] = config['wing_span']
        vsp_aero_data['s_ref'] = config['wing_area']
        vsp_aero_data['c_ref'] = config['MAC']
    elif config['model_unit'] == 'm':
        vsp_aero_data['x_rel'] = config['x_rel'] * 0.3048
        vsp_aero_data['b_ref'] = config['wing_span'] * 0.3048
        vsp_aero_data['s_ref'] = config['wing_area'] * 0.3048 * 0.3048
        vsp_aero_data['c_ref'] = config['MAC'] * 0.3048
    else:
        print("[WARNING]: No unit selected, defaulting to feet.")
        vsp_aero_data['x_rel'] = config['x_rel']
        vsp_aero_data['b_ref'] = config['wing_span']
        vsp_aero_data['s_ref'] = config['wing_area']
        vsp_aero_data['c_ref'] = config['MAC']

    # Flight conditions

    # Mach number
    mach_start = config['mach_start']
    mach_end = config['mach_end']
    mach_points = config['mach_points']

    # Air density [slugs/ft³] and temperature [R] from the standard atmosphere
    # at the analysis altitude
    altitude = config['altitude']
    rho, temp = get_atmosphere(altitude)
    rho = float(np.ravel(rho)[0])
    temp = float(np.ravel(temp)[0])

    # Speed of sound [ft/s]
    sos = np.sqrt(GAMMA * R * temp)

    # Dynamic viscosity [slug/(ft*s)] from the standard atmosphere
    altitude_meters = altitude * 0.3048
    mu = float(np.ravel(Atmosphere(altitude_meters).dynamic_viscosity)[0]) * PASCAL_SECOND_TO_SLUG_PER_FT_S

    # Reynolds number based on reference chord
    re_start = config['mach_start'] * sos * rho * config['MAC'] / mu
    re_end = config['mach_end'] * sos * rho * config['MAC'] / mu
    re_points = config['mach_points']

    # Aoa parms

    # For single point analysis:
    alpha_start = config['alpha_start']     # Angle of attack [deg]
    alpha_points = config['alpha_points']      # Number of alpha points
    alpha_end = config['alpha_end']      # End angle of attack [deg]

    # Beta parms

    # Sideslip angle [deg]
    beta_start = 0.0
    beta_points = 1

    vsp_aero_data['mach_start'] = mach_start
    vsp_aero_data['mach_points'] = mach_points
    vsp_aero_data['rho'] = rho
    vsp_aero_data['re_start'] = re_start
    vsp_aero_data['re_points'] = re_points
    vsp_aero_data['alpha_start'] = alpha_start
    vsp_aero_data['alpha_points'] = alpha_points
    vsp_aero_data['alpha_end'] = alpha_end
    vsp_aero_data['beta_start'] = beta_start
    vsp_aero_data['beta_points'] = beta_points

    # Analysis set
    thin_geom_set = vsp.SET_FIRST_USER  # For VLM: Thin geometry (lifting surfaces)

    vsp_aero_data['thin_geom_set'] = thin_geom_set

    # Wake setting

    # Number of wake iterations
    wake_iter = 10

    # Number of nodes in the wake
    wake_nodes = 20

    # Wake relaxation factor [0-1]
    wake_relax = 0.5

    vsp_aero_data['wake_iter'] = wake_iter
    vsp_aero_data['wake_nodes'] = wake_nodes
    vsp_aero_data['wake_relax'] = wake_relax

    # Stability off

    # Stability calculation type
    stability_type = 0

    vsp_aero_data['stability_type'] = stability_type

    # Convergence parms

    fwrd_convergence = 1.0      # ForwardGMRESConvergenceFactor
    adj_convergence = 1.0      # AdjointGMRESConvergenceFactor
    nonlinear_convergence = 1.0          # NonLinearConvergenceFactor
    size_factor = 1.0               # CoreSizeFactor

    vsp_aero_data['fwrd_convergence'] = fwrd_convergence
    vsp_aero_data['adj_convergence'] = adj_convergence
    vsp_aero_data['nonlinear_convergence'] = nonlinear_convergence
    vsp_aero_data['size_factor'] = size_factor

    stall_model_flag = vsp.STALL_ON

    # 2D airfoil maximum lift coefficient
    cl_max = 0.75

    # 2D lift coefficient at zero angle of attack
    cl0 = 0.0

    vsp_aero_data['stall_model_flag'] = stall_model_flag
    vsp_aero_data['cl_max'] = cl_max
    vsp_aero_data['cl0'] = cl0

    # Symmetry flags

    symmetry = 0  # No flow symmetry - required for CFy, CMx, CMz derivatives

    vsp_aero_data['symmetry'] = symmetry

    # Ref flag
    ref_flag = 0  # 0 = manual, 1 = from wing

    vsp_aero_data['ref_flag'] = ref_flag

    # Reference values (altitude-corrected: v_ref = mach * local speed of sound)

    vsp_aero_data['v_ref'] = mach_start * sos

# Outputs

    try:
        # Initialize and load geometry
        error_mgr = initialize_vsp(vsp_aero_data['filename'])

        # Run CompGeom analysis once (required before VSPAERO, geometry doesn't change)
        run_compgeom_analysis(vsp_aero_data)

        # Check for errors
        if check_errors(error_mgr):
            print("\nErrors occurred during CompGeom. Check geometry and settings.")
            return None

        # Setup VSPAERO inputs with current stability type
        setup_vspaero_inputs(vsp_aero_data=vsp_aero_data)

        # Run VSPAERO analysis
        results_id = run_vspaero_analysis(vsp_aero_data=vsp_aero_data)

        # Check for errors
        if check_errors(error_mgr):
            print(f"\nErrors occurred during VSPAERO analysis.")
            print("Continuing with remaining analyses...")

        # Extract aerodynamic results from VSPAEROSweep
        # CRITICAL: For sweeps, results are stored in ResultsVec, with one entry per sweep point!
        print("\n" + "=" * 60)
        print("Extracting Aerodynamic Results from Sweep")
        print("=" * 60)

        # Get the results vector - each element is one sweep point
        results_vec = vsp.GetStringResults(results_id, "ResultsVec")
        print(f"Number of sweep points: {len(results_vec)}")

        # Initialize arrays to store results
        alpha_data = []
        cl_data = []
        cd_data = []

        # Loop through each sweep point and extract results
        for i in range(vsp_aero_data['alpha_points']):
            # Each results_vec[i] is a result ID for one alpha point
            # GetDoubleResults returns an array, take the last element (final wake iteration)
            alpha_vec = vsp.GetDoubleResults(results_vec[i], "Alpha")
            alpha = alpha_vec[-1] if len(alpha_vec) > 0 else 0.0

            # IMPORTANT: Use CLi and CDi, don't use skin friction here
            cl_vec = vsp.GetDoubleResults(results_vec[i], "CLi")
            cl = cl_vec[-1] if len(cl_vec) > 0 else 0.0

            cd_vec = vsp.GetDoubleResults(results_vec[i], "CDi")
            cd = cd_vec[-1] if len(cd_vec) > 0 else 0.0

            alpha_data.append(alpha)
            cl_data.append(cl)
            cd_data.append(cd)

        print("\n" + "=" * 60)
        print("Alpha Sweep Results")
        print("=" * 60)
        print(f"{'Alpha (deg)':<12} {'CL':<10} {'CD':<10} {'L/D':<10}")
        print("-" * 78)

        for i in range(len(alpha_data)):
            ld_ratio = cl_data[i] / cd_data[i] if cd_data[i] != 0 else 0
            print(f"{alpha_data[i]:<12.2f} {cl_data[i]:<10.4f} {cd_data[i]:<10.5f} "
                  f" {ld_ratio:<10.2f}")

        # Store results in dictionary for return
        results_data = {
            'alpha': alpha_data,
            'CL': cl_data,
            'CD': cd_data
        }

        print("\n" + "=" * 60)
        print("Analysis Complete - Results Extracted Successfully")
        print("=" * 60)

        # Construct drag polar function
        # drag_polar = interp1d(cl_data, cd_data, kind='quadratic', fill_value="extrapolate")

        return results_data

    except Exception as e:
        print(f"\nError occurred: {e}")
        import traceback
        traceback.print_exc()
        return None
    finally:
        sys.stdout = old_stdout
        sys.stderr = old_stderr


if __name__ == "__main__":
    config={}
    main(config)