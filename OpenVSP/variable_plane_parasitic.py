#!/usr/bin/env python3
"""
MassProp Analysis Runner for OpenVSP
====================================

This script loads a .vsp3 file (output from create_variable_plane.py)
and runs MassProp analysis to compute aerodynamic coefficients and stability derivatives.
"""

import sys
import os

# Add OpenVSP Python path
sys.path.append('/Users/gabrielkern/Documents/Python/AgenticDesigner/OpenVSP-3.45.4-MacOS/python/openvsp')

import openvsp as vsp

VSP_FILE = "Mach1_Sizing.vsp3"
ANALYSIS_TYPE = "ParasiteDrag"

def initialize_vsp():
    """Initialize OpenVSP and load the geometry file."""
    print("=" * 60)
    print("Initializing VSPAERO Analysis")
    print("=" * 60)

    vsp.VSPCheckSetup()  # Initialize VSP system
    vsp.VSPRenew()  # Clear existing model

    version = vsp.GetVSPVersion()  # Get VSP version string
    print(f"OpenVSP Version: {version}")

    print(f"\nLoading file: {VSP_FILE}")
    vsp.ReadVSPFile(VSP_FILE)  # Load .vsp3 geometry file
    vsp.Update()  # Update all geometry
    print("File loaded successfully")

    return vsp.ErrorMgrSingleton.getInstance()  # Return error manager for checking

def run_parasitic_analysis(config):
    """Run parasitic drag analysis to obtain rough estimates of parasitic drag."""
    print("\n" + "=" * 60)
    print("Running Parasitic Drag Analysis")
    print("=" * 60)

    vsp.PrintAnalysisInputs(ANALYSIS_TYPE)

    # Set full payload as set
    vsp.SetIntAnalysisInput(ANALYSIS_TYPE, "GeomSet", [vsp.SET_FIRST_USER + 1], 0)

    # Set unit to feet
    vsp.SetIntAnalysisInput(ANALYSIS_TYPE, "LengthUnit", [4], 0)

    # Set laminar Cf eqn
    vsp.SetIntAnalysisInput(ANALYSIS_TYPE, "LamCfEqnChoice", [0], 0)

    # Set turbulent Cf eqn
    vsp.SetIntAnalysisInput(ANALYSIS_TYPE, "TurbCfEqnChoice", [1], 0)

    # Set reference area
    vsp.SetDoubleAnalysisInput(ANALYSIS_TYPE, "Sref", [config['wing_area']], 0)
    
    # Set atmosphere conditions
    vsp.SetDoubleAnalysisInput(ANALYSIS_TYPE, "Vinf", [config['mach_start']], 0)

    # mach
    vsp.SetIntAnalysisInput(ANALYSIS_TYPE, "VelocityUnit", [6], 0)

    # 0 ft of altitude
    vsp.SetDoubleAnalysisInput(ANALYSIS_TYPE, "Altitude", [0], 0)
    vsp.SetIntAnalysisInput(ANALYSIS_TYPE, "AltLengthUnit", [0], 0)

    print("\nParasitic Drag Analysis Inputs:")
    vsp.PrintAnalysisInputs(ANALYSIS_TYPE)  # Display configured inputs

    print("\nExecuting Parasitic Drag Analysiis...")
    paradrag_resid = vsp.ExecAnalysis(ANALYSIS_TYPE)  # Execute analysis
    print("Parasitic Drag Analysis COMPLETE")

    total_cd = vsp.GetDoubleResults(paradrag_resid, "Total_CD_Total")
    print(f"TOTAL CD: {total_cd[0]}")
    # vsp.PrintResults(paradrag_resid)  # Display results

    return total_cd[0]

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

def main(config):
    """Run complete Parasitic Drag analysis."""
    # Suppress all output by default
    old_stdout = sys.stdout
    old_stderr = sys.stderr

    try:
        # Initialize and load geometry
        error_mgr = initialize_vsp()
        # Run massprop analysis
        total_drag = run_parasitic_analysis(config)

        # Check for errors
        if check_errors(error_mgr):
            return None

        return total_drag

    except Exception as e:
        return None
    finally:
        # Restore stdout/stderr
        sys.stdout = old_stdout
        sys.stderr = old_stderr


if __name__ == "__main__":
    config = {'wing_area':1,'mach_start':0.7}
    return_val = main(config)
    print(return_val)