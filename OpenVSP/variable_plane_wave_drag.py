#!/usr/bin/env python3
"""
Wave Drag (Area-Rule) Analysis Runner for OpenVSP
==================================================

This script loads a .vsp3 file (from create_base_Mach1UAV.py) and runs
OpenVSP's built-in "WaveDrag" analysis -- the Eminton-Lord/supersonic-area-rule
method (Mach-angle oblique cutting planes, rolled through azimuth, fit via a
Fourier series) -- to compute zero-lift wave drag due to volume for the whole
combined wing+fuselage geometry.

This is a SUPERSONIC-linear-theory method: the Mach angle mu = arcsin(1/M) is
undefined for M <= 1, so this analysis should only be called at Mach numbers
comfortably above 1 (see wave_drag.py's suave_corrected_wave_drag(), which
uses this module's result only as the supersonic anchor of a transonic-gap
fairing curve, never evaluating it near or below M=1). See
wiki/concepts/full-configuration-transonic-wave-drag-estimation.md and
wiki/sources/suave-transonic-wave-drag-bridging-implementation.md in this
repository for the full background on why, and on the SUAVE-style bridging
this feeds into.

Mirrors the structure of variable_plane_analysis.py / variable_plane_parasitic.py
in this same directory: same initialize_vsp()/check_errors() helpers, same
main(config, filename=None) entry point and model_unit handling.
"""

import sys

ANALYSIS_TYPE = "WaveDrag"

# Add OpenVSP Python path
sys.path.append('/Users/gabrielkern/Documents/Python/AgenticDesigner/OpenVSP-3.45.4-MacOS/python/openvsp')

import openvsp as vsp


def initialize_vsp(filename):
    """Initialize OpenVSP and load the geometry file."""
    print("=" * 60)
    print("Initializing Wave Drag Analysis")
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


def run_wave_drag_analysis(config, mach, num_slices=20, num_rots=10):
    """Run OpenVSP's WaveDrag (supersonic area-rule) analysis at a single Mach number.

    mach should be > 1 (the Mach angle used internally is undefined at M<=1
    -- see the module docstring). num_slices/num_rots match the OpenVSP
    Wave Drag tool's own defaults (20 area slices per cutting-plane rotation,
    10 rotations sampling the Mach cone azimuth) and SUAVE's identical
    defaults for the same analysis (see
    wiki/sources/suave-transonic-wave-drag-bridging-implementation.md).

    Returns CD_wave, non-dimensionalized against config['wing_area'] (in
    config['model_unit']-consistent units, matching variable_plane_parasitic.py's
    Sref convention) rather than whatever internal reference area OpenVSP's
    WaveDrag analysis defaults to.
    """
    print("\n" + "=" * 60)
    print("Running Wave Drag Analysis")
    print("=" * 60)

    if config['model_unit'] == 'in':
        s_ref = config['wing_area'] * 12 * 12
    elif config['model_unit'] == 'ft':
        s_ref = config['wing_area']
    elif config['model_unit'] == 'm':
        s_ref = config['wing_area'] * 0.3048 * 0.3048
    else:
        print("[WARNING]: No unit selected, defaulting to feet.")
        s_ref = config['wing_area']

    vsp.SetAnalysisInputDefaults(ANALYSIS_TYPE)

    input_names = vsp.GetAnalysisInputNames(ANALYSIS_TYPE)
    print("Available inputs for WaveDrag:")
    for name in input_names:
        print(f"  - {name}")

    # Geometry set: use the same combined wing+fuselage set already defined
    # for parasitic drag in create_base_Mach1UAV.py (SET_FIRST_USER + 1,
    # "Para_Drag_Surfs") -- wave/area-rule drag needs the whole external
    # shape's combined cross-sectional area distribution, the same
    # requirement parasitic drag has, not just the thin VLM lifting
    # surfaces. Set defensively: only if the analysis actually exposes a
    # geometry-set input under one of these names (this codebase's own
    # ParasiteDrag analysis uses "GeomSet"; verify against the printed
    # input list above if WaveDrag turns out to use a different name).
    geom_set_input_candidates = ["GeomSet", "Set"]
    geom_set_input_name = next((n for n in geom_set_input_candidates if n in input_names), None)
    if geom_set_input_name is not None:
        vsp.SetIntAnalysisInput(ANALYSIS_TYPE, geom_set_input_name, [vsp.SET_FIRST_USER + 1], 0)
    else:
        print(f"[WARNING] None of {geom_set_input_candidates} found in WaveDrag's inputs "
              f"(see the printed list above) -- leaving geometry set at its analysis "
              f"default rather than guessing an unverified input name.")

    # Mach number (single point -- WaveDrag is not swept the way VSPAEROSweep is)
    vsp.SetDoubleAnalysisInput(ANALYSIS_TYPE, "Mach", [mach], 0)

    # Area-distribution discretization
    vsp.SetIntAnalysisInput(ANALYSIS_TYPE, "NumSlices", [num_slices], 0)
    vsp.SetIntAnalysisInput(ANALYSIS_TYPE, "NumRotSects", [num_rots], 0)
    vsp.SetIntAnalysisInput(ANALYSIS_TYPE, "UseModeFlag", [1], 0)

    print("\nWave Drag Analysis Inputs:")
    vsp.PrintAnalysisInputs(ANALYSIS_TYPE)

    print("\nExecuting Wave Drag Analysis...")
    wavedrag_resid = vsp.ExecAnalysis(ANALYSIS_TYPE)
    print("Wave Drag Analysis COMPLETE")

    cd_wave = vsp.GetDoubleResults(wavedrag_resid, "CDWave")[0]

    print(f"CDWave (Sref={s_ref}): {cd_wave}")

    return cd_wave


def check_errors(error_mgr):
    """Check for OpenVSP errors and display them."""
    num_errors = error_mgr.GetNumTotalErrors()
    if num_errors > 0:
        print("\n" + "=" * 60)
        print(f"OpenVSP Errors ({num_errors}):")
        print("=" * 60)
        while error_mgr.GetNumTotalErrors() > 0:
            err = error_mgr.PopLastError()
            print(f"  {err.m_ErrorString}")
        return True
    return False


def main(config, mach, filename=None, num_slices=20, num_rots=10):
    """Run complete Wave Drag analysis. mach should be > 1 (see module docstring)."""
    old_stdout = sys.stdout
    old_stderr = sys.stderr

    if filename is None:
        filename = "Mach1_Sizing.vsp3"

    try:
        error_mgr = initialize_vsp(filename)

        cd_wave = run_wave_drag_analysis(config, mach, num_slices=num_slices, num_rots=num_rots)

        if check_errors(error_mgr):
            print("Errors found.")
            return None

        return cd_wave

    except Exception:
        return None
    finally:
        sys.stdout = old_stdout
        sys.stderr = old_stderr


if __name__ == "__main__":
    filename = "/Users/gabrielkern/Documents/hypersonics/supersonicUAV/OpenVSP/OpenVSPConceptualDesign/Mach1UAV_V2.vsp3"
    config = {'wing_area': 3.2, 'model_unit': 'in'}
    result = main(config, mach=1.2, filename=filename)
