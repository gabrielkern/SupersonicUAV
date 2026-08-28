#!/usr/bin/env python3
"""
Variable Plane Generator for OpenVSP
==========================================
All dimensions are in feet. Dimensionality only matters for skin friction
"""

import sys
import os
from typing import Dict, Optional
import numpy as np

# Add OpenVSP Python path
sys.path.append('/Users/gabrielkern/Documents/Python/AgenticDesigner/OpenVSP-3.45.4-MacOS/python/openvsp')

import openvsp as vsp

def initialize_vsp():
    """Initialize OpenVSP and clear any existing geometry."""
    print("=" * 60)
    print("Initializing OpenVSP...")
    print("=" * 60)

    # Initialize OpenVSP
    vsp.VSPCheckSetup()
    vsp.VSPRenew()

    # Get version info
    version = vsp.GetVSPVersion()
    print(f"OpenVSP Version: {version}")

    return vsp.ErrorMgrSingleton.getInstance()

def create_wing(config):
    """
    Create the main wing.
    """
    print("\n" + "=" * 60)
    print("Creating Main Wing...")
    print("=" * 60)

    # Add wing geometry
    wing_id = vsp.AddGeom("WING")
    vsp.SetGeomName(wing_id, "Main_Wing")

    # Set absolute positioning
    vsp.SetParmVal(wing_id, "X_Rel_Location", "XForm", config['x_wing'])
    vsp.SetParmVal(wing_id, "Y_Rel_Location", "XForm", 0.0)
    vsp.SetParmVal(wing_id, "Z_Rel_Location", "XForm", 0.0)

    # Set rotation
    vsp.SetParmVal(wing_id, "X_Rel_Rotation", "XForm", 0.0)
    vsp.SetParmVal(wing_id, "Y_Rel_Rotation", "XForm", 0.0)
    vsp.SetParmVal(wing_id, "Z_Rel_Rotation", "XForm", 0.0)

    # Set symmetry to Y-planar (XZ plane symmetry = 2)
    vsp.SetParmVal(wing_id, "Sym_Planar_Flag", "Sym", vsp.SYM_XZ)

    # Get wing cross-section surface
    xsurf_id = vsp.GetXSecSurf(wing_id, 0)

    # Inboard properties
    vsp.SetParmVal(wing_id, "Root_Chord", "XSec_1", config['root_chord'])
    vsp.SetParmVal(wing_id, "Tip_Chord", "XSec_1", config['middle_chord'])
    vsp.SetParmVal(wing_id, "Span", "XSec_1", config['inboard_span'])
    vsp.SetParmVal(wing_id, "Sweep", "XSec_1", config['le_sweep'])
    vsp.SetParmVal(wing_id, "Sec_Sweep_Location", "XSec_1", 0.5)
    vsp.SetParmVal(wing_id, "Sweep_Location", "XSec_1", 0.0)
    vsp.SetParmVal(wing_id, "SectTess_U", "XSec_1", 6)
    vsp.SetParmVal(wing_id, "InCluster", "XSec_1", 1.0)
    vsp.SetParmVal(wing_id, "OutCluster", "XSec_1", 1.0)

    vsp.InsertXSec(wing_id, 1, vsp.XS_SIX_SERIES)
    vsp.SetParmVal(wing_id, "Root_Chord", "XSec_2", config['middle_chord'])
    vsp.SetParmVal(wing_id, "Tip_Chord", "XSec_2", config['tip_chord'])
    vsp.SetParmVal(wing_id, "Span", "XSec_2", config['outboard_span'])
    vsp.SetParmVal(wing_id, "Sweep", "XSec_2", config['le_sweep'])
    vsp.SetParmVal(wing_id, "Sec_Sweep_Location", "XSec_2", 0.5)
    vsp.SetParmVal(wing_id, "Sweep_Location", "XSec_2", 0.0)
    vsp.SetParmVal(wing_id, "SectTess_U", "XSec_2", 8)
    vsp.SetParmVal(wing_id, "InCluster", "XSec_2", 1.0)
    vsp.SetParmVal(wing_id, "OutCluster", "XSec_2", 1.0)
    vsp.SetParmVal(wing_id, "Dihedral", "XSec_2", -2.0)

    # Apply airfoil (file-based or CST)
    num_xsec = vsp.GetNumXSec(xsurf_id)
    print(f"DEBUG create_wing: Applying airfoil to {num_xsec} cross-sections")
    for i in range(num_xsec):
        vsp.ChangeXSecShape(xsurf_id, i, vsp.XS_SIX_SERIES)
        xsec_id = vsp.GetXSec(xsurf_id, i)
        vsp.SetParmVal(vsp.GetXSecParm(xsec_id, "Series"), 0.0)
        vsp.SetParmVal(vsp.GetXSecParm(xsec_id, "ThickChord"), config['thickness'])

    # Set tessellation parameters
    vsp.SetParmVal(wing_id, "Tess_W", "Shape", 13)
    vsp.SetParmVal(wing_id, "LECluster", "WingGeom", 1.0)
    vsp.SetParmVal(wing_id, "TECluster", "WingGeom", 1.0)

    # Set parasitic drag propertiess
    vsp.SetParmVal(vsp.FindParm(wing_id, "PercLam", "ParasiteDragProps"), 0.0)
    vsp.SetParmVal(vsp.FindParm(wing_id, "Q", "ParasiteDragProps"), 1.0)
    vsp.SetParmVal(vsp.FindParm(wing_id, "FFWingEqnType", "ParasiteDragProps"), 9) 

    print(f"Wing created with ID: {wing_id}")
    return wing_id

def create_fuselage(config):
    """
    Create the fuselage with multiple cross-sections.

    Fuselage specs:
    - Size determined by base and multiplier
    - 4 XSec stations:
      - Station 0: Circle (nose), 2 inch diameter
      - Station 1: Circle (main body start), 3*multiplier inch diameter
      - Station 2: Circle (main body mid), 3*multiplier inch diameter
      - Station 3: Circle (tail), 2 inch diameter
    """
    print("\n" + "=" * 60)
    print("Creating Fuselage...")
    print("=" * 60)

    # Add fuselage with Stack
    fuse_id = vsp.AddGeom("STACK")
    vsp.SetGeomName(fuse_id, "Fuselage")

    # Set position
    vsp.SetParmVal(fuse_id, "X_Rel_Location", "XForm", 0.0)
    vsp.SetParmVal(fuse_id, "Y_Rel_Location", "XForm", 0.0)
    vsp.SetParmVal(fuse_id, "Z_Rel_Location", "XForm", 0.0)

    # Set rotation to make it vertical (90° about X-axis)
    vsp.SetParmVal(fuse_id, "X_Rel_Rotation", "XForm", 0.0)
    vsp.SetParmVal(fuse_id, "Y_Rel_Rotation", "XForm", 0.0)
    vsp.SetParmVal(fuse_id, "Z_Rel_Rotation", "XForm", 0.0)

    # Set no symmetry
    vsp.SetParmVal(fuse_id, "Sym_Planar_Flag", "Sym", vsp.SYM_NONE)

    # Get cross-section surface
    xsurf_id = vsp.GetXSecSurf(fuse_id, 0)

    # Remove all but the first and last XSecs
    x_sec_surf_id = vsp.GetXSecSurf(fuse_id, 0)
    for i in range(vsp.GetNumXSec(x_sec_surf_id)-1,1,-1):
        vsp.CutXSec(fuse_id,i)

    # Station 0 (Nose) - Point
    vsp.ChangeXSecShape(xsurf_id, 0, vsp.XS_POINT)
    xsec_0 = vsp.GetXSec(xsurf_id, 0)

    vsp.SetParmVal(vsp.GetXSecParm(xsec_0, "AllSym"), 1)
    vsp.SetParmVal(vsp.GetXSecParm(xsec_0, "TopLAngle"), 15.0)

    # Station 1 (Main Body Start) - Circle
    vsp.ChangeXSecShape(xsurf_id, 1, vsp.XS_CIRCLE)
    xsec_1 = vsp.GetXSec(xsurf_id, 1)
    vsp.SetParmVal(vsp.GetXSecParm(xsec_1, "Circle_Diameter"), config['diameter'])
    vsp.SetParmVal(vsp.GetXSecParm(xsec_1, "XDelta"), config['body_length'] * (1/2.25))
    vsp.SetParmVal(vsp.GetXSecParm(xsec_1, "SectTess_U"), 6)

    vsp.SetParmVal(vsp.GetXSecParm(xsec_1, "AllSym"), 1)
    vsp.SetParmVal(vsp.GetXSecParm(xsec_1, "TopLAngle"), 0.0)

    # Station 2 (Main Body Mid) - Circle (same as Station 1)
    vsp.InsertXSec(fuse_id, 1, vsp.XS_CIRCLE)
    xsec_2 = vsp.GetXSec(xsurf_id, 2)
    vsp.SetParmVal(vsp.GetXSecParm(xsec_2, "Circle_Diameter"), config['diameter'])
    vsp.SetParmVal(vsp.GetXSecParm(xsec_2, "XDelta"), config['body_length'] * (0.75/2.25))
    vsp.SetParmVal(vsp.GetXSecParm(xsec_2, "SectTess_U"), 6)

    vsp.SetParmVal(vsp.GetXSecParm(xsec_2, "AllSym"), 1)
    vsp.SetParmVal(vsp.GetXSecParm(xsec_2, "TopLAngle"), 0.0)

    # Station 3 (Tail) - Point
    vsp.InsertXSec(fuse_id, 2, vsp.XS_POINT)
    xsec_3 = vsp.GetXSec(xsurf_id, 3)
    vsp.SetParmVal(vsp.GetXSecParm(xsec_3, "XDelta"), config['body_length'] * (0.5/2.25))
    vsp.SetParmVal(vsp.GetXSecParm(xsec_3, "AllSym"), 1)
    vsp.SetParmVal(vsp.GetXSecParm(xsec_3, "TopLAngle"), -15.0)

    # Set tessellation
    vsp.SetParmVal(fuse_id, "Tess_W", "Shape", 17)

    # Set parasitic drag properties
    vsp.SetParmVal(vsp.FindParm(fuse_id, "PercLam", "ParasiteDragProps"), 0.0)
    vsp.SetParmVal(vsp.FindParm(fuse_id, "Q", "ParasiteDragProps"), 1.1)
    vsp.SetParmVal(vsp.FindParm(fuse_id, "FFBodyEqnType", "ParasiteDragProps"), 3) 

    print(f"Fuselage created with ID: {fuse_id}")
    return fuse_id

def create_sets(wing_id, fuse_id):
    """Create the different sets that can be used for analysis."""
    print("\n" + "=" * 60)
    print("Creating Sets...")
    print("=" * 60)
    vsp.SetSetFlag(wing_id, vsp.SET_FIRST_USER, True)
    vsp.SetSetName(vsp.SET_FIRST_USER, "VLM_Surfs")

    vsp.SetSetFlag(wing_id, vsp.SET_FIRST_USER + 1, True)
    vsp.SetSetFlag(fuse_id, vsp.SET_FIRST_USER + 1, True)
    vsp.SetSetName(vsp.SET_FIRST_USER + 1, "Para_Drag_Surfs")

def export_model(filename="Mach1_Sizing.vsp3"):
    """Export the model to a .vsp3 file."""
    print("\n" + "=" * 60)
    print("Exporting Model...")
    print("=" * 60)

    try:
        vsp.WriteVSPFile(filename)
        print(f"Model successfully exported to: {filename}")
        return True
    except Exception as e:
        print(f"Error exporting model: {e}")
        return False

def main(config):
    """Main function to create the stability test plane."""
    print("=" * 60)
    print("Mach 1 UAV Test Plane Generator")
    print("=" * 60)

    # Initialize OpenVSP (suppress output)
    vsp.VSPCheckSetup()
    vsp.VSPRenew()
    error_mgr = vsp.ErrorMgrSingleton.getInstance()

    try:
        # Create all aircraft components (all these functions print, but we're not calling them with verbose)
        wing_id = create_wing(config)
        fuse_id = create_fuselage(config)

        # Create sets for analysiss
        create_sets(wing_id,fuse_id)

        # Update geometrys
        vsp.Update()

        # Export model
        _ = export_model()

    except Exception as e:
        raise