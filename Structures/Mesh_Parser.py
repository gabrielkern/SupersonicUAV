import numpy as np
import matplotlib.pyplot as plt

def load_degen_geom(filepath):
    with open(filepath, 'r') as f:
        lines = f.readlines()

    for i, line in enumerate(lines):
        line = line.strip()
        if line.startswith('BODY,') or line.startswith('SURFACE,') or line.startswith('LIFTING_SURFACE,'):
            comp_name = line.split(',')[1]
            print(f"Found component '{comp_name}' at line {i}")

            for j in range(i + 1, len(lines)):
                inner_line = lines[j].strip()
                if line.startswith('BODY,') or line.startswith('SURFACE,') or line.startswith('LIFTING_SURFACE,'):
                    note = 0 # no surface block
                    break
                elif line.startswith('SURFACE_NODE'):
                    parts = line.split(',')
                    n_xsecs = int(parts[1])   # '46' -> 46
                    n_pnts = int(parts[2])    # '25' -> 25




comps = load_degen_geom("./SupersonicUAV/Structures/Mach1UAV_V2_DegenGeom.csv")

# 1. Pull one station's ring of perimeter points for your fuselage
fuse = comps['Fuse']
station_10 = fuse['xyz'][10]        # shape (25, 3) — one cross-section, 25 points around it
print(station_10)

# 2. Panel segment lengths within one ring (perimeter point i -> i+1)
diffs = np.diff(station_10, axis=0)          # (24, 3)
panel_widths = np.linalg.norm(diffs, axis=1)  # (24,)
print(panel_widths)

# 3. Axial spacing between consecutive stations (station-to-station "length")
centroids = fuse['xyz'].mean(axis=1)          # (46, 3) — rough centroid per station
axial_spacing = np.linalg.norm(np.diff(centroids, axis=0), axis=1)  # (45,)
print(axial_spacing)

# 4. Sanity check on the nose-tip degeneracy we flagged earlier
print("Station 0 (nose) point spread:", fuse['xyz'][0].ptp(axis=0))  # should be ~0