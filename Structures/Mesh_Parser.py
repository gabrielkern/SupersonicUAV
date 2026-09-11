import numpy as np
import matplotlib.pyplot as plt

def load_degen_geom(filepath):
    with open(filepath, 'r') as f:
        lines = f.readlines()

    components = {}

    for i, line in enumerate(lines):
        line = line.strip()
        if line.startswith('BODY,') or line.startswith('SURFACE,') or line.startswith('LIFTING_SURFACE,'):
            comp_name = line.split(',')[1]

            for j in range(i + 1, len(lines)):
                inner_line = lines[j].strip()

                if inner_line.startswith('BODY,') or inner_line.startswith('SURFACE,') or inner_line.startswith('LIFTING_SURFACE,'):
                    break  # no surface block for this component

                elif inner_line.startswith('SURFACE_NODE'):
                    parts = inner_line.split(',')
                    n_xsecs = int(parts[1])
                    n_pnts = int(parts[2])
                    data_start = j + 2  # skip the x,y,z,u,w comment line (annoying ah line)

                    data_lines = lines[data_start : data_start + n_xsecs * n_pnts]
                    rows = [[float(v) for v in row.split(',')] for row in data_lines]
                    arr = np.array(rows).reshape(n_xsecs, n_pnts, 5)

                    xyz = arr[:, :, 0:3]

                    # handle duplicate names (symmetric left/right halves)
                    key = comp_name
                    suffix = 1
                    while key in components:
                        suffix += 1
                        key = f"{comp_name}_{suffix}"

                    # station spacing: distance between consecutive ring centroids
                    centroids = xyz.mean(axis=1)
                    station_spacing = np.linalg.norm(np.diff(centroids, axis=0), axis=1) # pain in the ah to wrap my head around

                    # panel widths: distance between consecutive perimeter points, per station
                    diffs = np.diff(xyz, axis=1)
                    panel_widths = np.linalg.norm(diffs, axis=2)

                    components[key] = {
                        'xyz': xyz,
                        'station_spacing': station_spacing,
                        'panel_widths': panel_widths,
                    }

                    break

    return components


comps = load_degen_geom("./SupersonicUAV/Structures/Mach1UAV_V2_DegenGeom.csv")

fig = plt.figure(figsize=(14, 8))
ax = fig.add_subplot(111, projection='3d')

colors = plt.cm.tab10(np.linspace(0, 1, len(comps)))

for (name, data), color in zip(comps.items(), colors):
    xyz = data['xyz']  # (nXsecs, nPnts, 3)

    # plot each cross-section ring
    for station in range(xyz.shape[0]):
        ring = xyz[station]
        ring_closed = np.vstack([ring, ring[0]])  # close the loop
        ax.plot(ring_closed[:, 0], ring_closed[:, 1], ring_closed[:, 2],
                 color=color, linewidth=0.6)

    # label the component once, at its first station
    ax.text(xyz[0, 0, 0], xyz[0, 0, 1], xyz[0, 0, 2], name, color=color, fontsize=8)

ax.set_xlabel('X (fore-aft)')
ax.set_ylabel('Y (spanwise)')
ax.set_zlabel('Z (vertical)')
ax.set_title('DegenGeom Reconstruction - Cross Sections by Component')

# equal aspect ratio so the geometry isn't visually distorted (shoutout claude on this)
all_pts = np.vstack([d['xyz'].reshape(-1, 3) for d in comps.values()])
max_range = (all_pts.max(axis=0) - all_pts.min(axis=0)).max() / 2.0
mid = all_pts.mean(axis=0)
ax.set_xlim(mid[0]-max_range, mid[0]+max_range)
ax.set_ylim(mid[1]-max_range, mid[1]+max_range)
ax.set_zlim(mid[2]-max_range, mid[2]+max_range)
ax.view_init(elev=25, azim=-50)

plt.tight_layout()
plt.show()