import numpy as np
import matplotlib.pyplot as plt

# flight vehcle structures E bruhn
# modulus weighted centroids
# treat as isotropic
# principal strains, use the strains to get sresses
#.002 strain roundabout for failure
# strain failure would be good to use instaead of stress

######################################################

# Material Properties guess
E1 = 20 *10**6 #psi
E2 = 20 *10**6 #psi
G12 = 0.6 *10**6 #psi
v12 = 0.3 
v21 = v12 * E2 / E1

# Material faulire loads guess
Xt = 200e3
Xc = 150e3
Yt = Xt
Yc = Xc
S12 = 14e3

# Maximum strain criterion
eps_max = 0.01  # 2% elongation

# Applied loads
Shear = 8
Moment = 0

# Ply List
t = 0.02  # inches, ply thickness
theta_groups = [
    [0,45,0]   # Group 1: Fuselage
]

# Diamter
D = 0.3 #inches
size = "full"

######################################################
thick = len(theta_groups[0])*t
r = D/2-thick
R = D/2
thick = len(theta_groups[0])*t
Inertia = np.pi/64 * ( D**4 - (D-thick*2)**4 )
Q_area = 2/3 * (R**3 - r**3)
if size == "half":      
    A = np.pi/2 * (R**2 - r**2)

    ybar = (4/(3*np.pi)) * (R**3 - r**3)/(R**2 - r**2)

    I_base = np.pi/64 * (D**4 - (D-thick*2)**4)

    Inertia = I_base - A*ybar**2

    Q_area = (R**3 - r**3)/3
    Q_areas = 2/3*(R**3-r**3) + np.pi*ybar/2 * (R**2-r**2)

elif size == "full":
    Inertia = np.pi/64 * ( D**4 - (D-thick*2)**4 )
    Q_area = 2/3 * (R**3 - r**3)   

q = Shear*Q_area / Inertia
Nxy = q                         # shear flow is the Nxy
sigma_m = Moment*(D/2)/2 / Inertia
Nx = -sigma_m*thick         # uses compression

# Each NM vector = [Nx, Ny, Nxy, Mx, My, Mxy]
NM_groups = [
    np.array([Nx, 0, Nxy, 0, 0, 0])    # shear and bending
]

def transform_Q(Q, theta_deg):
    theta = np.radians(theta_deg)
    m = np.cos(theta)
    n = np.sin(theta)
    
    Q11 = Q[0,0]; Q12 = Q[0,1]; Q22 = Q[1,1]; Q66 = Q[2,2]
    
    Qbar11 = Q11*m**4 + 2*(Q12+2*Q66)*m**2*n**2 + Q22*n**4 #
    Qbar12 = (Q11 + Q22 - 4*Q66)*m**2*n**2 + Q12*(m**4 + n**4) #
    Qbar22 = Q11*n**4 + 2*(Q12+2*Q66)*m**2*n**2 + Q22*m**4 #
    Qbar16 = (Q11 - Q12 - 2*Q66)*m**3*n + (Q12 - Q22 + 2*Q66)*m*n**3 #
    Qbar26 = (Q11 - Q12 - 2*Q66)*m*n**3 + (Q12 - Q22 + 2*Q66)*m**3*n #
    Qbar66 = (Q11 + Q22 - 2*Q12 - 2*Q66)*m**2*n**2 + Q66*(m**4 + n**4) #
    
    Qbar = np.array([
        [Qbar11, Qbar12, Qbar16],
        [Qbar12, Qbar22, Qbar26],
        [Qbar16, Qbar26, Qbar66]
    ])
    
    return Qbar

def global_to_material_strain(eps_global, theta_deg):
    theta = np.radians(theta_deg)
    m = np.cos(theta)
    n = np.sin(theta)
    
    T = np.array([
        [m**2, n**2, m*n],
        [n**2, m**2, -m*n],
        [-2*m*n, 2*m*n, m**2 - n**2]
    ])
    eps_material = T @ eps_global
    return eps_material

def tsai_wu_lambda(s1, s2, t12, XT, XC, YT, YC, S):
    F1  = 1/XT - 1/XC
    F11 = 1/(XT*XC)

    F2  = 1/YT - 1/YC
    F22 = 1/(YT*YC)

    F66 = 1/S**2
    F12 = -1/(2*np.sqrt(XT*XC*YT*YC))

    A = F1*s1 + F2*s2
    B = F11*s1**2 + F22*s2**2 + F66*t12**2 + 2*F12*s1*s2

    roots = np.roots([B, A, -1])
    roots = roots[np.isreal(roots)]
    roots = roots[roots > 0]
    if len(roots) == 0:
        return np.inf
    else:
        return np.min(roots)

delta = 1-v12*v21
S = np.array([[1/E1 , -v12/E1 , 0 ], [ -v21/E2 , 1/E2 , 0], [0 , 0 , 1/G12]])
Q = np.array([[E1/delta,v12*E2/delta,0],[v12*E2/delta,E2/delta,0],[0,0,G12]])

ply_stresses = []
lam_list_groups = []
failure_modes_groups = []


for group_idx, theta_list in enumerate(theta_groups):
    print(f"\nProcessing Group {group_idx+1}")

    N_group = len(theta_list)
    total_thickness = N_group * t
    z_group = [-total_thickness/2 + i*t for i in range(N_group+1)]
    
    # Initialize ABD matrices
    A = np.zeros((3,3))
    B = np.zeros((3,3))
    D = np.zeros((3,3))
    
    # Compute Qbar, A, B, D
    for k in range(N_group):
        theta = theta_list[k]
        Qbar = transform_Q(Q, theta)
        delta_z = z_group[k+1] - z_group[k]
        A += Qbar * delta_z
        B += 0.5 * Qbar * (z_group[k+1]**2 - z_group[k]**2)
        D += (1/3) * Qbar * (z_group[k+1]**3 - z_group[k]**3)
    
    ABD = np.block([[A,B],[B,D]])
    NM = NM_groups[group_idx]
    strain_curvature = np.linalg.inv(ABD) @ NM
    
    eps0 = strain_curvature[:3]
    kappa = strain_curvature[3:]
    
    lambdas = []
    failure_modes = []

    for k in range(N_group):
        z_mid = (z_group[k] + z_group[k+1])/2
        eps_ply_global = eps0 + z_mid * kappa
        eps_mat = global_to_material_strain(eps_ply_global, theta_list[k])
        sigma = Q @ eps_mat
        ply_stresses.append(sigma)
        
        s1, s2, t12 = sigma
        lam_tsai_wu = tsai_wu_lambda(s1, s2, t12, Xt, Xc, Yt, Yc, S12)

        # Strain-based failure
        eps1, eps2, gamma12 = eps_mat
        lam_strain = np.inf
        mode = "None"

        if abs(eps1) >= eps_max:
            lam_strain = eps_max / abs(eps1)
            mode = "Strain"

        if abs(eps2) >= eps_max:
            lam_strain2 = eps_max / abs(eps2)
            if lam_strain2 < lam_strain:
                lam_strain = lam_strain2
                mode = "Strain"

        # Determine final lambda and failure mode
        if lam_tsai_wu < lam_strain:
            lam = lam_tsai_wu
            mode = "Tsai-Wu"
        else:
            lam = lam_strain

        lambdas.append(lam)
        failure_modes.append(mode)
        print(f"Ply {k+1:2d}  θ={theta_list[k]:>4}°   λ = {lam:8.3f}   Mode: {mode}")
    
    lam_list_groups.append(np.array(lambdas))
    failure_modes_groups.append(failure_modes)

######################################################
# FIRST PLY FAILURE
all_theta = [theta for group in theta_groups for theta in group]
all_lambdas = np.concatenate(lam_list_groups)
all_modes = [mode for group in failure_modes_groups for mode in group]

fp = np.argmin(all_lambdas)

print("\nFIRST PLY FAILURE")
print("----------------")
print("Ply", fp+1, "at", all_theta[fp], "deg")
print("Load factor =", all_lambdas[fp])
print("Failure mode:", all_modes[fp])

# Determine which NM group this ply came from
group_idx = 0
ply_count = 0
for idx, group in enumerate(theta_groups):
    if fp < ply_count + len(group):
        group_idx = idx
        break
    ply_count += len(group)

NM_fail = NM_groups[group_idx]
print("Actual loads at failure:")
print("N =", all_lambdas[fp] * NM_fail[:3])
print("M =", all_lambdas[fp] * NM_fail[3:])


# WHAT THIS CODE DOES
# --------------------------------------------------------------------------
# Intake properties of composite as well as orientation
# Finds the basic Q stiffness matrix
# Solves for the ABD matrix for wach ply and adds the terms to create one ABD
# Inverts the ABD and multiplies it by the input loads for strain and bending
# Strain and bending combined for each ply to get stresses
# Stresses are converted to be in line with fiber direction
# The stresses are then placed through the Tsai-Wu failure criteria for FOS
# Strain elongation is also tested for failure and FOS
#
# This process is completed for top/bottom (bending) favoring compression due 
# to worse properties for composites in compression
# 
# The process is also completed for the sides (shear) with no favoring
# occuring
#
# This assumes the top and bottom will be carrying all of the bending load
# and the sides will be carrying all of the shear
#
# Safety factors need to be applied further for any complexities past square