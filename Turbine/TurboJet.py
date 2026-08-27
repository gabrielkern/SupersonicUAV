import numpy as np
import math as m
import matplotlib.pyplot as plt

############################################################################

##########################################
##############  INPUTS   #################
##########################################

vel = [700] # ft/s
h = [10000] # ft

#inlet
M_2 = 0.5 # mach at compressor front
n = 0.98 # recovery factor

#############################################################################


def atmosphere(h):
    psl = 2116.2
    Tsl = 518.69
    a1 = -3.56616e-3 / Tsl
    a2 = 0.54864e-3 / Tsl
    h1 = 36089
    h2 = 65617
    h3 = 104987
    g = 32.174
    R = 1716.56
    b1 = -g / (R * Tsl * a1)
    b2 = -g / (R * Tsl * a2)

    theta = np.zeros(len(h))
    delta = np.zeros(len(h))

    for i in range(len(h)):

        #
        # Troposphere
        #
        if h[i] <= h1:
            theta[i] = 1 + a1 * h[i]
            delta[i] = theta[i] ** b1

        #
        # Isothermal layer
        #
        elif h[i] > h1 and h[i] <= h2:
            theta[i] = 1 + a1 * h1
            delta[i] = theta[i] ** b1 * np.exp(
                g * (h1 - h[i]) /
                (R * Tsl * (1 + a1 * h1))
            )

        #
        # Stratosphere
        #
        elif h[i] > h2 and h[i] <= h3:
            theta[i] = 1 + a1 * h1 + a2 * (h[i] - h2)
            delta[i] = (
                (1 + a1 * h1) ** (b1 - b2)
                * np.exp(
                    g * (h1 - h2) /
                    (R * Tsl * (1 + a1 * h1))
                )
                * theta[i] ** b2
            )

    p = psl * delta
    T = Tsl * theta
    rho = p / (1716 * T)
    a = (1.4 * 1716 * T) ** 0.5

    T0 = 491.67
    S = 198.72
    mu0 = 3.584e-7

    mu = mu0 * (T / T0) ** (
        1.5 * (T0 + S) / (T + S)
    )

    return p, T, rho, a, mu

def altitude_corrections(vel,h):
    #T0/T = 1 + (γ-1)/2 · M²
    #P0/P = [1 + (γ-1)/2 · M²]^(γ/(γ-1))
    #ρ0/ρ = [1 + (γ-1)/2 · M²]^(1/(γ-1))
    GAM = 1.4
    P_stat,T_stat,rho,a,mu = atmosphere(h)
    M = vel/a
    T_stag = T_stat * (1+ (GAM-1)/2*M**2)
    P_stag = P_stat * (1+ (GAM-1)/2*M**2)**(GAM/(GAM-1))
    rho_stag = rho * (1+ (GAM-1)/2*M**2)**(1/(GAM-1))

    return T_stag, P_stag, P_stat, T_stat

def inlet(M_2,n,h,vel):

  GAM = 1.4
  R = 1716
  T1_stag,P1_stag, P1_stat,T1_stat = altitude_corrections(vel,h)
  T2_stag = T1_stag
  P2_stag = n*P1_stag
  T2_stat = T2_stag / (1+ (GAM-1)/2*M_2**2)
  P2_stat = P2_stag / (1+ (GAM-1)/2*M_2**2)**(GAM/(GAM-1))
  rho2_stat = P2_stat / (R*T2_stat)

  return T2_stat,P2_stat,rho2_stat,P1_stat,T1_stat

T2_stat,P2_stat,rho2_stat,P1_stat,T1_stat = inlet(M_2,n,h,vel)
print(T2_stat,P2_stat,rho2_stat,P1_stat,T1_stat)

