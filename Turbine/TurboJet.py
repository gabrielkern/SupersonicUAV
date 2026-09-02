import numpy as np
import math as m
import matplotlib.pyplot as plt

############################################################################

##########################################
##############  INPUTS   ################# T0 is stag or total temp
##########################################

vel = [700] # ft/s
h = [10000] # ft

#inlet
M_2 = 0.5 # mach at compressor front
nr = 0.98 # recovery factor
A0 = 0.05 #sqft reference area for mass flow

#comperessor
PR = 3 # p3_ideal/p2
nc = 0.87 # isentropic efficiency
M_3 = 0.2 # mach in combustion

#cumbust
M_4 = M_3
npres = 0.95 # pressure loss factor
T4_0 = 1600 # temperature at the turbine face
h_PR = 4.61e8 # fuel heating value based on Kerosene based JP-8

#turbine
nt = 0.9 # turbine efficiency
M_5 = 0.5

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

def altitude_corrections(vel,h,A0):
    #T0/T = 1 + (γ-1)/2 · M²
    #P0/P = [1 + (γ-1)/2 · M²]^(γ/(γ-1))
    #ρ0/ρ = [1 + (γ-1)/2 · M²]^(1/(γ-1))
    GAM = 1.4
    P_stat,T_stat,rho,a,mu = atmosphere(h)
    M = vel/a
    T_stag = T_stat * (1+ (GAM-1)/2*M**2)
    P_stag = P_stat * (1+ (GAM-1)/2*M**2)**(GAM/(GAM-1))
    rho_stag = rho * (1+ (GAM-1)/2*M**2)**(1/(GAM-1))
    m_dot = A0*rho*vel

    return T_stag, P_stag, P_stat, T_stat,m_dot

def inlet(M_2,nr,h,vel,A0):

  GAM = 1.4
  R = 1716
  T1_stag,P1_stag, P1_stat,T1_stat,m_dot = altitude_corrections(vel,h,A0)
  T2_stag = T1_stag
  P2_stag = nr*P1_stag
  T2_stat = T2_stag / (1+ (GAM-1)/2*M_2**2)
  P2_stat = P2_stag / (1+ (GAM-1)/2*M_2**2)**(GAM/(GAM-1))
  rho2_stat = P2_stat / (R*T2_stat)
  V2 =  (GAM*R*T2_stat)**0.5 * M_2
  return T1_stag,T1_stat,P1_stag,P1_stat,T2_stag,T2_stat,P2_stag,P2_stat,V2,rho2_stat,m_dot

def compressor(T2_0,P2_0,PR,nc,M_3):
    GAM = 1.4
    R = 1716
    cp = 6006
    T3_0_ideal = T2_0*PR**((GAM-1)/GAM)
    T3_0 = T2_0 + (T3_0_ideal-T2_0)/nc
    P3_0 = P2_0*PR
    T3 = T3_0 / (1+ (GAM-1)/2*M_3**2)
    P3 = P3_0 / (1+ (GAM-1)/2*M_3**2)**(GAM/(GAM-1))
    V3 = (GAM*R*T3)**0.5 * M_3
    rho3_stat = P3 / (R*T3)
    work_comp = cp*(T3_0-T2_0)

    return T3_0,T3,P3_0,P3,V3,rho3_stat,work_comp

def combustion(T3_0,P3_0,m_dot,npres,h_PR,T4_0,M_4):
    GAM = 1.4
    cp = 6006
    R = 1716
    f = cp*(T4_0-T3_0) / (h_PR-cp*T4_0)
    m4_dot = m_dot*(1+f) # mass flow is added through the fuel pretty cool
    P4_0 = npres*P3_0
    T4 = T4_0 / (1+ (GAM-1)/2*M_4**2)
    P4 = P4_0 / (1+ (GAM-1)/2*M_4**2)**(GAM/(GAM-1))
    rho4_stat = P4 / (R*T4)
    V4 = (GAM*R*T4)**0.5 * M_4

    return T4_0,T4,P4_0,P4,V4,rho4_stat,f,m4_dot

def turbine(T4_0,P4_0,work_comp,nt,M_5):
   cp = 6006
   GAM = 1.4
   R = 1716
   T5_0 = T4_0 - work_comp/cp
   T5_0_ideal = T4_0 - (T4_0-T5_0)/nt
   P5_0 = P4_0*(T5_0_ideal/T4_0)**(GAM/(GAM-1))
   T5 = T5_0 / (1+ (GAM-1)/2*M_5**2)
   P5 = P5_0 / (1+ (GAM-1)/2*M_5**2)**(GAM/(GAM-1))
   rho5_stat = P5 / (R*T5)
   V5 = (GAM*R*T5)**0.5 * M_5

   return T5_0,T5,P5_0,P5,V5,rho5_stat

def nozzle(T5_0,P5_0,h,m4_dot,m_dot,vel):
    GAM = 1.4
    R = 1716
    T6_0 = T5_0
    P6_0 = P5_0
    PR_crit = ((GAM+1)/2)**(GAM/(GAM-1))
    P_ambient,T_stat,rho,a,mu = atmosphere(h)
    PR_availible = P6_0 /P_ambient

    if PR_availible <= PR_crit: # not choked or fully expanded
        print("non-choked")
        M_6 = ( 2/(GAM-1) * ((PR_availible)**((GAM-1)/GAM)-1))**0.5
        T6 = T6_0 / (1+(GAM-1)/2*M_6**2)
        P6 = P6_0 / (1+(GAM-1)/2*M_6**2)**(GAM/(GAM-1))
        V6 = M_6*(GAM*R*T6)**0.5
        rho6_stat = P6 / (R*T6)
        P_thrust = 0 # pressure thrust
    else:                       # choked case means pressure thrust
        print("choked")
        M_6 = 1 # at throat
        T6 = T6_0/((GAM+1)/2)
        P6 = P6_0 /((GAM+1)/2)**(GAM/(GAM-1))
        V6 = (GAM*R*T6)**0.5
        rho6_stat = P6 / (R*T6)
        A6 = m4_dot/(V6*rho6_stat)
        P_thrust = (P6-P_ambient)*A6

    Thrust = m4_dot*V6-m_dot*vel+P_thrust

    return T6_0,T6,P6_0,P6,V6,rho6_stat,Thrust



    



T1_stag,T1_stat,P1_stag,P1_stat,T2_stag,T2_stat,P2_stag,P2_stat,V2,rho2_stat,m_dot= inlet(M_2,nr,h,vel,A0)
T3_0,T3,P3_0,P3,V3,rho3_stat,work_comp= compressor(T2_stag,P2_stag,PR,nc,M_3)
T4_0,T4,P4_0,P4,V4,rho4_stat,f,m4_dot = combustion(T3_0,P3_0,m_dot,npres,h_PR,T4_0,M_4)
T5_0,T5,P5_0,P5,V5,rho5_stat = turbine(T4_0,P4_0,work_comp,nt,M_5)
T6_0,T6,P6_0,P6,V6,rho6_stat,Thrust = nozzle(T5_0,P5_0,h,m4_dot,m_dot,vel)
print(T6_0,T6,P6_0,P6,V6,rho6_stat,Thrust)
