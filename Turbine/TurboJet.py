import numpy as np
import math as m
import matplotlib.pyplot as plt

# T0 is stag or total temp

'''
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
'''

# if try to push before pull run:     cd SupersonicUAV    then   git reset --soft HEAD~1
#############################################################################
class tubojet_calc:
    def __init__(self,vel,h,A0):
       self.M_2=0.5
       self.nr=0.98
       self.M_3=0.2
       self.PR=3
       self.nc=0.87
       self.M_4=self.M_3
       self.npres=0.95
       self.T4_0=1600
       self.h_PR=4.61e8
       self.M_5=0.5
       self.nt=0.9
       self.GAM=1.4
       self.R=1716
       self.cp=6006
       self.g=32.2

    def atmosphere(self,h):
        psl = 2116.2
        Tsl = 518.69
        a1 = -3.56616e-3 / Tsl
        a2 = 0.54864e-3 / Tsl
        h1 = 36089
        h2 = 65617
        h3 = 104987
        g = 32.174
        b1 = -g / (self.R * Tsl * a1)
        b2 = -g / (self.R * Tsl * a2)

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
                delta[i] = theta[i] ** b1 * np.exp(g * (h1 - h[i]) /(self.R * Tsl * (1 + a1 * h1)))
        #
        # Stratosphere
        #
            elif h[i] > h2 and h[i] <= h3:
                theta[i] = 1 + a1 * h1 + a2 * (h[i] - h2)
                delta[i] = ((1 + a1 * h1) ** (b1 - b2)* np.exp(g * (h1 - h2) /(self.R * Tsl * (1 + a1 * h1)))* theta[i] ** b2)

        p = psl * delta
        T = Tsl * theta
        rho = p / (1716 * T)
        a = (1.4 * 1716 * T) ** 0.5

        T0 = 491.67
        S = 198.72
        mu0 = 3.584e-7

        mu = mu0 * (T / T0) ** (1.5 * (T0 + S) / (T + S))

        return p, T, rho, a, mu

    def altitude_corrections(self,vel,h,A0):
        #T0/T = 1 + (γ-1)/2 · M²
        #P0/P = [1 + (γ-1)/2 · M²]^(γ/(γ-1))
        #ρ0/ρ = [1 + (γ-1)/2 · M²]^(1/(γ-1))
        P_stat,T_stat,rho,a,mu = self.atmosphere(h)
        M = vel/a
        T_stag = T_stat * (1+ (self.GAM-1)/2*M**2)
        P_stag = P_stat * (1+ (self.GAM-1)/2*M**2)**(self.GAM/(self.GAM-1))
        rho_stag = rho * (1+ (self.GAM-1)/2*M**2)**(1/(self.GAM-1))
        m_dot = A0*rho*vel

        return T_stag, P_stag, P_stat, T_stat,m_dot

    def inlet(self,h,vel,A0):
        T1_stag,P1_stag, P1_stat,T1_stat,m_dot = self.altitude_corrections(vel,h,A0)
        T2_stag = T1_stag
        P2_stag = self.nr*P1_stag
        T2_stat = T2_stag / (1+ (self.GAM-1)/2*self.M_2**2)
        P2_stat = P2_stag / (1+ (self.GAM-1)/2*self.M_2**2)**(self.GAM/(self.GAM-1))
        rho2_stat = P2_stat / (self.R*T2_stat)
        V2 =  (self.GAM*self.R*T2_stat)**0.5 * self.M_2
        return T1_stag,T1_stat,P1_stag,P1_stat,T2_stag,T2_stat,P2_stag,P2_stat,V2,rho2_stat,m_dot

    def compressor(self,T2_0,P2_0):
        T3_0_ideal = T2_0*self.PR**((self.GAM-1)/self.GAM)
        T3_0 = T2_0 + (T3_0_ideal-T2_0)/self.nc
        P3_0 = P2_0*self.PR
        T3 = T3_0 / (1+ (self.GAM-1)/2*self.M_3**2)
        P3 = P3_0 / (1+ (self.GAM-1)/2*self.M_3**2)**(self.GAM/(self.GAM-1))
        V3 = (self.GAM*self.R*T3)**0.5 * self.M_3
        rho3_stat = P3 / (self.R*T3)
        work_comp = self.cp*(T3_0-T2_0)

        return T3_0,T3,P3_0,P3,V3,rho3_stat,work_comp

    def combustion(self,T3_0,P3_0,m_dot):
        f = self.cp*(self.T4_0-T3_0) / (self.h_PR-self.cp*self.T4_0)
        m4_dot = m_dot*(1+f) # mass flow is added through the fuel pretty cool
        P4_0 = self.npres*P3_0
        T4 = self.T4_0 / (1+ (self.GAM-1)/2*self.M_4**2)
        P4 = P4_0 / (1+ (self.GAM-1)/2*self.M_4**2)**(self.GAM/(self.GAM-1))
        rho4_stat = P4 / (self.R*T4)
        V4 = (self.GAM*self.R*T4)**0.5 * self.M_4

        return self.T4_0,T4,P4_0,P4,V4,rho4_stat,f,m4_dot

    def turbine(self,P4_0,work_comp):
        T5_0 = self.T4_0 - work_comp/self.cp
        T5_0_ideal = self.T4_0 - (self.T4_0-T5_0)/self.nt
        P5_0 = P4_0*(T5_0_ideal/self.T4_0)**(self.GAM/(self.GAM-1))
        T5 = T5_0 / (1+ (self.GAM-1)/2*self.M_5**2)
        P5 = P5_0 / (1+ (self.GAM-1)/2*self.M_5**2)**(self.GAM/(self.GAM-1))
        rho5_stat = P5 / (self.R*T5)
        V5 = (self.GAM*self.R*T5)**0.5 * self.M_5

        return T5_0,T5,P5_0,P5,V5,rho5_stat

    def nozzle(self,T5_0,P5_0,h,m4_dot,m_dot,vel):
        T6_0 = T5_0
        P6_0 = P5_0
        PR_crit = ((self.GAM+1)/2)**(self.GAM/(self.GAM-1))
        P_ambient,T_stat,rho,a,mu = self.atmosphere(h)
        PR_availible = P6_0 /P_ambient

        if PR_availible <= PR_crit: # not choked or fully expanded
            print("non-choked")
            M_6 = ( 2/(self.GAM-1) * ((PR_availible)**((self.GAM-1)/self.GAM)-1))**0.5
            T6 = T6_0 / (1+(self.GAM-1)/2*M_6**2)
            P6 = P6_0 / (1+(self.GAM-1)/2*M_6**2)**(self.GAM/(self.GAM-1))
            V6 = M_6*(self.GAM*self.R*T6)**0.5
            rho6_stat = P6 / (self.R*T6)
            P_thrust = 0 # pressure thrust
        else:                       # choked case means pressure thrust
            print("choked")
            M_6 = 1 # at throat
            T6 = T6_0/((self.GAM+1)/2)
            P6 = P6_0 /((self.GAM+1)/2)**(self.GAM/(self.GAM-1))
            V6 = (self.GAM*self.R*T6)**0.5
            rho6_stat = P6 / (self.R*T6)
            A6 = m4_dot/(V6*rho6_stat)
            P_thrust = (P6-P_ambient)*A6

        Thrust = m4_dot*V6-m_dot*vel+P_thrust

        return T6_0,T6,P6_0,P6,V6,rho6_stat,Thrust

    def thrust(self, h, vel, A0):
        T1_stag, T1_stat, P1_stag, P1_stat, T2_stag, T2_stat, P2_stag, P2_stat, V2, rho2_stat, m_dot = self.inlet(h, vel,A0)
        T3_0, T3, P3_0, P3, V3, rho3_stat, work_comp = self.compressor(T2_stag, P2_stag)
        T4_0, T4, P4_0, P4, V4, rho4_stat, f, m4_dot = self.combustion(T3_0, P3_0, m_dot)
        T5_0, T5, P5_0, P5, V5, rho5_stat = self.turbine(P4_0, work_comp)
        T6_0, T6, P6_0, P6, V6, rho6_stat, Thrust = self.nozzle(T5_0, P5_0, h, m4_dot, m_dot, vel)
        TSFC = f*m_dot*self.g / Thrust # pounds/s

        T_stations = [T1_stat, T2_stat, T3, T4, T5, T6]
        P_stations = [P1_stat, P2_stat, P3, P4, P5, P6]

        return Thrust, T_stations, P_stations,TSFC

Teng = tubojet_calc(vel=[700], h=[10000], A0=0.05)
F, T_stations, P_stations, TSFC = Teng.thrust(h=[10000], vel=[700],A0=0.05)
print(F)
print(T_stations)
print(P_stations)
print(TSFC)
