#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 18 12:46:00 2021


Replication of first model in Nuraini et al,
"Mathematical Models for Assessing Vaccination Scenarios in Several Provinces in Indonesia"

This is the model with age compartments. It is an SIQRD model:
    
    - N
    - Susceptable
    - Infected
    - Quarantined
    - recovered
    - DEATH
    


@author: Jason Pekos
"""


import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.integrate import odeint
from typing import List, Callable

### This is graph formatting in SEABORN
rc={'lines.linewidth': 2, 'axes.labelsize': 18, 'axes.titlesize': 18}
sns.set(rc=rc)
sns.set_style("whitegrid", {'axes.grid' : False}) #remove grey grid
sns.set_context("paper") #changes the theme or the size or something ... not sure why I added this. 
plt.rcParams['figure.figsize'] = [16, 7] #changes the size


    
    
#params
pi:float    = 0.000001  #NEW ENTRY INTO S BOX
mu:float    = 0.000001  #DEATHS (NATURAL)
gamma:float = 0.08      #RECOVERY RATE
delta:float = 0.0028    #DEATH RATE (COVID) 
zeta:float  = 0.07      #REINFECTION RATE
beta:float  = 0.4815    #INFECTION RATE
eta:float   = 0.45      #VACCINE EFFICACY
vt:float    = 0.001     #VACCINE RATE
q:float     = 0.4       #QUARANTINE RATE

argList = [pi,mu,gamma,delta,zeta,beta,eta,vt,q]


dt:float    = 0.01 #TIMESTEP
y0:List     = [0.9999,0.0001,0.0,0.0,0.0,
               0.0,0.0,0.0,0.0,0.0,
               0.0,0.0,0.0,0.0,0.0,
               0.0,0.0,0.0,0.0,0.0,
               0.0,0.0,0.0,0.0,0.0] 



###Define Gradient
def SIQRDode(y0:List[float],
             t:np.ndarray,
             args:List[float]) -> List[float]:
    
    S1,I1,Q1,R1,D1 = y0[0], y0[1],y0[2],y0[3],y0[4]
    S2,I2,Q2,R2,D2 = y0[5], y0[6],y0[7],y0[8],y0[9]
    S3,I3,Q3,R3,D3 = y0[10], y0[11],y0[12],y0[13],y0[14]
    S2,I2,Q2,R2,D2 = y0[15], y0[16],y0[17],y0[18],y0[19]
    S5,I5,Q5,R5,D5 = y0[20], y0[21],y0[22],y0[23],y0[24]

    
    pi,mu,gamma,delta = args[0],args[1],args[2],args[3]
    zeta,beta,eta,vt  = args[4],args[5],args[6],args[7]
    q = args[8]
    
    N1 = S1 + I1 + Q1 + R1 + D1
    N1_hat  = N1 - D1
    
    N2 = S2 + I2 + Q2 + R2 + D2
    N2_hat  = N2 - D2

    N3 = S3 + I3 + Q3 + R3 + D3
    N3_hat  = N3 - D3
    
    N4 = S4 + I4 + Q4 + R4 + D4
    N4_hat  = N4 - D4

    N5 = S5 + I5 + Q5 + R5 + D5
    N5_hat  = N5 - D5
    
    V1 = vt
    V2 = vt
    V3 = vt
    V4 = vt
    V5 = vt
    
    
    ##Cross Relationship
    B11
    B21
    sumInfected = B*(I1/N) + B2*(I1/N) + B3*(I1/N) + B4*(I1/N) + B5*(I1/N)
    
 
    dS1:float = pi*N1_hat+ zeta*R1 - (beta*(I/N) + eta*V + mu)
    dI1:float = ((beta*S*I)/(N)) - (Q+I)*I
    dQ1:float = q*I - (gamma + delta +mu)*Q
    dR1:float = gamma * Q + eta*V*S - (zeta + mu)*R
    dD1:float = delta * Q
    
    dS2:float = pi*N_hat+ zeta*R - (beta*(I/N) + eta*V + mu)
    dI2:float = ((beta*S*I)/(N)) - (Q+I)*I
    dQ2:float = q*I - (gamma + delta +mu)*Q
    dR2:float = gamma * Q + eta*V*S - (zeta + mu)*R
    dD2:float = delta * Q
    
    dS3:float = pi*N_hat+ zeta*R - (beta*(I/N) + eta*V + mu)
    dI3:float = ((beta*S*I)/(N)) - (Q+I)*I
    dQ3:float = q*I - (gamma + delta +mu)*Q
    dR3:float = gamma * Q + eta*V*S - (zeta + mu)*R
    dD3:float = delta * Q
    
    dS4:float = pi*N_hat+ zeta*R - (beta*(I/N) + eta*V + mu)
    dI4:float = ((beta*S*I)/(N)) - (Q+I)*I
    dQ4:float = q*I - (gamma + delta +mu)*Q
    dR4:float = gamma * Q + eta*V*S - (zeta + mu)*R
    dD4:float = delta * Q
    
    dS4:float = pi*N_hat+ zeta*R - (beta*(I/N) + eta*V + mu)
    dI4:float = ((beta*S*I)/(N)) - (Q+I)*I
    dQ4:float = q*I - (gamma + delta +mu)*Q
    dR4:float = gamma * Q + eta*V*S - (zeta + mu)*R
    dD4:float = delta * Q
    
    dydt = [dS1, dI1, dQ1, dR1, dD1,
            dS2, dI2, dQ2, dR2, dD2,
            dS3, dI3, dQ3, dR3, dD3,
            dS4, dI4, dQ4, dR4, dD4,
            dS5, dI5, dQ5, dR5, dD5]
    
    return(dydt)

###RUN TEST

t = np.arange(0,100 + dt, dt )
sol = odeint(SIQRDode, y0, t, args = (argList,))

plt.plot(t,sol[:,1], label = "I", color = "red")
plt.plot(t,sol[:,3], label = "R", color = "green")
plt.plot(t,sol[:,4], label = "D", color = "black")
plt.legend()





 












