#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 17 23:11:46 2021


Replication of first model in Nuraini et al,
"Mathematical Models for Assessing Vaccination Scenarios in Several Provinces in Indonesia"

This is the model with no age compartments. It is an SIQRD model:
    
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
pi:float    = 0.000001 #NEW ENTRY INTO S BOX
mu:float    = 0.000001 #DEATHS (NATURAL)
gamma:float = 0.08 #RECOVERY RATE
delta:float = 0.0028 #DEATH RATE (COVID) 
zeta:float  = 0.07 #REINFECTION RATE
beta:float  = 0.4815 #INFECTION RATE
eta:float   = 0.45 #VACCINE EFFICACY
vt:float    = 0.001 #VACCINE RATE
q:float     = 0.4  #QUARANTINE RATE

argList = [pi,mu,gamma,delta,zeta,beta,eta,vt,q]


dt:float    = 0.01 #TIMESTEP
y0:List     = [0.9999,0.0001,0.0,0.0,0.0] 



###Define Gradient
def SIQRDode(y0:List[float],
             t:np.ndarray,
             args:List[float]) -> List[float]:
    
    S,I,Q,R,D = y0[0], y0[1],y0[2],y0[3],y0[4]
    
    pi,mu,gamma,delta = args[0],args[1],args[2],args[3]
    zeta,beta,eta,vt  = args[4],args[5],args[6],args[7]
    q = args[8]
    
    N = S + I + Q + R + D
    N_hat  = N - D
    V = vt
    

    
    dS:float = pi*N_hat+ zeta*R - (beta*(I/N) + eta*V + mu)
    dI:float = ((beta*S*I)/(N)) - (Q+I)*I
    dQ:float = q*I - (gamma + delta +mu)*Q
    dR:float = gamma * Q + eta*V*S - (zeta + mu)*R
    dD:float = delta * Q
    
    dydt = [dS, dI, dQ, dR, dD]
    
    return(dydt)

###RUN TEST

t = np.arange(0,100 + dt, dt )
sol = odeint(SIQRDode, y0, t, args = (argList,))

plt.plot(t,sol[:,1], label = "I", color = "red")
plt.plot(t,sol[:,3], label = "R", color = "green")
plt.plot(t,sol[:,4], label = "D", color = "black")
plt.legend()





 












