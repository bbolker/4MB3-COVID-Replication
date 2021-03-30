#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
NLS Optimizer
@author: Jason Pekos
"""

import numpy as np
import matplotlib.pyplot as plt
from lmfit import minimize, Parameters, Parameter, report_fit
from scipy.integrate import odeint


#fixed parameters
pi:float    = 0.000001 #NEW ENTRY INTO S BOX
mu:float    = 0.000001 #DEATHS (NATURAL)
#gamma:float = 0.08 #RECOVERY RATE
#delta:float = 0.0028 #DEATH RATE (COVID) 
zeta:float  = 0.07 #REINFECTION RATE
#beta:float  = 0.4815 #INFECTION RATE
eta:float   = 0.45 #VACCINE EFFICACY
vt:float    = 0.001 #VACCINE RATE
q:float     = 0.4  #QUARANTINE RATE

print("ji")
###Create Gradient Function
def f(siqrd, t, ps):
    """NoAgeStructure"""
    
    print(siqrd)
    print("ji")
    
    try:
        beta = ps['beta'].value
        gamma = ps['gamma'].value
        delta = ps['delta'].value
        m = ps['i0'].value
    except:
        s0,i0,beta, gamma, delta, m = ps
   
    S = siqrd[0]
    I = siqrd[1]
    Q = siqrd[2]
    R = siqrd[3]
    D = siqrd[4]  
    
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
#%





def g(t, y0, ps):
    """
    Solution to the ODE x'(t) = f(t,x,k) with initial condition x(0) = x0
    """
    x = odeint(f, y0, t, args=(ps,))
    return x



def residual(ps, ts, data):
    y0 = ps['s0'].value, ps['i0'].value,0,0,0
    model = g(ts, y0, ps)

    return (model - data).ravel()

t = np.linspace(0, 10, 100)
y0 = np.array([300000,1,0,0,0])


from NoAgeStructure import argList

data = g(t, y0, argList)
data += np.random.normal(size=data.shape)

# set parameters incluing bounds
params = Parameters()
params.add('s0', value= float(data[0, 0]), min=299000, max=300000)
params.add('i0', value=float(data[0, 1]), min=0, max=10)
params.add('beta', value=2.0, min=0, max=1)
params.add('gamma', value=1.0, min=0, max=1)
params.add('delta', value=1.0, min=0, max=1)
params.add('m', value=1.0, min=0, max=100)

# fit model and find predicted values
result = minimize(residual, params, args=(t, data), method='leastsq')
final = data + result.residual.reshape(data.shape)

# plot data and fitted curves
plt.plot(t, data, 'o')
plt.plot(t, final, '-', linewidth=2);

# display fitted statistics
report_fit(result)