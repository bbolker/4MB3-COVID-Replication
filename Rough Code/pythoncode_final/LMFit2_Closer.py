#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 29 21:02:24 2021

@author: work
"""

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
gammaTrue:float = 0.08 #RECOVERY RATE
deltaTrue:float = 0.0028 #DEATH RATE (COVID) 
zeta:float  = 0.07 #REINFECTION RATE
betaTrue:float  = 0.4815 #INFECTION RATE
eta:float   = 0.45 #VACCINE EFFICACY
vt:float    = 0.001 #VACCINE RATE
q:float     = 0.4  #QUARANTINE RATE
m:float = 100

print("ji")
###Create Gradient Function
def f(siqrd, t, ps):
    """NoAgeStructure"""
    
    print(ps)
    
    try:
        beta = ps['beta'].value
        gamma = ps['gamma'].value
        delta = ps['delta'].value
        m = ps['i0'].value
    except:
        s0,beta, gamma, delta, m = ps
   
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
    
    data = data.T

    return (model.T[3] - data[2]).ravel()

t = np.linspace(0, 1, 71)
y0 = np.array([2,0.001,0,0,0])


from NoAgeStructure import argList
from ParameterEstimation import DV

data = g(t, y0, [2,betaTrue, gammaTrue,deltaTrue,m])
data += np.random.normal(size=data.shape)
data = np.array(DV)
data = data.T

# set parameters incluing bounds
params = Parameters()
params.add('s0', value= 40000, min=0, max=300000)
params.add('i0', value=100, min=0, max=1000)
params.add('beta', value=2.0, min=0, max=1)
params.add('gamma', value=1.0, min=0, max=1)
params.add('delta', value=1.0, min=0, max=1)
params.add('m', value=1.0, min=0, max=100)

# fit model and find predicted values
result = minimize(residual, params, args=(t, data), method='leastsq')
data = data.T
#final = data + result.residual.reshape(data.shape)

# plot data and fitted curves
#plt.plot( data, 'o')
#plt.plot( final, '-', linewidth=2);

# display fitted statistics
report_fit(result)