#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 22 19:10:29 2021

This script estimates parameters for the Age-Structure and Non-Age-Structured
model based on the sources given in:
        https://www.medrxiv.org/content/10.1101/2020.12.21.20248241v1.full
        
Any changes are noted in comments

@author: Jason Pekos
"""

#%% Packages
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.integrate import odeint


#%% Assumed Parameters

#These values are given in the paper based on reasoning, not estimates from 
#data. 


#Quarantine Rate: no source given
q = 0.4


#Natural Recruitment Rate: set equal to the natural death rate, which is 
#estimated from the median life span of ~70 in indonesia. 
mu = 1/(360*70) #death rate
pi = mu         #recruitment rate


#Initial Vaccine Rate: set zero as there is no intial vaccine
v_0 = 0

#Later Vaccine Rate: set to a constant 0.8 for no reason. This seems high and 
#unreasonable. 
v_t = 0.8

##Reinfection Rate: Based on the following paper, they set at 1/150
#D.F. Gudbjartsson.,  DOI: 10.1056/NEJMoa2026116.
zeta = 1/150



#%% Derived Parameters

###Non-Linear Least Squares Implementation
#This is what is used in the paper, but the actual implementation does not
#have a lot to go on, and we have (probably) different time series for 
#indonesia, though not for CN


#Consider fitting some data-set (t,y) with a simple non-linear function
#Want to minimize sum of squared errors.


#First step is simply to define our gradient function. We import from the
#relevant files. Model W/O age structure first. 
from NoAgeStructure import SIQRDode
from NoAgeStructure import argList,dt,y0

t = np.arange(0,100 + dt, dt )
sol = odeint(SIQRDode, y0, t, args = (argList,))


#Test gradients --- should output graph
plt.plot(t,sol[:,1], label = "I", color = "red")
plt.plot(t,sol[:,3], label = "R", color = "green")
plt.plot(t,sol[:,4], label = "D", color = "black")
plt.legend()

#Create optimization vector omega; we are trying to solve for:
#Beta, gamma, delta, initial infections in our model
#So we need a four element optimization array. We compute over [0,1] for the 
#model paramters and [0,100] for the initial condition.

evalpoints:int = 10

omegaTest  = np.linspace(0,1,evalpoints)
betaTest   = np.linspace(0,1,evalpoints)
gammaTest  = np.linspace(0,1,evalpoints)
deltaTest  = np.linspace(0,1,evalpoints)

infectedTest = np.linspace(0,100,101)

#In the paper they define their error function as:
def PaperSquaredError(DerivedVecs, DataVecs) -> float:
    num = len(DerivedVecs[0])
    totalLoss = 0
    
    for j in range(num):
        sdQ  = (DataVecs[0][j] - DerivedVecs[0][j])**2
        sdCR = (DataVecs[1][j] - DerivedVecs[1][j])**2
        sdD  = (DataVecs[2][j] - DerivedVecs[2][j])**2
        totalLoss = totalLoss + sdQ + sdCR + sdD
        
    return(totalLoss)


#Data Import
ConData = pd.read_csv("ccv.csv")

#For some reason the time axis is out of order! what!
ConData['Date']=pd.to_datetime(ConData['Date'])

ConData = ConData.sort_values(by=['Date'])


#want a new dataframe with Quarantined, CR (recovered), Deaths
CDClean = pd.concat([ConData["Date"],
                     ConData["Total cases"],
                     ConData["Total cases"], 
                     ConData["Total deaths"]], axis = 1)


#Rename to correct names
CDClean.columns = ["Date", "Quarantined","Recovered","Deaths"]

#Paper assumes (zeta) percent of cases are quarantined
CDClean["Quarantined"] = CDClean["Quarantined"]*q


#There isn't any recovered data provided, so assume it is 
#the cases time-lagged by two weeks minus deaths. Created below:

CDClean['RC'] = (CDClean['Recovered'].shift(2)  - 
                 CDClean['Recovered'] * zeta    - 
                 CDClean['Deaths'])

CDClean["Date"] = pd.to_datetime(CDClean["Date"])


#They cropped so they only fit on april - october. We will do the same:
#CDClean = CDClean.set_index(['Date'])

CDClean.reset_index(drop=True)
df = CDClean.loc[133:165]
ax1 = CDClean.plot(x = "Date", y = "RC")   
#df.plot(x = "Date", y = "Recovered", ax = ax1)
#df.plot(x = "Date", y = "Deaths", ax = ax1)
#df.plot(x = "Date", y = "Quarantined", ax = ax1)

DV = [df["Quarantined"].tolist(),df["RC"].tolist(), df["Deaths"].tolist()]

#%% Fitting Data

from NoAgeStructure import eta, vt, q

#omegaTest  = np.linspace(0,1,evalpoints)
omegaTest = [0,1,2]
betaTest   = np.linspace(0,1,evalpoints)
gammaTest  = np.linspace(0,1,evalpoints)
deltaTest  = np.linspace(0,1,evalpoints)

infectedTest = np.linspace(0,100,101)

t = np.arange(0,100 + dt, dt )
sol = odeint(SIQRDode, y0, t, args = (argList,))

MSEmin = (0,0,0,0,100000000000000)

for i in omegaTest:
    for j in betaTest:
        for k in gammaTest:
            for l in deltaTest:
                for m in infectedTest:
                    y0 = [3565000 - m,m,0,0,0]
                    argList1 = [pi,mu,k,l,zeta,j,eta,vt,q]
                    sol =  odeint(SIQRDode,
                                  y0,
                                  t,
                                  args = (argList1,))
                    
                    EvalOn = [sol[2],sol[3],sol[4]]
                    MSEsol = PaperSquaredError(EvalOn,
                                               DV)
                    MSEnew = (j,k,l,m,MSEsol)
                    

                    if MSEnew[4] < MSEmin[4]:
                        MSEmin = MSEnew
                    


print(MSEmin)

beta  = MSEmin[0]
gamma = MSEmin[1]
delta = MSEmin[2]
m     = MSEmin[3]

argList1 = [pi,mu,gamma,delta,zeta,beta,eta,vt,q]

sol = odeint(SIQRDode, y0, t, args = (argList,))


#Test gradients --- should output graph
plt.plot(t,sol[:,1], label = "I", color = "red")
plt.plot(t,sol[:,3], label = "R", color = "green")
plt.plot(t,sol[:,4], label = "D", color = "black")
plt.legend()
        
        




