#The current script is an example on computing properties for
#a compressible boundary layer using the method described in
# Modular Method for Estimation of Velocity and Temperature Profiles in
#High-Speed Boundary Layers [INSERT PAGE AND VOLUME DETAILS]
#DOI: https://doi.org/10.2514/1.J061735

#Here, we compute the boundary layer properties using user-selected
#system parameters and modeling choices. 
#For an example with default modeling choices, see methodDemo1.py
#For more information, see README file

import os
import sys

# add path to BL libraries
pwd      = os.getcwd()
pathfuns = pwd+'/src'
sys.path.append(pathfuns)

#import Python libraries
import boundaryLayerPropFunctions as funs
import matplotlib.pyplot as plt
import numpy as np 

#define properties for compressible boundary layer
M = 5.84                         #Mach number
ReDelta2 = 2260                  #Reynolds number
Te = 55.2                        #Freestream temperature
Tw = 0.25*(1+0.896*0.2*M*M)*Te   #Wall temperature
Tw_Te = Tw/Te                    #Ratio of wall to freestream temperature
N = 3000                         #number of grid points in the boundary layer

#see README file for more information on modeling choice flags
[cf,Bq,ch,yPlus,uPlus,Tplus,mu_muw] = \
        funs.boundaryLayerProperties(M,ReDelta2,Tw_Te,Te,N, 
        flagViscTemp = 3, flagIncCf = 3, flagTempVelocity = 1, flagVelocityTransform = 3,\
                gridStretchPar = 1.005,underRelaxFactor = 1)

alf_plus = 2.0  # target resolution w.r.t. Kolmogorv length-scale in wall units  
jb       = 80   # grid index at which transition between the near-wall and the outer mesh stretching
ypw      = .5   # target y^* at the wall

[yPlus_j,yStar_j,jj,Ny,alf_opt,etaPlus_j] = \
        funs.gridProperties(yPlus,Tplus,mu_muw,alf_plus,jb,ypw,myNy=171)

#check grid resolution
resolution = np.gradient(yPlus_j[1:])/etaPlus_j[1:] #remove singularity at y=0

#print quantities
print('\n')
print('cf = '+str(cf))  #skin-friction coefficient
print('\n')
print('Bq = '+str(Bq))  #Wall heat transfer rate
print('\n')
print(f'ch = {ch}')     #Stanton number
print('\n')
print('Ny = '+str(Ny)) # N points in wall-normal direction up to boundary layer edge
print('\n')

#create plots
fig1,axs1 = plt.subplots()
axs1.plot(yPlus,uPlus)  #Velocity profile
axs1.set_xscale('log')
axs1.set_xlabel('$y^+$')
axs1.set_ylabel('$u^+$')
axs1.set_title('Boundary layer velocity profile')
axs1.set_xlim(left = 0.1)

fig2,axs2 = plt.subplots()
axs2.plot(yPlus,Tplus)  #Temperature profile
axs2.set_xscale('log')
axs2.set_xlabel('$y^+$')
axs2.set_ylabel('$T^+$')
axs2.set_title('Boundary layer temperature profile')
axs2.set_xlim(left = 0.1)

fig3,axs3 = plt.subplots()
axs3.semilogy(jj,yPlus_j)  #Grid profile yPlus
axs3.axhline(yPlus_j[-1], color="black",ls='dashed')
axs3.set_xlabel('$j$')
axs3.set_ylabel('$y^+$')
axs3.set_title('Boundary layer grid map, wall units')
axs3.set_xlim(left   = 1)

fig4,axs4 = plt.subplots()
axs4.semilogy(jj,yStar_j)  #Grid profile yStar
axs4.axhline(yStar_j[-1], color="black",ls='dashed')
axs4.axhline(30,color="red")
axs4.set_xlabel('$j$')
axs4.set_ylabel('$y^*$')
axs4.set_title('Boundary layer grid map, semi-local units')
axs4.set_xlim(left   = 1)

fig5,axs5 = plt.subplots()
axs5.semilogx(yPlus_j[1:], resolution)  #Resolution profile
axs5.axhline(alf_plus, color="red")
axs5.axvline(yPlus_j[-1], color="black",ls='dashed')
axs5.set_xscale('log')
axs5.set_xlabel('$y^+$')
axs5.set_ylabel('$\Delta^+/\eta^+$')
axs5.set_title('Boundary layer grid resolution')
axs5.set_xlim(left   = 1)
axs5.set_ylim(bottom = 0)
axs5.set_ylim(top    = alf_plus+2)


plt.show()
