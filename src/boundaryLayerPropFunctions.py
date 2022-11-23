"""
A Python script for predicting compressible boundary layer properties such as the skin friction coefficient, 
heat transfer (Wall heat transfer rate, Stanton number) along with the velocity and temperature profiles. \n

The current implementation is based on the method described in "Modular Method for Estimation of Velocity and Temperature Profiles in High-Speed Boundary Layers" [INSERT PAGE AND VOLUME DETAILS]\n
DOI: https://doi.org/10.2514/1.J061735 \n

The method seeks the following inputs: Mach number, Reynolds number (ReDelta2), Wall and freestream temperatures \n

The script contains multiple functions, with boundaryLayerProperties calling other functions for specific tasks \n
"""
###########################################################################################################
#import Python libraries
import numpy as np
import math
from scipy import integrate
from scipy import interpolate
import sys
###########################################################################################################
#define function blocks
def densityFromTemperature(T_Tw):
    
    """
    Compute density profile rho/rho_w using the temperature profile T/Tw\n

    Input parameters
    ----------------
    T_Tw: Boundary layer temperature profile normalized by the wall temperature\n
    
    Output results
    --------------
    rho_rhow: Boundary layer density profile normalized by density at the wall\n

    """
    
    #typecasting input data to float to prevent errors while computing reciprocal
    T_Tw = [float(i) for i in T_Tw]

    #in the current version, we only use equation of state
    #additional functions can be added with the selection process controlled by flags as shown below
    rho_rhow = np.reciprocal(T_Tw)
    
    return rho_rhow



def viscosityFromTemperature(T_Tw, Tw, flagViscTemp):
    
    """
    Compute dynamic viscosity profile mu/muw using the temperature profile T/Tw\n

    This function allows the user to choose from multiple viscosity-temperature (mu-T) definitions
    available in the literature to compute the viscosity profile. Typically, such mu-T relations rely
    on dimensional temperature input. Hence, apart from the normalized T/Tw profile, we also need to 
    provide the dimensional wall temperature (in K) and the choice of relation in the form of a flag \n
    
    Input parameters
    ----------------
    T_Tw: Boundary layer temperature profile normalized by the wall temperature\n
    Tw: Wall temperature (in K)\n
    flagViscTemp: Choice of mu-T relation\n

    Output results
    --------------
    mu_muw: Boundary layer viscosity profile normalized by dynamic viscosity at the wall\n

    """
    #typecasting input data to float
    T_Tw = [float(i) for i in T_Tw]
    #initialize a python list for the viscosity profile
    mu = np.zeros(np.size(T_Tw))
    #compute dimensional temperature profile by multiplying T/Tw with Tw
    T = np.multiply(T_Tw, Tw)
    
    #check for flag value for mu-T relation selection
    if flagViscTemp == 1:
        #Keyes' law
        for i in range(0, len(T)):
            #compute dynamic viscosity
            mu[i] = 1.488e-06 * np.sqrt(T[i]) / (1 + 122 / T[i] * np.power(10, -5 / T[i]))
        
        #normalize with dynamic viscosity at the wall
        mu_muw = np.divide(mu, mu[0])

    elif flagViscTemp == 2:
        #Sutherland's law
        for i in range(0, len(T)):
            #compute dynamic viscosity
            mu[i] = (1.458e-06 * T[i] ** 1.5)/(T[i] + 110.4)
        
        #normalize with dynamic viscosity at the wall
        mu_muw = np.divide(mu, mu[0])

    elif flagViscTemp == 3:
        #curve fit to CoolProp dataset
        #CoolProp Python Wrapper: http://www.coolprop.org/coolprop/wrappers/Python/index.html
        
        #curve-fit coefficients 
        A = 5.0612*1e-8
        D = 3.3125*1e-4
        alpha = 1.0958
        delta = 1.5415
        kappa = -0.3

        for i in range(0, len(T)):
            #compute dynamic viscosity
            mu[i] = (A*T[i]**alpha)*(1 + D*T[i]**delta)**kappa
        
        #normalize with dynamic viscosity at the wall
        mu_muw = np.divide(mu, mu[0])

    elif flagViscTemp == 4:
        #Power law
        #since power law only relies on the normalized temperature profile
        #we do not need dimensional temperature values

        #power law exponent
        #n = 3/4
        n = 2/3
        #coompute normalized dynamic viscosity profile
        mu_muw = np.power(T_Tw, n)

    else:
        #if the flag value is out of bounds, return an error message and exit program execution
        sys.exit('Invalid choice for viscosity-temperature relation - Check function viscosityFromTemperature')
    
    return mu_muw



def skinFrictionIncompressible(ReThetaHat, flagIncCf):

    """
    Compute the skin-friction coefficient for an incompressible boundary layer using Reynolds number\n

    This function uses a Reynolds number (Re_theta) to compute skin-friction coefficient using one of
    the multiple empirical curve-fits available in the literature. This function is only used for the
    calibration of the incompressible velocity profile in the function universalVelocityProfile.
    Note that the suffix 'Hat' in in variables denotes incompressible state.\n  

    Input parameters
    ----------------
    ReThetaHat: Reynolds number Re_theta for the incompressible boundary layer\n 
    flagIncCf: Choice of incompressible skin-friction (cfHat-ReThetaHat) relation\n

    Output results
    --------------
    cfHat: Skin friction coefficient for incompressible boundary layer\n

    """
    #check for flag value for cfHat-ReThetaHat relation selection  
    if flagIncCf == 1:
        #Karman-Schoenherr
        cfHat = 1 / (17.08*(np.log10(ReThetaHat))**2 + 25.11*np.log10(ReThetaHat) + 6.012)
    
    elif flagIncCf == 2:
        #Blasius
        cfHat = 0.026*ReThetaHat**(-0.25)
    
    elif flagIncCf == 3:
        #Smits
        cfHat = 0.024*ReThetaHat**(-0.25)
    
    else:
        #if the flag value is out of bounds, return an error message and exit program execution
        sys.exit('Invalid choice for incompressible skin friction relation -\
                Check function skinFrictionIncompressible')
    
    return cfHat



def temperatureFromVelocity(U_Ue, Tw_Te, Tr_Te, flagTempVelocity):
    
    """
    Compute the temperature profile T/Tw using the velocity profile U/Ue\n

    This function computes wall-normal temperature profile T/Tw using temperature-velocity (T-u) relations
    available in the literature. These relations, apart from the velocity profiles also typically require 
    additional temperature ratios described in the input parameters below. A flag has been added to allow
    the user to choose from multiple T-u relations implemented in the current function.\n

    Input parameters
    ----------------
    U_Ue:  Boundary layer velocity profile normalized with the freestream velocity Ue\n
    Tw_Te: Ratio of wall temperature Tw to freestream temperature Te\n
    Tr_Te: Ratio of recovery temperature Tr to freestream temperature Te\n
    flagTempVelocity: Choice of T-u relation\n

    Output results
    --------------
    T_Tw: Boundary layer temperature profile normalized by the wall temperature Tw\n
    """
    
    #compute required temperature ratios
    Te_Tw = 1 / Tw_Te
    Tr_Tw = Tr_Te * Te_Tw
    #initialize list for storing the temperature profile T/Tw
    T_Tw = np.zeros(np.size(U_Ue))
    
    #check for flag value for T-u relation selection
    if flagTempVelocity == 1:
        #Zhang's T-u relation
        #Reference: https://doi.org/10.1017/jfm.2013.620
        sPr = 0.792
        for i in range(0, len(U_Ue)):
            f = (1 - sPr) * U_Ue[i] ** 2 + sPr * U_Ue[i]
            T_Tw[i] = 1 + (Tr_Tw - 1) * f + (Te_Tw - Tr_Tw) * U_Ue[i] ** 2

    elif flagTempVelocity == 2:
        #Walz's T-u relation
        #Reference: https://doi.org/10.1017/jfm.2013.620
        for i in range(0, len(U_Ue)):
            T_Tw[i] = 1 + (Tr_Tw - 1) * U_Ue[i] + (Te_Tw - Tr_Tw) * U_Ue[i] ** 2

    else:
        #if the flag value is out of bounds, return an error message and exit program execution
        sys.exit('Invalid choice for temperature-velocity relation - Check function temperatureFromVelocity')
    
    return T_Tw


def TrTeFromMach(M, g, r):

    """
    Compute the ratio of recovery to freestream temperature (Tr/Te)\n

    This function uses the ratio of specific heats, gamma, and a recovery factor r to compute Tr/Te\n 

    Input parameters
    ----------------
    M: Mach number\n
    g: gamma, ratio of specific heats\n
    r: Recovery factor\n

    Output results
    --------------
    Tr_Te: Ratio of recovery to freestream temperature Tr/Te\n
    """
    
    Tr_Te = 1 + r * (g - 1) / 2 * np.power(M, 2)
    
    return Tr_Te



def wallToSemilocal(qPlus, T_Tw, mu_muw):

   """
   Compute semilocal scaled quantity from the wall scaled one\n

   Input parameters
   ----------------
   qPlus  : Wall scaled quantity\n 
   T_Tw   : Boundary layer temperature profile normalized by the wall temperature\n
   mu_muw : Boundary layer viscosity   profile normalized by the wall viscosity\n
   
   Output results
   --------------
   qStar : Semilocal scaled quantity\n

   """
   
   #compute density from temperature
   rho_rhow = densityFromTemperature(T_Tw)
   #transform wall quantity to semilocal quantity
   qStar = qPlus*np.sqrt(rho_rhow)/mu_muw
   return qStar



def SemilocalToWall(qStar, T_Tw, mu_muw):

   """
   Compute wall scaled quantity from the semilocal scaled one\n

   Input parameters
   ----------------
   qStar  : Semilocal scaled quantity\n 
   T_Tw   : Boundary layer temperature profile normalized by the wall temperature\n
   mu_muw : Boundary layer viscosity   profile normalized by the wall viscosity\n
   
   Output results
   --------------
   qPlus : Wall scaled quantity\n

   """
  
   #compute density from temperature
   rho_rhow = densityFromTemperature(T_Tw)
   #transform semilocal quantity to wall quantity
   qPlus = qStar*mu_muw/np.sqrt(rho_rhow)
   return qPlus



def StarMaps(yStar,T_Tw, mu_muw):

   """
   Compute mapping function interpolator w.r.t yStar\n

   Input parameters
   ----------------
   yStar  : Semilocal scaled wall distance\n 
   T_Tw   : Boundary layer temperature profile normalized by the wall temperature\n
   mu_muw : Boundary layer viscosity   profile normalized by the wall viscosity\n
   
   Output results
   --------------
   fmu_star : Semilocal scaled interpolator function for viscosity\n
   fT_star  : Semilocal scaled interpolator function for temperature\n

   """

   fT_star  = interpolate.interp1d(yStar, T_Tw  , fill_value="extrapolate", bounds_error=False) # do not bound interpolation 
   fmu_star = interpolate.interp1d(yStar, mu_muw, fill_value="extrapolate", bounds_error=False) # do not bound interpolation 
   return [fT_star,fmu_star]



def PlusMaps(yPlus,T_Tw, mu_muw):

   """
   Compute mapping function interpolator w.r.t yPlus\n

   Input parameters
   ----------------
   yStar  : Wall scaled wall distance\n 
   T_Tw   : Boundary layer temperature profile normalized by the wall temperature\n
   mu_muw : Boundary layer viscosity   profile normalized by the wall viscosity\n
   
   Output results
   --------------
   fmu_plus : Wall scaled interpolator function for viscosity\n
   fT_plus  : Wall scaled interpolator function for temperature\n

   """

   fT_plus  = interpolate.interp1d(yPlus, T_Tw  , fill_value="extrapolate", bounds_error=False) # do not bound interpolation 
   fmu_plus = interpolate.interp1d(yPlus, mu_muw, fill_value="extrapolate", bounds_error=False) # do not bound interpolation 
   return [fT_plus,fmu_plus]



def universalGridStretch(alf,jb,ysw,Retau_Star,fT_star,fmu_star,Nypoints):
    
    """
    Compute the universal grid stretching in semilocal units\n

    This function computes the universal wall-normal grid stretching profile in semilocal units
    which is then remapped into wall units to generate the physical grid for the boundary layer\n

    Input parameters
    ----------------
    alf : target resolution in semilocal Kolmogorov units\n
    jb  : transition node from viscous to outer stretching\n
    ysw : target wall distance from the wall in semilocal units\n
    Retau_Star : Retau_Star based on Retau on boundary layer edge\n
    Nypoints   : Ny points to use in wall-normal direction\n
 
    Output parameters
    ----------------
    yStar_j : semilocal scaled wall distance function of the number of points in the wall normal direction\n
    yPlus_j : wall scaled  wall distance function of the number of points in the wall normal direction\n
    jj      : index  of points in the wall normal direction\n
    Ny      : number of points in the wall normal direction\n
    """
    
    #von Karman constant
    k    = 0.41
    #grid constants
    ceta = k**0.25
    fac  = 4/(3*alf*ceta)
    #estimate point needed in wall-normal direction
    Ny_est  = fac*(Retau_Star**0.75)
    Ny      = int(np.ceil(Ny_est)+1)
    Ny      = np.max([Ny,jb])       # Ny cannot be lower than jb
    Ny      = np.max([Ny,Nypoints]) # Ny is the maximum between asymtotic and user defined value
    if Ny <= jb :
       print('WARNING: Predicted Ny points in natural stretching are less than jb')
       print('Bounding min(Ny) to jb')
       print('PLEASE VERIFY YOUR RESOLUTION TRESHOLD')
    #array of nodes
    j = np.asarray( range(0,Ny) )
    #viscous and outer grids
    ysvisc  = ysw*j
    yslog   = (0.75*alf*ceta*j)**1.333
    #blending factors
    blend1 =           1/(1+((j/jb)**2))
    blend2 = ((j/jb)**2)/(1+((j/jb)**2))
    #complete grid
    yStar_j = blend1*ysvisc+blend2*yslog
    yPlus_j = SemilocalToWall(yStar_j,fT_star(yStar_j),fmu_star(yStar_j))
    jj      = j
    return [yStar_j,yPlus_j,jj,Ny]

    

def universalVelocityProfile(yHatPlus, deltaHatPlus, wakeParameter):

    """
    Compute the universal incompressible velocity profile\n

    This function computes the velocity profile for an equivalent incompressible boundary layer
    on a wall-normal yHatPlus grid. The boundary layer edge is defined using deltaHatPlus. 
    For the universal profile definition used in the current study, we also require a 
    wake strength parameter calculated using function wakeParameterFromDeltaHatPlus.\n

    Input parameters
    ----------------
    yHatPlus: List of wall-normal grid points (in plus units) in the range [0,deltaHatPlus]\n
    deltaHatPlus: incompressible boundary layer thickness (in plus units)\n
    wakeParameter: Wake strength parameter\n

    Output results
    --------------
    uHatPlus: Universal incompressible boundary layer velocity profile (in plus units)\n

    """
   
    #Currently, the incompressible velocity profile is defined as
    #the sum of contribution from the inner and outer layers, respectively.
    #Reference: https://doi.org/10.2514/3.11820

    #initialize lists 
    uHatPlus = np.zeros(np.size(yHatPlus))
    functionValue = np.zeros(np.size(yHatPlus))
    velocityInner = np.zeros(np.size(yHatPlus))
    velocityOuter = np.zeros(np.size(yHatPlus))
 
    #constants used for inner layer velocity profile definition
    k = 0.41
    Aplus = 25.53

    #compute velocity profile at every grid point
    for i in range(0, len(yHatPlus)):
        
        #inner layer contribution
        lPlus = k * yHatPlus[i] * (1 - np.exp(-yHatPlus[i] / Aplus))
        functionValue[i] = 2 / (1 + np.sqrt(1 + 4 * lPlus ** 2))
        
        #since the velocity is zero at the wall (no-slip),
        #we begin integration from second grid point onwards
        if i > 0:
            #integrate using Trapezoidal rule
            velocityInner[i] = velocityInner[i-1] + \
                    ((functionValue[i-1] + functionValue[i])/2)\
                    *(yHatPlus[i] - yHatPlus[i-1])         
        
        #outer layer contribution
        etaHat = yHatPlus[i] / deltaHatPlus
        wakeFunction = 2 * np.sin(etaHat * np.pi / 2) ** 2
        velocityOuter[i] = (wakeParameter/k) * wakeFunction
        
        #sum of contributions from inner and outer layer
        uHatPlus[i] = velocityInner[i] + velocityOuter[i]
    
    return uHatPlus



def boundaryLayerGrid(N,gridStretchPar):

    """
    Generate boundary layer grid\n

    This function generates a wall-normal grid for an incompressible boundary layer in terms of 
    etaHat in the range [0,1]. Here, etaHat = y/deltaHat, where deltaHat is the incompressible 
    boundary layer thickness (the suffix 'Hat' denotes properties for the incompressible state). 
    Depending on the value of the grid stretching parameter (gridStretchPar), 
    one can generate both uniform (gridStretchPar = 1) and stretched grids (gridStretchPar>1).\n
    
    The grid stretching function: y_j = deltaYWall*(r^j - 1)/(r - 1) for j = 0 to N-1. Here:\n
    y_j: Coordinate of grid point at index j\n
    deltaYWall: Grid spacing at the wall\n
    r: Grid stretching parameter at the wall\n

    Reference: https://wmles.umd.edu/wall-stress-models/wall-model-ode/\n
   
    Input parameters
    ----------------
    N: Number of wall-normal grid points in boundary layer\n
    gridStretchPar: Grid stretching parameter\n

    Output results
    --------------
    etaHat: List of wall-normal grid points in the incompressible boundary layer\n 
    """

    #we will assume a constant grid stretching parameter  

    #the current stretched grid function will not work for uniform grids
    #Consequently, we add an if-else condition to define uniform/stretched grids
    if (gridStretchPar == 1):
        
        #uniform grid
        etaHat = np.linspace(0, 1, N, endpoint=True)

    else:

        #stretched grid
        #we first find the grid spacing at the wall for
        #the given number of grid points N and grid stretching parameter
        deltaEtaHatWall = (gridStretchPar - 1)/(gridStretchPar**(N-1) - 1)
        
        #initialize etaHat
        etaHat = np.array(np.zeros([N,]))

        #generate grid
        for i in range(0,N):
            
            etaHat[i] = deltaEtaHatWall*(gridStretchPar**i - 1)/(gridStretchPar - 1)


    etaHat = 1.*etaHat
    return etaHat



def thetaDelta(eta, rho_rhoe, U_Ue):
    
    """
    Compute the boundary layer thickness ratio theta/delta\n

    This function evaluates the ratio of momentum thickness theta to delta99 
    (wall-normal location, where U = 0.99 Ue, here U and Ue are the local 
    and freestream velocity, respectively).\n

    Input parameters
    ----------------
    eta: List of wall-normal grid points, normalized by boundary layer thickness (eta = y/delta)\n 
    rho_rhoe: Boundary layer density profile normalized by the freestream density\n
    U_Ue: Boundary layer velocity profile normalized by the freestream velocity\n

    Output results
    --------------
    thetaDeltaVal: ratio of boundary layer thickness ratio theta/delta\n

    """
    
    #initialize list for storing the integrand
    functionValue = np.zeros(np.size(U_Ue))
    
    for i in range(0, len(U_Ue)):
        #compute integrand
        functionValue[i] = rho_rhoe[i] * U_Ue[i] * (1 - U_Ue[i])
    
    #integrate using Trapezoidal rule
    thetaDeltaVal = integrate.trapz(functionValue, eta)
    
    return thetaDeltaVal



def inverseVelocityTransform(yHatPlus, uHatPlus, rho_rhow, mu_muw, flagVelocityTransform):

    """
    Computes the compressible state velocity by transforming the incompressible velocity profile\n

    The transformation process requires the definition of an inverse velocity transform function.
    Note that the suffix 'Hat' denotes properties for the incompressible state.\n

    Input parameters
    ----------------
    yHatPlus: Wall-normal grid points for the incompressible boundary layer (in plus units)\n
    uHatPlus: Velocity profile for incompressible boundary layer (in plus units)\n
    rho_rhow: Boundary layer density profile normalized by the density at the wall\n
    mu_muw: Boundary layer viscosity profile normalized by the viscosity at the wall\n
    flagVelocityTransform: Choice of inverse velocity transform function\n

    Output results
    --------------
    yPlus: Wall-normal grid points for the compressible boundary layer (in plus units)\n
    uPlus: Velocity profile for the compressible boundary layer (in plus units)\n
    """ 
    
    #initialize lists
    yPlus = np.zeros(len(yHatPlus))
    uPlus = np.zeros(len(yHatPlus))
    
    #check for flag selection
    if flagVelocityTransform == 1:
        #Inverse of the Van Driest velocity transform function
        #Reference for Van Driest transform: https://doi.org/10.2514/8.1895
         
        for i in range(0,len(yHatPlus)-1):

            #integration using Trapezoidal rule
            F = ( 1/np.sqrt(np.abs(rho_rhow[i])) + 1/np.sqrt(np.abs(rho_rhow[i+1])) )/2
             
            uPlus[i+1] = uPlus[i] + F*(uHatPlus[i+1]-uHatPlus[i])

        #in Van Driest transform, yHatPlus remains unchanged, hence:
        yPlus = yHatPlus.copy()


    elif flagVelocityTransform == 2:
        #Inverse of the Trettel-Larsson velocity transform function
        #Reference for Trettel-Larsson transform: https://doi.org/10.1063/1.4942022 
        
        #first transform yHatPlus
        for i in range(0,len(yHatPlus)):
            yPlus[i] = (1/(np.sqrt(rho_rhow[i])/mu_muw[i]))*yHatPlus[i]
         
        #now transform uHatPlus
        #integration is performed using trapezoidal rule
        for i in range(0,len(yHatPlus)-1):
            temp1 = (1/mu_muw[i] + 1/mu_muw[i+1])/2
            temp2 = (yPlus[i+1]-yPlus[i])/(yHatPlus[i+1]-yHatPlus[i])
            uPlus[i+1] = uPlus[i] + (temp1*temp2*(uHatPlus[i+1]-uHatPlus[i]))
        
 
    elif flagVelocityTransform == 3:
        #Inverse of the Volpiani velocity transform function
        #Reference for the Volpiani transform function: https://doi.org/10.1103/PhysRevFluids.5.052602
        
        for j in range(0,len(yHatPlus)-1):
        
            #mean of function values for yHatPlus to yPlus transformation
            F = (1/((rho_rhow[j]**0.5)*(mu_muw[j]**-1.5)) + \
                1/((rho_rhow[j+1]**0.5)*(mu_muw[j+1]**-1.5)))/2
        
            #mean of function values for uHatPlus to uPlus transformation
            G = (1/((rho_rhow[j]**0.5)*(mu_muw[j]**-0.5)) + \
                1/((rho_rhow[j+1]**0.5)*(mu_muw[j+1]**-0.5)))/2
        
            #integration is performed using trapezoidal rule
            
            #transform yHatPlus
            yPlus[j+1] = yPlus[j] + F * (yHatPlus[j+1]-yHatPlus[j])
        
            #transform velocity by integration
            uPlus[j+1] = uPlus[j] + G * (uHatPlus[j+1]-uHatPlus[j])


    else:
        #if the flag value is out of bounds, return an error message and exit program execution
        sys.exit('Invalid choice for inverse velocity transformation - Check function inverseVelocityTransform')
    
    return [yPlus, uPlus]



def computeBq(Pr,T_Tw,yPlus,gridStretchPar):
    
    """
    Compute the wall heat transfer rate Bq\n 

    This function computes the wall heat transfer rate Bq = qw/(Cp rhow uTau Tw),
    where the variable definitions are as follows:\n
    qw: Wall heat flux\n
    Cp: Specific heat at constant pressure\n
    rhow: Density at the wall location\n
    uTau: Friction velocity\n
    Tw: Wall temperature\n

    The definition can be rearranged to use only non-dimensional quantities, 
    which gives us Bq = (-1/Pr)*(dTPlus/dyPlus)|w. Here:\n
    Pr: Prandtl number\n
    (dTplus/dyPlus)|w: Wall temperature gradient (in plus units)\n

    Input parameters
    ----------------
    Pr: Prandtl number\n
    T_Tw: Boundary layer temperature profile, normalized by the wall temperature\n
    yPlus: Boundary layer grid points in the wall-normal direction (in plus units)\n
    gridStretchPar: Grid stretching parameter\n

    Output results
    --------------
    Bq: Wall heat transfer rate\n
    """ 
    
    #first, we need to check if we have a uniform or stretched grid 
    if (gridStretchPar == 1):

        #uniform grid
        #we can hence directly use finite difference 
        #approximations to find the wall temperature gradient

        #second-order accurate, forward-difference scheme
        Bq = -(1/Pr)*(-3*T_Tw[0]+4*T_Tw[1]-1*T_Tw[2])/(2*(yPlus[1] - yPlus[0]))
        
        ##first-order accurate, forward-difference scheme
        #Bq = -(1/Pr)*(-1*T_Tw[0] + 1*T_Tw[1])/(yPlus[1] - yPlus[0])

    else:

        #stretched grid
        #for stretched grids, we need to follow the generalised approach
        #assume a computational space S - here the grid is uniform
        #in the current function, we use the grid point numbers for S

        #find size of list
        gridSize = len(T_Tw)
        #create uniform grid in computational space
        S = np.linspace(0,gridSize-1,gridSize)
        
        #compute the wall temperature gradient in computational space
        #second-order accurate, forward-difference scheme
        dTp_dS = (-3*T_Tw[0]+4*T_Tw[1]-1*T_Tw[2])/(2*(S[1] - S[0]))

        #compute the derivative of physical space, yPlus, 
        #with respect to the computational space S
        #analytically differentiate the grid stretching function
        deltaYplusW = yPlus[1] - yPlus[0]
        dYp_dS =  (deltaYplusW/(gridStretchPar-1))*np.log(gridStretchPar)

        #now compute the wall temperature gradient in the physical space
        dTp_dYp = dTp_dS/dYp_dS

        #finally, compute wall heat transfer rate
        Bq = -(1/Pr)*dTp_dYp
 
    return Bq



def wakeParameterFromDeltaHatPlus(etaHat, deltaHatPlus, flagIncCf):
    
    """
    Compute wake parameter for the universal incompressible velocity profile\n

    This function uses a calibration process to compute the wake strength parameter.
    The calibration process begins with an initial guess to the wake parameter, 
    followed by comparing the skin-friction coefficient for the modeled incompressible 
    velocity profile with the one obtained using an incompressible skin-friction relation. 
    The misfit in the two coefficients is then used to update the guess value for the 
    wake parameter. The process is repeated until both the skin-friction coefficients 
    are within a preset tolerance to one another.\n
    
    Note that the use of suffix 'Hat' denotes properties for the incompressible state.\n

    Input parameters
    ----------------
    etaHat: List of wall-normal grid points in the incompressible boundary layer (etaHat = y/deltaHat)\n
    deltaHatPlus: Incompressible boundary layer thickness (in plus units)\n
    flagIncCf: Choice of incompressible skin-friction relation, see function skinFrictionIncompressible\n
   
    Output results
    --------------
    wakeParameterFinal: Converged value of the wake parameter\n 
    """ 

    def wakeParameterCore(yHatPlus, wakeParameter, deltaHatPlus, flagIncCf):

        """
        Core function for wake parameter estimation

        This function is placed inside the wakeParameterFromDeltaHatPlus function
        and involves computing the misfit in the skin-friction coefficients obtained from
        the modeled incompressible velocity profile and the chosen incompressible relation.

        Input parameters
        ----------------
        yHatPlus: wall-normal grid points for incompressible boundary layer (in plus units)
        wakeParameter: guess or current value of the wake parameter
        deltaHatPlus: Incompressible boundary layer thickness (in plus units)
        flagIncCf: Choice of incompressible skin-friction relation, see function skinFrictionIncompressible

        Output results
        --------------
        misfitValue: The misfit in the incompressible skin-friction coefficient for 
                     the modeled velocity profile and the one obtained using the 
                     chosen incompressible skin-friction relation 
        """ 
        
        #compute incompressible velocity profile
        uHatPlus = universalVelocityProfile(yHatPlus, deltaHatPlus, wakeParameter)
        #convert grid points from inner to outer layer scaling
        etaHat = np.divide(yHatPlus, deltaHatPlus)
        #density stays uniform for incompressible flow, hence density ratio = 1
        rho_rhoe = np.ones(np.size(etaHat))
        #normalize velocity profile (in plus units) with freestream velocity
        U_Ue = np.divide(uHatPlus, uHatPlus[(-1)])
        #compute the boundary layer thickness ratio theta/delta
        theta_delta = thetaDelta(etaHat, rho_rhoe, U_Ue)
        #compute Reynolds number Re_theta
        ReThetaHat = deltaHatPlus * uHatPlus[-1] * theta_delta
        #compute incompressible skin-friction using empirical curve fits
        cfHatReTheta = skinFrictionIncompressible(ReThetaHat, flagIncCf)
        #compute incompressible skin-friction using the modeled velocity profile uHatPlus
        cfHatModeled = np.divide(2, np.power(uHatPlus[-1], 2))

        #compute misfit in the two incompressible cf values
        misfitValue = cfHatModeled - cfHatReTheta

        return misfitValue


    #coming back to the original function
    
    #compute wall-normal grid points in inner scaling (plus units)
    yHatPlus = np.multiply(etaHat, deltaHatPlus)
    #initialize error, tolerance and iteration counter for the calibration process
    error = 1
    tol = 1e-09
    iterCount = 1

    #create an empty list to store the wake parameter at each iteration
    wakeParameter = {}
    #initialize wake parameter
    wakeParameter[0] = 0.5
    
    #create an empty list to store the misfit in incompressible skin-friction coefficients
    functionVal = {}

    #In the current implementation, we are using the secant method for the iterative process
    #Hence, we first need to generate the misfit for the first two iterations

    #call core function with initial guess value of wake parameter
    #store misfit to decide on second guess value and for subsequent iterations 
    functionVal[0] = wakeParameterCore(yHatPlus, wakeParameter[0], deltaHatPlus, flagIncCf)

    #now decide on second guess value
    
    #if we have a positive misfit, it implies that our modeled incompressible cf is larger
    #hence, we have a smaller \hat{U^+_e}, hence a smaller wake parameter --> INCREASE
    #Note that I have kept the greater than equal to sign here
    
    if functionVal[0] >= 0:
        #we choose second guess to be 20% larger than first guess  
        wakeParameter[1] = 1.2*wakeParameter[0]

    #otherwise, if we have a negative misfit, 
    #it implies that our modeled incompressible cf is smaller
    #hence, we have a larger \hat{u^+_e}, hence a larger wake parameter --> DECREASE

    elif functionVal[0] < 0:
        #we choose second guess to be 20% smaller than first guess
        wakeParameter[1] = 0.8*wakeParameter[0]

    #now compute misfit for second guess value
    functionVal[1] = wakeParameterCore(yHatPlus, wakeParameter[1], deltaHatPlus, flagIncCf)

    #now iterate until convergence using the secant method
    while error > tol:
        wakeParameter[iterCount + 1] = wakeParameter[iterCount] - functionVal[iterCount] * \
                (wakeParameter[iterCount] - wakeParameter[(iterCount - 1)]) / \
                (functionVal[iterCount] - functionVal[iterCount - 1])
        
        functionVal[iterCount + 1] = \
                wakeParameterCore(yHatPlus, wakeParameter[iterCount + 1], deltaHatPlus, flagIncCf)
        
        #error is the absolute value of misfit in cfHat for the current iteration
        error = np.abs(functionVal[iterCount + 1])
        #update iteration counter 
        iterCount = iterCount + 1

    #once converged, store the last value from the list    
    wakeParameterFinal = list(wakeParameter.values())[(-1)]    

    return wakeParameterFinal


def boundaryLayerProperties(M, ReDelta2, Tw_Te, Te, N, **kwargs):
    
    """
    Compute skin-friction, heat transfer along with boundary layer velocity and temperature profiles.\n
    
    This is the main function that computes the desired boundary layer properties. 
    The algorithm begins with an initial guess for the incompressible boundary layer thickness 
    (deltaHatPlus, in plus units). In the current implementation, the initial guess value is hardcoded
    as deltaHatPlus = ReDelta2/3, where ReDelta2 is the input Reynolds number for the compressible boundary layer.
    This is followed by using the function universalVelocityProfile to compute the incompressible velocity profile, 
    which is then transformed to the desired compressible state using the function inverseVelocityTransform. 
    The process is iterated until the Reynolds number for the modeled velocity profile is within a preset tolerance
    of the input Reynolds number.\n

    Input parameters
    ----------------
    M: Mach number\n
    ReDelta2: Reynolds number\n
    Tw_Te: Ratio of wall to freestream temperature Tw/Te\n
    Te: Freestream temperature (in K)\n 
    N: number of grid points\n
    [OPTIONAL KEYWORDS]\n 
    flagViscTemp: flag for choosing viscosity-temperature relation, Optional\n
    flagIncCf: flag for choosing incompressible skin-friction relation, Optional\n
    flagTempVelocity: flag for choosing temperature-velocity relation, Optional\n
    flagVelocityTransform: flag for choosing inverse velocity transform function, Optional\n
    gridStretchPar: grid stretching parameter\n
    underRelaxFactor: factor for under relaxing the speed of an iterative process\n
    
    Output results
    --------------
    cfFinal: Skin-friction coefficient\n
    Bq: Wall heat transfer rate\n
    ch: Stanton number\n
    yPlusCore: Wall-normal points for the compressible boundary layer (in plus units)\n
    uPlusCore: Boundary layer velocity profile (in plus units)\n
    T_Tw: Boundary layer temperature profile, normalized by the wall temperature Tw\n
    """

    #check for model choice flags - optional inputs
    if (kwargs == {}):
        #Notify that the code will run with default choices
        print('Running with default choice for models')
        
    #set default values for system parameters
    gridStretchPar = 1.016
    underRelaxFactor = 1
    
    #set default values for flags
    flagViscTemp = 3
    flagIncCf = 3
    flagTempVelocity = 1
    flagVelocityTransform = 3

    #check if some flags have been specified 
    #if yes, read and overwrite default value
    #if not, keep the default value
    flagViscTemp = kwargs.get('flagViscTemp',flagViscTemp)
    flagIncCf = kwargs.get('flagIncCf',flagIncCf)
    flagTempVelocity = kwargs.get('flagTempVelocity',flagTempVelocity)
    flagVelocityTransform = kwargs.get('flagVelocityTransform',flagVelocityTransform)
    
    #modify under-relaxation for Trettel-Larsson transform
    if (flagVelocityTransform == 2):
        underRelaxFactor = 0.5
    
    #overwrite quantities if user provides inputs
    gridStretchPar = kwargs.get('gridStretchPar',gridStretchPar)
    underRelaxFactor = kwargs.get('underRelaxFactor',underRelaxFactor)
    

    #define constants
    #gamma is ratio of specific heats. For the current work, we consider ideal diatomic gas
    gamma = 1.4
    #recovery factor
    r = 0.896
    #Prandtl number
    Pr = 0.72

    #threshold for grid resolution (in plus units)
    wallResolutionThreshold = 0.2

    #define core function
    def boundaryLayerPropertiesCore(etaHat, deltaHatPlus, T_Tw, Tw_Te, Tr_Te, Tw, underRelaxFactor, flagViscTemp, flagIncCf, flagTempVelocity, flagVelocityTransform):
        """
        Core function inside boundaryLayerProperties.

        This function performs the core steps of defining the universal incompressible velocity profile and then transforming
        it to the desired compressible state. A Reynolds number can then be computed for this modeled velocity profile,
        which is returned by this function for subsequent comparison with the input Reynolds number ReDelta2.
        
        Note that the suffix 'Hat' in variables corresponds to properties at the incompressible state

        Input parameters
        ----------------
        etaHat: Wall-normal grid points for the incompressible boundary layer in outer scaling (etaHat = y/deltaHat)
        deltaHatPlus: Incompressible boundary layer thickness (in plus units)
        T_Tw: Boundary layer temperature profile, normalized by the wall temperature Tw
        Tw_Te: Ratio of wall to freestream temperature, Tw/Te
        Tr_Te: Ratio of recovery to freestream temperature, Tr_Te
        Tw: Wall temperature (in K)
        underRelaxFactor: factor for under relaxing the speed of an iterative process
        flagViscTemp: flag for choosing viscosity-temperature relation
        flagIncCf: flag for choosing incompressible skin-friction relation
        flagTempVelocity: flag for choosing temperature-velocity relation
        flagVelocityTransform: flag for choosing inverse velocity transform function

        Output results
        --------------
        ReDelta2Modeled: Reynolds number ReDelta2 for the modeled boundary layer velocity profile
        T_Tw: Updated boundary layer temperature profile, normalized by the wall temperature Tw
        cf: Skin friction coefficient
        """ 

        #define global variables to allow usage outside the core function
        global yPlusCore
        global uPlusCore
        
        #scale incompressible boundary layer grid to plus units
        yHatPlus = np.multiply(etaHat, deltaHatPlus)
        #compute wake parameter
        wakeParameter = wakeParameterFromDeltaHatPlus(etaHat, deltaHatPlus, flagIncCf)
        #compute universal incompressible velocity profile
        uHatPlus = universalVelocityProfile(yHatPlus, deltaHatPlus, wakeParameter)
    
        #set/initialize error, tolerance and iteration counter
        errorTemperatureProfile = 1
        tolTemperatureProfile = 1e-09
        iterationTemperatureProfile = 0
         
        #we iterate until we get a converged temperature profile
        while (np.abs(errorTemperatureProfile) > tolTemperatureProfile):
            
            #compute density and viscosity profiles
            rho_rhow = densityFromTemperature(T_Tw)
            mu_muw = viscosityFromTemperature(T_Tw, Tw, flagViscTemp)
            #compute velocity profile for compressible state
            [yPlusCore, uPlusCore] = inverseVelocityTransform(yHatPlus, uHatPlus, rho_rhow, mu_muw, flagVelocityTransform)
            
            #scale velocity profile in plus units to be normalized with freestream velocity instead
            U_Ue = np.divide(uPlusCore, uPlusCore[(-1)])
            #store old temperature profile for error estimation
            T_TwOld = T_Tw.copy()
            #compute updated temperature profile
            T_Tw = temperatureFromVelocity(U_Ue, Tw_Te, Tr_Te, flagTempVelocity)
            #apply under-relaxation 
            T_Tw = np.multiply(T_TwOld,(1 - underRelaxFactor)) + np.multiply(T_Tw, underRelaxFactor) 
            #compute RMS error in temperature profile
            errorTemperatureProfile = np.power(np.mean(np.power(T_Tw - T_TwOld, 2)), 0.5)
            #update iteration counter - this is not essential and can hence be removed
            iterationTemperatureProfile = iterationTemperatureProfile + 1

        
        #store boundary layer edge (in plus units) to compute Reynolds number 
        deltaPlus = yPlusCore[(-1)]
        #normalize density profile by freestream density 
        rho_rhoe = np.divide(rho_rhow, rho_rhow[(-1)])
        #scale wall-normal grid from inner to outer scaling
        eta = np.divide(yPlusCore, deltaPlus)
        #compute boundary layer thickness ratio theta/delta for modeled velocity profile
        theta_delta = thetaDelta(eta, rho_rhoe, U_Ue)
        #update Reynolds number for modeled velocity profile
        ReDelta2Modeled = deltaPlus * uPlusCore[(-1)] * theta_delta * rho_rhow[(-1)]
        #compute skin friction coefficient
        cf = 1 / rho_rhow[-1] * 2 / np.power(uPlusCore[-1], 2)
        
        return [ReDelta2Modeled, T_Tw, cf]


    #back to the main function

    #compute wall temperature Tw from the input temperature ratio
    Tw = Tw_Te * Te
    #compute recovery to freestream temperature ratio Tr/Te
    Tr_Te = TrTeFromMach(M, gamma, r)

    #define wall-normal grid
    etaHat = boundaryLayerGrid(N,gridStretchPar)

    #set up dict list to store values of deltaHatPlus
    deltaHatPlus = {}

    #since we use the secant method to progress through the iterative process
    #we first need to run two iterations manually
    #first guess for deltaHatPlus is made using observations from incompressible flow
    deltaHatPlus[0] = ReDelta2/3
    tolReDelta2 = 1e-09 

    #for the first guess value, we need to compute some initial values as input for the core function
    #wall-normal grid points for incompressible boundary layer (in plus units)
    yHatPlus = np.multiply(etaHat, deltaHatPlus[0])
    #wake parameter
    wakeParameter = wakeParameterFromDeltaHatPlus(etaHat, deltaHatPlus[0], flagIncCf)
    #incompressible boundary layer velocity profile
    uHatPlus = universalVelocityProfile(yHatPlus, deltaHatPlus[0], wakeParameter)
    
    #for first iteration, we make an initial guess for the compressible velocity profile
    #where it is assumed to be the same as the incompressible velocity profile
    #this is necessary for computing the temperature profile
    #hence we assume for this case: u/u_e = \hat{u}+/\hat{u}+_e
    U_Ue = np.divide(uHatPlus, uHatPlus[-1])
    
    #**********************************************************************************************
    #we use Zhang's approximate T v/s U relation for this step - IT IS HARD CODED, CHANGE IF NEEDED
    T_Tw   = temperatureFromVelocity(U_Ue, Tw_Te, Tr_Te, 1)
    #**********************************************************************************************
    
    #define python dict list for cf
    cf = {}
    
    #call core function to run first iteration
    [ReDelta2Modeled, T_Tw, cf[0]] = \
            boundaryLayerPropertiesCore(etaHat, deltaHatPlus[0], T_Tw, Tw_Te, Tr_Te, Tw,\
            underRelaxFactor, flagViscTemp, flagIncCf, flagTempVelocity, flagVelocityTransform)
    
    #define python dict list for the error in ReDelta2 at each iteration
    errorReDelta2 = {}
    #compute mismatch - to be used as function value in secant method
    errorReDelta2[0] = ReDelta2Modeled - ReDelta2
    
    #initialize iteration counter
    iterationReDelta2 = 1

    #print to screen
    print('At iteration '+str(iterationReDelta2)+\
                ' , deltaHatPlus = '+str(deltaHatPlus[iterationReDelta2 - 1])+\
                ' and errorReDelta2 = '+str(errorReDelta2[iterationReDelta2 - 1]))

    #we now use the mismatch from first iteration to decide on the second guess value for the secant method
    if errorReDelta2[0] >= 0:
        #if we have positive error (also including zero error to prevent garbage values), 
        #we are overpredicting deltaHatPlus, hence reduce it by a factor of 2
        deltaHatPlus[1] = deltaHatPlus[0] / 2
    
    elif errorReDelta2[0] < 0:
        #if error is negative , we are underpredicting deltaHatPlus, hence increase it by a factor of 2
        deltaHatPlus[1] = deltaHatPlus[0] * 2
    
    #call core function using second guess value
    [ReDelta2Modeled, T_Tw, cf[1]] = \
            boundaryLayerPropertiesCore(etaHat, deltaHatPlus[1], T_Tw, Tw_Te, Tr_Te, Tw,\
            underRelaxFactor, flagViscTemp, flagIncCf, flagTempVelocity, flagVelocityTransform)

    #compute mismatch for second guess value
    errorReDelta2[1] = ReDelta2Modeled - ReDelta2
    #update iteration counter
    iterationReDelta2 = 2

    #print to screen
    print('At iteration '+str(iterationReDelta2)+\
                ' , deltaHatPlus = '+str(deltaHatPlus[iterationReDelta2 - 1])+\
                ' and errorReDelta2 = '+str(errorReDelta2[iterationReDelta2 - 1]))
    
    #now iterate until we have a converged ReDelta2
    while (np.abs(errorReDelta2[iterationReDelta2 - 1]) > tolReDelta2):
        
        #apply secant method to compute deltaHatPlus in current iteration
        deltaHatPlus[iterationReDelta2] = \
                deltaHatPlus[iterationReDelta2 - 1] - errorReDelta2[iterationReDelta2 - 1] * \
                (deltaHatPlus[iterationReDelta2 - 1] - deltaHatPlus[iterationReDelta2 - 2]) / \
                (errorReDelta2[iterationReDelta2 - 1] - errorReDelta2[iterationReDelta2 - 2])
       
        #under-relax solution to prevent solution blow-up
        deltaHatPlus[iterationReDelta2] = (1 - underRelaxFactor)*deltaHatPlus[iterationReDelta2 - 1] +\
                underRelaxFactor*deltaHatPlus[iterationReDelta2]

        #call core function
        [ReDelta2Modeled, T_Tw, cf[iterationReDelta2]] = \
                boundaryLayerPropertiesCore(etaHat, deltaHatPlus[iterationReDelta2], \
                T_Tw, Tw_Te, Tr_Te, Tw, underRelaxFactor, flagViscTemp, flagIncCf, flagTempVelocity, flagVelocityTransform)
        
        #update error
        errorReDelta2[iterationReDelta2] = ReDelta2Modeled - ReDelta2
        
        #update iteration counter
        iterationReDelta2 = iterationReDelta2 + 1
            
        #print to screen
        print('At iteration '+str(iterationReDelta2)+\
                ' , deltaHatPlus = '+str(deltaHatPlus[iterationReDelta2 - 1])+\
                ' and errorReDelta2 = '+str(errorReDelta2[iterationReDelta2 - 1]))


    #once converged, acquire final value of skin-friction coefficient
    cfFinal = list(cf.values())[(-1)]

    #ratio of wall to recovery temperature, to be used for subsequent computation
    Tw_Tr = Tw_Te/Tr_Te

    #compute wall heat transfer rate Bq
    Bq = computeBq(Pr,T_Tw,yPlusCore,gridStretchPar)

    #compute Stanton number c_h
    if (Tw_Tr == 1):
        #Stanton number is not defined for adiabatic walls
        ch = np.nan
    else:
        #compute Stanton number for non-adiabatic walls
        ch = Bq*(1/Tw_Te)*(1/uPlusCore[-1])*(1/(1 - 1/Tw_Tr))


    #check for grid size at the wall
    #since the first grid point is at the wall (yPlusCore = 0),
    #the grid size at the wall is just the location of the second grid point
    if yPlusCore[1]>wallResolutionThreshold:
        print('WARNING: Grid spacing at the wall = '+str(np.round(yPlusCore[1],3))+ \
                ' is GREATER than threshold value = '+str(wallResolutionThreshold)+' (in plus units)')
        print('RUN AGAIN WITH A FINER GRID AT THE WALL')


    #compute viscosity for output 
    mu_muw = viscosityFromTemperature(T_Tw, Tw, flagViscTemp)

    return [cfFinal,Bq,ch,yPlusCore,uPlusCore,T_Tw,mu_muw]



def gridProperties(yPlus,T_Tw,mu_muw,alf_plus,jb,ypw,**kwargs):
    
    """
    Compute natural wall-normal grid distribution.\n
    
    This is the main function that computes the natural grid stretching for compressible boundary laeyers. 
    The algorithm begins with an initial guess for the value of alpha. 
    In the current implementation, the initial guess value is hardcoded
    as alf = alf_plus/2, where alf is the outer layer constant resolution in semi-local Kolmogorov units
    and alf_plus is the target resolution in wall Kolmogorov units.
    This is followed by using the grid stretching function universalGridStretch which computes the wall-normal
    grid for a given alf parameter. The resulting spacing Delta y^* is then compared with the analytical formulation
    so that the resolution in Kolmogorov wall units does not exceed a certain treshold (alf_plus defined by the user) 
    The process is iterated until the alf parameted for the grid is not changing within a preset tolerance.\n

    Input parameters
    ----------------
    yPlus  : Wall-normal points for the compressible boundary layer (in plus units)\n
    T_Tw   : Boundary layer temperature profile, normalized by the wall temperature Tw
    mu_muw : Boundary layer viscosity   profile, normalized by the wall viscosity   muw
    alf_plus : target resolution in wall Kolmogorov units\n
    jb       : transition node from viscous to outer stretching\n
    ypw      : target wall distance from the wall in inner units\n
    myNy     : User defined number of points up to BL edge\n
    
    Output results
    --------------
    yStarCore_j   : semilocal scaled wall distance function of the number of points in the wall normal direction\n
    yPlusCore_j   : wall scaled  wall distance function of the number of points in the wall normal direction\n
    jjCore        : index  of points in the wall normal direction\n
    NyCore        : number of points in the wall normal direction\n
    alf_opt       : optimal alf star to respect threshold resolution in wall Kolmogorov units\n
    etaPlusCore_j : estimated etaPlus profile, useful to chek resolution requirements\n
    """

    print('\n')

    #check for model choice flags - optional inputs
    if (kwargs == {}):
        #Notify that the code will run with default choices
        print('Running with default choice for Ny points in wall normal direction')

    #set default values for system parameters
    myNy = jb

    #overwrite quantities if user provides inputs
    myNy = kwargs.get('myNy',myNy)

    #generate semilocal grid from the wall one
    yStar      = wallToSemilocal(yPlus, T_Tw, mu_muw)
    #measure Retau_Star, at boundary layer edge
    Retau_Star = yStar[-1]
    #generate interpolating functions to compute T_Tw and mu_muw on any grid spacing given yStar distribution
    [fT_star,fmu_star] = StarMaps(yStar,T_Tw, mu_muw)

    #optimize alf_star so that alf_plus resolution is not exceeded
    #first alf guess
    alf      = alf_plus/2
    #initialize error and tolerances
    error    = 10.
    mytol    = 1e-9
    iter_max = 1000
    count    = 1

    #von Karman constant
    k    = 0.41
    #relaxation factor to update alf
    relax_fac = 1. 

    #get the ysw in order to obtain the desired ysp
    # first guess
    ysw = ypw 
    # compute grid
    [yStar_j,yPlus_j,jj,Ny] = universalGridStretch(alf,jb,ysw,Retau_Star,fT_star,fmu_star,0)
    # adapt ysw for target ypw
    ysw = ypw*yStar_j[1]/yPlus_j[1]

    while (np.abs(error) > mytol) and (count < iter_max):
       #compute grid with guessed/corrected alf   
       [yStar_j,yPlus_j,jj,Ny] = universalGridStretch(alf,jb,ysw,Retau_Star,fT_star,fmu_star,0)
       etaStar_j = (k*yStar_j)**0.25
       etaPlus_j = SemilocalToWall(etaStar_j, fT_star(yStar_j), fmu_star(yStar_j))
       #check resolution requirement
       dyeta = np.gradient(yPlus_j[1:])/etaPlus_j[1:] #avoid singularity at y=0
       error = (alf_plus-max(dyeta))
       print( 'At iteration '+str(count)+\
              ', alf_star parameter = '+str(alf)+\
              ', error = '+str(error) )
       #update iteration counter
       count = count + 1
       #update alf
       alfnew = alf + error
       alfold = alf
       alf    = (1.-relax_fac)*alfold + relax_fac*alfnew
    print('\n')
    print("Computed ysw = "+str(ysw)+" for target ypw = "+str(ypw))
    print("Found optimal alf_star = "+str(alf)+", at iteration count = "+str(count))

    #final "optimal" grid
    [yStar_j,yPlus_j,jj,Ny] = universalGridStretch(alf,jb,ysw,Retau_Star,fT_star,fmu_star,myNy)

    #output variables
    yPlusCore_j   = yPlus_j
    yStarCore_j   = yStar_j
    jjCore        = jj
    NyCore        = Ny
    alf_opt       = alf
    etaPlusCore_j = SemilocalToWall((k*yStarCore_j)**0.25, fT_star(yStarCore_j), fmu_star(yStarCore_j)) 

    return [yPlusCore_j,yStarCore_j,jjCore,NyCore,alf_opt,etaPlusCore_j]

