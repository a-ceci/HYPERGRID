"""
A Python script for predicting compressible boundary layer properties such as the skin friction coefficient, \n 
heat transfer (Wall heat transfer rate, Stanton number) along with the velocity and temperature profiles. \n

The current implementation is based on the method described in "Modular Method for Estimation of Velocity and Temperature Profiles in High-Speed Boundary Layers" \n
DOI: https://doi.org/10.2514/1.J061735 \n

The method seeks the following inputs: Mach number, Reynolds number (ReDelta2), Wall and freestream temperatures \n

The script contains multiple functions, with boundaryLayerProperties calling other functions for specific tasks \n

Update 30th October 2024 by Alessandro Ceci: introduce direct transform apprach including HLPP transformation for version 2.0, \n
reference DOI: https://doi.org/10.1103/PhysRevFluids.8.L112601 \n
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
    try:
       T_Tw = [float(i) for i in T_Tw]
    except:
       T_Tw = float(T_Tw)

    #in the current version, we only use equation of state
    #additional functions can be added with the selection process controlled by flags as shown below
    rho_rhow = np.reciprocal(T_Tw)
    
    return rho_rhow



def viscosityFromTemperature(T_Tw,Tw,flagViscTemp):
    
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
        alpha_T = 1.0958
        delta_T = 1.5415
        kappa_T = -0.3

        for i in range(0, len(T)):
            #compute dynamic viscosity
            mu[i] = (A*T[i]**alpha_T)*(1 + D*T[i]**delta_T)**kappa_T
        
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



def skinFrictionIncompressible(ReThetaHat,flagIncCf):

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



def temperatureFromVelocity(U_Ue,Tw_Te,Tr_Te,flagTempVelocity):
    
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


def TrTeFromMach(M,g,r):

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



def wallToSemilocal(qPlus,T_Tw,mu_muw):

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



def SemilocalToWall(qStar,T_Tw,mu_muw):

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



def StarMaps(yStar,T_Tw,mu_muw):

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



def PlusMaps(yPlus,T_Tw,mu_muw):

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



def universalGridStretch(alf,jb,ysw,Retau_Star,fT_star,fmu_star,Nypoints,kappa):
    
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
    k    = kappa
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



def compressibleVelocityProfile(yPlus,MTau,rho_rhow,mu_muw,wakeParameter,Aplus,kappa,flagVelocityTransform):
 
    """
    Compute the target compressible boundary layer velocity profile by integrating an ODE \n

    This function takes in the wall-normal grid (in plus units) along with the density and 
    viscosity profiles to integrate an ODE described in Hasan et al. (https://doi.org/10.2514/1.J063335)
    which directly gives the compressible boundary layer velocity profile.\n 

    Input parameters
    ----------------
    yPlus: Wall-normal coordinates for the compressible boundary layer (in plus units)\n
    MTau: Friction Mach number = Friction velocity/speed of sound at the wall\n
    rho_rhow: Density profile normalized by density at the wall\n
    mu_muw: Viscosity profile normalized by viscosity at the wall\n
    wakeParameter: wake parameter Pi\n
    flagVelocityTransform: Choice of velocity transform function, used as a compressible eddy viscosity model\n

    Output results
    --------------
    uPlusInner: Inner layer component of the compressible velocity profile\n
    uPlusOuter: Outer layer component of the compressible velocity profile\n
    """


    Ny = len(yPlus)
    ReTau = np.copy(yPlus[-1])
    yod = yPlus/ReTau

    uPlusInner = np.zeros([Ny,])
    uPlusOuter = np.zeros([Ny,])

    dudyPlusInner = np.zeros([Ny,])
    dudyPlusOuter = np.zeros([Ny,])

    #compute yStar from yPlus
    plusStarScalingTerm = (mu_muw)*np.sqrt(1/rho_rhow)

    yStar = yPlus/plusStarScalingTerm

    #compute the eddy viscosity using choice of velocity transform function

    if (flagVelocityTransform == 2): # TL transform
        funMtau = 0
    elif (flagVelocityTransform == 3): # HLPP transform
        funMtau = 19.3*MTau
    else:
        sys.exit('Fix the compressible eddy viscosity calculation!')

    for i in range(Ny):

        D = (1 - np.exp(-yStar[i]/(Aplus + funMtau)))**2
        mut_muw = kappa*mu_muw[i]*yStar[i]*D

        dudyPlusInner[i] = 1/(mu_muw[i] + mut_muw)

        dudyPlusOuter[i] = \
                (np.sqrt(1/rho_rhow[i])/ReTau)*\
                (wakeParameter/kappa)*np.pi*np.sin(np.pi*yod[i])

    #now, both need to be integrated separately to get u^+_inner and u^+_outer
    #use trapezoidal rule
    for i in range(0,Ny-1):
        uPlusInner[i+1] = uPlusInner[i] + \
                ((dudyPlusInner[i] + dudyPlusInner[i+1])/2)*(yPlus[i+1] - yPlus[i])
        uPlusOuter[i+1] = uPlusOuter[i] + \
                ((dudyPlusOuter[i] + dudyPlusOuter[i+1])/2)*(yPlus[i+1] - yPlus[i])

    return [uPlusInner, uPlusOuter]

    

def forwardVelocityTransform(yPlusTemp,uPlusTemp,rho_rhow,mu_muw,MTau,Aplus,kappa,flagVelocityTransform):

    """
    Computes the equivalent incompressible state velocity profile by transforming the compressible velocity profile\n

    The transformation process requires the definition of a forward velocity transform function.
    Note that the suffix 'Hat' denotes properties for the incompressible state.\n

    Input parameters
    ----------------
    yPlusTemp: Wall-normal grid points for the compressible boundary layer (in plus units)\n
    uPlusTemp: Velocity profile for compressible boundary layer (in plus units)\n
    rho_rhow: Boundary layer density profile normalized by the density at the wall\n
    mu_muw: Boundary layer viscosity profile normalized by the viscosity at the wall\n
    flagVelocityTransform: Choice of forward velocity transform function\n

    Output results
    --------------
    yHatPlus: Wall-normal grid points for the incompressible boundary layer (in plus units)\n
    uHatPlus: Velocity profile for the incompressible boundary layer (in plus units)\n
    """

    #initialize variables
    uHatPlus = np.zeros(len(yPlusTemp))

    if (flagVelocityTransform == 1): # VD transform
        #Van Driest
        for i in range(0,len(yPlusTemp)-1):

            temp = (np.sqrt(rho_rhow[i]) + np.sqrt(rho_rhow[i+1]))/2
            uHatPlus[i+1] = uHatPlus[i] + temp*(uPlusTemp[i+1] - uPlusTemp[i])
        
        yHatPlus = np.copy(yPlusTemp)


    elif (flagVelocityTransform == 2): # TL transform
        #Trettel-Larsson
        yHatPlus = yPlusTemp*(1/mu_muw)*np.sqrt(rho_rhow)

        for i in range(0,len(yPlusTemp)-1):
            temp1 = (mu_muw[i] + mu_muw[i+1])/2
            temp2 = (yHatPlus[i+1] - yHatPlus[i])/(yPlusTemp[i+1] - yPlusTemp[i])
            uHatPlus[i+1] = uHatPlus[i] + temp1*temp2*(uPlusTemp[i+1] - uPlusTemp[i])


    elif (flagVelocityTransform == 3): # HLPP transform
        #HLPP
        #transform the wall-normal coordinate
        yHatPlus = yPlusTemp*(1/mu_muw)*np.sqrt(rho_rhow)

        funMTau = 19.3*MTau

        #initialize array for storing eddy viscosity ratio
        eddyViscRatio = np.zeros(len(yHatPlus))

        for ind in range(0,len(yPlusTemp)):
            
            #incompressible eddy viscosity term
            eddyViscInc = 1 + kappa*yHatPlus[ind]*\
                    (1 - np.exp(-yHatPlus[ind]/Aplus))**2

            #compressible eddy viscosity term
            eddyViscC = 1 + kappa*yHatPlus[ind]*\
                    (1 - np.exp(-yHatPlus[ind]/(Aplus + funMTau)))**2
   
            eddyViscRatio[ind] = eddyViscC/eddyViscInc

        #now integrate
        for i in range(0,len(yPlusTemp)-1):
            
            temp1 = (mu_muw[i] + mu_muw[i+1])/2
            temp2 = (yHatPlus[i+1] - yHatPlus[i])/(yPlusTemp[i+1] - yPlusTemp[i]) 
            uHatPlus[i+1] = uHatPlus[i] + temp1*temp2*eddyViscRatio[i]*\
                    (uPlusTemp[i+1] - uPlusTemp[i])

    else:
        sys.exit('Incorrect flag choice')


    return [yHatPlus,uHatPlus]



def universalVelocityProfile(yPlus,yHatPlus,wakeParameter,Aplus,kappa):

    """
    Compute the universal incompressible velocity profile\n

    This function computes the velocity profile for an equivalent incompressible boundary layer
    on a wall-normal yHatPlus grid. Since the outer layer is universal on the compressible y/delta 
    grid, the compressible yPlus grid is also needed as an input.
    For the universal profile definition used in the current study, we also require a 
    wake strength parameter which is computed either from a curve-fit or a calibration process.\n

    Input parameters
    ----------------
    yPlus: Wall-normal grid points (in plus units) for the compressible boundary layer\n
    yHatPlus: Wall-normal grid points (in plus units) for the incompressible boundary layer\n
    wakeParameter: Wake parameter Pi\n

    Output results
    --------------
    uHatPlusInner: Inner layer component of the incompressible boundary layer velocity profile (in plus units)\n
    uHatPlusOuter: Outer layer component of the incompressible boundary layer velocity profile (in plus units)\n

    """
   
    Ny = len(yHatPlus)

    #initialize variables
    dudy_HatPlusInnerArr = np.zeros([Ny,])
    uHatPlusInner = np.zeros([Ny,])

    #compute outer scaling coordinate
    yod = yPlus/yPlus[-1]
 
    #we need to define the integrand for the inner layer
    for i in range(0,Ny):

        D = (1 - np.exp(-yHatPlus[i]/Aplus))**2
        dudy_HatPlusInnerArr[i] = 1/(1 + kappa*yHatPlus[i]*D) 

    #now integrate using trapezoidal rule
    for i in range(0,Ny-1):

        uHatPlusInner[i+1] = uHatPlusInner[i] + \
                ((dudy_HatPlusInnerArr[i] + dudy_HatPlusInnerArr[i+1])/2)\
                *(yHatPlus[i+1] - yHatPlus[i]) 
   
    #call Coles' wake function for outer layer
    uHatPlusOuter = colesWakeFunction(yod,wakeParameter,kappa)

    return [uHatPlusInner,uHatPlusOuter]



def colesWakeFunction(etaTemp,wakeParameter,kappa):

    """
    Compute the Coles' wake function\n
    
    This function computes the Coles' wake function on a given grid and for a given wake strength parameter.\n

    Input parameters
    ----------------
    etaTemp: Wall-normal grid points (in outer scaling) for the compressible boundary layer\n
    wakeParameter: Wake parameter Pi\n

    Output results
    --------------
    uHatPlusOuter: Outer layer component of the incompressible boundary layer velocity profile (in plus units)\n

    """

    #initialize array
    uHatPlusOuter = np.zeros([len(etaTemp),])
    for i in range(len(etaTemp)):

        wakeFunction = 2*(np.sin(etaTemp[i] * np.pi/2))** 2
        uHatPlusOuter[i] = (wakeParameter/kappa)*wakeFunction

    return uHatPlusOuter



def boundaryLayerGrid(N,gridStretchPar):

    """
    Generate boundary layer grid\n

    This function generates a wall-normal grid for a boundary layer in terms of 
    eta in the range [0,1]. Here, eta = y/delta, where delta is the 
    boundary layer thickness. Depending on the value of the grid stretching parameter (gridStretchPar), 
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
    eta: List of wall-normal grid points in the compressible boundary layer\n 
    """

    #we will assume a constant grid stretching parameter  

    #the current stretched grid function will not work for uniform grids
    #Consequently, we add an if-else condition to define uniform/stretched grids
    if (gridStretchPar == 1):
        
        #uniform grid
        eta = np.linspace(0, 1, N, endpoint=True)

    else:

        #stretched grid
        #we first find the grid spacing at the wall for
        #the given number of grid points N and grid stretching parameter
        deltaEtaHatWall = (gridStretchPar - 1)/(gridStretchPar**(N-1) - 1)
        
        #initialize eta
        eta = np.array(np.zeros([N,]))

        #generate grid
        for i in range(0,N):
            
            eta[i] = deltaEtaHatWall*(gridStretchPar**i - 1)/(gridStretchPar - 1)


    return eta



def thetaDelta(eta,rho_rhoe,U_Ue):
    
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



def wakeParameterFromReTheta(ReTheta):

    """
    Compute the wake parameter Pi using a curve-fit\n 

    This function computes the wake parameter Pi using a curve-fit f(ReTheta),
    proposed by Hasan et al. (https://doi.org/10.2514/1.J063335)\n 

    Input parameters
    ----------------
    ReTheta: Reynolds number ReTheta for the compressible boundary layer \n 

    Output results
    --------------
    wakeParameter: Wake parameter Pi\n

    """

    z = ReTheta/425 - 1
    wakeParameter = 0.69*(1 - np.exp(-0.243*np.sqrt(z) - 0.15*z))

    return wakeParameter



def wakeParameterCalibration(yHatPlus,uHatPlus,flagIncCf):

    """
    Function used for calibration of wake parameter against an incompressible skin-friction relation\n 

    This function computes the misfit in the skin-friction coefficient for an incompressible boundary layer 
    computed using two methods:

        1. The velocity at the edge of the boundary layer, cfHat = 2/uHatPlusInf**2
        2. An incompressible skin-friction relation, cfHat = f(ReThetaHat)

    At the right wake parameter, both definitions should agree within a preset tolerance. Otherwise, 
    the misfit between the two predictions is used to guide a root-finding process for the wake parameter.
    This is called a calibration process to get the wake parameter for a given incompressible skin-friction relation.
    For more information, see: Kumar & Larsson (AIAA,2022), DOI: https://doi.org/10.2514/1.J061735 

    Input parameters
    ----------------
    yHatPlus: wall-normal grid (in plus units) for the incompressible boundary layer \n
    uHatPlus: incompressible boundary layer velocity profile \n
    flagIncCf: Flag for selecting an incompressible skin-friction relation \n

    Output results
    --------------
    misfitValue: Mismatch between the skin-friction coefficient defined using the two methods described above\n

    """

    #compute the freestream velocity
    uHatPlusInf = uHatPlus[-1]/0.99

    #compute ReTau for the incompressible boundary layer
    deltaHatPlus = np.copy(yHatPlus[-1])

    #compute the ratio of boundary layer thicknesses
    thetaDeltaHat = thetaDelta(yHatPlus/deltaHatPlus, \
            np.ones([len(yHatPlus),]), uHatPlus/uHatPlusInf)

    #compute the Reynolds number ReTheta for the incompressible boundary layer
    ReThetaHat = deltaHatPlus*uHatPlusInf*thetaDeltaHat

    #compute cf using empirical curve-fits
    cfHatReTheta = skinFrictionIncompressible(ReThetaHat, flagIncCf)

    #compute cf using the modeled velocity profile
    cfHatModeled = 2/(uHatPlusInf**2)

    #compute and return the misfit in the two cf values
    misfitValue = cfHatModeled - cfHatReTheta

    return misfitValue



def boundaryLayerProperties(M,ReDelta2,Tw_Te,Te,N,**kwargs):
    
    """
    Compute skin-friction, heat transfer along with boundary layer velocity and temperature profiles.\n
    
    This is the main function that computes the desired boundary layer properties. 
    The algorithm begins with an initial guess for the ReTau for the compressible boundary layer  
    (deltaPlus, in plus units). In the current implementation, the initial guess value is hardcoded
    as deltaPlus = ReDelta2/3, where ReDelta2 is the input Reynolds number for the compressible boundary layer.
    This is followed by using the function compressibleVelocityProfile to compute the compressible velocity profile,
    which is then used to compute the corresponding temperature, density and viscosity profiles. These new profiles 
    can now be used to recompute the velocity profile and the process is repeated until the system converges below 
    a preset tolerance. In order to accelerate the convergence process, a weaker convergence criterion is enforced 
    on the temperature profile, which could lead to smaller number of overall iterations of the method.\n

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
    calibrateWakePar: flag for choosing if the wake parameter should be calibrated, Optional\n
    gridStretchPar: grid stretching parameter\n
    underRelaxFactor: factor for under relaxing the speed of an iterative process\n
    
    Output results
    --------------
    cf: Skin-friction coefficient\n
    Bq: Wall heat transfer rate\n
    ch: Stanton number\n
    yPlus: Wall-normal points for the compressible boundary layer (in plus units)\n
    uPlus: Boundary layer velocity profile (in plus units)\n
    T_Tw: Boundary layer temperature profile, normalized by the wall temperature Tw\n
    """

    #set the tolerance for the iterative processes
    #fine global tolerance for the entire iterative process
    fineTolVal = 1e-09
    #coarse tolerance for just the temperature profile
    coarseTolVal = 1e-02

    #define constants
    #gamma is ratio of specific heats. \
    #For the current work, we consider ideal diatomic gas
    gamma = 1.4
    #recovery factor
    r = 0.896
    #Prandtl number
    Pr = 0.7
    #Damping constant
    Aplus = 17
    #set default Von Karman constant
    kappa = 0.41

    #compute wall temperature
    Tw = Tw_Te*Te 

    # dictionaries for outputs
    dictionary_ViscTemp = {}
    dictionary_ViscTemp[1] = "Keyes"; dictionary_ViscTemp[2] = "Sutherland"; dictionary_ViscTemp[3] = "CoolProp"; dictionary_ViscTemp[4] = "Power-Law"
    dictionary_IncCf = {}
    dictionary_IncCf[1] = "Karman-Schoenherr"; dictionary_IncCf[2] = "Blasius"; dictionary_IncCf[3] = "Smits"
    dictionary_TempVel = {}
    dictionary_TempVel[1] = "Zhang"; dictionary_TempVel[2] = "Walz"
    dictionary_VelTrans = {}
    dictionary_VelTrans[1] = "Van Driest"; dictionary_VelTrans[2] = "Trettel-Larsson"; dictionary_VelTrans[3] = "HLPP"

    #set default values for flags and other variables
    flagViscTemp = 3          # CoolProp
    flagIncCf = 3             # Smits
    flagTempVelocity = 1      # Zhang
    flagVelocityTransform = 3 # HLPP
    calibrateWakePar = 'No'

    gridStretchPar = 1.015

    #threshold for grid resolution (in plus units)
    wallResolutionThreshold = 0.2

    #set default value for the under-relaxation factor
    underRelaxFactor = 1 

    #check for model choice flags - optional inputs
    if (kwargs == {}):
        #Notify that the code will run with default choices
        print('Running with default choice for models') 
    
    else: 

        #check which flags have been specified 
        #if yes, read and overwrite default value
        #if not, keep the default value
        flagViscTemp          = kwargs.get('flagViscTemp',flagViscTemp)
        flagIncCf             = kwargs.get('flagIncCf',flagIncCf)
        flagTempVelocity      = kwargs.get('flagTempVelocity',flagTempVelocity)
        flagVelocityTransform = kwargs.get('flagVelocityTransform',flagVelocityTransform) 
        calibrateWakePar      = kwargs.get('calibrateWakePar',calibrateWakePar)
        
        gridStretchPar   = kwargs.get('gridStretchPar',gridStretchPar)
        underRelaxFactor = kwargs.get('underRelaxFactor',underRelaxFactor)    

    print('Von Karman constant k = '+str(kappa))

    print('Modeling choice for viscosity-temperature relation: ' + dictionary_ViscTemp[flagViscTemp])
    print('Modeling choice for friction: '                       + dictionary_IncCf[flagIncCf])
    print('Modeling choice for temperature-velocity relation: '  + dictionary_TempVel[flagTempVelocity])
    print('Modeling choice for velocity transformation: '        + dictionary_VelTrans[flagVelocityTransform])

    print('Calibrating wake parameter? ')
    print('Answer: '+ calibrateWakePar)

    #check if we want to calibrate the wake parameter
    if (calibrateWakePar == 'Yes'):
        #create a dictionary
        wakeParameterDict = {}
        misfitValue = {}
        errorWakeParameter = 1
    else:
        errorWakeParameter = 0

    #compute the ratio of recovery to freestream temperature
    Tr_Te = TrTeFromMach(M,gamma,r)

    #ratio of wall to recovery temperature, to be used for subsequent computation
    Tw_Tr = Tw_Te/Tr_Te

    print('Wall to free-stream temperature ration Tw_Te = '    +str(Tw_Te))
    print('Wall to recovery temperature ration Tw_Tr = '       +str(Tw_Tr))
    print('Recovery to free-stream temperature ration Tr_Te = '+str(Tr_Te))

    #compute dynamic viscosity ratio mue/muw
    mue_muw = viscosityFromTemperature(np.array([1/Tw_Te]),Tw,flagViscTemp)

    #compute the freestream to wall density ratio
    rhoe_rhow = densityFromTemperature(1/Tw_Te)

    #compute the Reynolds number ReTheta
    ReTheta = ReDelta2/mue_muw

    #initialize wake parameter using curve-fit from Hasan et al.
    wakeParameter = wakeParameterFromReTheta(ReTheta)

    #define wall-normal grid
    eta = boundaryLayerGrid(N,gridStretchPar)

    #initial guess for deltaPlus is made using observations from incompressible flow
    deltaPlus = ReDelta2/3

    #guess the turbulent Mach number
    MTau = 0.1

    #initialize temperature profile assuming incompressible profile
    T_Tw = np.ones([N,])
    
    #compute density and viscosity profiles
    rho_rhow = densityFromTemperature(T_Tw)
    mu_muw = viscosityFromTemperature(T_Tw,Tw,flagViscTemp)

    #initialize error variable and iteration counter
    errorTemperatureProfile = 1
    errorDeltaPlus = 1
    iterationTemperatureProfile = 0

    while (np.abs(errorTemperatureProfile) > fineTolVal or \
            np.abs(errorWakeParameter) > fineTolVal or
           np.abs(errorDeltaPlus) > fineTolVal):

        #reset the error for the coarse tolerance while loop
        errorTemperatureProfileCoarse = 1
        
        while (np.abs(errorTemperatureProfileCoarse) > coarseTolVal):

            #compute the inner-scaling wall coordinates
            yPlus = eta*deltaPlus

            #compute the compressible velocity profile
            [uPlusInner, uPlusOuter] = \
                    compressibleVelocityProfile(yPlus, MTau, rho_rhow, mu_muw,\
                    wakeParameter, Aplus, kappa, flagVelocityTransform)

            #sum up the inner and outer layer contributions
            uPlus = uPlusInner + uPlusOuter
            
            #the last grid point is at the edge, so compute the freestream value
            #this is one approach, other ways could also be thought of and implemented
            uPlusInf = uPlus[-1]/0.99

            #compute skin-friction coefficient
            cf = 2/(rhoe_rhow*(uPlusInf**2))

            #update the turbulent Mach number
            MTau = M*np.sqrt((cf/2)*rhoe_rhow*(1/Tw_Te))

            #compute the compressible velocity profile in outer scaling
            U_Ue = uPlus/uPlusInf

            #first copy the existing temperature profile
            T_TwOld = np.copy(T_Tw)

            #update the temperature profile
            T_Tw = temperatureFromVelocity(U_Ue, Tw_Te, Tr_Te, flagTempVelocity)

            #under relax the updation of the temperature profile
            T_Tw = (1-underRelaxFactor)*T_TwOld + underRelaxFactor*T_Tw

            #update the RMS error in temperature profile
            errorTemperatureProfileCoarse = \
                    np.power(np.mean(np.power(T_Tw - T_TwOld, 2)), 0.5)

            #update the density profiles
            rho_rhow = densityFromTemperature(T_Tw)
            rho_rhoe = rho_rhow/rhoe_rhow

            #update the viscosity profile    
            mu_muw = viscosityFromTemperature(T_Tw,Tw,flagViscTemp)

            #compute the boundary layer thickness ratio theta/delta
            thetaDeltaVal = thetaDelta(eta, rho_rhoe, U_Ue)

            #before updating ReTau, store its value in another variable
            deltaPlusOld = np.copy(deltaPlus)

            #update ReTau
            deltaPlus = ReDelta2*(1/thetaDeltaVal)*(1/(rhoe_rhow*uPlusInf))
       
            #update the error in deltaPlus
            errorDeltaPlus = deltaPlus - deltaPlusOld

        #the temperature profile is loosely converged
        #now check for calibration

        if (calibrateWakePar == 'Yes'):
            #compute the equivalent incompressible boundary layer velocity profile  

            if (flagVelocityTransform == 2 or flagVelocityTransform == 3):
                yHatPlus = yPlus*(1/mu_muw)*np.sqrt(rho_rhow)
            elif (flagVelocityTransform == 1):
                yHatPlus = np.copy(yPlus)

            #call function to compute the incompressible velocity profile
            [uHatPlusInner,uHatPlusOuter] = \
                    universalVelocityProfile(yPlus,yHatPlus,wakeParameter,Aplus,kappa)

            ##alternatively, the inner and outer layer components
            ##can also be recovered by forward transforming
            ##their compressible counterparts
            ##uncomment the lines below, if interested in that approach
            ##Although, this alternate approach seems to take longer
            ##and stronger under-relaxation for convergence

            ##forward transform the inner layer using choice of transform
            #[yHatPlus,uHatPlusInner] = forwardVelocityTransform(yPlus,uPlusInner,\
            #        rho_rhow,mu_muw,MTau,Aplus,kappa,flagVelocityTransform)
            
            ##forward transform the outer layer using Van Driest transform
            ##since Van Driest does not scale the wall-normal coordinate,
            ##we will just store the output in a temporary variable and delete it
            #[temp,uHatPlusOuter] = forwardVelocityTransform(yPlus,uPlusOuter,\
            #        rho_rhow,mu_muw,MTau,Aplus,kappa,1)
            #del temp

            #sum up the components
            uHatPlus = uHatPlusInner + uHatPlusOuter 
 
            #compute the cf-misfit in the current iteration
            misfitValue[iterationTemperatureProfile] = \
                    wakeParameterCalibration(yHatPlus,uHatPlus,flagIncCf)

            #perform a root finding iteration based on the misfit

            #check if this is the first iteration
            #if yes, then initialize the dictionary
            if (iterationTemperatureProfile == 0):
                
                wakeParameterDict[0] = np.copy(wakeParameter)

                #and based on the sign of misfit, create the second guess
                if (misfitValue[0] >= 0):
                    wakeParameterDict[1] = 1.2*wakeParameterDict[0]
                elif (misfitValue[0] < 0):
                    wakeParameterDict[1] = 0.8*wakeParameterDict[0]
                else:
                    sys.exit('Error in wake parameter calibration')
            
            else:
                #this is not the first iteration, so we have enough data for
                #root-finding using the secant method
                wakeParameterDict[iterationTemperatureProfile+1] = \
                        wakeParameterDict[iterationTemperatureProfile] - \
                        misfitValue[iterationTemperatureProfile] * \
                        (wakeParameterDict[iterationTemperatureProfile] - \
                        wakeParameterDict[iterationTemperatureProfile - 1])/\
                        (misfitValue[iterationTemperatureProfile] - \
                        misfitValue[iterationTemperatureProfile-1])
 

            #at the end, just update the wake parameter with latest guess
            wakeParameter = np.copy(wakeParameterDict[iterationTemperatureProfile+1])

            errorWakeParameter = float(abs(wakeParameterDict[iterationTemperatureProfile+1] - \
                    wakeParameterDict[iterationTemperatureProfile]))


        #we know that the RMS error at the last iteration of coarse while loop
        #is the error for the outer while loop as well!
        errorTemperatureProfile = np.copy(errorTemperatureProfileCoarse)
        
        #update iteration counter
        iterationTemperatureProfile = iterationTemperatureProfile + 1 

        print('At iteration {0},errors: T = {1:.4e} , ReTau = {2:.4e}, Pi = {3:.4e}'.
              format(iterationTemperatureProfile,errorTemperatureProfile,errorDeltaPlus,errorWakeParameter))

    #method converged, compute wall heat transfer rate
    Bq = computeBq(Pr,T_Tw,yPlus,gridStretchPar)

    #compute Stanton number c_h
    if (Tw_Tr == 1):
        #Stanton number is not defined for adiabatic walls
        ch = np.nan
    else:
        #compute Stanton number for non-adiabatic walls
        ch = Bq*(1/Tw_Te)*(1/uPlusInf)*(1/(1 - 1/Tw_Tr))

    #check for grid size at the wall
    #since the first grid point is at the wall (yPlusCore = 0),
    #the grid size at the wall is just the location of the second grid point
    if yPlus[1]>wallResolutionThreshold:
        print('WARNING: Grid spacing at the wall = '+str(np.round(yPlus[1],3))+ \
                ' is GREATER than threshold value = '+str(wallResolutionThreshold)+' (in plus units)')
        print('RUN AGAIN WITH A FINER GRID AT THE WALL')

    #return output quantities 
    return [cf,Bq,ch,yPlus,uPlus,T_Tw,mu_muw,kappa] 



def gridProperties(yPlus,T_Tw,mu_muw,alf_plus,jb,ypw,kappa,**kwargs):
    
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

    #set default values for system parameters
    myNy  = jb

    #check for model choice flags - optional inputs
    if (kwargs == {}):
        #Notify that the code will run with default choices
        print('Running with default choice for Ny points in wall normal direction')

    #overwrite quantities if user provides inputs
    myNy  = kwargs.get('myNy' ,myNy )

    print('Starting wall-normal grid generation')
    print('Von Karman constant k = '+str(kappa))

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

    #relaxation factor to update alf
    relax_fac = 1. 

    #get the ysw in order to obtain the desired ysp
    # first guess
    ysw = ypw 
    # compute grid
    [yStar_j,yPlus_j,jj,Ny] = universalGridStretch(alf,jb,ysw,Retau_Star,fT_star,fmu_star,0,kappa)
    # adapt ysw for target ypw
    ysw = ypw*yStar_j[1]/yPlus_j[1]

    while (np.abs(error) > mytol) and (count < iter_max):
       #compute grid with guessed/corrected alf   
       [yStar_j,yPlus_j,jj,Ny] = universalGridStretch(alf,jb,ysw,Retau_Star,fT_star,fmu_star,0,kappa)
       etaStar_j = (kappa*yStar_j)**0.25
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
    [yStar_j,yPlus_j,jj,Ny] = universalGridStretch(alf,jb,ysw,Retau_Star,fT_star,fmu_star,myNy,kappa)

    #output variables
    yPlusCore_j   = yPlus_j
    yStarCore_j   = yStar_j
    jjCore        = jj
    NyCore        = Ny
    alf_opt       = alf
    etaPlusCore_j = SemilocalToWall((kappa*yStarCore_j)**0.25, fT_star(yStarCore_j), fmu_star(yStarCore_j)) 

    return [yPlusCore_j,yStarCore_j,jjCore,NyCore,alf_opt,etaPlusCore_j]

