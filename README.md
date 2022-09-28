============================================
MOD-COMP
============================================
Modular computational tools for high speed compressible flows

A user guide and brief overview for a Python implementation of the method presented in 
Modular Method for Estimation of Velocity and Temperature Profiles in 
High-Speed Boundary Layers [INSERT PAGE AND VOLUME DETAILS]
DOI: https://doi.org/10.2514/1.J061735

Questions on the velocity profile calculation method can be sent to the authors: 
Vedant Kumar (vkumar20@umd.edu) and Johan Larsson (jola@umd.edu)

Questions on the wall-normal grid stretching method can be sent to the authors:
Alessandro Ceci (alessandro.ceci@uniroma1.it) and Sergio Pirozzoli (sergio.pirozzoli@uniroma1.it)

============================================
HOW TO USE
============================================
Beginner users are recommended to look at the Python script methodDemo1.py 
The script demonstrates an example of computing the skin friction coefficient, wall heat transfer
along with velocity and temperature profiles for a compressible boundary layer at Mach number = 5,
Reynolds number (ReDelta2) = 3000 with wall and freestream temperatures of 300 and 200 K, respectively.
For the sake of simplification, the script uses default modeling choices which has been observed to 
return the most accurate predictions at the time of the publication of the above manuscript.

Next, an example for the same boundary layer case as above however with the specification of 
modeling choices is presented in the Python script methodDemo2.py. 
More information on the available modeling choices and how to change them is available below.

============================================
FUNCTION DEFINITION
============================================

Function: boundaryLayerProperties in the Python script boundaryLayerPropFunctions.py

boundaryLayerProperties(M, ReDelta2, Tw_Te, Te, N, **kwargs)

Computes boundary layer properties for the prescribed flow parameters.
Subscripts _w (or w) and _e (or e) denote wall and edge (freestream) properties, respectively.

Parameters:
		M: float 
		Mach number (M = Ue/c_e, where c is the speed of sound)
		
		ReDelta2: float 
		Reynolds number (ReDelta2 = rho_e Ue theta / mu_w, where theta is the momentum thickness)
		
		Tw_Te: float 
		Ratio of wall (Tw) to freestream (Te) temperature
		
		Te: float 
		Freestream temperature	
		
		N: int 
		Number of grid points in the boundary layer	

		**kwargs: Optional inputs 

			
			Flags for different model choices
			
			flagViscTemp: int, optional 
			Flag for selecting viscosity-temperature relation. 

			flagIncCf: int, optional
			Flag for selecting an incompressible skin-friction relation

			flagTempVelocity: int, optional
			Flag for selecting a temperature-velocity relation

			flagVelocityTransform: int, optional
			Flag for selecting a velocity transform function

			
			Additional parameters

			gridStretchPar: float 
			Grid-stretching parameter for geometrical stretching.
			Hence, if gridStretchPar = 1, then we have a uniform grid and if
			gridStretchPar>1, then we have a grid stretching away from the wall

			underRelaxFactor: float 
			Extent of under-relaxation for the solution process such that
			x_new = (1 - underRelaxFactor)*x_old + underRelaxFactor*x_new


		Note that unspecified optional inputs switch to default values. 
		See below for more information.


Returns:
		cf: float 
		Skin friction coefficient
		
		Bq: float 
		Wall heat transfer rate

		ch: float 
		Stanton number, undefined for adiabatic walls

		yPlus: array
		Array of y-plus coordinates inside the boundary layer

		uPlus: array
		Array of scaled velocity (uPlus = u/u_tau) values in the boundary layer

		Tplus: array
		Array of T-plus values (Tplus = T/Tw) values in the boundary layer

		
		[Refer to the manuscript for the definition of the output parameters above]



============================================
METHOD DESCRIPTION
============================================

A detailed description of the method can be found in the paper referenced above. 

In brief, the properties for the compressible boundary layer are computed by relating it
with an equivalent incompressible state.

An incompressible boundary layer velocity profile is defined using a known model followed by
transforming it to the desired compressible state (defined using the function input parameters).

These transformation functions assume that the only effect of non-zero Mach numbers comes from
the boundary layer density profile as well as viscosity profile for some functions. 

The density and viscosity profiles are hence computed using models relating them with the 
boundary layer temperature which in turn is related to the boundary layer velocity profile using a
temperature-velocity modeling relation. 

The method follows an iterative process which, on convergence, returns the outputs described above.


============================================
MODELING CHOICES
============================================

The current work models the incompressible velocity profile using the definition presented in 
Huang et al. [AIAA Journal, Vol. 31, No. 9, 1993, pp. 1600– 1604].
It can be changed by editing the function universalVelocityProfile in boundaryLayerPropFunctions.py


Further, the following modeling choices are incorporated in the current implementation:

1. Viscosity-Temperature relation - Chosen by specifying value to flagViscTemp

[For implementation details, see function viscosityFromTemperature in boundaryLayerPropFunctions.py]

flagViscTemp = 1 : Keyes' viscosity law
flagViscTemp = 2 : Sutherland's law
flagViscTemp = 3 : Curve-fit to the CoolProp dataset [DEFAULT]
flagViscTemp = 4 : Power law, (mu/muw) = (T/Tw)^n, where n = 2/3



2. Incompressible skin-friction relation - Chosen by specifying value to flagIncCf

[For implementation details, see function skinFrictionIncompressible in boundaryLayerPropFunctions.py]

flagIncCf = 1 : Karman-Schoenherr's relation
flagIncCf = 2 : Blasius relation
flagIncCf = 3 : Smits' relation [DEFAULT]



3. Temperature-Velocity relation - Chosen by specifying value to flagTempVelocity

[For implementation details, see function temperatureFromVelocity in boundaryLayerPropFunctions.py]

flagTempVelocity = 1 : Zhang's relation [DEFAULT]
flagTempVelocity = 2 : Walz's relation



4. Velocity transform function - Chosen by specifying value to flagVelocityTransform

[For implementation details, see function inverseVelocityTransform in boundaryLayerPropFunctions.py]

flagVelocityTransform = 1 : Inverse of the Van Driest velocity transform
flagVelocityTransform = 2 : Inverse of the Trettel-Larsson velocity transform
flagVelocityTransform = 3 : Inverse of the Volpiani velocity transform function [DEFAULT]


============================================
POINTS TO NOTE
============================================

1. For grid-converged results, it is recommended to use grids with 
near-wall resolution < 0.2 plus units. The distribution of grid points can be 
controlled using the number of grid points "N" and, the grid-stretching parameter 
"gridStretchPar". By default, gridStretchPar = 1.016 and can be changed by specifying 
it as an input parameter to the function boundaryLayerProperties.

The following warning messages are displayed if the 
grid is coarser than the above threshold: 

"WARNING: Grid spacing at the wall = ___ is GREATER than threshold value = 0.2 (in plus units)"
"RUN AGAIN WITH A FINER GRID AT THE WALL"

The threshold for grid convergence check is hardcoded and can be altered by modifying the variable 
wallResolutionThreshold in the function boundaryLayerProperties.



2. The implementation has been found to be stable with the use of the Van Driest and Volpiani 
velocity transform functions. Hence, these cases are run without any under-relaxation 
(underRelaxFactor = 1). However, using the Trettel-Larsson transform typically needs moderate
to strong under-relaxation. Therefore, by default, underRelaxFactor = 0.5 for this case to
ensure stability of the solution process. These default values can be over-written by specifying 
the underRelaxFactor as an input parameter to the function boundaryLayerProperties.


============================================
ADDITIONAL DOCUMENTATION
============================================
A detailed description of each function implemented in boundaryLayerPropFunctions.py can be found in 
the file boundaryLayerPropFunctions.html. Please note that the function description in the html file 
is the same as the one found in the python script boundaryLayerPropFunctions.py.

