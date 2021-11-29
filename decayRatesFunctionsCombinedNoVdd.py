import numpy as np
import cProfile
import matplotlib.pyplot as plt
import scipy.special as sp
from scipy.integrate import quad, solve_ivp
from scipy.interpolate import interp1d, interp2d
from scipy.signal import argrelextrema
import tqdm
from tqdm.notebook import tqdm_notebook
from scipy.special import jv
import multiprocessing
import time
import sys

from config import k_B, h, hbar, mu_B, mu_0, m_unit, m, mu, g_J, J, a_dd, E_dd, Bnorm, OmegaNorm, patchingParameter_0, defaultKwargs, zPrecisionDefault, nmaxCap

def V_dd_3D(z, ρ):  # this should actually depend on m for the outcoming channel
    return 2*(1-3*z**2/(z**2+ρ**2))/(z**2+ρ**2)**(3/2) # + (2*4.26/10.2)**4/(z**2+ρ**2)**3

def hermite(z, n):
    hermiteFactors = np.zeros(n+1)
    hermiteFactors[n] = 1
    hermite_n = np.polynomial.hermite.Hermite(hermiteFactors)
    return hermite_n(z)

def harmonic_oscillator(z, n, D):
    return 1/np.sqrt(1.0*2**n*np.math.factorial(n))/(np.pi*D**2)**(1/4)*hermite(z/D, n)*np.exp(-z**2/(2*D**2))

def density(z, n, D):
    return harmonic_oscillator(z, n, D)**2

def computeZDimensions(D, zPrecision = zPrecisionDefault):
    # usually 5 / 0.1 is the optimal speed /result, it makes 100 points total for D not a list
    zMaxPrecision, dzPrecision = zPrecision
    # D can be a number or a list
    if type(D) != list:
        D = [D]
    Dmax = max(D)
    Dmin = min(D)
    z_max = zMaxPrecision*Dmax  
    dz = dzPrecision*Dmin 
    z_dimensions = [z_max, dz]
    return z_dimensions

def V_dd_averaged(ρ, n, D, ho_states = None, zPrecision = zPrecisionDefault, barrier = True):
    if not barrier:
        return 0
    if ho_states:
        dz = ho_states["dz"]
        z_max = ho_states["z_max"]
        Z = np.linspace(-z_max, z_max, round(2*z_max/dz)+1)
        ρ2D, Z2D = np.meshgrid(ρ, Z)
        dens = np.transpose(np.tile(ho_states["hos"][n]**2, (np.shape(Z2D)[1], 1)))
        return np.sum(V_dd_3D(Z2D, ρ2D)*dens, axis = 0)*dz
    else:
        z_max, dz = computeZDimensions(D, zPrecision)
        Z = np.linspace(-z_max, z_max, round(2*z_max/dz)+1)
        ρ2D, Z2D = np.meshgrid(ρ, Z)
        return np.sum(V_dd_3D(Z2D, ρ2D)*density(Z2D, n, D), axis = 0)*dz

def effectivePotential(ρ, m, n, D, ho_states = None, zPrecision = zPrecisionDefault, barrier = True):
    return (m**2-1/4)/ρ**2 + V_dd_averaged(ρ, n, D, ho_states, zPrecision, barrier)
    
def ho_list(nmax, D, zPrecision = [5, 0.1]):
    nmax = min(nmaxCap, nmax)
    z_max, dz = computeZDimensions(D, zPrecision)
    Z = np.linspace(-z_max, z_max, round(2*z_max/dz)+1)
    hos = []
    for n in range(nmax + 1):
        hos.append(harmonic_oscillator(Z, n, D))
    return {"dz": dz, "z_max": z_max, "hos": hos}
    
    
def normalize(ϕ, ρ_ar):
    density = ϕ**2
    N_total = len(ρ_ar)
    N_partial = N_total//10
    beta = np.sum(density[-N_partial:])/N_partial
    return ϕ/np.sqrt(beta*ρ_ar[-1])

def normalizeMax(ϕ, ρ_ar_large):
    maximums = argrelextrema(ϕ, np.greater)[0]
    minimums = argrelextrema(ϕ, np.less)[0]
    extremas = np.sort(np.concatenate((maximums, minimums)))
    phiNorm = ϕ/(np.abs(ϕ[extremas][-1])*np.sqrt(0.5*ρ_ar_large[-1]))
    return phiNorm

# def normalizePreintegral(ϕ, ρ_ar_init, ρ_ar_large):
#     phi = normalize(ϕ, ρ_ar_large)
#     dρ_init = ρ_ar_init[1] - ρ_ar_init[0]
#     dρ_large = ρ_ar_large[1] - ρ_ar_large[0]
#     dρArray = np.concatenate((np.ones(len(ρ_ar_init))*dρ_init, np.ones(len(ρ_ar_large)-1)*dρ_large))
#     return phi*dρArray

def checkNormalization(ϕ, ρ_init, ρ_large):
    density = ϕ**2
    dρ_init = ρ_init[1] - ρ_init[0]
    if ρ_large.size:
        dρ_large = ρ_large[1] - ρ_large[0]
        return np.sum(density[:len(ρ_init)])*dρ_init + np.sum(density[len(ρ_init):])*dρ_large
    else:
        return np.sum(density*dρ_init)
    
def checkNormalization(ϕ, ρ_tot):
    dρArray = createDRhoArray(ρ_tot)
    density = ϕ**2
    return np.sum(density*dρArray)
    
def ϕ_1array(ρ_ar, m, k, T, ϕ0, ϕ1):
    ϕm = np.zeros_like(ρ_ar)
    ϕm[0] = ϕ0
    ϕm[1] = ϕ1

    step = ρ_ar[1] - ρ_ar[0]
    Tar = T(ρ_ar, m, k, step)
    for i in range(2, len(ρ_ar)):
            ϕm[i] = ( ϕm[i-2] * (Tar[i - 2] - 1) + ϕm[i - 1] * (10*Tar[i - 1] + 2) ) / (1 - Tar[i])
    return ϕm

def ϕ_2arrayNumerov(ρ_ar_init, ρ_ar_large, m, k, n, D, ho_states = None, zPrecision = zPrecisionDefault, norm = False, barrier = True):
    def g(ρ, m, k):
        return (m**2 - 1/4.)/ρ**2 + V_dd_averaged(ρ, n, D, ho_states, zPrecision, barrier) - k**2
    def T(ρ, m, k, step):
        return step**2 * g(ρ, m, k) / 12.0

    ϕ0_init = 0.
    ϕ1_init = 1.
    ϕ_init = ϕ_1array(ρ_ar_init, m, k, T, ϕ0_init, ϕ1_init)
    step_init = ρ_ar_init[1] - ρ_ar_init[0]
    if ρ_ar_large.size: # if there is a \rho_large non zero
        step_large = ρ_ar_large[1] - ρ_ar_large[0]
        ϕ0_large = ϕ_init[-1]
        ϕ1_large = ϕ0_large + step_large*(ϕ_init[-1] - ϕ_init[-2])/step_init
        ϕ_large = ϕ_1array(ρ_ar_large, m, k, T, ϕ0_large, ϕ1_large)[1:]

        ϕm = np.concatenate((ϕ_init, ϕ_large))
    else:
        ϕm = ϕ_init
        ρ_ar_large = ρ_ar_init
    # check the correct lenght, cause I add a zero here to compensate, but then ends up being
    # one block too long...
    
    if norm:
        ϕm = normalizeMax(ϕm, ρ_ar_large)
        #ϕNorm = checkNormalization(ϕm, ρ_ar_init, ρ_ar_large)
        ϕNorm = checkNormalization(ϕm, createRhoTot(ρ_ar_init, ρ_ar_large))
        #print('Norm = ', round(ϕNorm, 5))
        if abs(ϕNorm - 1) > 0.05:
            print('Normalization issue for m, k, n, D = ', m, round(k, 3), n, round(D, 3))
            print('Norm = ', round(ϕNorm, 5))
    return ϕm

def ϕ_2arrayScipy(ρ_ar_init, ρ_ar_large, m, k, n, D, ho_states = None, zPrecision = zPrecisionDefault, norm = False, barrier = True):
    
    def g(ρ, m, k):
        return (m**2 - 1/4.)/ρ**2 + V_dd_averaged(ρ, n, D, ho_states, zPrecision, barrier) - k**2

    def secondDerivative(ρ, y, m, k):
        dydt = [y[1], y[0]*g(ρ, m, k)]
        return dydt
    
    ρ_tot = createRhoTot(ρ_ar_init, ρ_ar_large)
    y0 = [0,1]
    sol = solve_ivp(secondDerivative, t_span = (ρ_tot[0],ρ_tot[-1]), y0 = y0, t_eval = ρ_tot, args = (m, k))#, atol = 10**-20, method = 'DOP853')
    phi = sol.y[0,:]
    if norm:
        phi = normalizeMax(phi, ρ_tot)
        phiNorm = checkNormalization(phi, ρ_tot)
        if abs(phiNorm - 1) > 0.05:
            print('Normalization issue for m, k, n, D = ', m, round(k, 3), n, round(D, 3))
            print('Norm = ', round(phiNorm, 5))
    return phi

def ϕ_2array(ρ_ar_init, ρ_ar_large, m, k, n, D, ho_states = None, zPrecision = zPrecisionDefault, norm = False, barrier = True):
    if len(ρ_ar_init) + len(ρ_ar_large) > 20000:
        return ϕ_2arrayScipy(ρ_ar_init, ρ_ar_large, m, k, n, D, ho_states, zPrecision, norm, barrier)
    else:
        return ϕ_2arrayNumerov(ρ_ar_init, ρ_ar_large, m, k, n, D, ho_states, zPrecision, norm, barrier)
        

def ϕ_zeroEnergy(m, ρ):
    ρ0 = 1.
    return 2*np.sqrt(ρ) * (sp.kn(2*m, np.sqrt(8/ρ)) + np.exp(-2*np.sqrt(8/ρ0))*sp.iv(2*m, np.sqrt(8/ρ)))
        
def j0(ρ, z, J = J):
    return 2*(1-3*(z**2)/(ρ**2+z**2))/(ρ**2+z**2)**(3/2)

def j1(ρ, z, J = J):
    return 2*(-3)/J**(1/2)*z*ρ/(ρ**2+z**2)**(5/2)

def j2(ρ, z, J = J):
    return 2*(-3/2)/J*ρ**2/(ρ**2+z**2)**(5/2)

"""
def computeRhoDimensions(ki, kf, ρ_parameters):
    # this determines the correct ρ_dimension array to create based on the scattering situation
    ρmin_init, ρmax_init, dρ_init, dρ_large_ratio = ρ_parameters
    # The oscillation period in rho is 1/(2*pi*k)
    kmin, kmax = min(ki, kf), max(ki, kf)
    #ρmax_init = 1/(2*np.pi*kmin)
    dρ_large = dρ_large_ratio*2*np.pi/kmax
    ρmax_large = ρmax_init + int(2*2*np.pi/kmin) # this leaves 2 oscillation periode
    if ρmax_large == ρmax_init:
        ρmax_large = ρmax_init + 1
        dρ_large = dρ_init
    return [[ρmin_init, ρmax_init, dρ_init], [ρmax_large, dρ_large]]"""

def computeRhoDimensions(ki, kf, ρ_parameters):
    # this determines the correct ρ_dimension array to create based on the scattering situation
    ρmin_init, ρmax_init, dρ_init, dρ_large_ratio = ρ_parameters
    # The oscillation period in rho is 1/(2*pi*k)
    kmin, kmax = min(ki, kf), max(ki, kf)
    dρ_large = min(dρ_large_ratio*2*np.pi/kmax, 0.1)
    dρ_init = min(dρ_init, dρ_large)
    ρmax_large = ρmax_init + int(2*2*np.pi/kmin) # this leaves 2 oscillation periods
    if ρmax_large == ρmax_init:
        ρmax_large = ρmax_init + 1
        dρ_large = dρ_init
    return [[ρmin_init, ρmax_init, dρ_init], [ρmax_large, dρ_large]]

def createRhoTot(ρ_init, ρ_large):
    ρ_tot = np.concatenate((ρ_init, ρ_large[1:]))
    return ρ_tot

def createRhoArrays(ρ_dimensions):
    # this creates the numpy rho arrays based on the boundaries ρ_dimension
    # ρ_dimension = [[ρmin_init, ρmax_init, dρ_init], [ρmax_large, dρ_large]]
    ρ_init_dimensions, ρ_large_dimensions = ρ_dimensions
    ρmin_init, ρmax_init, dρ_init = ρ_init_dimensions
    ρmax_large, dρ_large = ρ_large_dimensions
    ρ_init = np.linspace(ρmin_init, ρmax_init, round(ρmax_init/dρ_init))
    ρ_large = np.linspace(ρmax_init, ρmax_large, round((ρmax_large-ρmax_init)/dρ_large)+1)
    ρ_tot = createRhoTot(ρ_init, ρ_large)
    return ρ_init, ρ_large, ρ_tot

def computeRhoArrays(ki, kf, ρ_parameters):
    ρ_dimensions = computeRhoDimensions(ki, kf, ρ_parameters)
    return createRhoArrays(ρ_dimensions)

def createDRhoArray(ρ_tot):
    dρArray = np.diff(ρ_tot)
    return np.concatenate((dρArray[:1], dρArray))

def interpolatePhi0(ki, kfMin, kfMax, D, ρ_parameters, ho_states = None, zPrecision = zPrecisionDefault, barrier = True):
    ρ_init, ρ_large, ρ_tot = computeRhoArrays(kfMin, kfMax, ρ_parameters)
    ϕ0_array = ϕ_2array(ρ_init, ρ_large, m = 0, k = ki, n = 0, D = D, ho_states = ho_states, norm = True, zPrecision = zPrecision, barrier = barrier)
    """
    ρmin_init, ρmax_init, dρ_init, dρ_large_ratio = ρ_parameters
    ρ_init = np.linspace(ρmin_init, ρmax_init, round(ρmax_init/dρ_init))
    dρ_large = dρ_large_ratio*2*np.pi/ki
    ρ_large = np.linspace(ρmax_init, ρmax_large, round((ρmax_large-ρmax_init)/dρ_large))
    ρ_tot = np.concatenate((ρ_init, ρ_large[1:]))
    ϕ0_array = ϕ_2array(ρ_init, ρ_large, m = 0, k = ki, n = 0, D = D, ho_states = ho_states, norm = True, zPrecision = zPrecision, barrier = barrier)"""
    return interp1d(ρ_tot, ϕ0_array)
    
def integrand_2array(D, j, n, m, ki, kf, ρ_parameters, ho_states = None, ϕ0_interpolated = None, zPrecision = zPrecisionDefault, barrier = True):
    ρ_init, ρ_large, ρ_tot = computeRhoArrays(ki, kf, ρ_parameters)
    if ho_states: # I'm not even sure this is necessary as the ho_states are in anyway only created for
                # only one D, which should be passed in argument anyway.
        dz = ho_states["dz"]
        z_max = ho_states["z_max"]
    else:
        z_max, dz = computeZDimensions(D, zPrecision)
    z = np.linspace(-z_max, z_max, round(2*z_max/dz))
    Rho, Z = np.meshgrid(ρ_tot, z, indexing = 'ij')
    
    prefactor_integral_z = 1/np.sqrt(1.0*2**n*np.math.factorial(n))/(np.pi*D**2)**(1/2)

    if j == 0:
        jfunc = j0
    if j == 1:
        jfunc = j1
    if j == 2:
        jfunc = j2
     
    shape = np.shape(Rho)
    if barrier:
        if ϕ0_interpolated:
            ϕ_in = normalizeMax(ϕ0_interpolated(ρ_tot), ρ_large)
        else:
            ϕ_in = ϕ_2array(ρ_init, ρ_large, m = 0, k = ki, n = 0, D = D, ho_states = ho_states, norm = True, zPrecision = zPrecision, barrier = barrier)
        ϕ_out = ϕ_2array(ρ_init, ρ_large, m = m, k = kf, n = n, D = D, ho_states = ho_states, norm = True, zPrecision = zPrecision, barrier = barrier)
    else:
        #ϕ_in = analyticFreeSolution(ρ_tot, ki, m = 0)
        #ϕ_out = analyticFreeSolution(ρ_tot, kf, m = 2)
        ϕ_in = ϕ_2array(ρ_init, ρ_large, m = 0, k = ki, n = 0, D = D, ho_states = ho_states, norm = True, zPrecision = zPrecision, barrier = barrier)
        ϕ_out = ϕ_2array(ρ_init, ρ_large, m = m, k = kf, n = n, D = D, ho_states = ho_states, norm = True, zPrecision = zPrecision, barrier = barrier)
    dρ_init = ρ_init[1] - ρ_init[0]
    dρ_large = ρ_large[1] - ρ_large[0]
    dρArray = createDRhoArray(ρ_tot)
    oneD_array = dρArray*ϕ_in*ϕ_out
    phi_function = np.transpose(np.tile(oneD_array, (shape[1], 1)), (1, 0))

    return dz*hermite(Z/D, n)*hermite(Z/D, 0)*np.exp(-Z**2/D**2)*jfunc(Rho, Z)*phi_function*prefactor_integral_z

def matrixElementSquared(D, j, n, m, ki, kf, ρ_parameters, ho_states = None, ϕ0_interpolated = None, zPrecision = zPrecisionDefault, barrier = True):
    matrix = integrand_2array(D, j, n, m, ki, kf, ρ_parameters, ho_states, ϕ0_interpolated, zPrecision, barrier)
    ρ_init, ρ_large, ρ_tot = computeRhoArrays(ki, kf, ρ_parameters)
    ρmax_large = ρ_large[-1]
    surface = ρmax_large**2 #there is no pi factor here, it is just \tilde{I} = rhomax*I by definition.
    sum_matrix = np.sum(matrix)
    return surface/(ki*kf)*np.linalg.norm(sum_matrix)**2

def statesAccessible(ki, B, Ω):
    listStates = {}  # dic[keys = 0, 1, 2 for magnetic change] and values = [list of n accessible]
    
    listStates[0] = [0]
    # j = 1
    listStates[1] = list(range(int((ki**2+B)/Ω + 1)))
    # j = 2
    listStates[2] = list(range(int((ki**2+2*B)/Ω + 1)))
    """
    listStates[0] = [0]
    # j = 1
    listStates[1] = []
    # j = 2
    listStates[2] = [0]
    # print(listStates)"""
    return listStates

def outputK(ki, B, Ω, j, n):
    kfsquared = ki**2 + B*j - Ω*n
    if kfsquared < 0:
        print("problem with imaginary kf")
        return 0
    else:
        return np.sqrt(kfsquared)

def findRhoMax(ki, BList, Ω, ρ_parameters):
    rhoMaxList = []
    for B in BList:
        listStates = statesAccessible(ki, B, Ω)
        for j in [1, 2]:
            for n in listStates[j]:
                kf = outputK(ki, B, Ω, j, n)
                rhoMaxList += [computeRhoDimensions(ki, kf, ρ_parameters)[1][0]]
    return np.max(np.array(rhoMaxList))

def findKfExtrema(ki, BList, Ω, ρ_parameters):
    kfList = [ki]
    for B in BList:
        listStates = statesAccessible(ki, B, Ω)
        for j in [1, 2]:
            for n in listStates[j]:
                kfList += [outputK(ki, B, Ω, j, n)]
    kfList = np.array(kfList)
    return np.min(kfList), np.max(kfList)

def findKfExtremaAllList(ki, BList, OmegaList, ρ_parameters):
    kfList = [ki]
    for omega in OmegaList:
        for B in BList:
            listStates = statesAccessible(ki, B, omega)
            for j in [1, 2]:
                for n in listStates[j]:
                    kfList += [outputK(ki, B, omega, j, n)]
    kfList = np.array(kfList)
    return np.min(kfList), np.max(kfList)

def createVddAveragedList(kfMin, kfMax, DList, ρ_parameters, ho_states = None, zPrecision = zPrecisionDefault, barrier = True, nmax = nmaxCap):
    ρ_init, ρ_large, ρ_tot = computeRhoArrays(ki, kf, ρ_parameters)
    Vdd_array = {}
    for n in range(nmax + 1):
        Vdd_2D_array = []
        for D in DList:
            Vdd_2D_array += [V_dd_averaged(ρ_tot, n, D, ho_states, zPrecision, barrier)]
        Vdd_array += np.array(Vdd_2D_array)

# Write a function that for each omega scanned, creates a subinterpolation of the Vdd averaged for this specific D, to then only have a 1

def decayOneChannel(j, n, ki, B, Ω, ρ_parameters, ho_states = None, ϕ0_interpolated = None, zPrecision = zPrecisionDefault, barrier = True):
    D = np.sqrt(2/Ω) # this is the ratio a_z/a_dd, it is different from Ticknor's D (which is the dipolar length)
    m = j
    kf = outputK(ki, B, Ω, j, n)
    return matrixElementSquared(D, j, n, m, ki, kf, ρ_parameters, ho_states, ϕ0_interpolated, zPrecision, barrier)

def decayAllChannel(ki, B, Ω, ρ_parameters, ho_states = None, ϕ0_interpolated = None, zPrecision = zPrecisionDefault, barrier = True):
    listStates = statesAccessible(ki, B, Ω)
    gamma = 0
    for j in [1, 2]:
        for n in listStates[j]:
            gamma += decayOneChannel(j, n, ki, B, Ω, ρ_parameters, ho_states, ϕ0_interpolated, zPrecision, barrier)
    return gamma

def normalizeDecay(gamma, Ω):
    D = np.sqrt(2/Ω)
    normDipolarRelaxation = E_dd/hbar*a_dd**3
    #numericalPrefactor = 2*np.sqrt(2*np.pi)
    numericalPrefactor = np.sqrt(2*np.pi)
    symmetrization = 2
    cm3 = (10**2)**3
    return gamma*normDipolarRelaxation*numericalPrefactor*symmetrization*D*cm3
# this is normalized and is expressed in cm**3 per second

def normalizedDecayAllChannel(ki, B, Ω, ρ_parameters, ho_states = None, ϕ0_interpolated = None, zPrecision = zPrecisionDefault, barrier = True):
    gamma = decayAllChannel(ki, B, Ω, ρ_parameters, ho_states, ϕ0_interpolated, zPrecision, barrier)
    return normalizeDecay(gamma, Ω)

# These are function for the multiprocess file
def createListϕ0(ki, BList, OmegaList, ρ_parameters, ho_states = None, zPrecision = zPrecisionDefault, barrier = True):
    ϕ0List = []
    for Ω in OmegaList:
        D = np.sqrt(2/Ω)
        kfMin, kfMax = findKfExtrema(ki, BList, Ω, ρ_parameters)
        ϕ0List += [interpolatePhi0(ki, kfMin, kfMax, D, ρ_parameters, ho_states, zPrecision, barrier)]
        # no need to pass ho states here, it would make this more complicated
    return ϕ0List

def createListHOstates(ki, BList, OmegaList, zPrecision = zPrecisionDefault):
    ho_statesList = []
    for Ω in OmegaList:
        D = np.sqrt(2/Ω)
        nmax = statesAccessible(ki, np.max(BList), Ω)[2][-1]
        ho_statesList += [ho_list(nmax, D, zPrecision)]
    return ho_statesList

def multiprocessing_func(decayRatesDict, ki, B, OmegaList, index, ρ_parameters, ho_statesList, ϕ0List, zPrecision, barrier):
    ho_states = ho_statesList[index]
    ϕ0_interpolated = ϕ0List[index]
    Omega = OmegaList[index]
    if Omega/B < 0.02:
        #print('Point skipped')
        # now that we observe that the coupling diminishes with n it would be nicer
        # to have a cutoff of the number of n considered rather than skipping the point.
        decayRatesDict[(Omega, B)] = np.nan
    else:
        #print('point registered, Omega = ', Omega, ' B = ', B)
        decayRatesDict[(Omega, B)] = normalizedDecayAllChannel(ki, B, Omega, ρ_parameters, ho_states, ϕ0_interpolated, zPrecisionDefault, barrier)
        print(decayRatesDict)

def dicToArray(decayRates, OmegaList, BList):
    try:
        decayRatesArray = []
        for Omega in OmegaList:
            decayRatesRunning = []
            for B in BList:
                decayRatesRunning += [decayRates[(Omega, B)]]
            decayRatesArray += [decayRatesRunning]
        return np.array(decayRatesArray)
    except KeyError:
        print('OmegaList, BList')
        print(OmegaList, BList)
        print('Omega, B')
        print(Omega, B)
        print('decayRatesArray')
        print(decayRatesArray)

def runSimulationListToDict(OmegaList, BList, ki, ρ_parameters, zPrecision = zPrecisionDefault, barrier = True):
        # precalculations
    print('calculating groud state wavefunction')
    ϕ0List = createListϕ0(ki, BList, OmegaList, ρ_parameters, ho_states = None, zPrecision = zPrecision, barrier = barrier)
    print('calculating ho wavefunction')
    ho_statesList = createListHOstates(ki, BList, OmegaList, zPrecision)
    # definitions of the multiprocesses
    processes = []
    manager = multiprocessing.Manager()
    decayRates = manager.dict()
    pool = multiprocessing.Pool(processes=8)
    for index, Omega in enumerate(OmegaList):
        # print(Omega)
        for B in BList:
            pool.apply_async(multiprocessing_func, args=(decayRates, ki, B, OmegaList, index, ρ_parameters, ho_statesList, ϕ0List, zPrecision, barrier,))
    pool.close()
    startTime = time.perf_counter()
    nJobs = len(pool._cache)
    while pool._cache:
        sys.stdout.flush()
        try:
            speed = (nJobs - len(pool._cache))/(time.perf_counter() - startTime)
            tau = int(len(pool._cache)/speed)
        except (ZeroDivisionError):
            tau = 0
            time.sleep(0.2)
        print("number of jobs pending: ", len(pool._cache), ", end expected in ", tau, " seconds")
        if tau > 20:
            time.sleep(5)
        else:
            time.sleep(1)
    return decayRates
        
def runSimulationList(OmegaList, BList, ki, ρ_parameters, zPrecision = zPrecisionDefault, barrier = True):
    decayRates = runSimulationListToDict(OmegaList, BList, ki, ρ_parameters, zPrecision = zPrecisionDefault, barrier = barrier)
    return dicToArray(decayRates, OmegaList, BList)



def saveResults(OmegaList, BList, decayRates, ρ_parameters, directory):
    with open(directory + '/saveRunDecays.npy', 'wb') as f:
        np.save(f, decayRates)
    with open(directory + '/saveRunOmegaList.npy', 'wb') as f:
        np.save(f, OmegaList)
    with open(directory + '/saveRunBList.npy', 'wb') as f:
        np.save(f, BList)
    with open(directory + '/saveRunRhoDimensions.npy', 'wb') as f:
        np.save(f, ρ_parameters)

def loadResults(directory):
    with open(directory + '/saveRunDecays.npy', 'rb') as f:
        decayRates =np.load(f, allow_pickle=True)
    with open(directory + '/saveRunOmegaList.npy', 'rb') as f:
        OmegaList = np.load(f, allow_pickle=True)
    with open(directory + '/saveRunBList.npy', 'rb') as f:
        BList = np.load(f, allow_pickle=True)
    with open(directory + '/saveRunRhoDimensions.npy', 'rb') as f:
        ρ_dimensions = np.load(f, allow_pickle=True)
    return OmegaList, BList, decayRates, ρ_dimensions

### Pure 2D functions ###

def integralPure2D(ki, kf, free):
    dx = 0.01
    xmax = 10*2*np.pi/ki
    X = np.linspace(dx, xmax, int(xmax/dx))
    phiIn = phiArrayAnalytic(X, k = ki, m = 0)
    if free:
        phiOut = analyticFreeSolution(X, k = kf, m = 2)
    else:
        phiOut = phiArrayAnalytic(X, k = kf, m = 2)
    integrantPure2D = xmax*phiIn*(-3/2)/J*(2/X**3)*phiOut
    return np.sum(integrantPure2D)*dx

def betaCoefficientPure2D(ki, B, free = False):
    kf = np.sqrt(2*B + ki**2)
    Integral = integralPure2D(ki, kf, free = free)
    prefactor = 4/(ki*kf)
    cm = 100
    units = E_dd*(a_dd*cm)**2/hbar
    return prefactor*units*Integral**2

def phiZero(ρ, m):
    return np.sqrt(ρ)*sp.kn(2*m, np.sqrt(8/ρ))

def rhoPatching(k, m, patchingParameter = patchingParameter_0):
    return patchingParameter*2*np.pi*max(np.sqrt(np.abs(m**2-1/4))/k, (2/k**2)**(1/3))

def delta(k, L, m):
    ρ_max = rhoPatching(k, m)
    dρ = 0.001
    M = phiZero(ρ_max, m)
    Mprim = (phiZero(ρ_max + dρ/2, m)-phiZero(ρ_max - dρ/2, m))/dρ
    r = (1/k)*(Mprim/M-1/(2*ρ_max))
    return np.arctan(((m/(k*ρ_max)-r)*sp.jv(m, k*ρ_max)-sp.jv(m+1, k*ρ_max))/((m/(k*ρ_max)-r)*sp.yn(m, k*ρ_max)-sp.yn(m+1, k*ρ_max)))

def gamma(k, L, m):
    ρ_max = rhoPatching(k, m)
    Delta = delta(k, L, m)
    alpha = np.cos(Delta)
    beta = -np.sin(Delta)
    return (alpha*sp.jv(m, k*ρ_max)+beta*sp.yn(m, k*ρ_max))*np.sqrt(np.pi*k*ρ_max/L)/phiZero(ρ_max, m)

def analyticFarDistance(ρ_tot, k, m):
    ρmax_init = rhoPatching(k, m)
    ρmax_large = ρ_tot[-1]
    Delta = delta(k, ρmax_large, m)
    Gamma = gamma(k, ρmax_large, m)
    phiEffectiveAnalytic = np.sqrt(np.pi*k*ρ_tot/ρmax_large)*(np.cos(Delta)*sp.jv(m, k*ρ_tot) - np.sin(Delta)*sp.yn(m, k*ρ_tot))
    return phiEffectiveAnalytic

def phiArrayAnalytic(ρ_tot, k, m):
    ρmax_init = rhoPatching(k, m)
    ρmax_large = ρ_tot[-1]
    Gamma = gamma(k, ρmax_large, m)
    if Gamma >= 0:
        phiAnalytic = Gamma*phiZero(ρ_tot, m)
        phiEffectiveAnalytic = analyticFarDistance(ρ_tot, k, m)
        return phiAnalytic*(ρ_tot <= ρmax_init) + phiEffectiveAnalytic*(ρ_tot > ρmax_init)
    else:
        phiAnalytic = -Gamma*phiZero(ρ_tot, m)
        phiEffectiveAnalytic = -analyticFarDistance(ρ_tot, k, m)
        return phiAnalytic*(ρ_tot <= ρmax_init) + phiEffectiveAnalytic*(ρ_tot > ρmax_init)
    
def analyticFreeSolution(ρ_tot, k, m):
    L = ρ_tot[-1]
    return np.sqrt(np.pi*k*ρ_tot/L)*sp.jv(m, k*ρ_tot)


# saveResults(OmegaList, BList, decayRates, ρ_dimensions, z_dimensions)
# OmegaList, BList, decayRates, ρ_dimensions, z_dimensions = loadResults()
#pool = Pool(processes=8)
#for _ in tqdm.tqdm(pool.imap_unordered(do_work, tasks), total=len(tasks)):
#    pass
if __name__ == '__main__':
    # WILL NOT WORK
    startTime = time.time()
    ki = 0.06
    # ki = 0.0068 # for chromium
    # Bmax = 1.5*10**(-3) for chromium
    # OmegaMax = 0.0009479741553371919
    Bmax = 2
    OmegaMax = 1
    BList = np.linspace(0.01, Bmax, 10)
    OmegaList = np.linspace(0.02, OmegaMax, 10)
    z_max = 8
    dz = 0.02
    decayRates = main(OmegaList, BList, ki)
    print('it took ', round(time.time() - startTime, 2), ' seconds')
    decayRatesArray = dicToArray(decayRates, OmegaList, BList)
    saveResults(OmegaList, BList, decayRatesArray)

    plt.pcolormesh(OmegaList*OmegaNorm/(2*np.pi*10**3), BList*Bnorm*1000, np.log10(np.transpose(decayRatesArray)), shading = 'auto')
    plt.xlabel("$\omega$ (kHz)")
    plt.ylabel("B (mG)")
    plt.title("Decay rate log$(\\beta_{3D})$  (cm$^3$/s)")
    plt.colorbar()
    plt.show()
    #plt.savefig("colorplot.jpg", dpi = 300)


