import numpy as np

patchingParameter_0 = 0.15

k_B = 1.38*10**(-23)
h = 6.626*10**(-34)
hbar = h/(2*np.pi)
mu_B = 9.27*10**(-24) # ampere metres carres
mu_0 = 1.25663706*10**(-6) # magnetic constant m kg s-2 A-2
m_unit = 1.6605*10**(-27) # kg
a_0 = 5.291772109*10**(-11) # m

atom = 'dysprosium'
#atom = 'chromium'

if atom == 'chromium':
    m = 52*m_unit
    mu = 6*mu_B
    g_J = 2
    J = 3
    
if atom == 'dysprosium':
    m = 161.926805*m_unit
    mu = 9.93*mu_B
    g_J = 9.93/8
    J = 8

a_dd = m/2*mu_0/(4*np.pi)*mu**2/hbar**2
E_dd = hbar**2/(2*m/2*a_dd**2)
Bnorm = E_dd/(g_J*mu_B)*10**4  # B*Bnorm is the magnetic field in gauss
OmegaNorm = E_dd/hbar # Omega*OmegaNorm is 2pi*frequency in Hz

defaultKwargs = {}
#zPrecisionDefault = [5, 0.1]
zPrecisionDefault = [8, 0.04]
#nmaxCap = 100
nmaxCap = 150