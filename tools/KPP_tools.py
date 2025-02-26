import numpy as np


nu0  = 50e-4
p1   = 3.0
Ri_0 = 0.7

def calRi_g(N2, dUdz, dVdz):
    return N2 / (dUdz**2 + dVdz**2)

 
def calShearInstabilityMixingShape(Ri_g):
    
    s = (1.0 - (Ri_g / Ri_0)**2)**p1

    s[Ri_g < 0]    = 1.0
    s[Ri_g > Ri_0] = 0.0
    
    return s 
    

def calKappa(N2, dUdz, dVdz):

    Ri_g = calRi_g(N2, dUdz, dVdz)
    
    return nu0 * calShearInstabilityMixingShape(Ri_g)
    

