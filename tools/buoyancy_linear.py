import numpy as np

g0 = 9.80616     # m / s**2      copied from models/csm_share/shr/shr_const_mod.F90

# Linear TS and buoyancy parameterization
# Data is from : https://www.nemo-ocean.eu/doc/node31.html
rho_sw  = 1029.0

alpha_T = 1.6550e-1 / rho_sw
alpha_S = 7.6554e-1 / rho_sw

T_ref = 20.0
S_ref = 35.0


def TS2b(T, S):

    dT = T - T_ref
    dS = S - S_ref

    return g0 * (alpha_T * dT - alpha_S * dS) 

def TS2rho(T, S):

    dT = T - T_ref
    dS = S - S_ref
    
    rho = rho_sw * ( 1.0 - TS2b(T, S) / g0 )

    return rho

