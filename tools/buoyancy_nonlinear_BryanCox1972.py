# Follow Bryan and Cox (1972) : An Approximate Equation of State for Numerical Models of Ocean Circulation
# Here we use Table 3, and arbitrary pick Z=0m as the formula.

g0               = 9.80616     # m / s**2      copied from models/csm_share/shr/shr_const_mod.F90

T_ref =   13.5
S_ref =   32.6
rho_ref = 1024.458

rho1 = -.20134    / rho_ref
rho2 =  .77096    / rho_ref
rho3 = -.49261e-2 / rho_ref
rho4 =  .46092e-3 / rho_ref
rho5 = -.20105e-2 / rho_ref
rho6 =  .36597e-4 / rho_ref
rho7 =  .47371e-5 / rho_ref
rho8 =  .37735e-4 / rho_ref
rho9 =  .65493e-5 / rho_ref

def TS2b(T, S):
    dT = T - T_ref
    dS = S - S_ref
    b  = - g0 * ( rho1 * dT + rho2 * dS +  rho3*(dT**2) + rho4*(dS**2)
        + rho5 * (dT * dS) + rho6 * (dT**3) + rho7 * (dS**2)*dT + rho8*(dT**2)*dS + rho9*(dS**3)
    )
    return b

def TS2rho(T, S):

    dT = T - T_ref
    dS = S - S_ref
    
    rho = rho_ref * ( 1.0 - TS2b(T, S) / g0 )

    return rho

