import numpy as np

S0     = 35.0
rho_fw = 1000.0
rho0 = 1026.0
cp = 3996.0
alpha = 2e-4
beta  = 8e-4
g0 = 9.80616     # m / s**2      copied from models/csm_share/shr/shr_const_mod.F90
zeta = 23.0



NK_m = 0.5
NK_n = 0.2

def calDeltaOnlyU(h, U10):
    
    u_star = calu_star(U10)

    Delta = 2 * NK_m * u_star**3 / h**2

    return Delta


def cal_hdbwe(h, U10, sfhf, pme, F_sol):
    
    # shfh : surface heat fluxes = longwave + sensible + latent (positive upwards)
    # pme  : Precipitation minus evaporation rate (kg / s)
    # F_sol : Shortwave radiation (positive downward)

    wb_prime = sfhf * alpha * g0 / (rho0 * cp) + pme * S0 * beta * g0 / rho_fw

    u_star = calu_star(U10)
    B = calB(h, wb_prime, zeta, F_sol)



    return 2 * NK_m * u_star**3 #- 0.5 * h * ( (1 - NK_n) * abs(B) + (1 + NK_n) * B )


def cal_we(h, U10, sfhf, pme, F_sol, db):
    
    # shfh : surface heat fluxes = longwave + sensible + latent (positive upwards)
    # pme  : Precipitation minus evaporation rate (kg / s)
    # F_sol : Shortwave radiation (positive downward)
    
    return cal_hdbwe(h, U10, sfhf, pme, F_sol) / ( h * db )


def cal_dSSTdt_e(h, U10, sfhf, pme, F_sol, db, dT):
    
    # shfh : surface heat fluxes = longwave + sensible + latent (positive upwards)
    # pme  : Precipitation minus evaporation rate (kg / s)
    # F_sol : Shortwave radiation (positive downward)
    
    dbhwe = cal_hdbwe(h, U10, sfhf, pme, F_sol)
    
    print("dbhwe: %.5f, db: %.5f, h: %.1f, dT: %.1f, dbhwe/db/h**2 = %.5f" % (np.mean(dbhwe), np.mean(db), np.mean(h), np.mean(dT), np.mean(dbhwe/db/h**2)))

    return cal_we(h, U10, sfhf, pme, F_sol, db) * dT / h 



def calu_star(U10):
 
    # Wu, J. (1982). Wind‐stress coefficients over sea surface from breeze to 
    # hurricane. Journal of Geophysical Research: Oceans, 87(C12), 9704-9706.
    u_star = U10 * np.sqrt( (0.8 + 0.065 * U10) * 1e-3 ) 
    return u_star
   
def calB(h, wb_prime, zeta, F_sol):

    B = - wb_prime + alpha * g0 / (rho0 * cp) * F_sol * (1 + calI(- h, zeta) - 2 / h * calIntI(- h, zeta))
    
    return B


def calI(z, zeta):
    return np.exp(z/zeta)

def calIntI(z, zeta):
    
    return zeta * (1.0 - np.exp(z / zeta))
