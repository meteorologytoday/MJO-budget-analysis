# Follow Millero & Poisson (1981): International one-atmosphere equation of state of seawater

# Notice that the formula typed in paper abstract is incorrect:
#
#   1. The C * S term on the right-hand-side of formula rho should be C * (S**2) just
#      like Equation (2)
#   2. The coefficient for the 5th power of rho0 is written 6.536336e-9 in abstract
#      but 6.536332e-9 in Equation (6). Since abstract formula is a typo, I would trust
#      the number in Equation (6) more. 
#

g0       = 9.80616     # m / s**2      copied from models/csm_share/shr/shr_const_mod.F90
rhoConst = 1029

def TS2rho(T, S):

    T1 = T
    T2 = T**2
    T3 = T**3
    T4 = T**4
    T5 = T**5
    
    A =  8.24493e-1 - 4.0899e-3 * T1 + 7.6438e-5 * T**2 - 8.2467e-7 * T**3 + 5.3875e-9 * T**4
    B = -5.72466e-3 + 1.0227e-4 * T1 - 1.6546e-6 * T**2
    C =  4.8314e-4
    rho0 = 999.842594 + 6.793952e-2 * T1 - 9.095290e-3 * T2 + 1.001685e-4 * T3 - 1.120083e-6 * T4 + 6.536332e-9 * T5

    rho = rho0 + A * S + B * S**(1.5) + C * S**2

    return rho

if __name__ == "__main__":

    print("In __main__. Making a density plot for checking...")

    print("Check value in Millero & Poisson (1981) Table 1")
    print("rho = ", TS2rho(5, 35))

    import matplotlib.pyplot as plt
    import numpy as np
    
    T_vec = np.linspace(0, 30, 121)
    S_vec = np.linspace(0, 40, 161)

    T_vec2 = np.linspace(0, 20, 121)
    S_vec2 = np.linspace(33.5, 36.5, 161)

    TT, SS = np.meshgrid(T_vec, S_vec, indexing='ij')
    TT2, SS2 = np.meshgrid(T_vec2, S_vec2, indexing='ij')

    print("Computing density...")
    rhorho = TS2rho(TT, SS)
    rhorho2 = TS2rho(TT2, SS2)

    print("Generating plot...")
    fig, ax = plt.subplots(1, 2, figsize=(10, 6))


    cs = ax[0].contour(S_vec, T_vec, rhorho, levels=20, colors="k")
    plt.clabel(cs)

    cs = ax[1].contour(S_vec2, T_vec2, rhorho2, levels=np.arange(1020, 1030, .5), colors="k")
    plt.clabel(cs)


    for _ax in ax.flatten():
        _ax.set_ylabel("Temperature [${}^\\circ \\mathrm{C}$]")
        _ax.set_xlabel("Salinity [ PSU ]")

    print("Displaying...")
    plt.show()





