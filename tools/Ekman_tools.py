import numpy as np
import earth_constants as ec 
import advection_tools

def calEkmanAdv(Q, taux, tauy, H, lat, lon, periodoc_lon=True, gap=30.0):
   
    Nlat, Nlon = taux.shape

    deg2rad = np.pi / 180.0
    f_co = 2 * ec.Omega * np.sin(deg2rad * lat)[:, np.newaxis]

    _tmp = ec.rho_sw * f_co * H
    u_Ek =   tauy / _tmp
    v_Ek = - taux / _tmp

    EkmanAdv = advection_tools.calAdv(Q, u_Ek, v_Ek, lat, lon, periodoc_lon=periodoc_lon, gap=gap)

    return EkmanAdv 

    
