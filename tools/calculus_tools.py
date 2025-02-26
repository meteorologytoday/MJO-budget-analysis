import numpy as np


def W_ddz_T(Q, z_W=None, z_T=None, clearNaN=True):
    
    Nz, Nlat, Nlon = Q.shape

    Nz_W = Nz+1

    dQdz_W = np.zeros((Nz_W, Nlat, Nlon))
    
    if z_T is None:
        if z_W is None:
            raise Exception("z_W must be provided if z_T is not provided.")
            
        z_T = (z_W[1:] + z_W[:-1]) / 2.0


    if len(z_T) != Nz:
        raise Exception("Length of z_T (%d) is not the same as input field (%d)." % (len(z_T), Nz))

    
    dz_W = np.zeros((Nz_W,))
    dz_W[1:-1] = z_T[:-1] - z_T[1:]
    dz_W[0]    = dz_W[1]
    dz_W[-1]   = dz_W[-2]
    dz_W = dz_W[:, None, None]   

 
    dQdz_W[1:-1] = Q[:-1, :, :] - Q[1:, :, :]
    dQdz_W /= dz_W

    if clearNaN is True:
        dQdz_W[np.isnan(dQdz_W)] = 0.0

    return dQdz_W

    
