import numpy as np
import earth_constants as ec 


def calLaplacian(Q, lat, lon, periodoc_lon=True, gap=30.0, mask=None):

    r_E = ec.r_E
    
    Nlat, Nlon = Q.shape

    if Nlat != len(lat):
        raise Exception("Length of lat not consistent")

    if Nlon != len(lon):
        raise Exception("Length of lon not consistent")


    deg2rad = np.pi / 180.0
    _cos  = np.cos(deg2rad * lat)

    dy = (0.5 * ( r_E * deg2rad ) * ( np.roll(lat, -1) - np.roll(lat, 1) ) )[:, np.newaxis]
    
    dlon_2 =  np.roll(lon, -1) - np.roll(lon, 1)

    # `gap` is just a number that is decidedly large to
    # determin that some dlon_2s are crossing 360
    dlon_2[dlon_2 < - gap] += 360.0
    
    
    if np.any(dlon_2 < - gap):
        raise Exception("Some dlon_2 is more negative than `-gap` = %f" % (gap,))
    
    if np.any(dlon_2 <= 0):
        raise Exception("Some dlon_2 is not positive. Please check.")
    
    if np.any(dlon_2 > gap):
        raise Exception("Some dlon_2 is larger than the gap %f . This should not happen please check." % gap)
    
    
    dlon_2 = dlon_2[np.newaxis, :]
    
    dx = (0.5* r_E * _cos[:, np.newaxis] * deg2rad ) * dlon_2
    
    d2Qdx2 = (np.roll(Q, -1, axis=1) + np.roll(Q, 1, axis=1) - 2*Q) / dx**2.0 
    d2Qdy2 = (np.roll(Q, -1, axis=0) + np.roll(Q, 1, axis=0) - 2*Q) / dy**2.0
   
    lapQ = d2Qdx2 + d2Qdy2
 
    mask_nsew = np.roll(mask, -1, axis=1) * np.roll(mask, 1, axis=1) * np.roll(mask, -1, axis=0) * np.roll(mask, 1, axis=0)
   
    for _var in [lapQ,]:
        _var[0,  :] = 0.0
        _var[-1, :] = 0.0
        
        if periodoc_lon == False:
            _var[:,  0] = 0.0
            _var[:, -1] = 0.0
   
        _var[mask_nsew == 0] = 0.0
 
    return lapQ

 
def calGrad(Q, lat, lon, periodoc_lon=True, gap=30.0, mask=None):

    r_E = ec.r_E
    
    Nlat, Nlon = Q.shape

    if Nlat != len(lat):
        raise Exception("Length of lat not consistent")

    if Nlon != len(lon):
        raise Exception("Length of lon not consistent")


    deg2rad = np.pi / 180.0
    _cos  = np.cos(deg2rad * lat)

    dy_2 = (( r_E * deg2rad ) * ( np.roll(lat, -1) - np.roll(lat, 1) ) )[:, np.newaxis]

    dlon_2 =  np.roll(lon, -1) - np.roll(lon, 1)

    # `gap` is just a number that is decidedly large to
    # determin that some dlon_2s are crossing 360
    dlon_2[dlon_2 < - gap] += 360.0
    
    
    if np.any(dlon_2 < - gap):
        raise Exception("Some dlon_2 is more negative than `-gap` = %f" % (gap,))
    
    if np.any(dlon_2 <= 0):
        raise Exception("Some dlon_2 is not positive. Please check.")
    
    if np.any(dlon_2 > gap):
        raise Exception("Some dlon_2 is larger than the gap %f . This should not happen please check." % gap)
    
    
    dlon_2 = dlon_2[np.newaxis, :]
    
    dx_2 = ( r_E * _cos[:, np.newaxis] * deg2rad ) * dlon_2
    
    dQdx = (np.roll(Q, -1, axis=1) - np.roll(Q, 1, axis=1)) / dx_2
    dQdy = (np.roll(Q, -1, axis=0) - np.roll(Q, 1, axis=0)) / dy_2
    
    mask_nsew = np.roll(mask, -1, axis=1) * np.roll(mask, 1, axis=1) * np.roll(mask, -1, axis=0) * np.roll(mask, 1, axis=0)
   
    for _var in [dQdx, dQdy]:
        _var[0,  :] = 0.0
        _var[-1, :] = 0.0
        
        if periodoc_lon == False:
            _var[:,  0] = 0.0
            _var[:, -1] = 0.0
   
        _var[mask_nsew == 0] = 0.0
 
    return dQdx, dQdy

    

def calAdv(Q, u, v, lat, lon, periodoc_lon=True, gap=30.0, mask=None):

    r_E = ec.r_E

    if mask is None:
        mask = np.ones((len(lat), len(lon)))
   
    dQdx, dQdy = calGrad(Q, lat, lon, periodoc_lon=periodoc_lon, gap=gap, mask=mask)
   
    for _var in [dQdx, dQdy]:
        _var[0,  :] = np.nan
        _var[-1, :] = np.nan
        
        if periodoc_lon == False:
            _var[:,  0] = np.nan
            _var[:, -1] = np.nan
    
    adv = - ( u * dQdx + v * dQdy )
    
    return adv

    
