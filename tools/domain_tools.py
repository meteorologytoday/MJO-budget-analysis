import numpy as np

def detectIndexRange(lat, lon, lat_rng, lon_rng):

    lat = np.array(lat)
    lon = np.array(lon) % 360

    lat_rng = np.array(lat_rng)
    lon_rng = np.array(lon_rng) % 360
    
    if lat_rng[1] < lat_rng[0]:  # across lon=0
        raise Exception("Latitude range should be lat_min, lat_max")

    lat_idx = (lat_rng[0] < lat) & (lat < lat_rng[1])

    if lon_rng[1] >= lon_rng[0]:
        lon_idx = (lon_rng[0] < lon) & (lon < lon_rng[1])
    
    else:  # across lon=0
        lon_idx = (lon_rng[0]) < lon | (lon < lon_rng[1])

    wgt = np.cos(lat * np.pi / 180)

    return lat_idx, lon_idx, wgt



