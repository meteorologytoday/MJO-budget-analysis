import netCDF5
import numpy as np

ref_filename = "./data/ERA5/sfc/ERA5_sfc_2015-01-01.nc"
out_filename = "./mask_ERA5.nc"
land_value = -32767


with netCDF4.Dataset(ref_filename, mode='r') as ds:

    sst = ds.variables['sst'][0, :, :]
    lat = ds.variables['latitude'][:]
    lon = ds.variables['longitude'][:]


mask = np.zeros((sst.shape[0], sst.shape[1]), dtype=np.int32)

mask[:] = 1
mask[sst.mask] = 0

with netCDF4.Dataset(out_filename, mode='w', format='NETCDF4_CLASSIC') as ds_out:

    lat_dim  = ds_out.createDimension('latitude', len(lat))
    lon_dim  = ds_out.createDimension('longitude', len(lon)) 


    var_lat = ds_out.createVariable('latitude', np.float32, ('latitude',))
    var_lon = ds_out.createVariable('longitude', np.float32, ('longitude',))
    
    var_lat[:] = lat
    var_lon[:] = lon

    var_mask = ds_out.createVariable("mask", np.int32, ('latitude', 'longitude'))
    var_mask[:] = mask



