import pandas as pd
import xarray as xr
import os
import numpy as np
from pathlib import Path

archive_root = Path("dataset") / "ERA5-derived-daily" / "1_hourly"


mapping_longname_shortname = {
    'total_precipitation'           : 'tp',
    'geopotential'                  : 'z',
    '10m_u_component_of_wind'       : 'u10',
    '10m_v_component_of_wind'       : 'v10',
    'mean_sea_level_pressure'       : 'msl',
    '2m_temperature'                : 't2m',
    'sea_surface_temperature'       : 'sst',
    'specific_humidity'             : 'q',
    'u_component_of_wind'           : 'u',
    'v_component_of_wind'           : 'v',
    'mean_surface_sensible_heat_flux'    : 'msshf',
    'mean_surface_latent_heat_flux'      : 'mslhf',
    'mean_surface_net_long_wave_radiation_flux'  : 'msnlwrf',
    'mean_surface_net_short_wave_radiation_flux' : 'msnswrf',
}

mapping_shortname_longname = { shortname : longname for longname, shortname in mapping_longname_shortname.items() }

file_prefix = "ERA5-derived-daily"
def generate_filename(varname, dt):
   
    if varname in mapping_longname_shortname.keys():
        long_varname = varname
        short_varname = mapping_longname_shortname[varname]
    else:
        long_varname = mapping_shortname_longname[varname] 
        short_varname = varname

    dt_str = pd.Timestamp(dt).strftime("%Y-%m-%d")

    filename = archive_root / long_varname / "{file_prefix:s}-{long_varname:s}-{time:s}.nc".format(
        file_prefix = file_prefix,
        long_varname = long_varname,
        time = dt_str,
    )
 
    return filename

def open_dataset(varname, dts):
  
    filenames = [
        generate_filename("total_precipitation", dt)
        for dt in dts
    ]
    
    ds = xr.open_mfdataset(filenames)
    
    lon_first_zero = np.argmax(ds.coords["longitude"].to_numpy() >= 0)
    #print("First longitude zero idx: ", lon_first_zero)
    ds = ds.roll(longitude=-lon_first_zero, roll_coords=True)


    lon = ds.coords["longitude"].to_numpy()  % 360
  
    # For some reason we need to reassign it otherwise the contourf will be broken... ??? 
    ds = ds.assign_coords(lon=lon) 
   
    return ds    

if __name__ == "__main__":   
    
    date = "2000-01-01"
    
    ds = open_dataset("v_component_of_wind", date, "inst")

    print(ds)
 
