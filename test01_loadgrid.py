

import glob
from os.path import join

import numpy as np
import xarray as xr

print("Start")

ECCO_dir = "data/ECCO_LLC"
curr_shortname = "ECCO_L4_TEMP_SALINITY_LLC0090GRID_SNAPSHOT_V4R4"

curr_dir = join(ECCO_dir, curr_shortname)

files_to_load = list(glob.glob(join(curr_dir,'*_snap_2017-*.nc')))

print("Load files")
# load file into workspace


grid_params_file = join(ECCO_dir, "ECCO_L4_GEOMETRY_LLC0090GRID_V4R4", "GRID_GEOMETRY_ECCO_V4r4_native_llc0090.nc")

grid_dataset = xr.open_dataset(grid_params_file)

theta_dataset = xr.open_dataset("data/ECCO_LLC/ECCO_L4_TEMP_SALINITY_LLC0090GRID_SNAPSHOT_V4R4/OCEAN_TEMPERATURE_SALINITY_snap_2017-12-26T000000_ECCO_V4r4_native_llc0090.nc")

#theta_dataset = xr.open_mfdataset(files_to_load, parallel=True, data_vars='minimal', coords='minimal', compat='override')


import matplotlib.pyplot as plt

fig=plt.figure(figsize=(8, 6.5))
theta_dataset.THETA.isel(k=0,tile=2,time=0).plot(vmin=-2, vmax=25, cmap='jet')

plt.show()
