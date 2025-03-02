from multiprocessing import Pool
import itertools
import numpy as np
import xarray as xr
import traceback
from pathlib import Path
import argparse
import pandas as pd

import bandpass_tools

def makeMJOsignal(
    input_file,
    output_file,
    varname,
):
    
    time_unit = pd.Timedelta(days=1)
    
    ds = xr.open_dataset(input_file)
    da = ds[varname]
    
    da = da.transpose("lat", "lon", "time")
    
    time = pd.DatetimeIndex(da.coords["time"][0:2].to_numpy())
    sampling_interval = (time[1] - time[0]) / time_unit
   
    print("sampling_interval = ", sampling_interval)
 
    period_rng = (
        pd.Timedelta(days=20) / time_unit,
        pd.Timedelta(days=90) / time_unit,
    )

    raw_data = da.to_numpy()
    raw_shape = raw_data.shape
    raw_data = np.reshape( raw_data, (-1, len(da.coords["time"])) )
    
    new_data = bandpass_tools.bandpass(raw_data, sampling_interval, period_rng)
    new_data = np.reshape(new_data, raw_shape)
    
    da_new = da.copy().load()
    da_new.values[:] = new_data[:]
  
    da_new = da_new.transpose("time", "lat", "lon") 
    print("Writing to file: ", output_file) 
    da_new.to_netcdf(output_file)


if __name__ == "__main__":
    

    varname = "ttr"
    input_file = Path(f"gendata/anomalies/1993-2017_31S-31N-n31_100E-100W-n80/anom_{varname:s}.nc")
    output_dir = Path("gendata/bandpass")
    output_file = output_dir / input_file.name

    output_dir.mkdir(exist_ok=True, parents=True)

    print(f"varname = {varname:s}")
    print("input_file = ", str(input_file))
    print("output_file = ", str(output_file))
    
    print("Doing math...")
    makeMJOsignal(input_file, output_file, varname)
    print("Done")
    
