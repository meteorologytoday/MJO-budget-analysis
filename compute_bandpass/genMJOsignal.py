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
    bandpass_rng = (pd.Timedelta(days=20), pd.Timedelta(days=90)),
):
    
    time_unit = pd.Timedelta(days=1)
    
    ds = xr.open_dataset(input_file)
    da = ds[varname]
    
    da = da.transpose("lat", "lon", "time")
    
    time = pd.DatetimeIndex(da.coords["time"][0:2].to_numpy())
    sampling_interval = (time[1] - time[0]) / time_unit
   
    print("sampling_interval = ", sampling_interval)
 
    period_rng = (
        bandpass_rng[0] / time_unit,
        bandpass_rng[1] / time_unit,
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
    
    parser = argparse.ArgumentParser(
                        prog = 'plot_skill',
                        description = 'Plot prediction skill of GFS on AR.',
    )

    parser.add_argument('--varname',    type=str, help='mask file of ERA', required=True)
    parser.add_argument('--input-dir',  type=str, help='mask file of ERA', required=True)
    parser.add_argument('--output-dir',  type=str, help='mask file of ERA', required=True)
    parser.add_argument('--bandpass-rng',  type=float, help='Bandpass in days', default=[20.0, 90.0])
    args = parser.parse_args()

    print(args)
    #

    varname = args.varname
    input_file  = Path(args.input_dir) / f"anom_{varname:s}.nc"
    output_file = Path(args.output_dir) / ("bandpass_%s" % str(input_file.name)[5:])

    #output_dir.mkdir(exist_ok=True, parents=True)

    print(f"varname = {varname:s}")
    print("input_file = ", str(input_file))
    print("output_file = ", str(output_file))
   
    bandpass_rng = [ pd.Timedelta(days=args.bandpass_rng[0]), pd.Timedelta(days=args.bandpass_rng[1]) ]
 
    print("Doing math...")
    makeMJOsignal(input_file, output_file, varname, bandpass_rng)
    print("Done")
    
