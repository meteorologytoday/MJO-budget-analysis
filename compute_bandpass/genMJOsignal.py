from multiprocessing import Pool
import itertools
import numpy as np
import xarray as xr
import traceback
from pathlib import Path
import argparse
import pandas as pd
import time as timeModule
#import bandpass_tools
import lanczos_filtering

def makeMJOsignal(
    input_file,
    output_file,
    varname,
    bandpass_algo,
    bandpass_rng = (pd.Timedelta(days=20), pd.Timedelta(days=90)),
):
    
    time_unit = pd.Timedelta(days=1)
    
    ds = xr.open_dataset(input_file)
    da = ds[varname]
  
    has_z = "ocn_z" in da.dims

    if has_z:
        da = da.transpose("ocn_z", "lat", "lon", "time")
    else:
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
   
    print("Start doing bandpass for variable `%s`" % (varname,)) 
    timer_beg= timeModule.time()

    if bandpass_algo == "lanczos":    
        new_data = lanczos_filtering.easy_lanczos_filter(raw_data, sampling_interval, lowpass_period=period_rng[0], highpass_period=period_rng[1])
    elif bandpass_algo == "hat":
        new_data = lanczos_filtering.easy_hat_filter(raw_data, sampling_interval, lowpass_period=period_rng[0], highpass_period=period_rng[1])
    elif bandpass_algo == "mavg":
        lowpass_half_window_size = int(period_rng[0] // 2)
        highpass_half_window_size = int(period_rng[1] // 2)
        new_data = lanczos_filtering.mavg_filter(raw_data, highpass_half_window_size = highpass_half_window_size, lowpass_half_window_size = lowpass_half_window_size, )
    
    new_data = np.reshape(new_data, raw_shape)
    timer_end = timeModule.time()
    print("Computational time for variable `%s`: %.1f min" % (
        varname,
        (timer_end - timer_beg) / 60.0,
    ))

    
    da_new = da.copy().load()
    da_new.values[:] = new_data[:]
 
    if has_z:
        da_new = da_new.transpose("time", "ocn_z", "lat", "lon")
    else:
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
    parser.add_argument('--bandpass-algo',  type=str, help='Bandpass in days', required=True, choices=["lanczos", "hat", "mavg"])
    args = parser.parse_args()

    print(args)
    #

    varname = args.varname
    input_file  = Path(args.input_dir) / f"anom_{varname:s}.nc"
    output_file = Path(args.output_dir) / ("bandpass-%s_%s" % (args.bandpass_algo, str(input_file.name)[5:],))

    #output_dir.mkdir(exist_ok=True, parents=True)

    print(f"varname = {varname:s}")
    print("input_file = ", str(input_file))
    print("output_file = ", str(output_file))
    print("bandpass_algo = ", args.bandpass_algo)
   
    bandpass_rng = [ pd.Timedelta(days=args.bandpass_rng[0]), pd.Timedelta(days=args.bandpass_rng[1]) ]
 
    print("Doing math...")
    makeMJOsignal(input_file, output_file, varname, args.bandpass_algo, bandpass_rng)
    print("Done")
    
