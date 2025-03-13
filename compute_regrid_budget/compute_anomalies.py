from multiprocessing import Pool
import itertools
import numpy as np
#import fmon_tools, watertime_tools
import anomalies
import xarray as xr
import traceback
from pathlib import Path
import argparse
import pandas as pd
import re
import time

def loadDatasets(input_dir, years, file_fmt="ocean_budget_stat_{year:04d}-{month:02d}.nc", data_vars="all"):
    
    input_dir = Path(input_dir)
    filenames = [ input_dir / file_fmt.format(year=year, month=month) for year, month in itertools.product( years, range(1, 13) ) ]
    ds = xr.open_mfdataset(filenames, data_vars=data_vars)
    
    return ds
 

def computeClimAnl(da, year_rng, moving_avg_days, vertical_layers_needed=None):

    varname = da.name

    # tm = time of climatology mean
    tm = pd.date_range("2021-01-01", "2021-12-31", freq="D", inclusive="both")
    ts = pd.DatetimeIndex(da.coords["time"].to_numpy())

    needed_dts = list(pd.date_range(
        pd.Timestamp(year=year_rng[0], month=1, day=1),
        pd.Timestamp(year=year_rng[1]+1, month=1, day=1),
        inclusive="left",
    ))

    if len(needed_dts) != len(ts):
        raise Exception("Length of loaded data is wrong. We need %d time points but we only have %d." % (len(needed_dts), len(ts)))
 
    # Test if time all exists
    if not np.all( [needed_dt == _ts for needed_dt, _ts in zip(needed_dts, ts) ] ):
        raise Exception("Loaded data has different time than expected")

    has_z = "ocn_z" in da.dims
   
    if has_z and vertical_layers_needed is not None:
        da = da.isel(ocn_z=slice(0, vertical_layers_needed))
 
    lat = da.coords["lat"]
    lon = da.coords["lon"]
    
    coords = dict(lat=lat, lon=lon)
    ocn_z = None
    if has_z:
        ocn_z = da.coords["ocn_z"]
        coords["ocn_z"] = ocn_z
        dims = ["time", "ocn_z", "lat", "lon"]

    else:
        dims = ["time", "lat", "lon"]
 
    # mean
    if has_z:
        shape = (len(tm), len(ocn_z), len(lat), len(lon))    
    else:
        shape = (len(tm), len(lat), len(lon))    

    coords["time"] = tm
    _da_mean = xr.DataArray(
        name = varname,
        data = np.zeros(shape),
        dims = dims,
        coords = coords,
    )

    # anom 
    if has_z:
        shape = (len(ts), len(ocn_z), len(lat), len(lon))    
    else:
        shape = (len(ts), len(lat), len(lon))    

    coords["time"] = ts
    _da_anom = xr.DataArray(
        name = varname,
        data = np.zeros(shape),
        dims = dims,
        coords = coords,
    )
 
    _da_ttl = _da_anom.copy()


    xs = da.to_numpy()
    tm, xm, xa, cnt, _ = anomalies.decomposeClimAnom_MovingAverage_all(ts, xs, n=moving_avg_days)


    _da_mean.values[:] = xm[:]
    _da_anom.values[:] = xa[:]
    _da_ttl.values[:]  = xs[:]


    """    
    for idx in itertools.product(*iter_rng):
        
        full_idx = (slice(None), *idx)

        xs = full_data[full_idx]
        tm, xm, xa, cnt, _ = anomalies.decomposeClimAnom_MovingAverage_all(ts, xs, n=moving_avg_days)

        _da_mean.values[full_idx] = xm
        _da_anom.values[full_idx] = xa[:]
        _da_ttl.values[full_idx]  = xs[:]
    """
    return _da_mean, _da_anom, _da_ttl



def main(
    varname,
    year_rng,
    moving_avg_days,
    input_dir,
    output_dir,
    overwrite=False,
    vertical_layers_needed=None,
):
    print("Processing varname", varname)

    input_dir       = Path(input_dir)
    output_dir      = Path(output_dir)
    output_dir_stat = output_dir 

    print("Planned output dir 1: %s" % (str(output_dir),))
    print("Planned output dir 2: %s" % (str(output_dir_stat),))

    output_dir.mkdir(parents=True, exist_ok=True)
    output_dir_stat.mkdir(parents=True, exist_ok=True)

    filename_clim = output_dir / ("clim_%s.nc" % (varname,))
    filename_anom = output_dir / ("anom_%s.nc" % (varname,))
    filename_ttl  = output_dir / ("ttl_%s.nc"  % (varname,))

    if np.all([ filename_clim.exists(), filename_ttl.exists(), filename_anom.exists() ]) and (overwrite == False):
        raise Exception("The target files %s, %s, %s all exist. " % (filename_clim, filename_ttl, filename_anom)) 

    years = list(range(year_rng[0], year_rng[1]+1))

    da = loadDatasets(input_dir, years, data_vars=[varname,])[varname]
  
    #MLG_frc = ( ds['MLG_frc_sw'] + ds['MLG_frc_lw'] + ds['MLG_frc_sh']  + ds['MLG_frc_lh'] + ds['MLG_frc_dilu'] ).rename('MLG_frc')
    #MLG_vmix = (ds['MLG_vdiff'] + ds['MLG_ent_wep']).rename('MLG_vmix')
    #MLG_nonfrc = (ds['MLG_adv'] + ds['MLG_hdiff'] + ds['MLG_vdiff'] + ds['MLG_ent_wep'] + ds['MLG_ent_wen']).rename('MLG_nonfrc')

    # Compute mean and anomaly
    print("Computate climate and anomalies for variable `%s`" % (varname,))
    timer_beg= time.time()
    da_clim, da_anom, da_ttl = computeClimAnl(da, year_rng, moving_avg_days, vertical_layers_needed=vertical_layers_needed)
    timer_end = time.time()
    print("Computational time for variable `%s`: %.1f min" % (
        varname,
        (timer_end - timer_beg) / 60.0,
    ))

    print("Output file 1: ", filename_ttl)
    print("Output file 2: ", filename_clim)
    print("Output file 3: ", filename_anom)

    da_ttl.to_netcdf(filename_ttl)
    da_clim.to_netcdf(filename_clim)
    da_anom.to_netcdf(filename_anom)


def distributedWork(details):

    year_rng = details["year_rng"]
    varname = details["varname"]
    moving_avg_days = details["moving_avg_days"]
    input_dir = Path(details["input_dir"])
    output_dir = Path(details["output_dir"])
    vertical_layers_needed = details["vertical_layers_needed"]


    detect_phase = details["detect_phase"]
    result = dict(
        details = details,
        status = "UNKNOWN",
    )

    try:
        
        work_label = f"{varname:s}"
 


        target_filenames = [
            output_dir / ("clim_%s.nc" % (varname,)),
            output_dir / ("anom_%s.nc" % (varname,)),
            output_dir / ("ttl_%s.nc" % (varname,)),
        ]

        all_exist = np.all( [ target_filename.exists() for target_filename in target_filenames ] )

        if detect_phase:

            result["need_work"] = not all_exist
            result["status"] = "OK"
            return result        
        
        print(f"##### Doing work of {work_label:s} #####")
        
        main(
            varname,
            year_rng,
            moving_avg_days,
            input_dir,
            output_dir,
            vertical_layers_needed=vertical_layers_needed,
            overwrite=True,
        )
        
        check_all_exist = np.all( [ target_filename.exists() for target_filename in target_filenames ] )
 
        if check_all_exist:
            result["status"] = "OK"
        else:
            result["status"] = "ERROR"
        
    except Exception as e:
        
        result["status"] = "ERROR"
        print(traceback.format_exc())

    return result


if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(
                        prog = 'plot_skill',
                        description = 'Plot prediction skill of GFS on AR.',
    )

    parser.add_argument('--nproc',   type=int, help='Date string: yyyy-mm-dd', default=1)
    parser.add_argument('--year-rng',   type=int, nargs=2, help='Year', required=True)
    parser.add_argument('--input-dir', type=str, help='Input file', required=True)
    parser.add_argument('--output-dir', type=str, help='Output directory', required=True)
    parser.add_argument('--vertical-layers-needed', type=int, help='how many vertical layers needed?', default=None)
    parser.add_argument('--ncpu', type=int, help='Number of CPUs.', default=4)
    parser.add_argument('--moving-avg-days', type=int, help='Number of days to do moving average. It has to be an odd number.', default=15)
    args = parser.parse_args()

    print(args)
    #                print("[%04d] File '%s' already exists. Skip." % (self.year, target_fullname, ))

    if args.vertical_layers_needed == -1:
        args.vertical_layers_needed = None

    input_args = []

    varnames = None
    with loadDatasets(args.input_dir, [args.year_rng[0], args.year_rng[0]]) as ds:
        varnames = list(ds.keys())

    needed_varnames = []
    print("##### We have the following variables: #####")
    for i, varname in enumerate(varnames):
        print("[%2d] %s" % (i+1, varname))

        if varname == "data_good":
            continue

        needed_varnames.append(varname)
    print("############################################")

    for varname in needed_varnames:

        details = dict(
            year_rng = args.year_rng,
            varname = varname,
            moving_avg_days = args.moving_avg_days,
            input_dir = args.input_dir,
            output_dir = args.output_dir,
            detect_phase = True,
            vertical_layers_needed = args.vertical_layers_needed,
        )

        result = distributedWork(details)
        if result["status"] in ["ERROR", "UNKNOWN"]:
            print("Something went wrong when detecting variable %s. Skip it." % (varname,))
            
        elif result["status"] == "OK":
            
            if result["need_work"]:

                details["detect_phase"] = False
                input_args.append((details,))

            else:
                print("Output files of %s already exists. Skip." % (varname,))


    failed_cases = []
    with Pool(processes=args.nproc) as pool:

        results = pool.starmap(distributedWork, input_args)
        for i, result in enumerate(results):
            if result["status"] != 'OK':
                print('!!! Failed to generate output of var %s.' % (result['details']['varname'], ))
                failed_cases.append(result['details'])


    print("Tasks finished.")

    if len(failed_cases) != 0:
        print("Failed cases: ")
        for i, failed_case in enumerate(failed_cases):
            print("Varname %s" % (failed_case["varname"],))


    print("Done.")

