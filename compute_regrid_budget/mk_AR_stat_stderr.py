from multiprocessing import Pool
import numpy as np
import os.path as path
#import fmon_tools, watertime_tools
import anomalies
import ARstat_tool
import xarray as xr
import watertime_tools
import traceback
from pathlib import Path
import argparse
import pandas as pd
import re


parser = argparse.ArgumentParser(
                    prog = 'plot_skill',
                    description = 'Plot prediction skill of GFS on AR.',
)

parser.add_argument('--input-dir', type=str, help='Input file', required=True)
parser.add_argument('--beg-year', type=int, help='Input file', required=True)
parser.add_argument('--end-year', type=int, help='Input file', required=True)
parser.add_argument('--ncpu', type=int, help='Number of CPUs.', default=4)
parser.add_argument('--overwrite', help='If we overwrite the output', action="store_true")
parser.add_argument('--AR-algo', type=str, required=True, choices=["HMGFSC24_threshold-1998-2017",])
parser.add_argument('--annual-cnt-threshold', type=int, help='Minimum number of AR-days in order to be included to compute the standard error.', default=10)
parser.add_argument('--moving-avg-days', type=int, help='Number of days to do moving average. It has to be an odd number.', default=15)

parser.add_argument('--output-dir', type=str, help='Output dir', default="")
parser.add_argument('--title', type=str, help='Output title', default="")
parser.add_argument('--no-display', action="store_true")
args = parser.parse_args()
print(args)

if args.output_dir == "":
    args.output_dir = "%s/climanom_%04d-%04d" % (args.input_dir, args.beg_year, args.end_year, )
    
output_dir_stat = "%s/%s_annual-cnt-threshold-%02d" % (args.output_dir, args.AR_algo, args.annual_cnt_threshold)

print("Planned output dir 1: %s" % (args.output_dir,))
print("Planned output dir 2: %s" % (output_dir_stat,))

print("Create dir: %s" % (args.output_dir,))
Path(args.output_dir).mkdir(parents=True, exist_ok=True)
Path(output_dir_stat).mkdir(parents=True, exist_ok=True)

filename_clim = "%s/clim.nc" % (args.output_dir,)
filename_anom = "%s/anom.nc" % (args.output_dir,)
filename_ttl  = "%s/ttl.nc" % (args.output_dir,)

yrs = list(range(args.beg_year, args.end_year+1))

ds = ARstat_tool.loadDatasets(args.input_dir, yrs)

MLG_frc = ( ds['MLG_frc_sw'] + ds['MLG_frc_lw'] + ds['MLG_frc_sh']  + ds['MLG_frc_lh'] + ds['MLG_frc_dilu'] ).rename('MLG_frc')
MLG_vmix = (ds['MLG_vdiff'] + ds['MLG_ent_wep']).rename('MLG_vmix')
MLG_nonfrc = (ds['MLG_adv'] + ds['MLG_hdiff'] + ds['MLG_vdiff'] + ds['MLG_ent_wep'] + ds['MLG_ent_wen']).rename('MLG_nonfrc')

ds = xr.merge(
    [
        ds,
        MLG_frc,
        MLG_nonfrc,
#        MLG_adv,
#        MLG_diff,
        MLG_vmix,
    ]
)

# Compute mean and anomaly


tm = pd.date_range("2021-01-01", "2021-12-31", freq="D", inclusive="both")
ts = pd.DatetimeIndex(ds.time.to_numpy())

ds_clim = []
ds_anom = []
ds_ttl  = []

target_varnames = list(ds.keys())

    
print("===== Target varnames list =====")
for i, varname in enumerate(target_varnames):
    print("(%2d) %s" % (i, varname,))

print("================================")
# For testing:
# target_varnames = ["IWV", "IVT", "dMLTdt", "MLG_frc", "MLG_nonfrc", "MXLDEPTH"]


def doStat(varname):

    print("Doing stat of variable: %s" % (varname,))

    global tm
    _da_mean = xr.DataArray(
        name = varname,
        data = np.zeros((len(tm), len(ds.coords["lat"]), len(ds.coords["lon"]))),
        dims = ["time", "lat", "lon"],
        coords = {
            "time" : tm,
            "lat"  : ds.coords["lat"],
            "lon"  : ds.coords["lon"],
        }
    )
   

 
    _da_anom = xr.DataArray(
        name = varname,
        data = np.zeros((len(ts), len(ds.coords["lat"]), len(ds.coords["lon"]))),
        dims = ["time", "lat", "lon"],
        coords = {
            "time" : ts,
            "lat"  : ds.coords["lat"],
            "lon"  : ds.coords["lon"],
        }
    )
 
    _da_ttl = xr.DataArray(
        name = varname,
        data = np.zeros((len(ts), len(ds.coords["lat"]), len(ds.coords["lon"]))),
        dims = ["time", "lat", "lon"],
        coords = {
            "time" : ts,
            "lat"  : ds.coords["lat"],
            "lon"  : ds.coords["lon"],
        }
    )
   
    for i in range(len(ds.coords["lon"])):
        for j in range(len(ds.coords["lat"])):
            
            _var = ds[varname][:, j, i]

            xs = _var.to_numpy()
            
            tm, xm, xa, cnt, _ = anomalies.decomposeClimAnom_MovingAverage(ts, xs, n=args.moving_avg_days)

            _da_mean[:, j, i] = xm
            _da_anom[:, j, i] = xa[:]
            _da_ttl[:, j, i] = xs[:]

            #print(ts[0], ";" , ts[-1])
    return varname, _da_mean, _da_anom, _da_ttl


if args.overwrite or (not path.exists(filename_clim)) or (not path.exists(filename_anom) or (not path.exists(filename_ttl))):

    print("Ready to multiprocess the statistical job.")
    with Pool(processes=args.ncpu) as pool:

        it = pool.imap(doStat, target_varnames)
        for (varname, _da_mean, _da_anom, _da_ttl) in it:

            ds_clim.append(_da_mean)
            ds_anom.append(_da_anom)
            ds_ttl.append(_da_ttl)

            #if re.match('^map_', varname):
            #    print("Need total field of AR object mask variable: %s" % (varname,))
            #    ds_ttl.append(_da_ttl)



    print("Stat all done. Merge the outcome")

    ds_clim = xr.merge(ds_clim)
    ds_anom = xr.merge(ds_anom)
    ds_ttl  = xr.merge(ds_ttl)

    ds_clim.to_netcdf(filename_clim)
    ds_anom.to_netcdf(filename_anom)
    ds_ttl.to_netcdf(filename_ttl)

else:
    print("Files %s, %s and %s already exists. Skip the computation." % (filename_clim, filename_anom, filename_ttl))

    ds_clim = xr.open_dataset(filename_clim)
    ds_anom = xr.open_dataset(filename_anom)
    ds_ttl  = xr.open_dataset(filename_ttl)
    

# Construct
# (Beginning water month, length of months)
time_constrains = [
    (1, 1),
    (2, 1),
    (3, 1),
    (4, 1),
    (5, 1),
    (6, 1),
    (1, 6),
    (1, 3),
    (4, 3),
    (1, 2),
    (3, 2),
    (5, 2),
]

time_labels = [
    "Oct",
    "Nov",
    "Dec",
    "Jan",
    "Feb",
    "Mar",
    "Oct-Mar",
    "Oct-Dec",
    "Jan-Mar",
    "Oct-Nov",
    "Dec-Jan",
    "Feb-Mar",
]


print("Variables in ds: ", list(ds.keys()))


ARobj_map = ds["map_%s" % (args.AR_algo,)]
AR_cond  = np.isfinite(ARobj_map) & (ARobj_map > 0)
ARf_cond  = np.isfinite(ARobj_map) & (ARobj_map == 0)

ds_stats = {}

            
stat_names = ["mean", "std", "var", "cnt", "annual_mean", "annual_std", "annual_var", "annual_cnt", "years_cnt",]
#for condition_name in ["clim", "ARf", "AR", "AR+ARf"]:
for condition_name in ["clim", "AR",]:

    print("Process condition: ", condition_name)

    _tmp = {}
    for varname, _ in ds_anom.items():
        _tmp[varname] = (["time", "lat", "lon", "stat"], np.zeros((len(time_constrains), len(ds.coords["lat"]), len(ds.coords["lon"]), len(stat_names))) )

    ds_stat = xr.Dataset(
        _tmp,

        coords = {
            "time" : time_labels,
            "lat"  : ds.coords["lat"],
            "lon"  : ds.coords["lon"],
            "stat" : stat_names,
        }
    )

    ds_stats[condition_name] = ds_stat


    for m, (beg_wm, stat_len_month) in enumerate(time_constrains): 
        
        print(watertime_tools.wm2m(beg_wm), "; ", watertime_tools.wm2m(beg_wm + stat_len_month))
        
        months = watertime_tools.wm2m(np.arange(beg_wm, beg_wm+stat_len_month))

        print("[%s-%d] Used MONTHS:" % (condition_name, m,), months)

        time_cond = ds.time.dt.month.isin(months)
        time_clim_cond = ds_clim.time.dt.month.isin(months)

        
        if condition_name == "clim":
            
            _ds_ref = ds
            total_cond = time_cond

        elif condition_name == "AR+ARf":
        
            _ds_ref = ds_anom
            total_cond = time_cond
        
        elif condition_name == "AR":

            _ds_ref = ds_anom
            total_cond = time_cond & AR_cond
 
        elif condition_name == "ARf":

            _ds_ref = ds_anom
            total_cond = time_cond & ARf_cond
 
        else:
            raise Exception("Unknown condition_name: ", condition_name) 

        
        # Construct n-days in a row selection
        #ds.time.dt.month.isin(watertime_tools.wm2m(wm))

        _ds = _ds_ref.where(total_cond)

        for varname, _ in ds_stat.items():

            _data = _ds[varname].to_numpy()


            ds_stat[varname][m, :, :, 0] = np.nanmean(_data, axis=0) #_data.mean( dim="time", skipna=True)
            ds_stat[varname][m, :, :, 1] = np.nanstd(_data,  axis=0) #_data.std(  dim="time", skipna=True)
            ds_stat[varname][m, :, :, 2] = np.nanvar(_data,  axis=0) #_data.std(  dim="time", skipna=True)
            ds_stat[varname][m, :, :, 3] = np.nansum(np.isfinite(_data),  axis=0)#_data.std(  dim="time", skipna=True)


            annual_means = np.zeros( ( len(yrs), len(ds.coords["lat"]), len(ds.coords["lon"])) )
            annual_vars  = np.zeros_like(annual_means)
            annual_stds  = np.zeros_like(annual_means)
            annual_cnts  = np.zeros_like(annual_means)
            for i, water_year in enumerate(yrs):
            
                # Find the mean and std of that particular water year
                _beg = pd.Timestamp("%04d-10-01" % (water_year-1)) + pd.DateOffset(months=beg_wm-1)
                _end = _beg + pd.DateOffset(months=stat_len_month)

                #print("[%d] _beg, _end = " % water_year, _beg, ", ", _end)

                year_cond = (ds.time >=_beg) & (ds.time < _end)
                _data = _ds[varname].where(year_cond).to_numpy()
                annual_means[i, :, :] = np.nanmean(_data, axis=0)
                annual_stds[i, :, :]  = np.nanstd(_data, axis=0)
                annual_vars[i, :, :]  = np.nanvar(_data, axis=0)
                annual_cnts[i, :, :]  = np.nansum(np.isfinite(_data), axis=0)


            idx_low_cnt = annual_cnts < args.annual_cnt_threshold
            annual_means[idx_low_cnt] = np.nan
            annual_stds[idx_low_cnt] = np.nan
            annual_vars[idx_low_cnt] = np.nan
            annual_cnts[idx_low_cnt] = np.nan

            ds_stat[varname][m, :, :, 4] = np.nanmean(annual_means, axis=0)
            ds_stat[varname][m, :, :, 5] = np.nanmean(annual_stds, axis=0)
            ds_stat[varname][m, :, :, 6] = np.nanmean(annual_vars, axis=0)
            ds_stat[varname][m, :, :, 7] = np.nanmean(annual_cnts, axis=0)
            ds_stat[varname][m, :, :, 8] = np.nansum(np.isfinite(annual_cnts), axis=0)


    output_filename = "%s/stat_%s.nc" % (output_dir_stat, condition_name,)
    print("Writing output file: %s" % (output_filename,))
    
    ds_stats[condition_name].to_netcdf(output_filename)

