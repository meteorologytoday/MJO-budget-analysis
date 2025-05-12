from multiprocessing import Pool
import numpy as np
import traceback
from pathlib import Path
import argparse
import itertools
import LPT_tools
import pandas as pd 
import ERA5_tools
from scipy import signal
import xarray as xr
 
R_earth = 6.371e6

def pleaseRun(cmd):
    print(">> %s" % cmd)
    os.system(cmd)



def work(details):

    
    output_root= details["output_root"]
    dataset        = details["dataset"]
    cent_dt        = details["dt"]
    
    threshold      = details["threshold"]
    lowpass_radius = details["lowpass_radius"]
    
    detect_phase = details["detect_phase"]
    result = dict(
        details = details,
        status = "UNKNOWN",
    )

    try:
        
        time_str = cent_dt.strftime("%Y-%m-%d")
        
        print(f"##### Doing work of {time_str:s} #####")

        target_filename = Path("LPO_%s.nc" % (time_str,))
        target_fullname = output_root / dataset / f"r-{lowpass_radius:.1f}_threshold-{threshold:.1f}" / target_filename
        
        result["output_file"] = target_fullname

        already_exists = target_filename.exists()

        if detect_phase:

            result["need_work"] = not target_fullname.exists()
            result["status"] = "OK"

            return result        
        
        print("[%s] Now generate file: %s" % (time_str, target_fullname,))

        day = pd.Timedelta(days=1)
        dts = pd.date_range(cent_dt - day, cent_dt + day, freq="D", inclusive="both")

        bounded_latitude = 20.0
        precip_factor = 1.0
        
        da = None
        area = None
        kernel = None
        if dataset == "ERA5":
             
            ds = ERA5_tools.open_dataset("total_precipitation", dts)

            ds = ds.mean(dim="valid_time")
            ds = ds.where(np.abs(ds.coords["latitude"]) < bounded_latitude, drop=True)
         
            lat = ds.coords["latitude"].to_numpy() 
            lon = ds.coords["longitude"].to_numpy()  % 360
          
            llat, llon = np.meshgrid(lat, lon, indexing='ij')
            

            dlat = lat[0] - lat[1]
            dlon = lon[1] - lon[0]

            dlat_rad = np.deg2rad(dlat)
            dlon_rad = np.deg2rad(dlon)
 
            half_Nlon = 50
            half_Nlat = 50
               
            da = ds["tp"] * 1e3 * 24
            area = R_earth**2 * np.cos(np.deg2rad(llat)) * dlon_rad * dlat_rad
            kernel = LPT_tools.genGaussianKernel(half_Nlon, half_Nlat, dlon, dlat, lowpass_radius, lowpass_radius)
            
        else:
            
            raise Exception("Error: unknown dataset `%s`. " % (dataset,))

        precip = da.to_numpy()
        
        #print("Shape of precip: ", precip.shape)
        
        print(f"[{time_str:s}] Compute LPO")
        
        precip_filtered = signal.convolve2d(precip, kernel, mode="same", fillvalue=0)
        labeled_array, LPOs = LPT_tools.detectLPOs(precip_filtered, llat, llon, area, threshold=threshold, weight=None, filter_func = LPT_tools.basicLPOFilter)
        
        max_LPO_N = 50
        feature_dict = dict(
            feature_n = np.zeros((max_LPO_N,), dtype=int),
            area      = np.zeros((max_LPO_N, ),),
            centroid_lat = np.zeros((max_LPO_N,),),
            centroid_lon = np.zeros((max_LPO_N,),),
        )

        for feature_name, feature in feature_dict.items():
            for i, LPO in enumerate(LPOs):
                feature[i] = LPO[feature_name]

        data_vars = dict(
            precip    = (["time", "lat", "lon"], np.expand_dims(precip_filtered, axis=0)),
            map_label = (["time", "lat", "lon"], np.expand_dims(labeled_array, axis=0)),
        )

        for feature_name, feature in feature_dict.items():
            data_vars[feature_name] = ( ["time", "LPO"], np.expand_dims(feature, axis=0) )

        data_vars["number_of_LPOs"] = np.array([len(LPOs),], dtype=int)

        new_ds = xr.Dataset(
            data_vars = data_vars,
            coords = dict(
                time = (["time"], [cent_dt,]),
                lat  = (["lat"], lat,),
                lon  = (["lon"], lon,),
            ),
            attrs = dict(
                lowpass_radius = lowpass_radius,
                threshold = threshold,
            ),
        )
       
        Path(target_fullname.parent).mkdir(exist_ok=True, parents=True)
        new_ds.to_netcdf(target_fullname, unlimited_dims="time")
        
        if target_fullname.exists():
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

    parser.add_argument('--nproc',   type=int, help='Date string: yyyy-mm-dd', default=4)
    parser.add_argument('--beg-time',   type=str, help='Wateryear', required=True)
    parser.add_argument('--end-time',   type=str, help='Wateryear', required=True)
    parser.add_argument('--output-root', type=str, help='Output directory', required=True)
    parser.add_argument('--dataset',  type=str, help='Options: ERA5 or ERAInterim', required=True, choices=["ERA5", "ERAInterim"])
    args = parser.parse_args()

    print(args)
    #                print("[%04d] File '%s' already exists. Skip." % (self.year, target_fullname, ))

    input_args = []
    
    radius_threshold_pairs = [
        (1.0, 12.0),
        (2.0, 12.0),
        (5.0, 12.0),
    ]


    dts = pd.date_range(args.beg_time, args.end_time, freq="D")
    for dt in dts:
        for lowpass_radius, threshold in radius_threshold_pairs:
            details = dict(
                dt = dt,
                output_root = Path(args.output_root),
                dataset = args.dataset,
                threshold = threshold,
                lowpass_radius = lowpass_radius, 
                detect_phase = True,
            )
            
            result = work(details)
             
            if result["status"] in ["ERROR", "UNKNOWN"]:
                print("[%s] Something went wrong when detecting %s." % (dt.strftime("%Y-%m-%d"),))
                
            elif result["status"] == "OK":
                
                if result["need_work"]:
                    details["detect_phase"] = False
                    input_args.append((details,))
    
                else:
                    print("[%s] Output file of date %s already exists. Skip." % (dt.strftime("%Y-%m-%d"),))
    
    failed_cases = []
    with Pool(processes=args.nproc) as pool:

        results = pool.starmap(work, input_args)
        for i, result in enumerate(results):
            if result["status"] != 'OK':
                print('!!! Failed to generate output of date %s.' % (result['details']['dt'].strftime("%Y-%m-%d"), ))
                failed_cases.append(result['details'])


    print("Tasks finished.")

    if len(failed_cases) != 0:
        print("Failed cases: ")
        for i, failed_case in enumerate(failed_cases):
            print("Date: %s" % (failed_case["dt"].strftime("%Y-%m-%d"),))


    print("Done.")
