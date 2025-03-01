from multiprocessing import Pool
import numpy as np
from datetime import (datetime, timedelta, timezone)
from pathlib import Path
import os.path
import os
import netCDF4
import argparse
print("Loading libraries completed.")

def pleaseRun(cmd):
    print(">> %s" % cmd)
    os.system(cmd)


parser = argparse.ArgumentParser(
                    prog = 'plot_skill',
                    description = 'Plot prediction skill of GFS on AR.',
)

parser.add_argument('--ncpu',   type=int, help='Date string: yyyy-mm-dd', default=4)
parser.add_argument('--beg-year',   type=int, help='Wateryear', required=True)
parser.add_argument('--end-year',   type=int, help='Wateryear', required=True)
parser.add_argument('--output-dir', type=str, help='Output directory', default="")
parser.add_argument('--lat-rng',    type=float, nargs=2, help='Latitude  range', required=True)
parser.add_argument('--lon-rng',    type=float, nargs=2, help='Longitude range. 0-360', required=True)
parser.add_argument('--lat-nbox',   type=int, help='Latitude  range', required=True)
parser.add_argument('--lon-nbox',   type=int, help='Longitude range. 0-360', required=True)
parser.add_argument('--mask-ERA',  type=str, help='mask file of ERA', required=True)
parser.add_argument('--mask-ECCO',  type=str, help='mask file of ECCO', required=True)
parser.add_argument('--ERA-type',  type=str, help='Options: ERA5 or ERAInterim', required=True, choices=["ERA5", "ERAInterim"])
parser.add_argument('--ignore-empty-box',  action="store_true")
args = parser.parse_args()

print(args)
#                print("[%04d] File '%s' already exists. Skip." % (self.year, target_fullname, ))

def work(details):

    year = details["year"]
    detect_phase = details["detect_phase"]
    result = dict(
        details = details,
        status = "UNKNOWN",
    )

    try:
 
        print("##### Doing work of year %04d #####" % (self.year,))

        target_filename = Path("ocean_budget_stat_year%04d.nc" % (self.year,))
        target_fullname = Path(args.output_dir) / target_filename

        already_exists = target_filename.exists()

        if detect_phase:

            result["need_work"] = not target_filename.exists()
            result["status"] = "OK"

            return result        
        
        
        print("[%04d] Now generate file: %s" % (self.year, target_fullname,))

        cmd = [
            "python3", "construct_timeseries_by_boxes_3.py",
            "--wateryear", "%d" % (self.year,),
            "--output-dir", args.output_dir,
            "--output-filename", target_filename,
            "--lat-rng", "%f %f" % tuple(args.lat_rng),
            "--lon-rng", "%f %f" % tuple(args.lon_rng),
            "--lat-nbox", "%d" % args.lat_nbox,
            "--lon-nbox", "%d" % args.lon_nbox,
            "--mask-ERA", args.mask_ERA,
            "--mask-ECCO", args.mask_ECCO,
            "--ERA-type", args.ERA_type,
        ]

        if args.ignore_empty_box:
            cmd.append("--ignore-empty-box")

        cmd = " ".join(cmd)

        pleaseRun(cmd)

        result["status"] = "OK"
        
    except Exception as e:
        
        result["status"] = "ERROR"
        print(traceback.format_exc())

    return result

input_args = []

for year in range(args.beg_year, args.end_year+1):

    details = dict(year = year, detect_phase = True)

    result = work(details)
    
    if result["status"] in ["ERROR", "UNKNOWN"]:
        
        print("Something went wrong when detecting year %d. Skip it." % (year,))
        
    elif result["status"] == "OK":
        
        if result["need_work"]:

            details["detect_phase"] = False
            input_args.append((details,))

        else:
            print("[%04d] Output file for year %d already exists. Skip." % (year,))


failed_cases = []
with Pool(processes=nproc) as pool:

    results = pool.starmap(work, input_args)
    for i, result in enumerate(results):
        if result["status"] != 'OK':
            print('!!! Failed to generate output of year %d.' % (result['details']['year'], ))
            failed_cases.append(result['details'])


print("Tasks finished.")

if len(failed_cases) != 0:
    print("Failed cases: ")
    for i, failed_case in enumerate(failed_cases):
        print("Year %d" % (failed_case["year"],))


print("Done.")
