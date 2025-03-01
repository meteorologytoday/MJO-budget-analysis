from multiprocessing import Pool
import numpy as np
import regrid_budget_core
import traceback
from pathlib import Path
import os.path
import os
import argparse
import itertools

def pleaseRun(cmd):
    print(">> %s" % cmd)
    os.system(cmd)



def work(details):

    year = details["year"]
    month = details["month"]

    regrid_file_ECCO = details["regrid_file_ECCO"]
    regrid_file_ERA5 = details["regrid_file_ERA5"]

    detect_phase = details["detect_phase"]
    result = dict(
        details = details,
        status = "UNKNOWN",
    )

    try:
        time_str = f"{year:04d}-{month:02d}"
 
        print(f"##### Doing work of {time_str:s} #####")

        target_filename = Path("ocean_budget_stat_%s.nc" % (time_str,))
        target_fullname = Path(args.output_dir) / target_filename
            
        result["output_file"] = target_fullname

        already_exists = target_filename.exists()

        if detect_phase:

            result["need_work"] = not target_fullname.exists()
            result["status"] = "OK"

            return result        
        
        
        print("[%s] Now generate file: %s" % (time_str, target_fullname,))

        regrid_budget_core.main(
            target_fullname,
            year,
            month,
            regrid_file_ERA5,        
            regrid_file_ECCO,
        )

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
    parser.add_argument('--beg-year',   type=int, help='Wateryear', required=True)
    parser.add_argument('--end-year',   type=int, help='Wateryear', required=True)
    parser.add_argument('--output-dir', type=str, help='Output directory', default="")
    parser.add_argument('--lat-rng',    type=float, nargs=2, help='Latitude  range', required=True)
    parser.add_argument('--lon-rng',    type=float, nargs=2, help='Longitude range. 0-360', required=True)
    parser.add_argument('--lat-nbox',   type=int, help='Latitude  range', required=True)
    parser.add_argument('--lon-nbox',   type=int, help='Longitude range. 0-360', required=True)
    parser.add_argument('--mask-ERA',  type=str, help='mask file of ERA', required=True)
    parser.add_argument('--mask-ECCO',  type=str, help='mask file of ECCO', required=True)
    parser.add_argument('--regrid-file-ECCO',  type=str, help='mask file of ECCO', required=True)
    parser.add_argument('--regrid-file-ERA5',  type=str, help='mask file of ECCO', required=True)
    parser.add_argument('--ERA-type',  type=str, help='Options: ERA5 or ERAInterim', required=True, choices=["ERA5", "ERAInterim"])
    parser.add_argument('--ignore-empty-box',  action="store_true")
    args = parser.parse_args()

    print(args)
    #                print("[%04d] File '%s' already exists. Skip." % (self.year, target_fullname, ))

    input_args = []

    for year, month in itertools.product( range(args.beg_year, args.end_year+1), range(1, 13) ):

        details = dict(
            year = year,
            month = month,
            detect_phase = True,
            regrid_file_ECCO = args.regrid_file_ECCO,
            regrid_file_ERA5 = args.regrid_file_ERA5,
        )

        result = work(details)
        
        if result["status"] in ["ERROR", "UNKNOWN"]:
            print("Something went wrong when detecting year %d. Skip it." % (year,))
            
        elif result["status"] == "OK":
            
            if result["need_work"]:

                details["detect_phase"] = False
                input_args.append((details,))

            else:
                print("[%04d-%02d] Output file %s already exists. Skip." % (year, month, result["output_file"]))


    failed_cases = []
    with Pool(processes=args.nproc) as pool:

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
