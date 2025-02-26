with open("shared_header.py", "rb") as source_file:
    code = compile(source_file.read(), "shared_header.py", "exec")
exec(code)
import pandas as pd
import numpy as np
import argparse
import ECCO_helper, ECCO_computeTendency
import xarray as xr
import postprocess_ECCO_tools
import traceback

parser = argparse.ArgumentParser(
                    prog = 'postprocess_ECCO.py',
                    description = 'Postprocess ECCO data (Mixed-Layer integrated).',
)

parser.add_argument('--MLD-method', required=True, help="If set then use ECCO MLD instead.", type=str, choices=["RHO", "FIXED500m"])
parser.add_argument('--nproc', type=int, default=2)
args = parser.parse_args()
print(args)

output_root_dir = "data/ECCO_LLC"

MLD_dev = 0.03



def ifSkip(dt):

    skip = False

    #if 5 <= dt.month and dt.month <= 8 :
    #    skip = True
 
    # We need extra days to compute dSST/dt
    #if dt.month == 4 and dt.day != 1:
    #    skip = True
 
    #if dt.month == 9 and dt.day != 30:
    #    skip = True


    return skip




def work(dt, output_filename_G_terms, output_filename_MXLANA):

    result = dict(status="OK", dt=dt, target_files = [output_filename_G_terms, output_filename_MXLANA])
  
    try: 

        time_now_str = dt.strftime("%Y-%m-%d")

        print("[%s] Work starts." % (time_now_str,))
        
        dir_name = os.path.dirname(output_filename_G_terms)
        if not os.path.isdir(dir_name):
            print("Create dir: %s" % (dir_name,))
            Path(dir_name).mkdir(parents=True, exist_ok=True)
        
        
        print("[%s] Now computing G terms..." % (time_now_str,))
        tends = ECCO_computeTendency.computeTendency(dt)

        ds = xr.Dataset(data_vars={})
        for varname, G in tends.items():
            ds[varname] = G

        ds.time.encoding = {}
        ds.reset_coords(drop=True)

        print("Output: ", output_filename_G_terms)
        ds.to_netcdf(output_filename_G_terms, format='NETCDF4')
        ds.close()
           
        """
        # Phase 2
        # This one computes the advection. Does not depend on MLD_method.
        _tmp = ECCO_helper.getECCOFilename("HADV_g", "DAILY", dt)
        output_filename_ADV = "%s/%s/%s" % (output_root_dir, _tmp[0], _tmp[1])
        
        dir_name = os.path.dirname(output_filename_ADV)
        if not os.path.isdir(dir_name):
            print("Create dir: %s" % (dir_name,))
            Path(dir_name).mkdir(parents=True, exist_ok=True)
        
        if os.path.isfile(output_filename_ADV):
            print("[%s] File %s already exists. Skip." % (time_now_str, output_filename_ADV))

        else:
            print("[%s] Now compute the advection." % (time_now_str, ))
            
            ds = ECCO_computeTendency.computeTendencyAdv(dt)
            print("Output: ", output_filename_ADV)
            ds.to_netcdf(output_filename_ADV, format='NETCDF4')
        """

        # Phase 3
        # This one computes the mixed-layer integrated quantities
        _tmp = ECCO_helper.getECCOFilename("MLT", "DAILY", dt, extra_dirsuffix=extra_dirsuffix)
        output_filename_MXLANA = "%s/%s/%s" % (output_root_dir, _tmp[0], _tmp[1])
        
        dir_name = os.path.dirname(output_filename_MXLANA)
        if not os.path.isdir(dir_name):
            print("Create dir: %s" % (dir_name,))
            Path(dir_name).mkdir(parents=True, exist_ok=True)
      
        print("[%s] Now compute the mixed-layer integrated quantities. Method = %s" % (time_now_str, args.MLD_method))
        
        if args.MLD_method == "RHO":
            fixed_MLD = -1.0

        elif args.MLD_method == "FIXED500m":
            fixed_MLD = 500.0

        
        postprocess_ECCO_tools.processECCO(
            dt,
            output_filename_MXLANA,
            fixed_MLD=fixed_MLD, 
        ) 


    except Exception as e:

        result['status'] = "ERROR"
        traceback.print_stack()
        print(e)


    print("[%s] Job done." % (time_now_str,))



    return result


failed_dates = []
    
dts = pd.date_range(beg_time.strftime("%Y-%m-%d"), end_time.strftime("%Y-%m-%d"), inclusive="both")
input_args = []
for dt in dts:

    y = dt.year
    m = dt.month
    d = dt.day
    time_now_str = dt.strftime("%Y-%m-%d")

    if ifSkip(dt):
        
        print("Skip the date: %s" % (time_now_str,))
        continue

    if args.MLD_method == "FIXED500m":
        
        extra_dirsuffix = "_500m"

    else:
        
        extra_dirsuffix = ""
    
            
    _tmp = ECCO_helper.getECCOFilename("Gs_ttl", "DAILY", dt)
    output_filename_G_terms = "%s/%s/%s" % (output_root_dir, _tmp[0], _tmp[1])

    _tmp = ECCO_helper.getECCOFilename("MLT", "DAILY", dt, extra_dirsuffix=extra_dirsuffix)
    output_filename_MXLANA = "%s/%s/%s" % (output_root_dir, _tmp[0], _tmp[1])

    all_exists = True

    for output_filename in [output_filename_G_terms, output_filename_MXLANA]:
        if not os.path.isfile(output_filename):
            all_exists = False

    if all_exists:        
        print("[%s] File all exists. Skip." % (time_now_str, ))
    else:
        input_args.append((dt, output_filename_G_terms, output_filename_MXLANA))
        


with Pool(processes=args.nproc) as pool:

    results = pool.starmap(work, input_args)

    for i, result in enumerate(results):
        if result['status'] != 'OK':
            print('[%s] Failed to generate output %s and %s' % (
                result['dt'].strftime("%Y-%m-%d_%H"),
                *result['target_files'],
            ))

            failed_dates.append(result['dt'])


print("Tasks finished.")

print("Failed dates: ")
for i, failed_date in enumerate(failed_dates):
    print("%d : %s" % (i+1, failed_date.strftime("%Y-%m-%d"),))
