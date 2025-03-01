import xarray as xr
import numpy as np

def loadDatasets(input_dir, yrs, file_fmt="AR_statistics_wateryear%04d.nc"):
    filenames = [ "%s/%s" % (input_dir, file_fmt % yr) for yr in yrs  ]
    data = xr.open_mfdataset(filenames)
    return data
   

def countInNdays(cond, ndays=1):

    if ndays <= 0:
        raise Exception("Parameter `ndays` must be a positive integer.")
    
    count = cond.copy().astype(int)
    new_count = count.copy()
    
    for nday in range(1, ndays):
        shifts = dict(time= - nday)

        new_count += (
            count.shift(shifts=shifts, fill_value=0) 
            & ( cond.time.shift(shifts=shifts) == (cond.time + np.timedelta64(nday, 'D')) ) 
        )
        
    return new_count
        


def ifNdaysInARow(cond, ndays=1):

    if ndays <= 0:
        raise Exception("Parameter `ndays` must be a positive integer.")
    
    new_cond = cond.copy()

    for nday in range(1, ndays):
        shifts = dict(time= - nday)

        new_cond = (
            new_cond 
            & cond.shift(shifts=shifts, fill_value=False) 
            & ( cond.time.shift(shifts=shifts) == (cond.time + np.timedelta64(nday, 'D')) ) 
        )
        
    return new_cond
        

if __name__ == "__main__":
     

    ds = xr.open_dataset("output_ECCO/1993-2017_10N-60N-n25_120E-120W-n60/AR_statistics_yr2005.nc")
    ivt = ds.IVT[:20, 5, 5]
    cond = ivt >= 250
   
    print("IVT: ") 
    print(ivt)
    print("IVT condition flags: ") 
    print(cond)

    for n in range(1, 6):
        print("IVT count in %d day(s): " % (n,)) 
        print(countInNdays(cond, n))



    for n in range(1, 6):
        print("IVT condition %d day(s) in a row: " % (n, )) 
        print(ifNdaysInARow(cond, n))

    
