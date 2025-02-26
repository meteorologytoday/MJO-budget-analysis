import glob
import numpy as np
import xarray as xr
import ecco_v4_py as ecco

import ECCO_helper

from datetime import (datetime, timedelta)
print("Start")

beg_datetime = datetime(2017, 12, 26)
end_datetime = datetime(2017, 12, 28)

total_days = int((end_datetime - beg_datetime).total_seconds() / 86400)

for _t in range(total_days):

    _now_datetime = beg_datetime + _t * timedelta(days=1)

    print("Test datetime: ", _now_datetime)
    tends = ECCO_helper.computeTendency( _now_datetime)



    ds = xr.Dataset(data_vars={})
    for varname, G in tends.items():
        ds[varname] = G

    ds.time.encoding = {}
    ds.reset_coords(drop=True)

    filename = "G_%s.nc" % _now_datetime.strftime("%Y-%m-%d") 
    print("Output: ", filename)
    ds.to_netcdf(filename, format='NETCDF4')
