from multiprocessing import Pool
import multiprocessing
import datetime
from pathlib import Path
import os.path
import os
import netCDF4

def pleaseRun(cmd):
    print(">> %s" % cmd)
    os.system(cmd)

beg_time = datetime.datetime(1992,     8, 1)
end_time = datetime.datetime(2017,     6, 1)




g0 = 9.81
