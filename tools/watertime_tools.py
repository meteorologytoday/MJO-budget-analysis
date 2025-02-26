from datetime import datetime, timedelta
import numpy as np

# watermonth 1 = Oct

m2wm = np.vectorize( lambda m: ((m - 10) % 12 + 1) )
wm2m = np.vectorize( lambda wm:  ((wm + 9) - 1) % 12 + 1)


def doy_leap(t):
    ref_t = datetime.datetime(2020, t.month, t.day)
    return int(ref_t.strftime('%j'))

def doy_noleap(t):

    ref_t = datetime.datetime(2022, t.month, t.day)
    return int(ref_t.strftime('%j'))

def getWateryear(d):
   
    return d.year + 1 if d.month in [10, 11, 12] else d.year

def getWaterday(d, no_leap=False):


    wy = getWateryear(d)

    if no_leap is False:
        wd = (d - datetime(wy-1, 10, 1)).total_seconds() / 86400

    elif no_leap is True:
        
        if d.month == 2 and d.day == 29:

            return np.nan
            
        else:
        

            if d.month >= 10:
                y = 2020
            else:
                y = 2021

            wd = (datetime(y, d.month, d.day) - datetime(2020, 10, 1)).total_seconds() / 86400
            
    return wd



"""
def getWatertime(d):

    wateryear = getWateryear(d)
    beg_of_wateryear = datetime(wateryear-1, 10, 1)
    end_of_wateryear = datetime(wateryear, 4, 1)
   
    normalized_time = (d - beg_of_wateryear) / (end_of_wateryear - beg_of_wateryear)

    return normalized_time
"""

def getWatertime(d):

    wateryear = getWateryear(d)
    beg_of_wateryear = datetime(wateryear-1, 10, 1)
    end_of_wateryear = datetime(wateryear, 10, 1)
   
    normalized_time = wateryear + (d - beg_of_wateryear) / (end_of_wateryear - beg_of_wateryear)

    return normalized_time

