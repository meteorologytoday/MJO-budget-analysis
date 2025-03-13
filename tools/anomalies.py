import numpy as np
import datetime
from datetime import timedelta, datetime
import pandas as pd

def doy_noleap(t):

    ref_t = datetime(2022, t.month, t.day)

    return int(ref_t.strftime('%j'))

def doy_leap(t):
    ref_t = datetime(2020, t.month, t.day)
    return int(ref_t.strftime('%j'))

def total_doy(t):
    return int((datetime(t.year+1, 1, 1) - datetime(t.year, 1, 1) ).total_seconds()/86400)

def fraction_of_year(t):

    return ( doy_leap(dt) - 1 ) / total_doy(dt)


def decomposeClimAnom(ts, xs: np.ndarray, assist = None):

    if assist is None:

        tm = pd.date_range("2021-01-01", "2021-12-31", freq="D", inclusive="both")

        doy  = np.zeros(len(ts), dtype=np.int32)
        skip = np.zeros(len(ts), dtype=np.bool_)
        cnt  = np.zeros(len(tm), dtype=np.int32)
        
        doy[:] = -1
        for i, t in enumerate(ts):

            m = t.month
            d = t.day

            if m == 2 and d == 29:
                skip[i] = True
            else:
                skip[i] = False
                doy[i] = doy_noleap(t)

                cnt[doy[i]-1] += 1

        assist = {
            'doy'  : doy,
            'skip' : skip,
            'cnt'  : cnt,
            'tm'   : tm,
        }

    tm = assist['tm']
    doy  = assist['doy']       
    cnt  = assist['cnt']       
    skip = assist['skip']       
   
    xm       = np.zeros((len(tm),))

    for i, t in enumerate(ts):

        if skip[i] :
            continue
        
        xm[doy[i]-1]  += xs[i]
        

    cnt_nz = cnt != 0
    xm[cnt_nz] /= cnt[cnt_nz]
    xm[cnt == 0] = np.nan

    xa = np.zeros((len(xs),))
    for i, t in enumerate(ts):

        m = t.month
        d = t.day

        if m == 2 and d == 29:

            # Interpolate Feb 29 if Feb 28 and Mar 1 exist.
            if i > 0 and i < (len(ts) - 1):
                xa[i] = (xa[i-1] + xa[i+1]) / 2.0
            else:
                xa[i] = np.nan
        
        else:
           
            #print("xs[%d]: " % i, xs[i], "; xm[doy[i]-1]: ", xm[doy[i]-1]) 
            xa[i] = xs[i] - xm[doy[i]-1]


    return tm, xm, xa, cnt, assist


def decomposeClimAnom_MovingAverage_all(ts, xs: np.ndarray, assist = None, n=1):



    if len(xs.shape) <= 2:
        raise Exception("The parameter `xs` should be a 2-dim array (time, space). Now we have the shape: %s" % (str(xs.shape),))
    
    xs = np.ascontiguousarray(xs) # So that space index moves the fastest

    if n % 2 != 1:
        raise Exception("The parameter `n` has to be an odd number. Now it is {n:s}".format(n=str(n)))

    n_half = int( (n - 1) / 2)

    if assist is None:

        # 2021 is not a leap year
        tm = pd.date_range("2021-01-01", "2021-12-31", freq="D", inclusive="both")
        

        xs_original_shape = np.array(xs.shape)
        
        spatial_pts = xs.size // len(ts)
        xs_shape = ( len(ts),  spatial_pts )
        xm_shape = ( len(tm),  spatial_pts )

        doy  = np.zeros(len(ts), dtype=np.int32)
        skip = np.zeros(len(ts), dtype=np.bool_)
        cnt  = np.zeros(len(tm), dtype=np.int32)
        
        doy[:] = -1
        for i, t in enumerate(ts):

            m = t.month
            d = t.day

            if m == 2 and d == 29:
                skip[i] = True
            else:
                skip[i] = False
                doy[i] = doy_noleap(t)

                cnt[doy[i]-1] += 1

        assist = {
            'doy'  : doy,
            'skip' : skip,
            'cnt'  : cnt,
            'tm'   : tm,
            'spatial_shape' : xs_original_shape[1:],
            'xs_shape' : xs_shape,
            'xm_shape' : xm_shape,
        }


    tm = assist['tm']
    doy  = assist['doy']
    cnt  = assist['cnt']
    skip = assist['skip']
    xm_shape = assist['xm_shape']
    xs_shape = assist['xs_shape']
    spatial_shape = assist['spatial_shape']
   

    xs = np.reshape(xs, xs_shape)
    xm = np.zeros(xm_shape)
    xa = np.zeros(xs_shape)


    for i, t in enumerate(ts):

        # Skip Feb 29
        if skip[i] :
            continue
        
        xm[doy[i]-1, :]  += xs[i, :]
        

    # Use the cyclic shift
    roll_sum_xm  = np.zeros_like(xm)
    roll_sum_cnt = np.zeros_like(cnt)
    for _shift in range(- n_half, n_half+1):
        roll_sum_xm += np.roll(xm, _shift, axis=0)
        roll_sum_cnt += np.roll(cnt, _shift)

    xm = roll_sum_xm / roll_sum_cnt[:, None]
    xm[roll_sum_cnt==0] = np.nan



    # First round, compute anomalies
    for i, t in enumerate(ts):
        xa[i, :] = xs[i, :] - xm[doy[i]-1, :]


    # Second round, fill up Feb 29
    for i, t in enumerate(ts):

        m = t.month
        d = t.day

        if m == 2 and d == 29:
            # Interpolate Feb 29 if Feb 28 and Mar 1 exist.
            if i > 0 and i < (len(ts) - 1):
                xa[i, :] = (xa[i-1, :] + xa[i+1, :]) / 2.0
            else: # if cannot be interpolated because at the beginning or end of the data
                xa[i, :] = np.nan



    xm = np.reshape(xm, (len(tm), *spatial_shape))
    xa = np.reshape(xa, (len(ts), *spatial_shape))
    
    return tm, xm, xa, cnt, assist


def decomposeClimAnom_MovingAverage(ts, xs: np.ndarray, assist = None, n=1):

    if n % 2 != 1:
        raise Exception("The parameter `n` has to be an odd number. Now it is {n:s}".format(n=str(n)))

    n_half = int( (n - 1) / 2)

    if assist is None:

        # 2021 is not a leap year
        tm = pd.date_range("2021-01-01", "2021-12-31", freq="D", inclusive="both")

        doy  = np.zeros(len(ts), dtype=np.int32)
        skip = np.zeros(len(ts), dtype=np.bool_)
        cnt  = np.zeros(len(tm), dtype=np.int32)
        
        doy[:] = -1
        for i, t in enumerate(ts):

            m = t.month
            d = t.day

            if m == 2 and d == 29:
                skip[i] = True
            else:
                skip[i] = False
                doy[i] = doy_noleap(t)

                cnt[doy[i]-1] += 1

        assist = {
            'doy'  : doy,
            'skip' : skip,
            'cnt'  : cnt,
            'tm'   : tm,
        }


    tm = assist['tm']
    doy  = assist['doy']
    cnt  = assist['cnt']
    skip = assist['skip']
   
    xm       = np.zeros((len(tm),))

    for i, t in enumerate(ts):

        # Skip Feb 29
        if skip[i] :
            continue
        
        xm[doy[i]-1]  += xs[i]
        


    # Use the cyclic shift

    roll_sum_xm  = np.zeros_like(xm)
    roll_sum_cnt = np.zeros_like(cnt)
    for _shift in range(- n_half, n_half+1):
        roll_sum_xm += np.roll(xm, _shift)
        roll_sum_cnt += np.roll(cnt, _shift)

    xm = roll_sum_xm / roll_sum_cnt
    xm[roll_sum_cnt==0] = np.nan

    xa = np.zeros((len(xs),))

    # First round, compute anomalies
    for i, t in enumerate(ts):
        xa[i] = xs[i] - xm[doy[i]-1]


    # Second round, fill up Feb 29
    for i, t in enumerate(ts):

        m = t.month
        d = t.day

        if m == 2 and d == 29:
            # Interpolate Feb 29 if Feb 28 and Mar 1 exist.
            if i > 0 and i < (len(ts) - 1):
                xa[i] = (xa[i-1] + xa[i+1]) / 2.0
            else:
                xa[i] = np.nan
        

    return tm, xm, xa, cnt, assist



if __name__ == "__main__":

    # Creating fake data
    import pandas as pd
   
    beg_year = 2021
    end_year = 2030

    expected_cnt = end_year - beg_year + 1
 
    dates    = pd.date_range(start="%04d-01-01" % (beg_year,), end="%04d-12-31" % (end_year,), freq="D")
    noise    = np.random.randn(len(dates))
    raw_data = np.zeros((len(dates),))
    
    amp = 10.0
    wnm = 1.0

    for (i, _dt) in enumerate(dates):

        dt = pd.to_datetime(_dt)
        frac = fraction_of_year(dt)

        raw_data[i] = np.sin(2*np.pi * wnm * frac ) * amp + noise[i]

        
        #print("%d : %d-%d-%d, fraction of that year: %.4f" % (i, dt.year, dt.month, dt.day, ( doy_leap(dt) - 1 ) / total_doy(dt)))
        


    t_clim, clim, anom, cnt, _ = decomposeClimAnom_MovingAverage(dates, raw_data, n=15)

    if np.any(cnt != expected_cnt):
        print("[Debug] Expected cnt: ", expected_cnt)
        print("[Debug] Computed cnt = ", cnt)
        raise Exception("Count should be %d but we don't get it correct.")


    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(3, 1)
        
    ax[0].plot(dates, raw_data, label="Raw data")

    ax[1].plot(dates, noise, "k-", label="actual noise")
    ax[1].plot(dates, anom, "r--", label="computed anomalies")
    
    ax[2].plot(t_clim, clim, "k-", label="computed climatology")

    for _ax in ax:
        _ax.legend()

    plt.show()
     


