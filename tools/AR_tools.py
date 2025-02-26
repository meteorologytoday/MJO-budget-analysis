import numpy as np
from datetime import datetime

def findTimeInd(t, t_segs):
    
    t_inds = np.zeros((len(t_segs), len(t),), dtype=bool)
    
    for i, t_seg in enumerate(t_segs):
        
        for j, _t in enumerate(t):
            
            if (_t > t_seg[0]) and (_t < t_seg[1]): 

                t_inds[i, j] = True
        

    return t_inds
    

def glue(t_segs, glue_threshold):

    # first, test each gaps

    #if len(t_segs) % 2 != 0:
    #    raise Exception("Length of t_segs is not an even number.")



    N = len(t_segs)
    
    if N == 0:
        return t_segs

    keep_gap = [ (False if ( t_segs[i+1][0] - t_segs[i][1] < glue_threshold) else True ) for i in range(N-1) ]
    
    new_t_segs = []
    

    new_t_seg = [t_segs[0][0], np.nan]
    for i in range(N-1):

        if keep_gap[i]:

            new_t_seg[1] = t_segs[i][1]
            new_t_segs.append(new_t_seg)

            new_t_seg = [t_segs[i+1][0], np.nan]
        else:
            pass

    new_t_seg[1] = t_segs[-1][1]
    new_t_segs.append(new_t_seg)
    
    return new_t_segs

def detectAbove(_t, x, x_threshold, glue_threshold=0):

    if np.any(np.isnan(x)):
        raise Exception("Data cannot contain NaN")

    #if np.any(np.isnan(_t)):
    #    raise Exception("Time cannot contain NaN")


    # convert everything into real number
    dtype_t = type(_t[0])

    if dtype_t is datetime:
        t = [ _tt.timestamp() for _tt in _t ]
    else:
        t = _t


    t_segs  = []
    seg_i = 0

    i     = 0
    flag = False

    t_seg = [np.nan, np.nan]
    for i in range(len(x)):
            
        if flag is False:
                    
            if x[i] >= x_threshold:
                flag = True
                if i == 0:
                    t_seg[0] = t[0]
                else:
                    t_seg[0] = t[i-1] + (t[i] - t[i-1]) * (x_threshold - x[i-1]) / (x[i] - x[i-1])
                
            #else: # flag is True, meaning it keeps to be above threshold
            #    if i == len(x) - 1: # last element
            #        t_seg[1] = t[-1]
            #        break

            #i#else: # x[i] < x_threshold
            
        else: # Flag is True
            
            if x[i] >= x_threshold:

                if i == len(x) - 1:
            
                    flag = False
                    t_seg[1] = t[-1]

            else:

                flag = False
                t_seg[1] = t[i] - (t[i] - t[i-1]) * (x_threshold - x[i]) / (x[i-1] - x[i])

        if not np.isnan(t_seg[0]) and not np.isnan(t_seg[1]):
            
            if dtype_t is datetime:
                t_seg[0] = datetime.fromtimestamp(t_seg[0])
                t_seg[1] = datetime.fromtimestamp(t_seg[1])

            t_segs.append(t_seg)

            t_seg = [np.nan, np.nan]
    
        
    t_segs = glue(t_segs, glue_threshold)
    t_inds = findTimeInd(_t, t_segs)   
 
    return t_segs, t_inds



def detectAbove_isolated(_t, x, x_threshold, dt):

    if np.any(np.isnan(x)):
        raise Exception("Data cannot contain NaN")

    dtype_t = type(_t[0])

    if dtype_t is datetime:
        t = [ _tt.timestamp() for _tt in _t ]
    else:
        t = _t


    dt_seconds = dt.total_seconds()
    half_dt_seconds = dt_seconds / 2

    t_segs  = []
    seg_i = 0

    t_seg = [np.nan, np.nan]
    for i in range(len(x)):
        
        if x[i] >= x_threshold:
            t_seg[0] = t[i] - half_dt_seconds
            t_seg[1] = t[i] + half_dt_seconds
                
            
            if dtype_t is datetime:
                t_seg[0] = datetime.fromtimestamp(t_seg[0])
                t_seg[1] = datetime.fromtimestamp(t_seg[1])

            t_segs.append(t_seg)

        t_seg = [np.nan, np.nan]

    
    t_inds = findTimeInd(_t, t_segs)   
 
    return t_segs, t_inds
