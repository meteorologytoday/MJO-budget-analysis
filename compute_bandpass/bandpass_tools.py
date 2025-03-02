import numpy as np

def bandpass(arr, sampling_interval, period_rng):
    
    # Assuming the first dimension is space, and second dimension is time
    
    if len(arr.shape) == 1:
        arr = np.expand_dims(arr, axis=0)
    
    if len(arr.shape) != 2:
        raise Exception("Input array can only be 1- or 2-dimensional.")

    N = arr.shape[1]
    
    freq =np.fft.fftfreq(N, d=sampling_interval)
    periods = np.abs( 1 / freq )

    print(periods)

    period_mask = np.zeros((N,))
    period_mask[ (periods > period_rng[0]) & (periods < period_rng[1]) ] = 1.0
   
    period_mask[0] = 0.0 
    
    #period_mask[:] = 0.0 
  
    print(np.sum(period_mask)) 
    spec = np.fft.fft(arr, axis=1)
    spec_new = spec * period_mask[None, :]
    
    arr_new = np.fft.ifft(spec_new, axis=1)
   
    return arr_new 
    
    
    
    
    
    
    
    
    
