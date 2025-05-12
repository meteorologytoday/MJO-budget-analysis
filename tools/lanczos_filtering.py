import numpy as np

def easy_lanczos_filter(y, sampling_interval, lowpass_period=0, highpass_period=np.inf, axis=-1):

    window_length = y.shape[axis] * sampling_interval
     
    lowpass_wvn  = window_length / lowpass_period
    highpass_wvn = window_length / highpass_period
    
    return lanczos_filter(y, lowpass_wvn=lowpass_wvn, highpass_wvn=highpass_wvn, axis=axis)

def easy_hat_filter(y, sampling_interval, lowpass_period=0, highpass_period=np.inf, axis=-1):

    window_length = y.shape[axis] * sampling_interval
     
    lowpass_wvn  = window_length / lowpass_period
    highpass_wvn = window_length / highpass_period
    
    return hat_filter(y, lowpass_wvn=lowpass_wvn, highpass_wvn=highpass_wvn, axis=axis)

   
    
    
def mavg_filter(y, highpass_half_window_size, lowpass_half_window_size, axis=-1):
    
    y = np.array(y)
    highpass_window_size = 2*highpass_half_window_size + 1   
    highpass_kernel = np.ones(highpass_window_size) / highpass_window_size  # Uniform kernel
    mavg = np.apply_along_axis(lambda m: np.convolve(m, highpass_kernel, mode='same'), axis=axis, arr=y)
    
    y_anom = y - mavg
    
    lowpass_window_size = 2*lowpass_half_window_size + 1   
    lowpass_kernel = np.ones(lowpass_window_size) / lowpass_window_size  # Uniform kernel
    y_truncated = np.apply_along_axis(lambda m: np.convolve(m, lowpass_kernel, mode='same'), axis=axis, arr=y_anom)
    
    return y_truncated
   

# it is assumed that domain length is 2 pi
def lanczos_filter(y, lowpass_wvn=np.inf, highpass_wvn=0, axis=-1):
    
    y = np.array(y)
    N = y.shape[axis]
    
    wvn  = np.fft.fftfreq(N, d=1/N)
    y_spec = np.fft.fft(y, axis=axis)



    _tmp = np.pi * wvn / lowpass_wvn
    sigma_factor_lowpass    = np.sin(_tmp) / _tmp
    sigma_factor_lowpass[0] = 1.0


    _tmp = np.pi * wvn / highpass_wvn
    sigma_factor_highpass    = np.sin(_tmp) / _tmp
    sigma_factor_highpass[0] = 1.0

    sigma_factor_bandpass = sigma_factor_lowpass - sigma_factor_highpass

    sigma_factor_bandpass[np.abs(wvn) >= lowpass_wvn]   = 0.0
    sigma_factor_bandpass[np.abs(wvn)  < highpass_wvn]  = 0.0



    for _ in range(y.ndim - 1):
        sigma_factor_bandpass = np.expand_dims(sigma_factor_bandpass, 0)

    y_spec *= sigma_factor_bandpass

    truncated_y = np.fft.ifft(y_spec)

    return truncated_y 

def hat_filter(y, lowpass_wvn=np.inf, highpass_wvn=0, axis=-1):
    
    y = np.array(y)
    
    N = y.shape[axis]
    
    wvn = np.fft.fftfreq(N, d=1/N)
    y_spec = np.fft.fft(y, axis=axis)


    _tmp = np.ones_like(wvn)
    _tmp[np.abs(wvn) >= lowpass_wvn]  = 0.0
    _tmp[np.abs(wvn) <  highpass_wvn] = 0.0

    factor_bandpass = _tmp
    for _ in range(y.ndim - 1):
        factor_bandpass = np.expand_dims(factor_bandpass, 0)

    print(factor_bandpass.shape)
    y_spec *= factor_bandpass
    
    truncated_y = np.fft.ifft(y_spec)

    return truncated_y 



if __name__ == "__main__":
   


    cutoff_period_lowpass  = 30.0 # day
    #cutoff_period_highpass = np.inf # day
    cutoff_period_highpass = 120.0
    dt = 1.0 # day
    
    t = np.arange(0, 360*2, dtype=float)
    y = t.copy()
    y[:] = 0.0
    y[(t > 50) & (t < 100)] = 3.0
  
    y += np.random.normal(0, 0.5, len(t))
    y += 2.0 * np.sin(2*np.pi/360*t)
    y += 1.0 * np.sin(2*np.pi/25*t)

 
    y -= y.mean()

    window_length = dt * len(t)
    cutoff_period_lowpass_wvn  = window_length / cutoff_period_lowpass
    cutoff_period_highpass_wvn = window_length / cutoff_period_highpass


    print("cutoff_period_lowpass_wvn = ", cutoff_period_lowpass_wvn)
    print("cutoff_period_highpass_wvn = ", cutoff_period_highpass_wvn)


    y1 = hat_filter(y, lowpass_wvn=cutoff_period_lowpass_wvn,   highpass_wvn=cutoff_period_highpass_wvn)
    y2 = lanczos_filter(y, lowpass_wvn=cutoff_period_lowpass_wvn, highpass_wvn=cutoff_period_highpass_wvn)


    y_sp = np.fft.fft(y)
    y1_sp = np.fft.fft(y1)
    y2_sp = np.fft.fft(y2)


    rng = slice(1, len(t)//2)
    y_resp = np.real(y_sp / y_sp)[rng]
    y1_resp = np.real(y1_sp / y_sp)[rng]
    y2_resp = np.real(y2_sp / y_sp)[rng]




    print("Plotting...")
    import matplotlib as mplt
    import matplotlib.pyplot as plt

    
    fig, ax = plt.subplots(2, 1)
    
    ax[0].plot(t, y, label="original")
    ax[0].plot(t, y1, label="hat")
    ax[0].plot(t, y2, label="lanczos")
    ax[0].legend()

    ax[1].plot(np.abs(y_resp), label="original")
    ax[1].plot(np.abs(y1_resp), label="hat")
    ax[1].plot(np.abs(y2_resp), label="lanczos")
    ax[1].legend()

    for _ax in ax.flatten():
        _ax.grid()

    fig.suptitle("Bandpass period = (%.1f, %.1f)" % (cutoff_period_lowpass, cutoff_period_highpass) )

    plt.show()
     
