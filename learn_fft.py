"""In this script I use Fourier analysis to remove noise from a dataset.

by Neil Campbell
August 25, 2023
"""


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os

from scipy.optimize import curve_fit
from scipy.signal import hilbert


def g(r, z):
    """ This is the function describing the magnetic flux v vertical position.
    """
    return 1./(r**2. + z**2.)**(1.5) 

def noise_osc_sum(time, amps, freqs, offset):
    """ This function models the noise that remains after the signal is
    subtratced. Its amplitude is enough to make a difference based on where
    in the signal the beats are located."""

    #freq2 = 2.41*np.pi

    values = pd.Series(np.zeros(len(time))) 
    for a, f in zip(amps, freqs):
        values += (a[0]*np.cos(f*2.*np.pi*time) + a[1]*np.sin(f*2.*np.pi*time))

    return values


def signal_v_pos_ngc(z, a, b, c, d):
    """ This function describes the total flux measured by the squid coils,
    and has terms for constant and linear deviations. The QD parameter r in
    this function have been changed by Neil Campbell to better fit the data.
    """

    # The QD MPMS values for b0 and r are fixed
    r = 8.7 #mm # let it be a free parameter and took the mode value.
                # This needs to be fixed b/c the c paramter depends on
                # its value.
    b0 = 15.2 #mm
    
    cpart = c*(g(r, z-d-b0/2.) - 2.*g(r, z-d) + g(r, z-d+b0/2.))

    #b=0.
    return a + b*z + cpart

def signal_v_pos_freer(z, a, b, c, d, r):
    """ This function describes the total flux measured by the squid coils,
    and has terms for constant and linear deviations. The QD parameter r in
    this function have been changed by Neil Campbell to better fit the data.
    """

    # The QD MPMS values for b0 and r are fixed
    #r = 8.7 #mm # let it be a free parameter and took the mode value.
                # This needs to be fixed b/c the c paramter depends on
                # its value.
    b0 = 15.2 #mm
    
    cpart = c*(g(r, z-d-b0/2.) - 2.*g(r, z-d) + g(r, z-d+b0/2.))

    #b=0.
    return a + b*z + cpart



def main():


    directory = (r"R:\Lab Member Files\Pratap Pal\SQUID\Yuchuan"
                r"\MNN_048_out-of-plane M_v_H_5 temperatures")
    os.chdir(directory)
    filename = "iis70_data.csv"

    num_fqs = 219
    num_blank = 30

    data = pd.read_csv(filename)
    print(data.head())

    raw_voltage = data["scan2.V"]
    noise = data["scan2.res"]
    x = data["scan2.z"] - min(data["scan2.z"])
    signal = data["scan2.fit"]
    print(x[:3])
    lenx = len(x)
    
    hilb = hilbert(noise)
    print(len(hilb))
    plt.plot(x,noise)
    plt.plot(x, np.abs(hilb))
    plt.show()
    # Fit raw data to the signal with r as a free parameter.
    # returns r=8.70-8.71 mm

    p0 = [0, -0.0002, -3000, 16, 9]
    popt, pcov = curve_fit(signal_v_pos_freer, x, raw_voltage, p0=p0)
    print(popt)
    plt.plot(x, raw_voltage)
    plt.plot(x, signal_v_pos_freer(x, *popt))
    plt.show()
    my_residuals = raw_voltage - signal_v_pos_freer(x, *popt)
    plt.plot(noise) 
    plt.plot(my_residuals)
    plt.plot()
    plt.show()

    fft_values = np.fft.fft(noise)
    freqs = np.fft.fftfreq(len(noise), (np.max(x) - np.min(x))/len(noise))
    magnitude = np.abs(fft_values)
    fft_values_signal = np.fft.fft(signal)
    magnitude_signal = np.abs(fft_values_signal)
    print(fft_values[:4])
    print(freqs[:4], freqs[-4:])


    fft_values_list = []
    for val in fft_values:
        fft_values_list.append([np.real(val), np.imag(val)])

    args_asc = np.argsort(magnitude)
    my_args = args_asc[::-1]
    #print(my_args)
    blank_ind = [i+num_blank for i in range(num_fqs-2*num_blank)]
    fft_to_invert = fft_values.copy()
    fft_to_invert[:num_blank] = 0
    fft_to_invert[-num_blank:] = 0
    #fft_to_invert[blank_ind] = 0
    #fft_to_invert[-num_blank:] = 0
    
    #print(magnitude[my_args[:num_fqs]])
    #print(fft_values[my_args[:num_fqs]])
    #print([fft_values_list[i] for i in my_args[:num_fqs]])
    #print(freqs[my_args[:num_fqs]])

    amps = [fft_values_list[i] for i in my_args[:num_fqs]]
    fqs = [freqs[i] for i in my_args[:num_fqs]]

    predicted = noise_osc_sum(x, amps, fqs, 0)
    ift = np.fft.ifft(fft_to_invert)*(lenx/(lenx-num_blank))
    my_clean_voltage = raw_voltage - ift
    plt.plot(x, abs(noise))
    plt.plot(x, abs(my_clean_voltage-signal_v_pos_freer(x, *popt)))
    plt.show()

    # Apply FFT to whole signal 


    plt.plot(x, noise)
    #plt.plot(x, predicted)
    plt.plot(x, ift)
    plt.show()

    #magnitude = np.abs(fft_to_invert)
    plt.stem(freqs, magnitude)
    plt.stem(freqs, magnitude_signal, linefmt="m", markerfmt="mo")
    plt.show()


if __name__ == "__main__":
    main()
