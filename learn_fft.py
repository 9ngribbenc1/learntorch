"""In this script I use Fourier analysis to remove noise from a dataset.

by Neil Campbell
August 25, 2023
"""


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
 

def noise_osc_sum(time, amps, freqs, offset):
    """ This function models the noise that remains after the signal is
    subtratced. Its amplitude is enough to make a difference based on where
    in the signal the beats are located."""

    #freq2 = 2.41*np.pi

    values = pd.Series(np.zeros(len(time))) 
    for a, f in zip(amps, freqs):
        values += (a[0]*np.cos(f*2.*np.pi*time) + a[1]*np.sin(f*2.*np.pi*time))

    return values



def main():


    directory = (r"R:\Lab Member Files\Pratap Pal\SQUID\Yuchuan"
                r"\MNN_048_out-of-plane M_v_H_5 temperatures")
    os.chdir(directory)
    filename = "iis70_data.csv"

    num_fqs = 219
    num_blank = 30

    data = pd.read_csv(filename)
    print(data.head())

    noise = data["scan0.res"]
    x = data["scan0.z"] - min(data["scan0.z"])
    print(x[:3])
    lenx = len(x)

    fft_values = np.fft.fft(noise)
    freqs = np.fft.fftfreq(len(noise), (np.max(x) - np.min(x))/len(noise))
    magnitude = np.abs(fft_values)
    print(fft_values[:4])
    print(freqs[:4], freqs[-4:])


    fft_values_list = []
    for val in fft_values:
        fft_values_list.append([np.real(val), np.imag(val)])

    args_asc = np.argsort(magnitude)
    my_args = args_asc[::-1]
    #print(my_args)
    fft_to_invert = fft_values.copy()
    fft_to_invert[:num_blank] = 0
    fft_to_invert[-num_blank:] = 0
    
    #print(magnitude[my_args[:num_fqs]])
    #print(fft_values[my_args[:num_fqs]])
    #print([fft_values_list[i] for i in my_args[:num_fqs]])
    #print(freqs[my_args[:num_fqs]])

    amps = [fft_values_list[i] for i in my_args[:num_fqs]]
    fqs = [freqs[i] for i in my_args[:num_fqs]]

    predicted = noise_osc_sum(x, amps, fqs, 0)
    ift = np.fft.ifft(fft_to_invert)*(lenx/(lenx-num_blank))


    plt.plot(x, noise)
    #plt.plot(x, predicted)
    plt.plot(x, ift)
    plt.show()

    plt.stem(freqs, magnitude)
    plt.show()


if __name__ == "__main__":
    main()
