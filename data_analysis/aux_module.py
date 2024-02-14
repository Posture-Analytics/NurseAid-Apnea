import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
from statsmodels.tsa.stattools import adfuller

def simulate_data(duration=60, sampling_rate=30):
    """
    Simulates a noisy 0.3Hz oscillation signal for a given duration and sampling rate.
    
    Parameters:
    duration (int): Duration of the signal in seconds.
    sampling_rate (int): Sampling rate in Hz.
    
    Returns:
    t (numpy.ndarray): Time vector.
    signal (numpy.ndarray): The simulated signal.
    """
    t = np.linspace(0, duration, int(duration * sampling_rate), endpoint=False)
    frequency = 0.3  # 0.3Hz frequency
    signal = np.sin(2 * np.pi * frequency * t)  # Sine wave at 0.3Hz
    noise = np.random.normal(0, 0.5, signal.shape)  # Gaussian noise
    noisy_signal = signal + noise
    return t, noisy_signal

# Function definitions
def filter_frequencies(signal, fs, band_mask, n_most_significant):
    """
    Filter frequencies of a signal based on power spectral density.

    Parameters:
    - signal: 1D array-like
        The input signal.
    - fs: float
        The sampling frequency of the signal.
    - band_mask: tuple
        A tuple representing the frequency band to keep, e.g., (low_cutoff, high_cutoff).
    - n_most_significant: int
        Number of most significant frequencies to keep.

    Returns:
    - filtered_signal: 1D array
        The filtered signal in the specified frequency band.
    - filtered_frequencies: 1D array
        The frequencies corresponding to the n most significant peaks in power spectral density.
    - filtered_psd: 1D array
        The power spectral density values corresponding to the n most significant peaks.
    - filtered_fft: 
    
    - new_values: 1D array
        Array that generates filtered_signal via FFT
    """

    # ===========================================================================================================================
    # = Attention!!!! The data should be conjugate symmetric in order for the time series to be real! Must be implemented soon. =
    # ===========================================================================================================================

    # Compute the FFT of the signal
    n = len(signal)

    freqs = np.fft.fftfreq(n, 1/fs)
    fft_values = np.fft.fft(signal)

    # Apply band mask
    mask = (freqs >= band_mask[0]) & (freqs <= band_mask[1])
    filtered_frequencies = freqs[mask]
    filtered_fft = fft_values[mask]
    #print("Filtered frequencies: "+ str(filtered_frequencies))

    # Compute power spectral density
    psd = np.abs(filtered_fft) ** 2

    # Select the n most significant frequencies
    indices = np.argsort(psd)[-n_most_significant:]
    filtered_frequencies = filtered_frequencies[indices]
    filtered_fft = filtered_fft[indices]
    filtered_psd = psd[indices]
    
    # Create a filter in the frequency domain
    low_cutoff, high_cutoff = band_mask

    new_values = []
    for i in range(len(fft_values)):
        if (freqs[i] < low_cutoff) or (freqs[i] > high_cutoff) or (i in indices):
            new_values.append(0)
        else:
            new_values.append(fft_values[i])
    
    filtered_signal = np.fft.ifft(new_values)

    return filtered_signal, filtered_frequencies, filtered_psd, filtered_fft, new_values

def plot_signals(original_signal, filtered_signal, fs, label):
    """
    Plot the original and filtered signals, as well as the power spectral density.

    Parameters:
    - original_signal: 1D array-like
        The original input signal.
    - filtered_signal: 1D array-like
        The filtered signal in the specified frequency band.
    - fs: float
        sample frequency
    - label: string
        label of the plot
    """

    # Plot the original and filtered signals
    plt.figure(figsize=(12, 6))

    plt.subplot(1, 1, 1)
    plt.plot(np.arange(len(original_signal)) / fs, original_signal, label='Original Signal')
    plt.plot(np.arange(len(filtered_signal)) / fs, filtered_signal, label='Filtered Signal')
    plt.title('Original and Filtered Signals: ' + label)
    plt.xlabel('Time (s)')
    plt.ylabel('Amplitude')
    plt.legend()

    plt.tight_layout()
    plt.show()

def plot_spectral_densities(freqs, psd, filtered_frequencies, filtered_psd, fs, label):
    """
    Plot the original and filtered signals, as well as the power spectral density.

    Parameters:
    - freqs: 1D array-like
        The frequency values.
    - psd: 1D array-like
        The power spectral density of the original signal.
    - filtered_frequencies: 1D array-like
        The frequencies corresponding to the n most significant peaks in power spectral density.
    - filtered_psd: 1D array-like
        The power spectral density values corresponding to the n most significant peaks.
    - fs: float
        sample frequency
    - label: string
        label of the plot
    """

    # Plot the original and filtered signals
    plt.figure(figsize=(12, 6))

    # Plot the power spectral density
    plt.subplot(1, 1, 1)
    plt.plot(freqs[:int(freqs.size/2)], psd[:int(freqs.size/2)], label='Original PSD')
    plt.plot(filtered_frequencies, filtered_psd, 'ro', label='Filtered PSD Peaks')
    plt.title('Power Spectral Density - ' + label)
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Power/Frequency')
    plt.legend()

    plt.tight_layout()
    plt.show()

def adf_test(time_series, critical_value=0.05):
    """
    Perform Augmented Dickey-Fuller test for stationarity.

    Parameters:
    - time_series: pandas Series or NumPy array, the time series to be tested.
    - critical_value: float, significance level for the test (default is 0.05).

    Returns:
    - result: pandas Series, containing test statistics and critical values.
    - is_stationary: bool, True if the time series is stationary, False otherwise.
    """

    # Perform Augmented Dickey-Fuller test
    result = adfuller(time_series)
    
    # Extract test statistics and critical values
    test_statistic = result[0]
    p_value = result[1]
    critical_values = result[4]
    
    # Check if the test statistic is less than the critical value
    ## IT IS NEEDED TO CHECK IF THIS DECISION HOLDS TRUE
    critical_value_at_significance = result[4][str(int(critical_value*100)) + "%"]
    is_stationary = test_statistic < critical_value_at_significance
    
    return result, is_stationary

def random_resample(original_sample, n_resamples=1000):
    """
    Perform bootstrap resampling on a one-dimensional sample.

    Parameters:
    - original_sample: np.array, the original one-dimensional sample
    - n_resamples: int, the number of resamples to generate

    Returns:
    - np.array, resampled data
    """

    resamples = np.random.choice(original_sample, size= n_resamples, replace=True)
    
    return resamples

def plot_normalized_histogram(data, bins=100):
    """
    Automates the simple plotting of an histogram.

    Parameters:
    - data: np.array
        The one-dimensional sample
    - n_resamples: int
        The number of bins

    Returns:
    - plot
    """


    plt.hist(data, bins=bins, density=True, alpha=0.7, color='blue', edgecolor='black')

    plt.xlabel('Value')
    plt.ylabel('Probability Density')
    plt.title('Normalized Histogram')
    
    plt.show()

def finding_peaks(time_series, fs):

    x = np.linspace(0, int(time_series.size/fs), int(time_series.size))
    y = time_series  # Adding some noise to the sine wave

    # Find peaks in the data
    peaks, _ = find_peaks(y)

    # Plot the curve
    plt.plot(x, y, label='Curve')

    # Highlight the peaks on the plot
    plt.plot(x[peaks], y[peaks], 'ro', label='Peaks')

    # Add labels and title
    plt.xlabel('X-axis')
    plt.ylabel('Y-axis')
    plt.title('Curve with Peaks - ' + str(len(peaks)) + " peaks")

    # Add legend
    plt.legend()

    # Show the plot
    plt.show()

def generate_signal(frequencies, fft_coefficients, n, fs):
    
    if(len(frequencies) != len(fft_coefficients)):
        raise Exeption("Inputs must have same size!")

    result = []
    freq = np.array(frequencies)
    fft_coef = np.array(fft_coefficients)
    ## t = np.arange(0, n, 1/fs)  # Time array
    t = np.linspace(0, n/fs, num=n, endpoint=True, dtype=None, axis=0)
    element = 0

    for i in t:
        element = 0

        for j in range(len(frequencies)):

            element += fft_coef[j] * np.exp(1j * 2 * np.pi * freq[j] * i)

        result.append(element/(len(t)))
    
    return result

def recursive_binary_TEST(intervals, time_series, a, b, fs):

    if (b-a)/fs<20: # Verify if the interval is less than the mean duration of apnea events (verifying of the number is needed)
        # If small anough ....
        intervals.append((a, b, False, None))
        return 
    
    result, is_stationary = adf_test(time_series[a:b])    
    # Check if the test function returns True for the mid element
    if is_stationary:
        # If true, return the interval
        intervals.append((a, b, True, result[1]))
        return
    
    # Calculate mid index
    mid = (a + b) // 2
    
    # If test function returns False, recursively search in the left and right halves
    recursive_binary_TEST(intervals, time_series, a, mid, fs)
    recursive_binary_TEST(intervals, time_series,  mid + 1, b, fs)

    # If not found in either, return
    return 

def intervals_cleaner(clean_intervals, intervals):
    
    '''
        deprecated!
    '''

    i = 0
    while i < len(intervals)-1:
    
        j=i
        if (intervals[i][2] == True):
            while(j < len(intervals)-1):
                if intervals[j+1][2] == False:
                    break
                j+=1
            clean_intervals.append((intervals[i][0], intervals[j][1], True))
            i = j
            
        else:
            while(j < len(intervals)-1):
                if intervals[j+1][2] == True:
                    break
                j+=1
            clean_intervals.append((intervals[i][0], intervals[j][1], False))
            i = j
        
        if (i == len(intervals)-1):
            clean_intervals.append((intervals[i][0], intervals[i][1], intervals[i][2]))
        
        i+=1
    return

def plot_intervals(time_series, intervals, decimals):
    """
    Plot time series with intervals.

    Parameters:
    - time_series (list): List representing the time series.
    - intervals (list): List of tuples with elements in the format (initial index, final index, boolean, float).
                        Boolean indicates the color of the interval (1 for blue, 0 for red).
                        Float is used as the label for the interval.
    """
    fig, ax = plt.subplots(figsize=(20, 6))

    # Plot the time series
    ax.plot(time_series, color='black')

    # Plot intervals
    for start, end, boolean, label in intervals:
        color = 'blue' if boolean else 'red'
        ax.axvspan(start, end, alpha=0.3, color=color)
        midpoint = (start + end) / 2
        
        if label == None:
            label = ""
        else: 
            label = "Pv: " + str(round(label, decimals))
        
        ax.text(midpoint, max(time_series), label, ha='center', va='bottom', color='black')

    ax.legend()
    plt.xlabel('Time Index')
    plt.ylabel('Value')
    plt.title('Time Series with Intervals')
    plt.show()
    
    return

def nyquist_truncator(intervals, nyquist_magic_number):
    size = len(intervals)
    
    for i in range(size): # goes through every element in intervals
        if (intervals[i][2] == True): # if stationary, subdivides
            prev_value = intervals[i][0]
            if (intervals[i][1] - intervals[i][0] > nyquist_magic_number): # does the element have more then the magic number?
                for j in range(intervals[i][0], intervals[i][1] - 2*nyquist_magic_number - intervals[i][0], nyquist_magic_number): # truncates 
                    next_value = j + nyquist_magic_number
                    intervals.append((prev_value, next_value, True, intervals[i][3]))
                    prev_value = next_value
                # now the end of the element
                intervals.append((prev_value, intervals[i][1], True, intervals[i][3]))
            else: # just append the value already present
                intervals.append((intervals[i][0], intervals[i][1], True, intervals[i][3]))
            
    for i in range(size):
        intervals.pop(0)