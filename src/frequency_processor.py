import numpy as np

from scipy.fft import fft, fftfreq

from scipy.signal import find_peaks

import matplotlib.pyplot as plt

from src import aux_module as aux

class FrequencyProcessor:
    """
    A class used to process frequency signals.

    Attributes:
        window_sample (dict): A dictionary containing timestamps as keys and signal values as values.
        window_duration (float): The duration of the signal window in seconds.
        window_period (float): The period of the signal window in seconds.
        window_sample_rate (float): The sample rate of the signal window in Hz.
        spectrum (list): A list of tuples containing frequency and amplitude pairs.
        highest_frequencies (list): A list of tuples containing the n highest frequencies and their amplitudes.
    """

    def __init__(self, window_sample: dict, show_plots: bool = True):
        """
        Initializes a new instance of the FrequencyProcessor class.

        Args:
            window_sample (dict): A dictionary containing timestamps as keys and signal values as values.
        """

        self.window_sample = window_sample
        self.show_plots = show_plots

        self.window_duration = None

        self.window_period = None
        self.window_sample_rate = None

        self.spectrum = None
        self.highest_frequencies = None

        # Get sample statistics
        self.get_sample_statistics()

    def get_period(self) -> float:
        """
        Calculate the period of the signal.

        Returns:
            float: The period of the signal.
        """

        # Get the number of samples
        num_samples = len(self.window_sample)

        # Get the first and last timestamps
        timestamps = list(self.window_sample.keys())

        first_timestamp = int(timestamps[0])
        last_timestamp = int(timestamps[-1])

        # TODO: add a flag to indicate inesperable stops that can invalidate the calculation

        self.window_duration = (last_timestamp - first_timestamp) / 1000 # in seconds

        # Calculate the period
        self.window_period = (self.window_duration) / num_samples

        return self.window_period

    def get_sample_rate(self) -> float:
        """
        Calculate the sample rate of the signal.

        Returns:
            float: The sample rate of the signal.
        """

        assert self.window_period is not None

        # Calculate the sample rate
        self.window_sample_rate = 1.0 / self.window_period

        return self.window_sample_rate

    def get_sample_statistics(self) -> None:
        """
        Calculate and set the period and sample rate of the signal.
        """

        # Get the period
        self.period = self.get_period()

        # Get the sample rate
        self.sample_rate = self.get_sample_rate()

    def process_signal(self, n_freqs: int = 5, lowcut: float = 0.1, highcut: float = 1.0) -> None:

        # Remove the mean from the signal
        noisy_signal = list(self.window_sample.values())
        noisy_signal = noisy_signal - np.mean(noisy_signal)
        
        # Process the signal
        filtered_signal, filtered_frequencies, filtered_psd, filtered_fft, new_values = aux.filter_frequencies(noisy_signal, self.sample_rate, (lowcut, highcut), n_freqs)

        # Plot the results
        if self.show_plots:
            freqs = np.fft.fftfreq(len(noisy_signal), 1/self.sample_rate)
            psd = np.abs(np.fft.fft(noisy_signal)) ** 2
            aux.plot_signals(noisy_signal, filtered_signal, fs=self.sample_rate, label="Dados uniformes, 0,x Hz")
            aux.plot_spectral_densities(freqs, psd, filtered_frequencies, filtered_psd, fs=self.sample_rate, label="Dados uniformes, 0,x Hz")

        # # Print a summary of the results
        # for i in range(n_freqs):
        #     print("freq: ", filtered_frequencies[i], "- psd: ", filtered_psd[i])

        filtered_results = list(zip(filtered_frequencies, filtered_psd))
        filtered_results = sorted(filtered_results, key=lambda x: x[1], reverse=True)

        # Store the results
        self.spectrum = filtered_results
    
    def count_peaks(self) -> int:
        """
        Count the number of peaks in the reconstructed time-domain signal.

        Returns:
            int: The number of peaks.
        """

        assert self.ifft_y is not None

        # Find the peaks
        peaks, _ = find_peaks(self.ifft_y)

        # Count the peaks
        self.num_peaks = len(peaks)

        return self.num_peaks
    
    def get_respiratory_rate(self, unit: str = "Hz") -> float:
        """
        Calculate the respiratory rate from the number of peaks.

        Args:
            unit (str): The unit for respiratory rate calculation. Default is "Hz".

        Returns:
            float: The respiratory rate.
        """

        assert self.num_peaks is not None

        # Calculate the respiratory rate
        self.respiratory_rate = self.num_peaks / self.window_duration # in Hz

        if unit == "BPM":
            self.respiratory_rate *= 60

        return self.respiratory_rate
    
    def plot_signals(self) -> None:
        """
        Plot the original signal (pixel sums) and the reconstructed time-domain signal.
        """

        fig, axs = plt.subplots(2,1)

        # Plot the original signal (pixel sums)
        timestamps = list(self.window_sample.keys())
        pixel_sums = list(self.window_sample.values())

        axs[0].plot(timestamps, pixel_sums)
        axs[0].set_title("Original signal (pixel sums over the mask)")

        # Reconstruction the time-domain signal from the 

