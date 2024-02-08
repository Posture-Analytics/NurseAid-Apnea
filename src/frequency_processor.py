import numpy as np

from scipy.fft import fft, fftfreq

from scipy.signal import find_peaks

import matplotlib.pyplot as plt

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

    def __init__(self, window_sample: dict):
        """
        Initializes a new instance of the FrequencyProcessor class.

        Args:
            window_sample (dict): A dictionary containing timestamps as keys and signal values as values.
        """

        self.window_sample = window_sample

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

    def get_frequency_spectrum(self) -> None:
        """
        Calculate the frequency spectrum of the signal.
        """

        print("Calculating the frequency spectrum...")
        print("Period: {} seconds".format(self.period))
        print("Sample rate: {} Hz".format(self.sample_rate))

        # Get the pixel sums of the samples (y)
        y = np.array(list(self.window_sample.values()))

        # Subtract the mean from the pixel sums
        y = y - np.mean(y)

        # Get the frequencies (x) and get the detecable frequencies (according to the Nyquist frequency)
        x = fftfreq(len(y), d=self.period)[:len(y) // 2]

        # Apply the Fast Fourier Transform (FFT)
        y = fft(y)[:len(y) // 2]

        # Normalize the FFT
        y = 2.0 / len(self.window_sample) * np.abs(y)

        # Make the pairs (frequency, amplitude)
        self.spectrum = list(zip(x, y))

        # Sort the spectrum by amplitude
        self.spectrum.sort(key=lambda x: x[1], reverse=True)

    def get_highest_frequencies(self, n: int = 5, bandpass_filter: bool = True) -> list:
        """
        Get the n highest frequencies in the spectrum.

        Args:
            n (int): The number of highest frequencies to retrieve.
            bandpass_filter (bool): Whether to apply bandpass filtering. Default is True.

        Returns:
            list: A list of tuples containing the n highest frequencies and their amplitudes.
        """

        if bandpass_filter:
            assert self.bandpass_lowcut is not None
            assert self.bandpass_highcut is not None

            # Apply the bandpass filter
            analyzed_spectrum = self.apply_bandpass_filter()
        else:
            analyzed_spectrum = self.spectrum

        # Select the n highest frequencies
        self.highest_frequencies = analyzed_spectrum[:n]

        return self.highest_frequencies
    
    def set_bandpass_filter(self, lowcut: float, highcut: float) -> None:
        """
        Set the bandpass filter cutoff frequencies.

        Args:
            lowcut (float): The low cutoff frequency.
            highcut (float): The high cutoff frequency.
        """

        self.bandpass_lowcut = lowcut
        self.bandpass_highcut = highcut

    def apply_bandpass_filter(self) -> list:
        """
        Apply the bandpass filter to the spectrum.

        Returns:
            list: A list of tuples containing the filtered frequencies and their amplitudes.
        """

        assert self.spectrum is not None

        assert self.bandpass_lowcut is not None
        assert self.bandpass_highcut is not None

        # # Get the frequencies and amplitudes
        # frequencies, amplitudes = zip(*self.spectrum)

        # Apply the bandpass filter
        self.filtered_spectrum = list(filter(lambda x: self.bandpass_lowcut <= x[0] <= self.bandpass_highcut, self.spectrum))

        # Sort the bandpass filter by amplitude
        self.filtered_spectrum.sort(key=lambda x: x[1], reverse=True)

        return self.filtered_spectrum

    # def apply_IFFT(self):

    #     assert self.spectrum is not None

    #     # Get the frequencies and amplitudes
    #     self.highest_frequencies, amplitudes = zip(*self.highest_frequencies)

    #     # Apply the IFFT
    #     self.ifft_y = np.fft.ifft(self.highest_frequencies)

    #     return self.ifft_y

    def reconstruct_time_domain_signal(self) -> None:
        """
        Reconstruct the time-domain signal from the selected n highest frequencies.
        """

        assert self.spectrum is not None

        # Get the frequencies and amplitudes
        frequencies, amplitudes = zip(*self.highest_frequencies)

        # Create an array of zeros with the same length as the original signal
        reconstructed_signal = np.zeros(len(self.window_sample))

        # Place the selected significant frequency components at the corresponding positions in the array
        for i in range(len(frequencies)):
            reconstructed_signal[int(frequencies[i])] = amplitudes[i]

        # To ensure the signal is centered correctly for the IFFT, you need to mirror the frequencies in the second half of the spectrum. This is because the FFT coefficients you have correspond to positive and negative frequencies. For each selected frequency component, also place its complex conjugate at the mirrored position in the array.
        for i in range(len(frequencies)):
            # Skip DC component
            if i != 0:
                mirrored_index = len(reconstructed_signal) - int(frequencies[i])
                if mirrored_index < 0:
                    mirrored_index = 0
                if mirrored_index >= len(reconstructed_signal):
                    mirrored_index = len(reconstructed_signal) - 1
                reconstructed_signal[mirrored_index] = np.conj(amplitudes[i])

        # Apply the IFFT (Inverse Fast Fourier Transform) to the modified array to obtain the time-domain signal
        self.ifft_y = np.fft.ifft(reconstructed_signal)

        # Plot the IFFT
        plt.plot(self.ifft_y)
        plt.show()

    # def process_signal(self, n_freqs: int = 5, lowcut: float = 0.1, highcut: float = 1.0) -> None:
    #     """
    #     Process the signal to extract features and plot the results.

    #     Args:
    #         n_freqs (int): The number of top frequencies to consider. Default is 5.
    #         lowcut (float): The low cutoff frequency for bandpass filtering. Default is 0.1 Hz.
    #         highcut (float): The high cutoff frequency for bandpass filtering. Default is 1.0 Hz.
    #     """

    #     # Get the signal and the timestamps
    #     timestamps = np.array(list(self.window_sample.keys()))
    #     signal = np.array(list(self.window_sample.values()))

    #     # FFT of the signal
    #     fft_result = np.fft.rfft(signal)
    #     freqs = np.fft.rfftfreq(len(signal), d=1/self.window_sample_rate)
    #     psd = np.abs(fft_result) ** 2

    #     # Identify the top frequencies
    #     top_indices = np.argsort(psd)[-top_freq:]
    #     top_frequencies = freqs[top_indices]

    #     # Filter design to keep only the top frequencies
    #     nyquist_rate = self.window_sample_rate / 2.0
    #     filter_band = [(f - 0.05) / nyquist_rate for f in top_frequencies] + \
    #                   [(f + 0.05) / nyquist_rate for f in top_frequencies]
    #     filter_band = np.clip(filter_band, 0, 1) # Ensure within [0, 1] range
    #     filter_coefs = np.zeros(len(freqs))
    #     for i in range(top_freq):
    #         filter_coefs[(freqs >= filter_band[2*i]) & (freqs <= filter_band[2*i+1])] = 1

    #     # Apply the filter in the frequency domain
    #     filtered_fft = fft_result * filter_coefs
    #     filtered_signal = np.fft.irfft(filtered_fft)

    #     # Print the top frequencies and their corresponding amplitudes
    #     print("Top frequencies:")
    #     for i in range(top_freq):
    #         print("{:.2f}Hz: {:.2f}".format(top_frequencies[i], psd[top_indices[i]]))

    #     # Find and count the peaks
    #     peaks, _ = find_peaks(filtered_signal)

    #     # Count the peaks
    #     num_peaks = len(peaks)


    #     print("Number of peaks: {}".format(num_peaks))
    #     print("Frequency of peaks: {:.2f}Hz".format(num_peaks / self.window_duration))

    #     # Plot the original and filtered signals
    #     plt.figure(figsize=(15, 5))
    #     plt.plot(timestamps, signal, label='Original Signal')
    #     plt.plot(timestamps, filtered_signal, label='Filtered Signal')

    #     # Plot a red cross at the location of each peak
    #     plt.plot(peaks, filtered_signal[peaks], "x", color="red")

    #     plt.xlabel('Time (s)')
    #     plt.ylabel('Signal Amplitude')
    #     plt.legend()
    #     plt.show()


    def process_signal(self, n_freqs: int = 5, lowcut: float = 0.1, highcut: float = 1.0) -> None:
        """
        Process the signal to extract features and plot the results.

        Args:
            n_freqs (int): The number of top frequencies to consider. Default is 5.
            lowcut (float): The low cutoff frequency for bandpass filtering. Default is 0.1 Hz.
            highcut (float): The high cutoff frequency for bandpass filtering. Default is 1.0 Hz.
        """

        # Get the signal and the timestamps
        timestamps = np.array(list(self.window_sample.keys()))
        original_signal = np.array(list(self.window_sample.values()))

        # Subtracting the mean from the signal to remove the zero-frequency component
        signal = original_signal - np.mean(original_signal)

        # FFT
        fft_result = np.fft.rfft(signal)
        fft_freq = np.fft.rfftfreq(len(signal), d=1/self.window_sample_rate)
        fft_ampl = np.abs(fft_result)

        # Bandpass filter
        # Create a boolean mask for frequencies within the desired range
        band_mask = (fft_freq >= lowcut) & (fft_freq <= highcut)
        # Apply mask to keep only the desired frequencies
        fft_result_filtered = fft_result * band_mask
        
        # Inverse FFT to get the filtered time domain signal
        filtered_signal = np.fft.ifft(fft_result_filtered)
        
        # Select the top N frequencies
        # Find indices of the top N amplitudes within the bandpass range
        top_indices = np.argsort(fft_ampl[band_mask])[-n_freqs:]
        top_freqs = fft_freq[band_mask][top_indices]
        top_ampls = fft_ampl[band_mask][top_indices]

        # Print the top frequencies and their corresponding amplitudes
        print("Top frequencies:")
        for i in range(n_freqs):
            print("{:.2f}Hz: {:.2f}".format(top_freqs[i], top_ampls[i]))

        # Find the peaks
        peaks, _ = find_peaks(filtered_signal)

        # Count the peaks
        num_peaks = len(peaks)

        print("Number of peaks: {} = {:.2f}Hz = {:.2f}BPM".format(num_peaks, num_peaks / self.window_duration, num_peaks / self.window_duration * 60))
    
        # Plot the original, filtered signal and the top frequencies
        plt.figure(figsize=(12, 6))

        # Original signal
        plt.subplot(3, 1, 1)
        plt.title("Original Signal")
        plt.plot(timestamps, original_signal, label='Original')
        plt.legend()

        # Filtered signal
        plt.subplot(3, 1, 2)
        plt.title("Filtered Signal")

        # raise ValueError(f"x and y must have same first dimension, but "
        # ValueError: x and y must have same first dimension, but have shapes (232,) and (117,)
        # Fix this issue generaing a new array for the x axis

        x = np.linspace(0, len(filtered_signal.real), len(filtered_signal.real))
        plt.plot(x, filtered_signal.real, label='Filtered', color='orange')
        plt.legend()

        # Top frequencies
        plt.subplot(3, 1, 3)
        plt.title("Top Frequencies")
        plt.stem(top_freqs, top_ampls, 'r', markerfmt='r*')
        plt.xlabel('Frequency (Hz)')
        plt.ylabel('Amplitude')
        plt.tight_layout()
        plt.show()
    
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

