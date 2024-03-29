import cv2
import json
import os

import numpy as np

from PIL import Image

from src import masks
from src import frequency_processor

import matplotlib.pyplot as plt


class WindowProcessor:
    """
    A class used to process windows of thermal data.

    Attributes:
        thermal_data (ThermalData): An instance of the ThermalData class containing thermal data.
        window_parameter_set (WindowParameterSet): An instance of the WindowParameterSet class containing window parameters.
        window_size (int): The size of the window in seconds.
        window_step (int): The step size between windows in seconds.
        window_start_index (int): The start index of the current window.
        window_end_index (int): The end index of the current window.
        data_duration (float): The duration of the thermal data in seconds.
        window_indexes (list): A list of tuples containing the start and end indices of each window.
        window_results (list): A list of results obtained from processing each window.
    """

    def __init__(self, thermal_data, window_parameter_set, show_plots=True):
        """
        Initializes a new instance of the WindowProcessor class.

        Args:
            thermal_data (ThermalData): An instance of the ThermalData class containing thermal data.
            window_parameter_set (WindowParameterSet): An instance of the WindowParameterSet class containing window parameters.
        """

        self.thermal_data = thermal_data
        self.window_parameter_set = window_parameter_set
        self.show_plots = show_plots

        self.window_size = self.window_parameter_set.get_parameter("window_size")
        self.window_step = self.window_parameter_set.get_parameter("window_step")

        self.window_start_index = None
        self.window_end_index = None

        self.data_duration = thermal_data.get_data_duration()

        self.window_indexes = None

        self.window_results = None

        self.window_sum = None
        self.window_mean = None

        self.evaluate_windows()

    def evaluate_windows(self) -> None:
        """
        Evaluate the start and end indices of each window, and generate a list of tuples containing the start and end indices.
        """

        assert self.thermal_data is not None
        assert self.window_size <= self.data_duration

        # Get the timestamps
        timestamps = list(self.thermal_data.samples.keys())
        timestamps = [int(timestamp) for timestamp in timestamps]

        # Get the start index of the first window as the first timestamp
        window_start_index = 0
        window_start_timestamp = timestamps[window_start_index]

        # Estimate the end timestamp of the first window
        window_end_timestamp = window_start_timestamp + (self.window_size * 1000)

        # Find the index of the end timestamp of the first window (the biggest timestamp before the end timestamp of the first window)
        window_end_index = timestamps.index(max([timestamp for timestamp in timestamps if timestamp < window_end_timestamp]))

        # Update the start and end indices of the first window
        self.window_start_index = window_start_index
        self.window_end_index = window_end_index

        # Initialize the list of window indexes
        self.window_indexes = [(window_start_index, window_end_index)]

        # Slide the window, according to the window step, until the end of the data
        while (window_end_index + self.window_step) < len(timestamps) - 1:

            # Update the indexes of the next window
            window_start_index += self.window_step
            window_end_index += self.window_step

            # Append the indexes of the next window to the list
            self.window_indexes.append((window_start_index, window_end_index))

    def get_window(self, window_index: int) -> dict:
        """
        Get a window of thermal data based on the window index.

        Args:
            window_index (int): The index of the window.

        Returns:
            dict: A dictionary containing timestamps as keys and frames as values for the window.
        """

        assert self.window_indexes is not None

        # Get the start and end indexes of the window
        window_start_index, window_end_index = self.window_indexes[window_index]

        # Get the timestamps
        timestamps = list(self.thermal_data.samples.keys())

        # Get the window
        window = {}

        for index in range(window_start_index, window_end_index + 1):
            # Get the timestamp
            timestamp = timestamps[index]
            # Get the frame
            frame = self.thermal_data.samples[timestamp]
            # Add the frame to the window
            window[timestamp] = frame

        return window

    def next_window(self) -> None:
        """
        Update the start and end indices of the next window.
        """

        # Update the start and end indices of the next window
        self.window_start_index += self.window_step
        self.window_end_index += self.window_step

    def process(self) -> None:
        """
        Process all windows and save the results to a JSON file.
        """
            
        assert self.window_indexes is not None

        # Initialize the list of window results
        self.window_results = []

        self.pre_process_thermal_data()

        # Iterate over each window
        for window_index in range(len(self.window_indexes)):

            # Get the window
            window = self.get_window(window_index)

            # Process the window
            window_results = self.process_window(window)

            # Append the window results to the list
            self.window_results.append(window_results)

        # TODO: improve logging system
        print("Window processing finished.")

        # Dump all the results to a json file
        with open(f"results/{self.window_parameter_set.get_parameter('id')}.json", "w") as file:
            json.dump(self.window_results, file)

    def pre_process_thermal_data(self) -> None:
        """
        Apply pre-processing steps to the data that will be used for all windows.
        """

        # Normalize the temperature values into the range (0, 255)
        self.thermal_data.normalize_temperatures(self.window_parameter_set.get_parameter("min_temp"), 
                                                           self.window_parameter_set.get_parameter("max_temp"))

        # Apply a gaussian blur filter to the images
        self.thermal_data.apply_gaussian_filter(self.window_parameter_set.get_parameter("gaussian_blur_kernel_size"))

    def process_window(self, window: dict, debug: bool = True) -> None:
        """
        Process a window of thermal data.

        Args:
            window (dict): The window of thermal data to be processed.
            debug (bool): Whether to enable debugging. Default is True.
        """
        
        # Get the mean frame of the window
        mean_frame = self.thermal_data.get_mean_frame(window)

        # Generate the mask
        mask_generator = masks.FrameMask()

        if self.window_parameter_set.get_parameter("mask_type") == "box":
            mask_generator.generate_box_mask(box_coordinates=self.window_parameter_set.get_parameter("box_coordinates"))
        elif self.window_parameter_set.get_parameter("mask_type") == "otsu":
            mask_generator.generate_otsu_mask(mean_frame, 
                                    dilate_kernel_size=self.window_parameter_set.get_parameter("dilate_kernel_size"))
        else:
            raise Exception("Invalid mask type")
        
        # Mask the low accuracy area
        if self.window_parameter_set.get_parameter("mask_low_accuracy"):
            mask_generator.generate_low_accuracy_mask()

        # Mask the columns to isolate the torax and abdomen
            
        # if columns_to_mask := self.window_parameter_set.get_parameter("mask_columns"):
        #     mask_box_coordinates = (columns_to_mask[0], 0, columns_to_mask[1], 24)
        #     mask_generator.generate_box_mask(mask_box_coordinates)
            
        if border_to_mask := self.window_parameter_set.get_parameter("mask_borders"):
            mask_generator.generate_box_mask((border_to_mask, border_to_mask, 31-border_to_mask, 23-border_to_mask))

        # Plot the mask and the masked mean frame side by side
        if debug and self.window_parameter_set.get_parameter("mask_type") == "otsu" and self.show_plots:
            mask_generator.plot_mask(mean_frame)

        # Apply the mask to the images
        window = mask_generator.apply_mask_over_window(window)
        
        # TODO: Generate images

        # Sum pixels (TODO: plot the sum of pixels)
        self.window_sum = self.thermal_data.get_window_sum(window)

        self.window_mean = self.window_sum.copy()
        # Divide each values of the window_mean dict by the number of white pixels on the mask
        for key in self.window_mean.keys():
            self.window_mean[key] = self.window_mean[key] / mask_generator.num_white_pixels

        # DEBUG
        # # Save the timestamps and the sum of pixels into a csv file
        # with open("samples.csv", 'w') as f: # TODO: fix this path
        #     for key in self.window_sum.keys():
        #         f.write("%s,%s\n"%(key,self.window_sum[key]))
                

        # Get frequency spectrum (optional: plot the frequency spectrum)
        frequency_processor_unit = frequency_processor.FrequencyProcessor(self.window_sum, self.show_plots)
        frequency_processor_unit.process_signal()

        # Get the spectrum as a result
        spectrum = frequency_processor_unit.spectrum

        return spectrum

# TODO: find a place to plot the pixels amplitude
    
    def merge_sliding_window_results(self, respiratory_freq, only_first_frequency: bool = True) -> None:
        """
        Merge the results of the windows into statistics
        """

        # Evaluate the RMSE, MAE and R2 of the frequency peaks, compared to the respiratory frequency
        rmse = 0
        mae = 0
        r2 = 0

        for window_result in self.window_results:
            # Get the frequency peaks
            frequency_peaks = [peak[0] for peak in window_result]

            # Get the first frequency peak
            if only_first_frequency:
                frequency_peaks = frequency_peaks[0]

            # Get the RMSE of the frequency peaks
            rmse += np.sqrt(np.mean((np.array(frequency_peaks) - respiratory_freq) ** 2))
            # Get the MAE of the frequency peaks
            mae += np.mean(np.abs(np.array(frequency_peaks) - respiratory_freq))
            # Get the R2 of the frequency peaks
            r2 += 1 - (np.sum((np.array(frequency_peaks) - respiratory_freq) ** 2) / np.sum((np.array(frequency_peaks) - np.mean(frequency_peaks)) ** 2))
        
        # Get the mean RMSE
        self.rmse = rmse / len(self.window_results)
        # Get the mean MAE
        self.mae = mae / len(self.window_results)
        self.r2 = r2 / len(self.window_results)

    def save_window_sum(self, file_name: str) -> None:
        """
        Save the sum of pixels of each window into a csv file.

        Args:
            file_name (str): The name of the file.
        """

        # Check if the window sum is not None
        assert self.window_sum is not None

        # Checj if the results folder exists
        os.makedirs("results", exist_ok=True)

        # Save the timestamps and the sum of pixels into a csv file
        with open(f"./results/{file_name}_sum.csv", 'w') as f:
            for key in self.window_sum.keys():
                f.write("%s,%s\n"%(key,self.window_sum[key]))

        # Convert the dictionary keys into integers
        self.window_sum = {int(k): v for k, v in self.window_sum.items()}

        # Clear any previous plot
        plt.clf()

        # Save a plot showing the sum of pixels
        plt.plot(list(self.window_sum.keys()), list(self.window_sum.values()))
        plt.title(f"Sum of pixels ({file_name})")
        plt.savefig(f"./results/{file_name}_sum.png")
        if self.show_plots:
            plt.show()

    def save_window_mean(self, file_name: str) -> None:
        """
        Save the mean of pixels of each window into a csv file.

        Args:
            file_name (str): The name of the file.
        """

        # Check if the window mean is not None
        assert self.window_mean is not None

        # Checj if the results folder exists
        os.makedirs("results", exist_ok=True)

        # Save the timestamps and the mean of pixels into a csv file
        with open(f"./results/{file_name}_mean.csv", 'w') as f:
            for key in self.window_mean.keys():
                f.write("%s,%s\n"%(key,self.window_mean[key]))

        # Convert the dictionary keys into integers
        self.window_mean = {int(k): v for k, v in self.window_mean.items()}

        # Clear any previous plot
        plt.clf()

        # Save a plot showing the mean of pixels
        plt.plot(list(self.window_mean.keys()), list(self.window_mean.values()))
        plt.title(f"Mean of pixels ({file_name})")
        plt.savefig(f"./results/{file_name}_mean.png")
        if self.show_plots:
            plt.show()

    # Iterate over the frames and store the maximum amplitude of each pixel
    def get_window_amplitude(self, window: dict) -> None:
        """
        Get the amplitude of each pixel in the window.

        Args:
            window (dict): The window of thermal data.
        """

        # Initialize the minimum and maximum temperatures array
        min_frame = np.ones_like(list(window.values())[0]) * np.inf
        max_frame = np.zeros_like(list(window.values())[0])

        # Get the mimimum temperature for each pixel over the window
        for timestamp, frame in window.items():
            min_frame = np.minimum(min_frame, frame)
            max_frame = np.maximum(max_frame, frame)

        # Get the amplitude of each pixel
        amplitude = max_frame - min_frame

        return amplitude

    def save_window_amplitude(self, file_name: str, supress_low_accuracy_areas: bool = False) -> None:
        """
        Save the amplitude of each pixel of each window into a csv file.

        Args:
            file_name (str): The name of the file.
        """

        # Checj if the results folder exists
        os.makedirs("results", exist_ok=True)

        # Get the window
        window = self.get_window(0)

        # Get the amplitude of each pixel in the window
        amplitude = self.get_window_amplitude(window)
        self.save_simple_images(amplitude, file_name + "_amplitude_raw")

        # Supress low accuracy areas
        if supress_low_accuracy_areas:
            mask_generator = masks.FrameMask()
            mask_generator.generate_low_accuracy_mask()
            
            # Apply the mask to the images
            amplitude = mask_generator.apply_mask(amplitude)
            
        self.save_simple_images(amplitude, file_name + "_amplitude_no_corners", mask="corners")

        # Supress borders
        mask_generator = masks.FrameMask()
        mask_generator.generate_box_mask((5, 5, 27, 19))
        amplitude = mask_generator.apply_mask(amplitude)

        self.save_simple_images(amplitude, file_name + "_amplitude_no_borders", mask="borders")

        # Plot and save the amplitude of each pixel
        plt.imshow(amplitude, cmap="gray")
        plt.colorbar()
        plt.title(f"Amplitude of each pixel ({file_name})")
        plt.savefig(f"./results/{file_name}_amplitude.png")
        if self.show_plots:
            plt.show()

    def save_mean_frame(self, file_name: str) -> None:
        """
        Save the mean frame of each window into a csv file.

        Args:
            file_name (str): The name of the file.
        """

        # Checj if the results folder exists
        os.makedirs("results", exist_ok=True)

        # Get the window
        window = self.get_window(0)

        # Get the mean frame of the window
        mean_frame = self.thermal_data.get_mean_frame(window)

        self.save_simple_images(mean_frame, file_name + "_mean_frame", normalize=False)

    def save_simple_images(self, frame, name, upscale_factor=10, mask=None, normalize=True):
        """
        Save the frame as a simple image.

        Args:
            frame (np.array): The frame to be saved.
            upscale_factor (int): The upscale factor. Default is 10.
        """

        
        # Converting the list of temperatures into a 32x24 array
        temperature_array = np.array(frame).reshape(24, 32)

        if normalize:

            min_temp = temperature_array.min()
            max_temp = temperature_array.max()

            if min_temp == 0:
                # Catch the next minimum value
                min_temp = temperature_array[temperature_array != 0].min()

            # Normalize the temperature values into the range (0, 255)
            temperature_array = (temperature_array - min_temp) / (max_temp - min_temp) * 255

        if mask:

            # Apply the mask to the images
            if mask == "corners":
                mask_generator = masks.FrameMask()
                mask_generator.generate_low_accuracy_mask()
                temperature_array = mask_generator.apply_mask(temperature_array)

            if mask == "borders":
                mask_generator = masks.FrameMask()
                mask_generator.generate_box_mask((5, 5, 27, 19))
                temperature_array = mask_generator.apply_mask(temperature_array)

        # Upscale the frame
        temperature_array = np.repeat(np.repeat(temperature_array, upscale_factor, axis=0), upscale_factor, axis=1)

        print(temperature_array.shape)

        # Create and save the image
        image = Image.fromarray(temperature_array.astype(np.uint8), 'L')  # 'L' mode is for grayscale
        image.save(f"./results/{name}.png")

    def rotate_and_mirror_samples(self) -> None:
        """
        Rotate and mirror the samples to increase the number of samples.
        """

        # Rotate 180 degrees and mirror the samples
        self.thermal_data.rotate_and_mirror_samples()