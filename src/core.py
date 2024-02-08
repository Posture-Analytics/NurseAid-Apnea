import cv2
import json
import os


import numpy as np

from tqdm import tqdm

import src.decoders as decoder

import numpy as np
import matplotlib.pyplot as plt

from PIL import Image


class ThermalProcessor:
    """
    A class used to process thermal data.

    Attributes:
        input_folder (str): The path to the folder containing input samples.
        output_folder (str): The path to the folder where results will be saved.
        filename (str): The name of the file being processed.
        samples (dict): A dictionary containing timestamps as keys and temperature arrays as values.
        min_temp (float): The minimum temperature value in the samples.
        max_temp (float): The maximum temperature value in the samples.
        normalized_min_temp (float): The minimum temperature value after normalization.
        normalized_max_temp (float): The maximum temperature value after normalization.
        mask (None or Mask): An optional mask object associated with the thermal data.
    """

    def __init__(self, input_folder: str = "./samples/", output_folder: str = "./results/") -> None:
        """
        Initializes a new instance of the ThermalProcessor class.

        Args:
            input_folder (str, optional): The path to the folder containing input samples. Defaults to "./samples/".
            output_folder (str, optional): The path to the folder where results will be saved. Defaults to "./results/".
        """
            
        self.filename = None
        self.input_folder = input_folder
        self.output_folder = output_folder
        os.makedirs(self.output_folder, exist_ok=True)

        self.samples = None
        self.min_temp = None
        self.max_temp = None

        self.normalized_min_temp = None
        self.normalized_max_temp = None

        self.mask = None

    def load_samples(self, filename: str = None) -> None:
        """
        Load samples from a JSON file.

        Args:
            filename (str, optional): The name of the file to load samples from. Defaults to None.
        """

        assert filename is not None or self.filename is not None

        input_filepath = self.input_folder + filename + ".json"

        # Read the json file with the samples into a dictionary
        with open(input_filepath, 'r') as f:
            samples = json.load(f)

        decoded_samples = {}

        # Initialize the min and max temperatures
        min_temp = np.inf
        max_temp = -np.inf

        # Iterate over each sample
        for timestamp, encoded_string in samples.items():

            # Decode the string
            decoded_values = decoder.decode_base64(encoded_string, 0, 100.0)

            # Convert the values into a 24x32 numpy array
            decoded_values = np.array(decoded_values).reshape(24, 32)

            # Rotate the array 180 degrees clockwise
            decoded_values = np.rot90(decoded_values, k=2, axes=(1, 0))

            # Update the min and max temperatures
            min_temp = min(min_temp, np.min(decoded_values))
            max_temp = max(max_temp, np.max(decoded_values))

            # Assign the decoded values to the dictionary
            decoded_samples[timestamp] = decoded_values

        self.samples = decoded_samples
        self.min_temp = min_temp
        self.max_temp = max_temp

        self.output_folder = self.output_folder + filename + "/"

    def normalize_temperatures(self, new_min_temp: float, new_max_temp: float) -> None:
        """
        Normalize temperature values to a new range.

        Args:
            new_min_temp (float): The minimum temperature value after normalization.
            new_max_temp (float): The maximum temperature value after normalization.
        """

        assert self.samples is not None

        self.normalized_min_temp = new_min_temp
        self.normalized_max_temp = new_max_temp

        normalized_samples = {}

        # Iterate over each sample
        for timestamp, sample in self.samples.items():

            # Clip the sample to the new range
            sample = np.clip(sample, new_min_temp, new_max_temp)

            # Normalize the values to the range [0, 255]
            sample = (sample - new_min_temp) * (255 / (new_max_temp - new_min_temp))

            # Assign the normalized sample to the dictionary
            normalized_samples[timestamp] = sample

        # Update the samples
        self.samples = normalized_samples

    def get_pseudoframes(self, frame: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """
        Generate pseudoframes from a given frame.

        Args:
            frame (np.ndarray): The frame to generate pseudoframes from.

        Returns:
            tuple[np.ndarray, np.ndarray]: A tuple containing two pseudoframes.
        """
        
        # Create the arrays for the pseudoframes
        pseudoframe_0 = np.zeros((24, 32))
        pseudoframe_1 = np.zeros((24, 32))

        # Iterate over the rows and collumns of the frame, filling the pseudoframes
        # Page 0: even rows and even columns
        # Page 1: even rows and odd columns
        for row in range(24):
            for col in range(32):
                if (row + col) % 2 == 0:
                    pseudoframe_0[row][col] = frame[row][col]
                else:
                    pseudoframe_1[row][col] = frame[row][col]

        # Interpolate the black pixels according to the surrounding pixels
        pseudoframe_0 = self.interpolate_black_pixels(pseudoframe_0)
        pseudoframe_1 = self.interpolate_black_pixels(pseudoframe_1)

        return self.pseudoframe_0, self.pseudoframe_1

    def interpolate_black_pixels(self, pseudoframe: np.ndarray) -> np.ndarray:
        """
        Interpolate black pixels in a pseudoframe.

        Args:
            pseudoframe (np.ndarray): The pseudoframe to interpolate.

        Returns:
            np.ndarray: The interpolated pseudoframe.
        """

        # Get the number of rows and columns of the pseudoframe
        rows, cols = pseudoframe.shape
        
        # Iterate over the rows and collumns of the pseudoframe
        for row in range(rows):
            for col in range(cols):
                if pseudoframe[row][col] == 0:

                    surrounding_pixels = []

                    # Get the surrounding pixels, if they exist
                    if row > 0:
                        surrounding_pixels.append(pseudoframe[row - 1][col])

                    if row < rows - 1:
                        surrounding_pixels.append(pseudoframe[row + 1][col])

                    if col > 0:
                        surrounding_pixels.append(pseudoframe[row][col - 1])

                    if col < cols - 1:
                        surrounding_pixels.append(pseudoframe[row][col + 1])

                    # If there are no surrounding pixels, skip
                    if len(surrounding_pixels) == 0:
                        continue
                    else:
                        # Calculate the mean of the surrounding pixels
                        mean = np.mean(surrounding_pixels)

                        # Assign the mean to the current pixel
                        pseudoframe[row][col] = mean

        return pseudoframe

    def get_mean_frame(self, window: dict = None) -> np.ndarray:
        """
        Get the mean frame from a window of samples.

        Args:
            window (dict, optional): The window of samples. Defaults to None.

        Returns:
            np.ndarray: The mean frame.
        """

        assert self.samples is not None

        if window:
            samples = window
        else:
            samples = self.samples

        # Get the mean frame
        mean_frame = np.mean(np.array(list(samples.values())), axis=0)

        return mean_frame
    
    def get_window_sum(self, window: dict = None) -> dict:
        """
        Calculate the sum of temperatures in a window.

        Args:
            window (dict, optional): The window of samples. Defaults to None.

        Returns:
            dict: A dictionary containing timestamps as keys and the sum of temperatures as values.
        """
            
        assert self.samples is not None
    
        if window:
            samples = window
        else:
            samples = self.samples

        # Get the sum of the window as a dict
        window_sum_dict = {}

        for timestamp, frame in samples.items():
            window_sum_dict[timestamp] = np.sum(frame)

        return window_sum_dict
    
    def get_data_duration(self) -> int:
        """
        Get the duration of the data.

        Returns:
            int: The duration of the data in milliseconds.
        """
            
        assert self.samples is not None

        # Get the timestamps
        timestamps = list(self.samples.keys())

        # Get the start and end timestamps
        start_timestamp = int(timestamps[0])
        end_timestamp = int(timestamps[-1])

        # Calculate the duration
        duration = end_timestamp - start_timestamp

        return duration
    
    def generate_images(self, generate_pseudoframes: bool = False, upscale_factor: int = 1) -> None:
        """
        Generate images from the samples.

        Args:
            generate_pseudoframes (bool, optional): Whether to generate pseudoframes. Defaults to False.
            upscale_factor (int, optional): The factor to upscale the images by. Defaults to 1.
        """

        output_image_directory = self.output_folder + "images/"

        # Ensure the directory for saving images exists
        os.makedirs(output_image_directory, exist_ok=True)

        # Ensure the directory for saving frames and pseudoframes exists
        os.makedirs(f"{output_image_directory}/frames", exist_ok=True)
        os.makedirs(f"{output_image_directory}/pseudoframes", exist_ok=True)

        # Initialize the progress bar
        pbar = tqdm(total=len(self.samples))

        # Iterate over each sample
        for timestamp, temperature_array in self.samples.items():

            # Update the progress bar
            pbar.update(1)

            if generate_pseudoframes:
                pseudoframe_0, pseudoframe_1 = self.get_pseudoframes()

            if upscale_factor > 1:
                # Upscale the image by repeating each pixel 4 times in both 
                temperature_array = np.repeat(np.repeat(temperature_array, upscale_factor, axis=0), upscale_factor, axis=1)

                if generate_pseudoframes:
                    pseudoframe_0 = np.repeat(np.repeat(pseudoframe_0, upscale_factor, axis=0), upscale_factor, axis=1)
                    pseudoframe_1 = np.repeat(np.repeat(pseudoframe_1, upscale_factor, axis=0), upscale_factor, axis=1)

            # Create and save the image
            image = Image.fromarray(temperature_array.astype(np.uint8), 'L')
            image.save(f"{output_image_directory}/frames/{timestamp}.png")

            if generate_pseudoframes:
                pseudoframe_0 = Image.fromarray(pseudoframe_0.astype(np.uint8), 'L')
                pseudoframe_1 = Image.fromarray(pseudoframe_1.astype(np.uint8), 'L')

                pseudoframe_0.save(f"{output_image_directory}/pseudoframes/{timestamp}_0.png")
                pseudoframe_1.save(f"{output_image_directory}/pseudoframes/{timestamp}_1.png")

        # Close the progress bar
        pbar.close()

    def copy(self) -> 'ThermalProcessor':
        """
        Create a copy of the ThermalProcessor object.

        Returns:
            ThermalProcessor: A copy of the ThermalProcessor object.
        """

        # Create a copy of the thermal data
        new_thermal_data = ThermalProcessor()

        new_thermal_data.filename = self.filename
        new_thermal_data.input_folder = self.input_folder
        new_thermal_data.output_folder = self.output_folder

        new_thermal_data.samples = self.samples.copy()
        new_thermal_data.min_temp = self.min_temp
        new_thermal_data.max_temp = self.max_temp

        new_thermal_data.normalized_min_temp = self.normalized_min_temp
        new_thermal_data.normalized_max_temp = self.normalized_max_temp

        new_thermal_data.mask = self.mask

        return new_thermal_data
    
    def apply_gaussian_filter(self, kernel_size: int = 1) -> None:
        """
        Apply a Gaussian filter to the samples.

        Args:
            kernel_size (int, optional): The size of the Gaussian kernel. Defaults to 1.
        """
            
        assert self.samples is not None

        # Initialize the dictionary for the filtered samples
        filtered_samples = {}

        # Iterate over each sample
        for timestamp, sample in self.samples.items():

            # Apply the gaussian filter
            filtered_sample = cv2.GaussianBlur(sample, (kernel_size, kernel_size), 0)

            # Assign the filtered sample to the dictionary
            filtered_samples[timestamp] = filtered_sample

        # Update the samples
        self.samples = filtered_samples

    # def save_samples_into_csv(self):
    #     # Save the (timestamp, normalized temperature) pairs into a csv file
    #     with open("samples.csv", 'w') as f: # TODO: fix this path
    #         for timestamp, sample in self.samples.items():
    #             f.write(f"{timestamp},{sample}\n")
