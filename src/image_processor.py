import cv2
import os

import numpy as np
import matplotlib.pyplot as plt

from PIL import Image
from scipy.fft import fft, fftfreq
from tqdm import tqdm

from typing import Optional, Union

def plot_frame(image: np.ndarray, mask: Optional[np.ndarray] = None, show: bool = True, title: Optional[str] = None,
               save_path: Optional[str] = None) -> None:
    """
    Plot the image with an optional mask overlay.

    Args:
        image (np.ndarray): The image to be plotted.
        mask (Optional[np.ndarray]): The mask to be applied to the image. Default is None.
        show (bool): Whether to show the plot. Default is True.
        title (Optional[str]): The title of the plot. Default is None.
        save_path (Optional[str]): The path to save the plot. Default is None.
    """

    if isinstance(mask, np.ndarray):
        # Apply the mask to the image
        image = cv2.bitwise_and(image, image, mask=mask)

    # Plot the image
    plt.imshow(image, cmap='gray')

    # Set the title
    if title is not None:
        plt.title(title)

    # Save the plot
    if save_path is not None:
        plt.savefig(save_path)

    # Show the plot
    plt.show()

def plot_line_plot(x: Union[list, np.ndarray], y: Union[list, np.ndarray], show: bool = True,
                   title: Optional[str] = None, save_path: Optional[str] = None) -> None:
    """
    Plot a line plot.

    Args:
        x (Union[list, np.ndarray]): The x-axis values.
        y (Union[list, np.ndarray]): The y-axis values.
        show (bool): Whether to show the plot. Default is True.
        title (Optional[str]): The title of the plot. Default is None.
        save_path (Optional[str]): The path to save the plot. Default is None.
    """

    # Plot the line plot
    plt.plot(x, y)

    # Set the title
    if title is not None:
        plt.title(title)

    # Save the plot
    if save_path is not None:
        plt.savefig(save_path)

    # Show the plot
    plt.show()

def plot_pixel_sum(pixel_sums: dict, show: bool = True, title: Optional[str] = None,
                    save_path: Optional[str] = None) -> None:
    """
    Plot pixel sums over time.

    Args:
        pixel_sums (dict): A dictionary containing timestamps as keys and pixel sums as values.
        show (bool): Whether to show the plot. Default is True.
        title (Optional[str]): The title of the plot. Default is None.
        save_path (Optional[str]): The path to save the plot. Default is None.
    """

    # Preprocess the pixel sums
    timestamps = list(pixel_sums.keys())
    timestamps = [int(timestamp) for timestamp in timestamps]
    sums = list(pixel_sums.values())

    # Set the start time (t0) as the first timestamp
    t0 = timestamps[0]
    
    # Convert all timestamps to milliseconds since t0
    for i in range(len(timestamps)):
        timestamps[i] = timestamps[i] - t0

    # Plot the pixel sums
    plot_line_plot(timestamps, sums, show, title, save_path)
