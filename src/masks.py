import cv2

import numpy as np

import src.image_processor as image_processor

from typing import Tuple

class FrameMask:
    """
    A class used to represent a frame mask.

    Attributes:
        mask (np.ndarray): The mask generated.
        num_white_pixels (int): The count of white pixels in the mask.
    """

    def __init__(self) -> None:
        """
        Initializes a new instance of the FrameMask class.
        """

        self.mask = np.ones((24, 32), dtype=np.uint8)
        self.num_white_pixels = 0

    def generate_box_mask(self, box_coordinates: Tuple[int, int, int, int]) -> None:
        """
        Generate a mask from the box coordinates.

        Args:
            box_coordinates (Tuple[int, int, int, int]): The box coordinates.
        """

        # Get the box coordinates
        x1, y1, x2, y2 = box_coordinates

        # Create an empty mask
        mask = np.zeros((24, 32), dtype=np.uint8)

        # Set the box region to white
        mask[y1:y2, x1:x2] = 255

        # Update the mask (compose with the last mask)
        self.mask = self.apply_mask(mask)

        # Update the number of white pixels
        self.update_white_pixels_count()

    def generate_otsu_mask(self, frame: np.ndarray, gaussian_blur_kernel_size: int = 5, dilate_kernel_size: int = 5) -> None:
        """
        Generate a mask using Otsu's thresholding.

        Args:
            frame (np.ndarray): The frame to be masked.
            gaussian_blur_kernel_size (int): The size of the Gaussian blur kernel. Default is 5.
            dilate_kernel_size (int): The size of the dilation kernel. Default is 5.
        """

        # Blur the frame to reduce noise
        blurred_frame = cv2.GaussianBlur(frame, (gaussian_blur_kernel_size, gaussian_blur_kernel_size), 0)

        # Convert the frame to uint8
        blurred_frame = blurred_frame.astype(np.uint8)

        # Apply Otsu's thresholding
        _, mask = cv2.threshold(blurred_frame, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        # Invert the mask
        mask = cv2.bitwise_not(mask)

        # Get the edges of the mask
        mask = cv2.Canny(mask, 100, 200)

        # Close the edges
        kernel = np.ones((3, 3), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

        # Dilate the mask
        kernel = np.ones((dilate_kernel_size, dilate_kernel_size), np.uint8)
        mask = cv2.dilate(mask, kernel, iterations=1)

        # Update the mask (compose with the last mask)
        self.mask = self.apply_mask(mask)

        # Update the number of white pixels
        self.update_white_pixels_count()
    
    def generate_low_accuracy_mask(self, side_length: int = 6) -> None:
        """
        Generate a mask to remove the corners of the frame, presented as low accuracy regions.

        Args:
            side_length (int): The side length of the triangles. Default is 6.
        """

        # Create an empty mask
        mask = np.ones((24, 32), dtype=np.uint8) * 255

        # Set the corners to black
        # Top left corner
        for i in range(side_length):
            for j in range(side_length - i):
                mask[j, i] = 0

        # Bottom left corner
        for i in range(side_length):
            for j in range(side_length - i):
                mask[-j-1, i] = 0

        # Top right corner
        for i in range(side_length):
            for j in range(side_length - i):
                mask[i, -j-1] = 0    

        # Bottom right corner
        for i in range(side_length):
            for j in range(side_length - i):
                mask[-i-1, -j-1] = 0    

        # Update the mask (compose with the last mask)
        self.mask = self.apply_mask(mask)

        # Update the number of white pixels
        self.update_white_pixels_count()

    def update_white_pixels_count(self) -> None:
        """Update the count of white pixels in the mask."""
        self.num_white_pixels = np.sum(self.mask == 255)

    def apply_mask(self, frame: np.ndarray) -> np.ndarray:
        """
        Apply the mask to the frame.

        Args:
            frame (np.ndarray): The frame to be masked.

        Returns:
            np.ndarray: The masked frame.
        """

        masked_frame = cv2.bitwise_and(frame, frame, mask=self.mask)

        return masked_frame
    
    def apply_mask_over_window(self, window: dict) -> dict:
        """
        Apply the mask to a window of frames.

        Args:
            window (dict): The window of frames.

        Returns:
            dict: The masked window.
        """

        masked_window = {}
        for timestamp, frame in window.items():
            masked_window[timestamp] = self.apply_mask(frame)

        return masked_window
    
    def plot_mask(self, frame: np.ndarray) -> None:
        """
        Plot the mask and the masked frame side by side.

        Args:
            frame (np.ndarray): The frame.
        """
        
        # Apply the mask to the frame
        masked_frame = self.apply_mask(frame)

        # Plot the mask and the masked frame side by side
        image_processor.plot_frame(np.hstack((self.mask, frame, masked_frame)), title="Mask and masked frame")

        # TODO: add a save path