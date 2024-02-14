from src import core

from src import parameters
from src import window_processor

import matplotlib.pyplot as plt

sample_files = ["02_13_lucas_back_02hz",
                "02_13_lucas_back_04hz",
                "02_13_lucas_back_05hz",
                "02_13_lucas_left_02hz",
                "02_13_lucas_left_04hz",
                "02_13_lucas_left_05hz"]

for file in sample_files:

    thermal_data = core.ThermalProcessor()
    thermal_data.load_samples(file)

    # Generate a custom parameter set
    parameter_set = parameters.ParameterSet()

    # Evaluate the maximum window size by taking the number of samples and dividing by 8
    parameter_set.add_parameter("window_size", int(len(thermal_data.samples) / 8))

    parameter_set.add_parameter("window_step", 10000) # Very high value to avoid overlapping windows
    parameter_set.add_parameter("min_temp", 19)
    parameter_set.add_parameter("max_temp", 35)
    parameter_set.add_parameter("gaussian_blur_kernel_size", 3)
    parameter_set.add_parameter("mask_type", "otsu")
    parameter_set.add_parameter("dilate_kernel_size", 5)
    parameter_set.add_parameter("bandpass_low_cut", 0.0)
    parameter_set.add_parameter("bandpass_high_cut", 1.0)
    parameter_set.add_parameter("mask_low_accuracy", True)

    # Process the window
    window_processor_unit = window_processor.WindowProcessor(thermal_data, parameter_set)

    print("Processing window {}...".format(file))
    print("Number of windows: ", len(window_processor_unit.window_indexes))

    window_processor_unit.process()

    # Save the widnow_sum into a csv file
    window_processor_unit.save_window_sum(file)

    # Save the window_amplitude
    window_processor_unit.save_window_amplitude(file)