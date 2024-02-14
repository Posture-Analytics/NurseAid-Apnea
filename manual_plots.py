from src import core

from src import parameters
from src import window_processor

import matplotlib.pyplot as plt

# Disable globally the matplotlib show function
plt.ioff()

sample_files = ["02_13_lucas_back_02hz",
                "02_13_lucas_back_04hz",
                "02_13_lucas_back_05hz",
                "02_13_lucas_left_02hz",
                "02_13_lucas_left_04hz",
                "02_13_lucas_left_05hz"]

window_sizes = [5, 10, 15, 20, 25, 30]

for window_size in window_sizes:

    for file in sample_files:

        thermal_data = core.ThermalProcessor()
        thermal_data.load_samples(file)

        # Add a sufix on the file name to indicate the window size
        file = file + "_" + str(window_size) + "sec"

        # Generate a custom parameter set
        parameter_set = parameters.ParameterSet()

        # Evaluate the maximum window size by taking the number of samples and dividing by 8
        # parameter_set.add_parameter("window_size", int(len(thermal_data.samples) / 8))
        parameter_set.add_parameter("window_size", window_size)

        parameter_set.add_parameter("window_step", 10000) # Very high value to evaluate just one window
        parameter_set.add_parameter("min_temp", 21)
        parameter_set.add_parameter("max_temp", 37)
        parameter_set.add_parameter("gaussian_blur_kernel_size", 3)
        parameter_set.add_parameter("mask_type", "otsu")
        parameter_set.add_parameter("dilate_kernel_size", 5)
        parameter_set.add_parameter("bandpass_low_cut", 0.0)
        parameter_set.add_parameter("bandpass_high_cut", 1.0)
        parameter_set.add_parameter("mask_low_accuracy", True)

        # Process the window
        window_processor_unit = window_processor.WindowProcessor(thermal_data, parameter_set, show_plots=False)

        print("Processing window {}...".format(file))
        print("Number of windows: ", len(window_processor_unit.window_indexes))

        window_processor_unit.process()

        # Save the widnow_sum into a csv file
        window_processor_unit.save_window_sum(file)

        # Save the window_mean into a csv file
        window_processor_unit.save_window_mean(file)

        # Save the window_amplitude
        window_processor_unit.save_window_amplitude(file, supress_low_accuracy_areas=True)

        print("Window {} processed.".format(file), "\n\n")