from src import core

from src import parameters
from src import window_processor

import matplotlib.pyplot as plt

thermal_data = core.ThermalProcessor()
thermal_data.load_samples("coleta_almir_apnea_29_01_24_deitado_1_0.3")

grid_search_parameters = parameters.generate_parameter_sets()
print(len(grid_search_parameters), "parameter sets generated.")

for pameter_set in grid_search_parameters:

    print("Running parameter set:")
    print(pameter_set)

    thermal_data_copy = thermal_data.copy()

    # Create a new window processor
    window_processor_unit = window_processor.WindowProcessor(thermal_data_copy, pameter_set)

    window_processor_unit.process()
    # Get the results
    window_results = window_processor_unit.window_results

    # break



