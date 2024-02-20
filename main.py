from src import core

from src import parameters
from src import window_processor

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

import matplotlib.pyplot as plt
import pandas as pd

from tqdm import tqdm

filenames = [#"01_24_almir_back_03hz",
             #"02_09_almir_back_03hz",
             #"02_09_almir_left_03hz",
             #"02_09_lucas_back_03hz",
             #"02_09_lucas_left_03hz",
             "02_13_lucas_back_02hz",
             #"02_13_lucas_back_04hz",
             #"02_13_lucas_back_05hz",
             "02_13_lucas_left_02hz",
             #"02_13_lucas_left_04hz",
             #"02_13_lucas_left_05hz"
             ]

grid_search_parameters = parameters.generate_parameter_sets()
print(len(grid_search_parameters), "parameter sets generated.")

pbar = tqdm(total=len(grid_search_parameters)*len(filenames))

for filename in filenames:

    thermal_data = core.ThermalProcessor()
    thermal_data.load_samples(filename)

    # Get the list of parameters names
    parameter_names = list(grid_search_parameters[0].parameters.keys())

    columns = parameter_names + ["rmse", "mae", "r2", "true_respiratory_frequency", "author", "position", "date"]

    grid_search_results = pd.DataFrame(columns=columns)

    for parameter_set in grid_search_parameters:

        ##print("Running parameter set:")
        ##print(parameter_set)

        thermal_data_copy = thermal_data.copy()

        # Create a new window processor
        window_processor_unit = window_processor.WindowProcessor(thermal_data_copy, parameter_set, show_plots=False)

        window_processor_unit.process()

        # Get the results
        window_results = window_processor_unit.window_results

        # Extract the original respiratory frequency from the filename
        respiratory_frequency = int(filename.split("_")[-1].split("hz")[0])/10.0
        ##print("True respiratory frequency: ", respiratory_frequency)

        window_processor_unit.merge_sliding_window_results(respiratory_frequency)

        ##print("RMSE: ", window_processor_unit.rmse)
        ##print("MAE: ", window_processor_unit.mae)
        ##print("R2: ", window_processor_unit.r2)

        # Store the results into a dictionary
        results_dict = {}

        for parameter_name in parameter_names:
            results_dict[parameter_name] = parameter_set.get_parameter(parameter_name)
        
        results_dict["rmse"] = window_processor_unit.rmse
        results_dict["mae"] = window_processor_unit.mae
        results_dict["r2"] = window_processor_unit.r2
        results_dict["true_respiratory_frequency"] = respiratory_frequency

        results_dict["author"] = filename.split("_")[2]
        results_dict["position"] = filename.split("_")[3]
        results_dict["date"] = filename.split("_")[0]

        # Convert the dictionary into a pandas dataframe
        results_df = pd.DataFrame([results_dict])

        # Concatenate the results dataframe with the grid search results dataframe
        grid_search_results = pd.concat([grid_search_results, results_df], ignore_index=True)

        pbar.update(1)
        # break

pbar.close()

##print(grid_search_results)

# Save the results to a CSV file
grid_search_results.to_csv("grid_search_results.csv", index=False)

# Save the serialized results to a pickle file
grid_search_results.to_pickle("grid_search_results.pkl")


