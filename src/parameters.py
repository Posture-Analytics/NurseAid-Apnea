from typing import List, Tuple, Union
from itertools import product

# Constant parameters
BANDPASS_LOW_CUT: List[float] = [0.1]
BANDPASS_HIGH_CUT: List[float] = [0.7]

# Variable parameters
# WINDOW_SIZE: List[int] = [5, 10, 15, 20, 25, 30]  # in seconds
WINDOW_SIZE: List[int] = [60]  # in seconds
# WINDOW_STEP: List[int] = [1, 3, 5, 7, 9, 11]  # in seconds
MINIMUM_TEMPERATURE: List[int] = [18, 20, 22, 24]
MAXIMUM_TEMPERATURE: List[int] = [30, 32, 34, 36]
GAUSSIAN_BLUR_KERNEL_SIZES: List[int] = [1, 3, 5]
MASK_TYPE: List[str] = ["otsu"]
MASK_LOW_ACCURACY_AREA: List[bool] = [True]
# MASK_COLUMNS: List[Tuple[int, int]] = [(4, 25)]
MASK_BORDERS: List[int] = [5]

# Conditional parameters
DILATE_KERNEL_SIZES: List[int] = [3, 5]  # if MASK_TYPE == "otsu"
# BOX_COORDINATES: List[Tuple[int, int, int, int]] = [(0, 0, 32, 24)]  # if MASK_TYPE == "box"


class ParameterSet:
    """
    A class used to represent a set of parameters.

    Attributes:
        parameters (dict): A dictionary storing parameter names and their values.
        results (dict or None): A dictionary storing results (if available).
    """

    def __init__(self) -> None:
        """
        Initializes a new instance of the ParameterSet class.
        """

        self.parameters = {}
        self.results = None

    def add_parameter(self, parameter_name: str, parameter_values: Union[int, float, str, bool, Tuple[int, int, int, int]]) -> None:
        """
        Add a parameter to the parameter set.

        Args:
            parameter_name (str): Name of the parameter.
            parameter_values (int, float, str, bool, Tuple[int, int, int, int]): Value(s) of the parameter.
        """

        self.parameters[parameter_name] = parameter_values

    def get_parameter(self, parameter_name: str) -> Union[None, int, float, str, bool, Tuple[int, int, int, int]]:
        """
        Get the value of a parameter from the parameter set.

        Args:
            parameter_name (str): Name of the parameter.

        Returns:
            int, float, str, bool, Tuple[int, int, int, int] or None: Value(s) of the parameter if found, None otherwise.
        """

        return self.parameters.get(parameter_name)
    
    def copy(self) -> 'ParameterSet':
        """
        Create a copy of the parameter set.

        Returns:
            ParameterSet: A copy of the parameter set.
        """

        new_parameter_set = ParameterSet()
        new_parameter_set.parameters = self.parameters.copy()

        # Check if the parameter set has results
        if self.results != None:
            new_parameter_set.results = self.results.copy()

        return new_parameter_set
    
    def __str__(self) -> str:
        """
        Return the string representation of the parameter set.

        Returns:
            str: String representation of the parameter set.
        """

        string = ""

        string += "PARAMETERS:\n"
        for parameter_name, parameter_value in self.parameters.items():
            string += f"   -{parameter_name}: {parameter_value}\n"

        if self.results != None:
            string += "RESULTS:\n"
            for result_name, result_value in self.results.items():
                string += f"   -{result_name}: {result_value}\n"

        string += "\n"

        return string
    
def generate_parameter_set_code(parameter_set: ParameterSet) -> ParameterSet:
    """
    Generate a unique ID for the parameter set.

    Args:
        parameter_set (ParameterSet): The parameter set for which the ID is generated.

    Returns:
        ParameterSet: The parameter set with the unique ID added.
    """

    # Generate a code for the parameter set
    code = ""
    for parameter_name, parameter_value in parameter_set.parameters.items():
        code += f"{parameter_name}={parameter_value}_"
    code = code[:-1]

    # Hash the code to generate a unique ID with a fixed length (20)
    parameter_set_id = hash(code) % (10 ** 20)

    # Add the ID to the parameter set
    parameter_set.add_parameter("id", parameter_set_id)

    return parameter_set

def generate_parameter_sets() -> List[ParameterSet]:
    """
    Generate a list of parameter sets based on defined parameters.

    Returns:
        List[ParameterSet]: A list of parameter sets.
    """

    parameter_sets = []

    # Generate the parameter sets
    # for window_size, window_step, min_temp, max_temp, gaussian_blur_kernel_size, mask_type, bandpass_low_cut, bandpass_high_cut, mask_low_accuracy \
    #     in product(WINDOW_SIZE, WINDOW_STEP, MINIMUM_TEMPERATURE, MAXIMUM_TEMPERATURE, GAUSSIAN_BLUR_KERNEL_SIZES, MASK_TYPE, BANDPASS_LOW_CUT, BANDPASS_HIGH_CUT, MASK_LOW_ACCURACY_AREA):

    # for window_size, min_temp, max_temp, gaussian_blur_kernel_size, mask_type, bandpass_low_cut, bandpass_high_cut, mask_low_accuracy, mask_columns \
    #     in product(WINDOW_SIZE, MINIMUM_TEMPERATURE, MAXIMUM_TEMPERATURE, GAUSSIAN_BLUR_KERNEL_SIZES, MASK_TYPE, BANDPASS_LOW_CUT, BANDPASS_HIGH_CUT, MASK_LOW_ACCURACY_AREA, MASK_COLUMNS):

    for window_size, min_temp, max_temp, gaussian_blur_kernel_size, mask_type, bandpass_low_cut, bandpass_high_cut, mask_low_accuracy, mask_borders \
        in product(WINDOW_SIZE, MINIMUM_TEMPERATURE, MAXIMUM_TEMPERATURE, GAUSSIAN_BLUR_KERNEL_SIZES, MASK_TYPE, BANDPASS_LOW_CUT, BANDPASS_HIGH_CUT, MASK_LOW_ACCURACY_AREA, MASK_BORDERS):

        base_parameter_set = ParameterSet()

        # Add the parameters to the parameter set
        base_parameter_set.add_parameter("window_size", window_size)
        # base_parameter_set.add_parameter("window_step", window_step)
        base_parameter_set.add_parameter("window_step", window_size)
        base_parameter_set.add_parameter("min_temp", min_temp)
        base_parameter_set.add_parameter("max_temp", max_temp)
        base_parameter_set.add_parameter("gaussian_blur_kernel_size", gaussian_blur_kernel_size)
        base_parameter_set.add_parameter("mask_type", mask_type)
        base_parameter_set.add_parameter("bandpass_low_cut", bandpass_low_cut)
        base_parameter_set.add_parameter("bandpass_high_cut", bandpass_high_cut)
        base_parameter_set.add_parameter("mask_low_accuracy", mask_low_accuracy)
        base_parameter_set.add_parameter("mask_borders", mask_borders)

        complete_parameter_sets = []

        # Add the conditional parameters to the parameter set
        if (mask_type == "otsu"):
            for dilate_kernel_size in DILATE_KERNEL_SIZES:
                # Create a copy of the parameter set
                new_parameter_set = base_parameter_set.copy()
                # Add the conditional parameter to the parameter set
                new_parameter_set.add_parameter("dilate_kernel_size", dilate_kernel_size)

                # Append the parameter set to the list
                complete_parameter_sets.append(new_parameter_set)

        # elif (mask_type == "box"):
        #     for box_coordinates in BOX_COORDINATES:
        #         # Create a copy of the parameter set
        #         new_parameter_set = base_parameter_set.copy()
        #         # Add the conditional parameter to the parameter set
        #         new_parameter_set.add_parameter("box_coordinates", box_coordinates)

        #         # Append the parameter set to the list
        #         complete_parameter_sets.append(new_parameter_set)

        for parameter_set in complete_parameter_sets:
            # Generate a code for the parameter set
            parameter_set = generate_parameter_set_code(parameter_set)

            # Add the parameter set to the list
            parameter_sets.append(parameter_set)

    return parameter_sets