# This script is used to extract timing data from synthesis reports and save it in a JSON format.
import os
import re
import json

# Constants for parsing the report that specify which line 
# contains the delay information.
# This pattern is specific to the Vivado synthesis report format.
# If a different synthesis tool is used, you might have to define a new pattern.
PATTERN_DELAY_INFO = "Data Path Delay:"

# This function extracts the delay from a line in the report file.
# It uses a regular expression to find the delay value in nanoseconds.
# It is specific to the Vivado synthesis report format.
# If a different synthesis tool is used, this function may need to be modified.
def extract_delay(line):
    """
    Extract the delay from a line in the report.
    
    Args:
        line (str): A line from the report file.
        
    Returns:
        float: The extracted delay in nanoseconds.
    """
    match = re.search(r'Data Path Delay:\s+([\d.]+)ns', line)
    assert match, f"Could not find data path delay in line: {line}"
    return float(match.group(1))

def extract_single_rpt(rpt_file):
    """
    Extract data from the report file.
    
    Args:
        rpt_file (str): Path to the report file.
        
    Returns:
        delay (float): The extracted delay in nanoseconds.
    """    
    max_delay = 0.0
    # Read the report file and extract the required data
    with open(rpt_file, 'r') as f:
        for line in f:
            # Extract delay of the data path
            if PATTERN_DELAY_INFO in line:
                delay = extract_delay(line)
                max_delay = max(max_delay, delay)
                
    return max_delay  # Return 0.0 if no delay is found

def extract_rpt_data(map_unit_to_list_unit_chars, json_output):
    """
    Extract the data from the map_unit_to_list_unit_chars dictionary and save it to a JSON file.
    IMPORTANT: For now we assume that only DATA_TYPE is the only parameter that can be used to characterize the unit.
    
    Args:
        map_unit_to_list_unit_chars (dict): Dictionary mapping unit names to a list of UnitCharacterization objects.
        json_output (str): Path to the output JSON file.
    """
    # Create the output data structure
    output_data = {}
    for unit_name, list_unit_chars in map_unit_to_list_unit_chars.items():
        dataDict = {}
        validDict = {"1": 0.0}
        readyDict = {"1": 0.0}
        VRDelayFinal = 0.0
        CVDelayFinal = 0.0
        CRDelayFinal = 0.0
        VCDelayFinal = 0.0
        VDDelayFinal = 0.0
        traversedUnitOnce = False
        for unit_char in list_unit_chars:
            traversedParamOnce = False
            for delay_type, rpt_filename in unit_char.get_signals_type_to_rpt().items():
                # Check if the report file exists
                if not os.path.exists(rpt_filename):
                    continue
                traversedParamOnce = True
                traversedUnitOnce = True
                # Extract delay from the report file
                delay = extract_single_rpt(rpt_filename)
                if delay_type == ("data", "data"):
                    dataDict[str(unit_char.get_parameter_value("DATA_TYPE"))] = delay
                elif delay_type == ("valid", "valid"):
                    validDict["1"] = max(validDict["1"], delay)
                elif delay_type == ("ready", "ready"):
                    readyDict["1"] = max(readyDict["1"], delay)
                elif delay_type == ("valid", "ready"):
                    VRDelayFinal = max(VRDelayFinal, delay)
                elif delay_type == ("control", "valid"):
                    CVDelayFinal = max(CVDelayFinal, delay)
                elif delay_type == ("control", "ready"):
                    CRDelayFinal = max(CRDelayFinal, delay)
                elif delay_type == ("valid", "control"):
                    VCDelayFinal = max(VCDelayFinal, delay)
                elif delay_type == ("valid", "data"):
                    VDDelayFinal = max(VDDelayFinal, delay)
                else:
                    print("\033[91m" + f"[ERROR] Unknown delay type {delay_type} in report file {rpt_filename} for unit {unit_name}. Skipping." + "\033[0m")
                    continue
            if traversedParamOnce == False:
                param_value = unit_char.get_parameter_value("DATA_TYPE")
                print("\033[93m" + f"[WARNING] Report file for unit {unit_name} for bitwidth {param_value} does not exist({rpt_filename}). Skipping." + "\033[0m")

        if traversedUnitOnce == False:
            print("\033[91m" + f"[ERROR] No reports found for unit {unit_name}." + "\033[0m")
            continue

        output_data[unit_name] = {"delay":{"data": dataDict,
                                       "valid": validDict,
                                       "ready": readyDict,
                                       "VR": VRDelayFinal,
                                       "CV": CVDelayFinal,
                                       "CR": CRDelayFinal,
                                       "VC": VCDelayFinal,
                                       "VD": VDDelayFinal}}


    # Save the output data to the JSON file
    with open(json_output, 'w') as f:
        json.dump(output_data, f, indent=2)