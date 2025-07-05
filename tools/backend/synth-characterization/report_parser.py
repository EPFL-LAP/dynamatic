# This script is used to extract timing data from synthesis reports and save it in a JSON format.
import os
import re
import json

# Constants for parsing the report that specify which line 
# contains the connection information and which line contains the delay information.
# These patters are specific to the Vivado synthesis report format.
# If a different synthesis tool is used, new patterns may need to be defined.
PATTERN_CONNECTION_INFO = "Command      :"
PATTERN_DELAY_INFO = "Data Path Delay:"

# This function extracts the connection type and delay from a line in the report file.
# It uses regular expressions to find the relevant information.
# It is specific to the Vivado synthesis report format.
# If a different synthesis tool is used, this function may need to be modified.
def extract_connection_type(line):
    """
    Extract the connection type from a line in the report.
    
    Args:
        line (str): A line from the report file.
        
    Returns:
        tuple: A tuple containing the from_port and to_port.
    """
    match = re.search(r'report_timing\s+-from\s+\[get_ports\s+{?([\w\[\]]+)}?\]\s+-to\s+\[get_ports\s+{?([\w\[\]]+)}?\]', line)
    assert match, f"Could not find connection type in line: {line}"
    from_port = match.group(1)
    to_port = match.group(2)
    return from_port, to_port

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
        tuple: A tuple containing dataDelay, validDelay, readyDelay, VRDelay, CVDelay, CRDelay, VCDelay, VDDelay.
    """
    # Initialize variables
    dataDelay = -1
    validDelay = -1
    readyDelay = -1
    VRDelay = -1
    CVDelay = -1
    CRDelay = -1
    VCDelay = -1
    VDDelay = -1
    
    # Read the report file and extract the required data
    with open(rpt_file, 'r') as f:
        for line in f:
            # Extract connection type
            if PATTERN_CONNECTION_INFO in line:
                from_port, to_port = extract_connection_type(line)
            # Extract delay of the data path
            if PATTERN_DELAY_INFO in line:
                delay = extract_delay(line)
                connection_found = False
                # Determine the connection type based on the ports names
                validSignalFrom = "_valid" in from_port
                validSignalTo = "_valid" in to_port
                readySignalFrom = "_ready" in from_port
                readySignalTo = "_ready" in to_port
                conditionFrom = ("condition" in from_port or "index" in from_port)and not validSignalFrom and not readySignalFrom
                conditionTo = ("condition" in to_port or "index" in to_port) and not validSignalTo and not readySignalTo
                dataFrom = False
                dataTo = False
                if not validSignalFrom and not readySignalFrom and not conditionFrom:
                    assert "lhs" in from_port or "rhs" in from_port or "trueValue" in from_port or "falseValue" in from_port or "ins" in from_port or "data" in from_port or "addrIn" in from_port, f"Unexpected port `{from_port}` without valid or ready signal."
                    dataFrom = True
                if not validSignalTo and not readySignalTo and not conditionTo:
                    assert "result" in to_port or "outs" in to_port or "trueOut" in to_port or "falseOut" in to_port or "addrOut" in to_port or "dataOut" in to_port, f"Unexpected port `{to_port}` without valid or ready signal."
                    dataTo = True

                if (dataFrom and dataTo) or (conditionFrom and dataTo):
                    # Data to data connection
                    connection_found = True
                    dataDelay = max(dataDelay, delay)
                elif validSignalFrom and validSignalTo:
                    # Valid to valid connection
                    connection_found = True
                    validDelay = max(validDelay, delay)
                elif validSignalFrom and dataTo:
                    # Valid to data connection (VD)
                    connection_found = True
                    VDDelay = max(VDDelay, delay)
                elif validSignalFrom and readySignalTo:
                    # Valid to ready connection (VR)
                    connection_found = True
                    VRDelay = max(VRDelay, delay)
                elif validSignalFrom and conditionTo:
                    # Valid to condition connection (CV)
                    connection_found = True
                    VCDelay = max(VCDelay, delay)
                elif readySignalFrom and readySignalTo:
                    # Ready to ready connection
                    connection_found = True
                    readyDelay = max(readyDelay, delay)
                elif conditionFrom and validSignalTo:
                    # Condition to valid connection (CV)
                    connection_found = True
                    CVDelay = max(CVDelay, delay)
                elif conditionFrom and readySignalTo:
                    # Condition to ready connection (CR)
                    connection_found = True
                    CRDelay = max(CRDelay, delay)

                assert connection_found, f"Could not determine connection type for ports `{from_port}` and `{to_port}`"

    return dataDelay, validDelay, readyDelay, VRDelay, CVDelay, CRDelay, VCDelay, VDDelay

def extract_rpt_data(map_unit2rpts, json_output):
    """
    Extract the data from the map_unit2rpts dictionary and save it to a JSON file.
    IMPORTANT: For now we assume that only DATA_TYPE is the only parameter that can be used to characterize the unit.
    
    Args:
        map_unit2rpts (dict): Dictionary containing unit names as keys and their reports as values.
        json_output (str): Path to the output JSON file.
    """
    # Create the output data structure
    output_data = {}
    for unit_name, map_rpt2params in map_unit2rpts.items():
        dataDict = {}
        validDict = {"1": 0.0}
        readyDict = {"1": 0.0}
        VRDelayFinal = 0.0
        CVDelayFinal = 0.0
        CRDelayFinal = 0.0
        VCDelayFinal = 0.0
        VDDelayFinal = 0.0
        traversedOnce = False
        # Extract the data from the reports
        for rpt_file, params in map_rpt2params.items():
            # Check if the report file exists
            if not os.path.exists(rpt_file):
                print("\033[93m" + f"[WARNING] Report file for unit {unit_name} parameters {params} does not exist({rpt_file}). Skipping." + "\033[0m")
                continue
            traversedOnce = True
            # Extract data2data, valid2valid, ready2ready, VR, CV, CR, VC and VD
            dataDelay, validDelay, readyDelay, VRDelay, CVDelay, CRDelay, VCDelay, VDDelay = extract_single_rpt(rpt_file)
            dataDict[str(params["DATA_TYPE"])] = dataDelay
            validDict["1"] = max(validDict["1"], validDelay)
            readyDict["1"] = max(readyDict["1"], readyDelay)
            VRDelayFinal = max(VRDelayFinal, VRDelay)
            CVDelayFinal = max(CVDelayFinal, CVDelay)
            CRDelayFinal = max(CRDelayFinal, CRDelay)
            VCDelayFinal = max(VCDelayFinal, VCDelay)
            VDDelayFinal = max(VDDelayFinal, VDDelay)

        if traversedOnce == False:
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