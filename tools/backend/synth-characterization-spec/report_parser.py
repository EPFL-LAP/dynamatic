# Extract per-(in_port, out_port) timing data and save as a JSON matrix.
import os
import re
import json

PATTERN_DELAY_INFO = "Data Path Delay:"


def extract_delay(line):
    match = re.search(r'Data Path Delay:\s+([\d.]+)ns', line)
    assert match, f"Could not find data path delay in line: {line}"
    return float(match.group(1))


def extract_single_rpt(rpt_file):
    """Return the max Data Path Delay in the report file, or None if absent."""
    if not os.path.exists(rpt_file):
        return None
    max_delay = None
    with open(rpt_file, 'r') as f:
        for line in f:
            if PATTERN_DELAY_INFO in line:
                d = extract_delay(line)
                max_delay = d if max_delay is None else max(max_delay, d)
    return max_delay


def extract_rpt_data(map_unit_to_list_unit_chars, json_output):
    """
    Build a nested per-port-pair delay matrix per bitwidth.

    output[unit_name]["delays"][in_port][out_port][bitwidth_str] = delay_ns
    """
    output_data = {}
    for unit_name, list_unit_chars in map_unit_to_list_unit_chars.items():
        delays = {}
        for unit_char in list_unit_chars:
            bw = str(unit_char.get_parameter_value("BITWIDTH"))
            pair_to_rpt = getattr(unit_char, "map_pair_to_rpt", {})
            if not pair_to_rpt:
                print(
                    "\033[91m" + f"[ERROR] No pair_to_rpt for unit {unit_name} at BITWIDTH={bw}." + "\033[0m")
                continue
            for (iport, oport), rpt_path in pair_to_rpt.items():
                delay = extract_single_rpt(rpt_path)
                if delay is None:
                    continue
                delays.setdefault(iport, {}).setdefault(oport, {})[bw] = delay
        output_data[unit_name] = {"delays": delays}

    with open(json_output, 'w') as f:
        json.dump(output_data, f, indent=2)
