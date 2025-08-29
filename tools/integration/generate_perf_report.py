"""
This script outputs a performance report generated from 
GoogleTest's .xml outputs.
"""

import os
import sys
import json
import xml.etree.ElementTree as ET
from pathlib import Path


DYNAMATIC_ROOT = Path(__file__).parent.parent.parent
RESULTS_DIR = DYNAMATIC_ROOT / "build" / "tools" / "integration" / "results"


def find_files_ext(directory, ext):
    """
    Finds all files with a given extension in a directory.

    Arguments:
    `directory` -- Path of the directory
    `ext` -- File extension to be searched for

    Returns: List of paths to all files in directory with extension ext
    """
    c_files = []
    for root, _, files in os.walk(directory):
        for file in files:
            if file.endswith(ext):
                c_files.append(os.path.join(root, file))
    return c_files


def parse_xml(path):
    """
    Parses a single .xml report.

    Arguments:
    `path` -- XML file path

    Returns: List of dictionaries describing the logged information
    for each test case found in the report. 
    """
    tree = ET.parse(path)
    root = tree.getroot()

    res = []
    for test in root.iter("testcase"):
        row = {}

        test_name = test.attrib["classname"] + "." + test.attrib["name"]

        fail_messages = []
        for f in test.findall("failure"):
            fail_messages.append(f.attrib["message"])

        if len(fail_messages) > 0:
            result = "**fail** :x:"
        else:
            result = "pass :heavy_check_mark:"

        row["name"] = test_name
        row["result"] = result

        for property in test.iter("property"):
            prop_name = property.attrib["name"]
            prop_val = property.attrib["value"]

            if prop_name != "cycles" or prop_val != "-1":
                row[prop_name] = prop_val

        res.append(row)

    return res


def parse_results(path):
    """
    Collects data from all reports in a folder and its subtree.

    Arguments:
    `path` -- Directory path

    Returns: List of dictionaries describing the logged information
    found in all reports.
    """
    xml_files = find_files_ext(path, ".xml")

    columns = ["name", "result", "cycles"]
    data = []

    for file in xml_files:
        data += parse_xml(file)

    return data


def table(header, data):
    """
    Generates a markdown table with the given data.

    Arguments:
    `header` -- List of column names
    `data` -- List of dictionaries representing rows in the table;
    keys correspond to elements of header (i.e. column names)

    Returns: A string representing a markdown table containing the
    given data.
    """
    res = ""
    for elem in header:
        res += f"| {elem} "
    res += "|\n"

    for elem in header:
        res += "| ---- "
    res += "|\n"

    for row in data:
        for elem in header:
            if elem in row:
                res += f"| {row[elem]} "
            else:
                res += f"| N/A "
        res += "|\n"

    return res


def main():
    """
    Entry point.

    Arguments:
    `save_path` -- Path to which the json performance report will be saved.
    `compare_path` -- Path of perf. report against which this one will be compared.
    """
    if len(sys.argv) < 3:
        print("Error: Not enough arguments")
        exit(-1)

    data = {
        "columns": ["name", "cycles", "result", "old_cycles", "comparison"],
        "data": parse_results(RESULTS_DIR)
    }

    with open(sys.argv[1], "w") as f:
        json.dump(data, f)

    with open(sys.argv[2], "r") as f:
        old_data = json.load(f)

    for old_row in old_data["data"]:
        if "fail" in old_row["result"]:
            continue

        test_name = old_row["name"]
        old_cycles = old_row["cycles"]

        for row in data["data"]:
            if row["name"] == test_name:
                row["old_cycles"] = old_cycles

                try:
                    diff = int(row["cycles"]) - int(row["old_cycles"])
                    if diff < 0:
                        row["comparison"] = f"{diff} :heavy_check_mark:"
                    elif diff > 0:
                        row["comparison"] = f"**{diff}** :heavy_exclamation_mark:"
                    else:
                        row["comparison"] = f"{diff} :heavy_minus_sign:"

                except ValueError:
                    pass

    print("## Performance Report")
    print(table(
        data["columns"],
        data["data"]
    ))


if __name__ == "__main__":
    main()
