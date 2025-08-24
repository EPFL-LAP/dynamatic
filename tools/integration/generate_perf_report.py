"""
This script outputs a performance report generated from 
GoogleTest's .xml outputs.
"""

import os
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
    """
    print("## Performance Report")
    print(table(
        ["name", "cycles", "result"],
        parse_results(RESULTS_DIR)
    ))


if __name__ == "__main__":
    main()
