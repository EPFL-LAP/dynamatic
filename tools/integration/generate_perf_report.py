"""
This script outputs a performance report generated from 
GoogleTest's .xml outputs.
"""

import os
import sys
import pickle
import argparse
import xml.etree.ElementTree as ET
from pathlib import Path


DYNAMATIC_ROOT = Path(__file__).parent.parent.parent
RESULTS_DIR = DYNAMATIC_ROOT / "build" / "tools" / "integration" / "results"


class CLIHandler:
    """
    This class parses the script's command line arguments.
    """

    def __init__(self):
        self.parser = argparse.ArgumentParser()
        self.add_arguments()

    def add_arguments(self):
        """
        Configures all available command line arguments.
        """
        self.parser.add_argument(
            "-c",
            "--compare",
            nargs="?",
            type=str,
            default=None,
            help="Path of another performance report against which to compare the generated one.",
        )
        self.parser.add_argument(
            "-s",
            "--save",
            nargs="?",
            type=str,
            default=None,
            help="Path to which the generated performance report should be saved (as binary data).",
        )
        self.parser.add_argument(
            "-f",
            "--fail",
            nargs="?",
            type=float,
            default=5.0,
            help="Smallest percent difference between current and old report \
                that should return a failure exit code. Default is 5, i.e. if any test \
                has a cycle performance at least 5%% worse than the previous report \
                (specified with -c), the generator will return non-zero.",
        )

    def parse_args(self, args=None):
        """
        Parses the command-line arguments.

        Arguments:
        `args` -- List of arguments to parse (default: sys.argv)

        Returns: Parsed arguments namespace
        """
        return self.parser.parse_args(args)


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
        res += "| ----"
        if elem in ["cycles", "old_cycles", "comparison"]:
            res += ":"
        res += " "
    res += "|\n"

    for row in data:
        for elem in header:
            if elem in row:
                try:
                    res += f"| {int(row[elem]):,} "
                except ValueError:
                    res += f"| {row[elem]} "
            else:
                res += f"| N/A "
        res += "|\n"

    return res


def main():
    """
    Entry point.
    """
    cli = CLIHandler()
    args = cli.parse_args()

    data = {
        "columns": ["name", "cycles", "result", "old_cycles", "comparison"],
        "data": parse_results(RESULTS_DIR)
    }

    if args.save:
        with open(args.save, "wb") as f:
            pickle.dump(data, f)

    if args.compare:
        with open(args.compare, "rb") as f:
            old_data = pickle.load(f)

    failed = False

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
                    percent = diff / int(row["old_cycles"]) * 100.0

                    if args.fail and percent >= args.fail:
                        failed = True

                    if diff < 0:
                        row["comparison"] = f":heavy_check_mark: {diff:,} ({percent:.2f}%)"
                    elif diff > 0:
                        row["comparison"] = f":heavy_exclamation_mark: **{diff:,} ({percent:.2f}%)**"
                    else:
                        row["comparison"] = f":heavy_minus_sign: {diff:,} ({percent:.2f}%)"

                except ValueError:
                    pass

    if failed:
        print("## Performance Report :heavy_exclamation_mark:")
    else:
        print("## Performance Report :heavy_check_mark:")
    print(table(
        data["columns"],
        data["data"]
    ))

    if failed:
        sys.exit(-1)


if __name__ == "__main__":
    main()
