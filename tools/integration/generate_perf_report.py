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
            result = "fail"
        else:
            result = "pass"

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
    xml_files = find_files_ext(path, ".xml")

    columns = ["name", "result", "cycles"]
    data = []

    for file in xml_files:
        data += parse_xml(file)

    return data


def table(header, data):
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
    print("## Performance Report")
    print(table(
        ["name", "cycles", "result"],
        parse_results(RESULTS_DIR)
    ))


if __name__ == "__main__":
    main()
