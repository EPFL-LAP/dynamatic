import os as os
import csv
import re as re


def find_files(root_dir, file_pattern):
    """
    Returns: A list of paths to all files whose name matches the given file_pattern
    """
    file_paths = []
    for dirpath, _, filenames in os.walk(root_dir):
        for filename in filenames:
            if re.match(file_pattern, filename):
                file_paths.append(os.path.join(dirpath, filename))
    return file_paths


def extract_info_from_files_n(patterns_identifier, n, cp, patterns, pattern_names, files):
    res = {"Test id": patterns_identifier, "N": n, "CP used for buffering": cp}

    if len(patterns) != len(pattern_names) or len(files) != len(patterns):
        print("Invalid argument for extract_info_from_files")
        return 0

    for pattern, key, file_path in zip(patterns, pattern_names, files):
        try:
            with open(file_path, "r") as file:
                content = file.read()

                match = re.search(pattern, content)

                if match:
                    res[key] = match.group(1)
                else:
                    res[key] = ""
        except FileNotFoundError:
            # Handle the case when the file doesn't exist
            print(f"Error: The file {file_path} was not found.")
            res[key] = "???"

    return res

def extract_info_from_files(patterns_identifier, load, store, cp, patterns, pattern_names, files):
    """
    Extract relevant information from the given files

    Parameters:
        patterns_identifier (string): Key used to identify all patterns associated with it (first entry in dictionary)
        patterns (list of regular expressions): Pattern that must be looked for with one group inside
        pattern_names (list of strings): Must have the same length as patterns.
            Key that will be associated to the corresponding pattern.
        files (list of file paths): Must have the same length as patterns.
            Pathfiles to the file we are trying to extract information from for each pattern.

    Returns:
        res(dictionary): A dictionary containing the results of the search, with each pattern group identified by the names given in pattern names.
    """
    res = {"Test id": patterns_identifier, "Load queue size": load, "Store queue size": store, "CP used for buffering": cp}

    if len(patterns) != len(pattern_names) or len(files) != len(patterns):
        print("Invalid argument for extract_info_from_files")
        return 0

    for pattern, key, file_path in zip(patterns, pattern_names, files):
        try:
            with open(file_path, "r") as file:
                content = file.read()

                match = re.search(pattern, content)

                if match:
                    res[key] = match.group(1)
                else:
                    res[key] = ""
        except FileNotFoundError:
            # Handle the case when the file doesn't exist
            print(f"Error: The file {file_path} was not found.")
            res[key] = "???"

    return res


def collect_data(file_paths):
    """Collects all data in the given file_paths by calling extract_info_from_files on each given path
    and returns all data in an array.
    """
    data = []
    for file_path in file_paths:
        info = extract_info_from_files(file_path)
        if info:
            data.append(info)
    return data


def write_to_csv(data, output_csv):
  """Write the data array to a csv file"""
  with open(output_csv, "a", newline="") as csvfile:
      fieldnames = data[0].keys()  # Use the keys of the first dict as column headers
      writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

      csvfile.seek(0, 2)  # Move to the end of the file
      if csvfile.tell() == 0:
          writer.writeheader()  # Write the header only if the file is empty

      writer.writerows(data)
