import argparse
import os
import sys
import shutil
import subprocess
import re

class CLIHandler:
  def __init__(self):
    self.parser = argparse.ArgumentParser()
    self.add_arguments()

  def add_arguments(self):
    self.parser.add_argument(
      "-l", "--list", 
      help="Path to a text file with names of tests to run. If not given, runs all tests."
    )
    self.parser.add_argument(
      "-i", "--ignore",
      help="Path to a text file with names of tests to ignore (i.e. skip)."
    )
    self.parser.add_argument(
      "-t", "--timeout",
      help="Custom timeout value for a single test. If not given, 500 seconds is used."
    )

  def parse_args(self, args=None):
    """
    Parses the command-line arguments.
    :param args: List of arguments to parse (default: sys.argv).
    :return: Parsed arguments namespace.
    """
    return self.parser.parse_args(args)

INTEGRATION_FOLDER = "../../integration-test/"
SCRIPT_CONTENT = """set-dynamatic-path ../..
set-src {src_path}
compile
write-hdl
simulate
exit
"""
DYN_FILE = "../../build/run_test.dyn"
DYNAMATIC_COMMAND = "../../bin/dynamatic --run {script_path}"

# Class to have different colors while writing in terminal
class bcolors:
  HEADER = "\033[95m"
  OKBLUE = "\033[94m"
  OKCYAN = "\033[96m"
  OKGREEN = "\033[92m"
  WARNING = "\033[93m"
  FAIL = "\033[91m"
  ENDC = "\033[0m"
  BOLD = "\033[1m"
  UNDERLINE = "\033[4m"

# Print with a certain color
def color_print(string: str, color: str):
  print(f"{color}{string}{bcolors.ENDC}")

def find_files_ext(directory, ext):
  c_files = []
  for root, dirs, files in os.walk(directory):
    for file in files:
      if file.endswith(ext) and root.find("/out/") == -1:
        c_files.append(os.path.join(root, file))
  return c_files

def write_string_to_file(content, file_path):
  with open(file_path, "w") as file:
    file.write(content)


def read_file(file_path):
  with open(file_path, "r") as file:
    return file.read()

def modify_line(file_path, new_first_line, line_number):
  with open(file_path, "r") as file:
    lines = file.readlines()

  if len(lines) >= line_number and line_number > 0:
    lines[line_number - 1] = new_first_line + "\n"

  with open(file_path, "w") as file:
    file.writelines(lines)

def run_command_with_timeout(command, timeout=500):
  try:
    proc = subprocess.run(
      command,
      shell=True,
      timeout=timeout,
      check=True,
      stdout=subprocess.PIPE,  # Suppress standard output
      stderr=subprocess.PIPE,  # Suppress standard error
    )
    stderr = proc.stdout.decode()
    if "FATAL" in stderr or "FAIL" in stderr:
      return 1
    return 0
  except (subprocess.TimeoutExpired, subprocess.CalledProcessError):
    return 2

def replace_filename_with(file_path, to_add):
  directory = os.path.dirname(file_path)
  new_path = os.path.join(directory, to_add)
  return new_path

def append_to_file(filename, text):
  with open(filename, "a") as file:
    file.write(text + "\n")

def get_sim_time(log_path):
  # Regular expression to match the desired line format
  pattern = re.compile(r"Time: (\d+) ns")
    
  last_time = None

  # Open the file and read it in reverse line order
  with open(log_path, "r") as file:
    for line in reversed(file.readlines()):
      match = pattern.search(line)
      if match:
        last_time = int(match.group(1))
        break

  if last_time is not None:
    return last_time
  else:
    return None

def main():
  cli = CLIHandler()
  args = cli.parse_args()  # Parse the CLI arguments

  c_files = []
  if args.list:
    test_names = read_file(args.list).strip().split("\n")
    c_files = [os.path.join(INTEGRATION_FOLDER, name, f"{name}.c") for name in test_names]
  else:
    c_files = find_files_ext(INTEGRATION_FOLDER, ".c")

  ignored_tests = []
  if args.ignore:
    test_names = read_file(args.ignore).strip().split("\n")
    ignored_tests = [os.path.join(INTEGRATION_FOLDER, name, f"{name}.c") for name in test_names]

  print("========= INTEGRATION TEST =========")

  test_cnt = 0
  passed_cnt = 0
  ignored_cnt = 0

  for c_file in c_files:
    # Write .dyn script with appropriate source file name
    write_string_to_file(SCRIPT_CONTENT.format(src_path=c_file), DYN_FILE)

    # Get out dir name
    out_dir = replace_filename_with(c_file, "out")

    # Check if test is supposed to be ignored
    if c_file in ignored_tests:
      ignored_cnt += 1
      color_print(f"[IGNORED] {c_file}", bcolors.OKGREEN)
      continue

    # One more test to handle
    test_cnt += 1

    # Remove previous out directory
    if os.path.isdir(out_dir):
      shutil.rmtree(out_dir)
    
    # Run test and output result
    if args.timeout:
      result = run_command_with_timeout(DYNAMATIC_COMMAND.format(script_path=DYN_FILE), timeout=int(args.timeout))
    else:
      result = run_command_with_timeout(DYNAMATIC_COMMAND.format(script_path=DYN_FILE))
    
    if result == 0:
      sim_log_path = os.path.join(out_dir, "sim", "report.txt")
      sim_time = get_sim_time(sim_log_path)

      if sim_time != None: 
        color_print(f"[PASS] {c_file} (simulation time: {sim_time} ns)", bcolors.OKGREEN)
      else:
        color_print(f"[PASS] {c_file}", bcolors.OKGREEN)
      
      passed_cnt += 1
    elif result == 1:
      color_print(f"[FAIL] {c_file}", bcolors.FAIL)
    else:
      color_print(f"[TIMEOUT] {c_file}", bcolors.WARNING)

  print(f"** Integration testing finished: passed {passed_cnt}/{test_cnt} tests ({100 * passed_cnt / test_cnt : .2f}% ), {ignored_cnt} ignored **")
  if passed_cnt == test_cnt:
    sys.exit(0)
  else:
    sys.exit(1)


if __name__ == "__main__":
  main()