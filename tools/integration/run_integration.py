"""
Script for running Dynamatic integration tests.
"""

import argparse
import os
import sys
import shutil
import subprocess
import re
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor


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
            "-l",
            "--list",
            help="Path to a text file with names of tests to run. If not given, runs all tests.",
        )
        self.parser.add_argument(
            "-i",
            "--ignore",
            help="Path to a text file with names of tests to ignore (i.e. skip).",
        )
        self.parser.add_argument(
            "-t",
            "--timeout",
            nargs="?",
            type=int,
            default=500,
            help="Custom timeout value for a single test. Default is 500 seconds.",
        )
        self.parser.add_argument(
            "-w",
            "--workers",
            nargs="?",
            type=int,
            help="Number of workers to run in parallel for testing. Default is os.cpu_count().",
        )

    def parse_args(self, args=None):
        """
        Parses the command-line arguments.

        Arguments:
        `args` -- List of arguments to parse (default: sys.argv)

        Returns: Parsed arguments namespace
        """
        return self.parser.parse_args(args)


DYNAMATIC_ROOT = Path(__file__).parent.parent.parent
INTEGRATION_FOLDER = DYNAMATIC_ROOT / "integration-test"
SCRIPT_CONTENT = (
    f"set-dynamatic-path {DYNAMATIC_ROOT}"
    + """
set-src {src_path}
compile
write-hdl
simulate
exit
"""
)

# Note: Must use --exit-on-failure in order for run_command_with_timeout
#       to be able to detect the status code properly
DYNAMATIC_COMMAND = (
    str(DYNAMATIC_ROOT / "bin" / "dynamatic") +
    " --exit-on-failure --run {script_path}"
)


class TermColors:
    """
    Contains ANSI color escape sequences for colored terminal output.
    """

    HEADER = "\033[95m"
    OKBLUE = "\033[94m"
    OKCYAN = "\033[96m"
    OKGREEN = "\033[92m"
    WARNING = "\033[93m"
    FAIL = "\033[91m"
    ENDC = "\033[0m"
    BOLD = "\033[1m"
    UNDERLINE = "\033[4m"


def color_print(string: str, color: str):
    """
    Prints colored text to stdout using ANSI escape sequences.

    Arguments:
    `string` -- The text to be outputted
    `color` -- ANSI escape seq. for the desired color; use TermColors constants
    """
    print(f"{color}{string}{TermColors.ENDC}")


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
            if file.endswith(ext) and root.find("/out/") == -1:
                c_files.append(os.path.join(root, file))
    return c_files


def write_string_to_file(content, file_path):
    """
    Opens a text file and writes a string to it.

    Arguments:
    `content` -- String to be written
    `file_path` -- Path of the file to be written to
    """
    with open(file_path, "w", encoding="UTF-8") as file:
        file.write(content)


def read_file(file_path):
    """
    Opens a text file and reads all of its contents.

    Arguments:
    `file_path` -- Path to the file to be read from

    Returns: File contents as a string
    """
    with open(file_path, "r", encoding="UTF-8") as file:
        return file.read()


def run_command_with_timeout(
    command, timeout, stdout=subprocess.PIPE, stderr=subprocess.PIPE
):
    """
    Runs a command with a time limit for execution.

    Arguments:
    `command` -- Shell command to be executed
    `timeout` -- Execution time limit
    `stdout`  -- File to redirect stdout to
    `stderr`  -- File to redirect stderr to

    Returns: An integer representing the execution result
    0 -- if process completed without errors
    1 -- if process returned a non-zero exit code
    2 -- if process execution time exceeded timeout
    """
    try:
        subprocess.run(
            command,
            shell=True,
            timeout=timeout,
            check=True,
            stdout=stdout,
            stderr=stderr,
        )
        return 0
    except subprocess.CalledProcessError:
        # Test returned non-zero
        return 1
    except subprocess.TimeoutExpired:
        # Test timed out
        return 2


def replace_filename_with(file_path, suffix):
    """
    Replaces the file name in the path with another string.

    Arguments:
    `file_path` -- A path to a file
    `suffix` -- String to replace the file name with

    Returns: A new path with the file name replaced with suffix.

    Example: If called with "/home/user/a.txt" and "b.py", returns "/home/user/b.py"
    """
    directory = os.path.dirname(file_path)
    new_path = os.path.join(directory, suffix)
    return new_path


def get_sim_time(log_path):
    """
    Returns the simulation time from a Modelsim log file.

    Arguments:
    `log_path` -- Path to the Modelsim simulation log.

    Returns:
    The simulation duration found in the log file.
    If this value is not present in the log file, throws a `ValueError`.
    """

    # Regular expression to match the desired line format
    pattern = re.compile(r"Time: (\d+) ns")

    last_time = None

    # Open the file and read it in reverse line order
    with open(log_path, "r", encoding="UTF-8") as file:
        for line in reversed(file.readlines()):
            match = pattern.search(line)
            if match:
                last_time = int(match.group(1))
                break

    if last_time is not None:
        return last_time
    else:
        raise ValueError("Log file does not contain simulation time!")


def run_test(c_file, id, timeout):
    """
    Runs the specified integration test.

    Arguments:
    `c_file`   -- Path to .c source file of integration test.
    `id`       -- Index used to identify the test.
    `timeout`  -- Timeout in seconds for running the test.

    Returns:
    Dictionary with the following keys:
    `id`      -- Index of the test that was given as argument.
    `msg`     -- Message indicating the result of the test.
    `status`  -- One of 'pass', `fail` or `timeout`.
    """

    # Write .dyn script with appropriate source file name
    dyn_file = DYNAMATIC_ROOT / "build" / f"test_{id}.dyn"
    write_string_to_file(SCRIPT_CONTENT.format(src_path=c_file), dyn_file)

    # Get out dir name
    out_dir = replace_filename_with(c_file, "out")

    # Remove previous out directory
    if os.path.isdir(out_dir):
        shutil.rmtree(out_dir)

    Path(out_dir).mkdir()

    with open(Path(out_dir) / "dynamatic_out.txt", "w") as stdout, open(
        Path(out_dir) / "dynamatic_err.txt", "w"
    ) as stderr:
        # Run test and output result
        exit_code = run_command_with_timeout(
            DYNAMATIC_COMMAND.format(script_path=dyn_file),
            timeout=timeout,
            stdout=stdout,
            stderr=stderr,
        )

        name = Path(c_file).name[:-2]
        if exit_code == 0:
            sim_log_path = os.path.join(out_dir, "sim", "report.txt")
            try:
                sim_time = get_sim_time(sim_log_path)
                return {
                    "id": id,
                    "msg": f"[PASS] {name} (simulation duration: "
                    f"{round(sim_time / 4)} cycles)",
                    "status": "pass",
                }
            except ValueError:
                # This should never happen
                return {
                    "id": id,
                    "msg": f"[PASS] {name} (simulation duration: NOT FOUND)",
                    "status": "pass",
                }

        elif exit_code == 1:
            return {"id": id, "msg": f"[FAIL] {name}", "status": "fail"}
        else:
            return {"id": id, "msg": f"[TIMEOUT] {name}", "status": "timeout"}


def main():
    """
    Entry point for the script.
    """
    cli = CLIHandler()
    args = cli.parse_args()  # Parse the CLI arguments

    c_files = find_files_ext(INTEGRATION_FOLDER, ".c")
    test_names = []
    if args.list:
        test_names = read_file(args.list).strip().split("\n")
        filtered_c_files = []
        for test in test_names:
            found = False
            for file in c_files:
                if Path(file).name == test + ".c":
                    filtered_c_files.append(file)
                    found = True
                    break

            if not found:
                color_print(f"[WARNING] Test '{test}' not found", TermColors.WARNING)

        c_files = filtered_c_files

    ignored_tests = []
    if args.ignore:
        ignored_tests = read_file(args.ignore).strip().split("\n")

    print("========= INTEGRATION TEST =========")

    test_cnt = 0
    passed_cnt = 0
    ignored_cnt = 0

    # ProcessPoolExecutor creates subprocesses and not threads, and as such,
    # is not limited by Python's Global Interpreter Lock.
    # Hence, it allows the full performance gain from parallelism to be achieved,
    # unlike ThreadPoolExecutor, which would not make execution any faster whatsoever.
    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        # Note: _max_workers is a private variable, as indicated by the underscore.
        # However, we access it to get the number of workers used, since the default
        # number (which is used when the ctor is called with max_workers=None) is
        # os.cpu_count() or os.process_cpu_count(), depending on the Python version.
        # This is "cheating", but is independent of the Python version.
        color_print(
            f"[INFO] Running with {executor._max_workers} worker(s).", TermColors.OKBLUE
        )

        processes = []
        for idx, c_file in enumerate(c_files):
            # Check if test is supposed to be ignored
            name = Path(c_file).name[:-2]
            if name in ignored_tests:
                ignored_cnt += 1
                color_print(f"[IGNORED] {name}", TermColors.OKGREEN)
                continue

            # One more test to handle
            test_cnt += 1

            # Run the test
            processes.append(executor.submit(run_test, c_file, idx, args.timeout))

            color_print(f"[INFO] Submitted {name} for execution", TermColors.OKBLUE)
            sys.stdout.flush()

        for p in processes:
            result = p.result()
            if result["status"] == "pass":
                color_print(result["msg"], TermColors.OKGREEN)
                passed_cnt += 1
            elif result["status"] == "fail":
                color_print(result["msg"], TermColors.FAIL)
            else:
                color_print(result["msg"], TermColors.WARNING)

            sys.stdout.flush()

    print(
        f"** Integration testing finished: "
        f"passed {passed_cnt}/{test_cnt} tests "
        f"({100 * passed_cnt / test_cnt: .2f}% ), "
        f"{ignored_cnt} ignored **"
    )
    if passed_cnt == test_cnt:
        sys.exit(0)
    else:
        sys.exit(1)


if __name__ == "__main__":
    main()
