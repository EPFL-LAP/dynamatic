import argparse
import os
import shutil
import subprocess
from random import shuffle


class CLIHandler:
    def __init__(self):
        self.parser = argparse.ArgumentParser(
            description="Run integration tests."
        )
        self.add_arguments()

    def add_arguments(self):
        self.parser.add_argument(
            "--skippable-seq-n",
            type=int,
            metavar="N",
            help="Run with --skippable-seq-n N.",
        )

    def parse_args(self, args=None):
        return self.parser.parse_args(args)


INTEGRATION_FOLDER = "./integration-test/"
SCRIPT_CONTENT = """set-src integration-test/if_loop_3/if_loop_3.c
compile --buffer-algorithm on-merges
write-hdl
simulate
exit
"""

DYN_FILE = "./build/original_run.dyn"

COMPILE_COMMAND = "compile --buffer-algorithm on-merges"
SKIPPABLE_SEQ_COMMAND = (
    "compile --buffer-algorithm on-merges --skippable-seq-n {}"
)

SET_SRC_COMMAND = "set-src "
DYNAMATIC_COMMAND = "./bin/dynamatic --run "
TESTS_FAIL_FILE = "./tests_to_skip.txt"


class bcolors:
    OKGREEN = "\033[92m"
    WARNING = "\033[93m"
    FAIL = "\033[91m"
    ENDC = "\033[0m"


def color_print(string: str, color: str):
    print(f"{color}{string}{bcolors.ENDC}")


def find_files_ext(directory, ext):
    c_files = []
    for root, _, files in os.walk(directory):
        for file in files:
            if file.endswith(ext) and "/out/" not in root:
                c_files.append(os.path.join(root, file))
    return c_files


def write_string_to_file(content, file_path):
    with open(file_path, "w", encoding="utf-8") as file:
        file.write(content)


def read_skip_list(file_path):
    if not os.path.exists(file_path):
        return set()
    with open(file_path, "r", encoding="utf-8") as file:
        return {line.strip() for line in file if line.strip()}


def modify_line(file_path, new_line, line_number):
    with open(file_path, "r", encoding="utf-8") as file:
        lines = file.readlines()
    lines[line_number - 1] = new_line + "\n"
    with open(file_path, "w", encoding="utf-8") as file:
        file.writelines(lines)


def run_command_with_timeout(command, timeout=220):
    try:
        proc = subprocess.run(
            command,
            shell=True,
            timeout=timeout,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
        )
    except subprocess.TimeoutExpired:
        return "timeout"

    # compile.sh propagates the solver's nonzero status. Inspect its explicit
    # result first so UNSAT is not reported as a generic compilation failure.
    output = proc.stdout.decode(errors="replace")
    if "UNSAT" in output:
        return "unsat"
    if proc.returncode == 0:
        return "success"
    return "fail"


def main():
    args = CLIHandler().parse_args()

    dyn_file = DYN_FILE
    write_string_to_file(SCRIPT_CONTENT, dyn_file)

    if args.skippable_seq_n is not None:
        mode = f"WITH SKIPPABLE SEQ N={args.skippable_seq_n}"
        modify_line(
            dyn_file,
            SKIPPABLE_SEQ_COMMAND.format(args.skippable_seq_n),
            2,
        )
    else:
        mode = "BASE"
        modify_line(dyn_file, COMPILE_COMMAND, 2)

    print(f"========= INTEGRATION TEST {mode} =========")

    c_files = find_files_ext(INTEGRATION_FOLDER, ".c")
    shuffle(c_files)

    skipped_tests = read_skip_list(TESTS_FAIL_FILE)
    unsat_tests = []
    failed_tests = []
    timeout_tests = []
    passed_tests = 0

    for test_number, c_file in enumerate(c_files, start=1):
        if c_file in skipped_tests:
            print(f"[SKIP] {c_file}")
            continue

        modify_line(dyn_file, SET_SRC_COMMAND + c_file, 1)

        out_dir = os.path.join(os.path.dirname(c_file), "out")
        if os.path.isdir(out_dir):
            shutil.rmtree(out_dir)

        result = run_command_with_timeout(DYNAMATIC_COMMAND + dyn_file)
        progress = f"({test_number}/{len(c_files)})"

        if result == "success":
            passed_tests += 1
            color_print(f"[SUCCESS] {c_file} {progress}", bcolors.OKGREEN)
        elif result == "unsat":
            unsat_tests.append(c_file)
            color_print(f"[UNSAT] {c_file} {progress}", bcolors.FAIL)
        elif result == "timeout":
            timeout_tests.append(c_file)
            color_print(f"[TIMEOUT] {c_file} {progress}", bcolors.WARNING)
        else:
            failed_tests.append(c_file)
            color_print(f"[FAIL] {c_file} {progress}", bcolors.FAIL)

    print(
        f"Summary: {passed_tests} passed, {len(unsat_tests)} UNSAT, "
        f"{len(failed_tests)} failed, {len(timeout_tests)} timed out."
    )

    if unsat_tests:
        print("Tests reported UNSAT by handshake_count_solver.py:")
        for c_file in unsat_tests:
            print(c_file)


if __name__ == "__main__":
    main()