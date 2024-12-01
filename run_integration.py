import os
import shutil
import subprocess
from random import shuffle

INTEGRATION_FOLDER = "./integration-test/"
SCRIPT_CONTENT = """set-src integration-test/if_loop_3/if_loop_3.c
compile --buffer-algorithm on-merges 
write-hdl
simulate
exit
"""
FTD_DYN_FILE = "./build/ftd_run.dyn"
DYN_FILE = "./build/original_run.dyn"
FTD_COMPILE_COMMAND = "compile --fast-token-delivery --buffer-algorithm on-merges"
SET_SRC_COMMAND = "set-src "
DYNAMATIC_COMMAND = "./bin/dynamatic --run "
TESTS_FAIL_FILE = "./tests_to_skip.txt"
TEST_CORRECT_FILE = "./tests_correct.txt"


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


def get_string_result(total, original, ftd):
    return f"({total}, {original}, {ftd})"


def run_command_with_timeout(command, timeout=220):
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


def main():
    c_files = find_files_ext(INTEGRATION_FOLDER, ".c")
    shuffle(c_files)
    write_string_to_file(SCRIPT_CONTENT, FTD_DYN_FILE)
    write_string_to_file(SCRIPT_CONTENT, DYN_FILE)
    modify_line(FTD_DYN_FILE, FTD_COMPILE_COMMAND, 2)

    counter_test, counter_pass_original, counter_pass_ftd = 0, 0, 0
    failed_ftd = []

    print("========= INTEGRATION TEST WITH FAST TOKEN DELIVERY =========")

    for c_file in c_files:

        # Change source file in .dyn scripts
        modify_line(DYN_FILE, SET_SRC_COMMAND + c_file, 1)
        modify_line(FTD_DYN_FILE, SET_SRC_COMMAND + c_file, 1)

        # Get out dir name
        out_dir = replace_filename_with(c_file, "out")

        # Get files to skip
        if c_file in read_file(TESTS_FAIL_FILE):
            continue

        # One more test to handle
        counter_test += 1

        # Remove previous out directory
        if os.path.isdir(out_dir):
            shutil.rmtree(out_dir)

        # Check whether either the input file works on dynamatic or if it was in the list of files already checked
        is_original_ok = False
        if c_file in read_file(TEST_CORRECT_FILE):
            is_original_ok = True
        else:
            result = run_command_with_timeout(DYNAMATIC_COMMAND + DYN_FILE)
            if result == 0:
                is_original_ok = True
                append_to_file(TEST_CORRECT_FILE, c_file)
            elif result == 1:
                append_to_file(TESTS_FAIL_FILE, c_file)
                color_print(
                    f"[FAIL ORIGINAL] {c_file} {get_string_result(counter_test, counter_pass_original, counter_pass_ftd)}",
                    bcolors.FAIL,
                )
            else:
                append_to_file(TESTS_FAIL_FILE, c_file)
                color_print(
                    f"[TIMEOUT ORIGINAL] {c_file} {get_string_result(counter_test, counter_pass_original, counter_pass_ftd)}",
                    bcolors.WARNING,
                )

        if is_original_ok:

            # in this case, the file works, and we can check it over FTD version
            counter_pass_original += 1

            color_print(f"[SUCCESS ORIGINAL] {c_file} ", bcolors.OKGREEN)

            # Remove out dir
            if os.path.isdir(out_dir):
                shutil.rmtree(out_dir)

            # Run the file with FTD
            error = run_command_with_timeout(DYNAMATIC_COMMAND + FTD_DYN_FILE)
            if error == 0:
                counter_pass_ftd += 1
                color_print(
                    f"[SUCCESS FTD] {c_file} {get_string_result(counter_test, counter_pass_original, counter_pass_ftd)}",
                    bcolors.OKGREEN,
                )
            elif error == 1:
                color_print(
                    f"[FAIL FTD] {c_file} {get_string_result(counter_test, counter_pass_original, counter_pass_ftd)}",
                    bcolors.FAIL,
                )
                failed_ftd.append(c_file)
            else:
                color_print(
                    f"[TIMEOUT FTD] {c_file} {get_string_result(counter_test, counter_pass_original, counter_pass_ftd)}",
                    bcolors.WARNING,
                )
                failed_ftd.append(c_file)

    if counter_pass_original != counter_pass_ftd:
        print(
            "The following tests were succcesful on dynamatic, and failed with Fast Token Delivery"
        )
        for c_file in failed_ftd:
            print(c_file)


if __name__ == "__main__":
    main()
