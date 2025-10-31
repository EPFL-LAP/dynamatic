from pathlib import Path
import subprocess
from sys import stdout
from subprocess import run


import sys

# === CONFIG ===
TIMEOUT = 20  # You can convert to seconds if using Python timeout
DYNAMATIC_PATH = Path("./bin/dynamatic")
DYNAMATIC_OPT_BIN = Path("./bin/dynamatic-opt")
TEST_DIR = Path("./integration-test")
COMPILE_TEMPLATE = Path("./skip/templates/compile-with-dynamatic-compile-template.dyn")
SIMULATE_TEMPLATE = Path("./skip/templates/compile-with-dynamatic-simulate-template.dyn")
TCL_TEMPLATE = Path("./skip/templates/call-vivado-impl-for-dynamatic.tcl")
CONSTRAINTS_TEMPLATE = Path("./skip/templates/call-vivado-impl-for-dynamatic-contraints.xdc")
BUFFERING_ALG = "cpbuf"

RTL_BASE_PATH = "tb/duv_inst"

COMP_DIR = Path("out/comp")
SIM_DIR = Path("out/sim")
VIVADO_DIR = Path("out/vivado")
RUNS_BASE = Path("./skip/runs")  # where all timestamped folders go
RUN_LOG_NAME = "run.log"


def shell(*args, **kwargs):
    stdout.flush()
    return run(*args, **kwargs, check=True)


# def run_cmd(cmd, cwd=None, timeout=None, log_file=None):
#     """Run a shell command safely with timeout and optional logging."""
#     try:
#         result = subprocess.run(
#             cmd,
#             cwd=cwd,
#             shell=True,
#             text=True,
#             timeout=timeout,
#             capture_output=True
#         )
#         if log_file:
#             with open(log_file, "a") as f:
#                 f.write(f"\n[CMD] {cmd}\n")
#                 f.write(result.stdout)
#                 f.write(result.stderr)
#         return result.returncode
#     except subprocess.TimeoutExpired:
#         if log_file:
#             with open(log_file, "a") as f:
#                 f.write(f"\n[TIMEOUT] {cmd}\n")
#         return 124



def run_cmd(cmd, cwd=None, timeout=None, log_file=None):
    try:
        with subprocess.Popen(
            cmd,
            cwd=cwd,
            shell=True,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            bufsize=1,  # line-buffered
            universal_newlines=True,
        ) as proc:

            for line in proc.stdout:
                if log_file:
                    with open(log_file, "a") as f:
                        f.write(line)

            try:
                proc.wait(timeout=timeout)
            except subprocess.TimeoutExpired:
                proc.kill()
                if log_file:
                    with open(log_file, "a") as f:
                        f.write(f"\n[TIMEOUT] {cmd}\n")
                return 124

            return proc.returncode

    except Exception as e:
        if log_file:
            with open(log_file, "a") as f:
                f.write(f"\n[ERROR] {cmd} -> {e}\n")
        return 1

    

def log(msg: str, log_file: str):
    with open(log_file, "a") as f:
        f.write(f"{msg}\n")
