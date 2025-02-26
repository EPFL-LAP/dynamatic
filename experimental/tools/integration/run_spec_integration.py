"""
Script for running Dynamatic speculative integration tests.
"""
import os
import shutil
import subprocess
import sys
from pathlib import Path

DYNAMATIC_ROOT = Path(__file__).parent.parent.parent.parent
INTEGRATION_FOLDER = DYNAMATIC_ROOT / "integration-test"

DYNAMATIC_OPT_BIN = DYNAMATIC_ROOT / "build" / "bin" / "dynamatic-opt"


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

  print(c_file)

  # Get the c_file directory
  c_file_dir = os.path.dirname(c_file)

  # Get out dir name
  out_dir = os.path.join(c_file_dir, "out")

  # Remove previous out directory
  if os.path.isdir(out_dir):
    shutil.rmtree(out_dir)

  Path(out_dir).mkdir()

  # Custom compilation flow

  # Start with copying the .cf file to the out_dir
  cf_file_base = os.path.join(c_file_dir, "cf.mlir")
  cf_file = os.path.join(out_dir, "cf.mlir")
  shutil.copy(cf_file_base, cf_file)

  # cf transformations (standard)
  cf_transformed = os.path.join(out_dir, "cf_transformed.mlir")
  with open(cf_transformed, "w") as f:
    result = subprocess.run([
        DYNAMATIC_OPT_BIN, cf_file,
        "--canonicalize", "--cse", "--sccp", "--symbol-dce",
        "--control-flow-sink", "--loop-invariant-code-motion", "--canonicalize"],
        stdout=f,
        stderr=sys.stdout
    )
    if result.returncode == 0:
      print("Applied standard transformations to cf")
    else:
      return {
          "id": id,
          "msg": "Failed to apply standard transformations to cf",
          "status": "fail"
      }

  # cf transformations (dynamatic)
  cf_dyn_transformed = os.path.join(out_dir, "cf_dyn_transformed.mlir")
  with open(cf_dyn_transformed, "w") as f:
    result = subprocess.run([
        DYNAMATIC_OPT_BIN, cf_transformed,
        "--arith-reduce-strength=max-adder-depth-mul=1",
        "--push-constants",
        "--mark-memory-interfaces"
    ],
        stdout=f,
        stderr=sys.stdout
    )
    if result.returncode == 0:
      print("Applied Dynamatic transformations to cf")
    else:
      return {
          "id": id,
          "msg": "Failed to apply Dynamatic transformations to cf",
          "status": "fail"
      }

  # cf level -> handshake level
  handshake = os.path.join(out_dir, "handshake.mlir")
  with open(handshake, "w") as f:
    result = subprocess.run([
        DYNAMATIC_OPT_BIN, cf_dyn_transformed,
        "--lower-cf-to-handshake"
    ],
        stdout=f,
        stderr=sys.stdout
    )
    if result.returncode == 0:
      print("Compiled cf to handshake")
    else:
      return {
          "id": id,
          "msg": "Failed to compile cf to handshake",
          "status": "fail"
      }

  # handshake transformations
  handshake_transformed = os.path.join(out_dir, "handshake_transformed.mlir")
  with open(handshake_transformed, "w") as f:
    result = subprocess.run([
        DYNAMATIC_OPT_BIN, handshake,
        "--handshake-analyze-lsq-usage", "--handshake-replace-memory-interfaces",
        "--handshake-minimize-cst-width", "--handshake-optimize-bitwidths",
        "--handshake-materialize", "--handshake-infer-basic-blocks"
    ],
        stdout=f,
        stderr=sys.stdout
    )
    if result.returncode == 0:
      print("Applied transformations to handshake")
    else:
      return {
          "id": id,
          "msg": "Failed to apply transformations to handshake",
          "status": "fail"
      }

  # Buffer placement (Simple buffer placement)
  handshake_buffered = os.path.join(out_dir, "handshake_buffered.mlir")
  timing_model = DYNAMATIC_ROOT / "data" / "components.json"
  with open(handshake_buffered, "w") as f:
    result = subprocess.run([
        DYNAMATIC_OPT_BIN, handshake_transformed,
        "--handshake-set-buffering-properties=version=fpga20",
        f"--handshake-place-buffers=algorithm=on-merges timing-models={timing_model}"
    ],
        stdout=f,
        stderr=sys.stdout
    )
    if result.returncode == 0:
      print("Placed simple buffers")
    else:
      return {
          "id": id,
          "msg": "Failed to place simple buffers",
          "status": "fail"
      }

  # handshake canonicalization
  handshake_canonicalized = os.path.join(
      out_dir, "handshake_canonicalized.mlir")
  with open(handshake_canonicalized, "w") as f:
    result = subprocess.run([
        DYNAMATIC_OPT_BIN, handshake_buffered,
        "--handshake-canonicalize",
        "--handshake-hoist-ext-instances",
        "--handshake-reshape-channels"
    ],
        stdout=f,
        stderr=sys.stdout
    )
    if result.returncode == 0:
      print("Canonicalized handshake")
    else:
      return {
          "id": id,
          "msg": "Failed to canonicalize Handshake",
          "status": "fail"
      }

  # Speculation
  handshake_speculation = os.path.join(out_dir, "handshake_speculation.mlir")
  spec_json = os.path.join(c_file_dir, "spec.json")
  with open(handshake_speculation, "w") as f:
    result = subprocess.run([
        DYNAMATIC_OPT_BIN, handshake_canonicalized,
        f"--handshake-speculation=json-path={spec_json}",
        "--handshake-materialize",
        "--handshake-canonicalize"
    ],
        stdout=f,
        stderr=sys.stdout
    )
    if result.returncode == 0:
      print("Added speculative units")
    else:
      return {
          "id": id,
          "msg": "Failed to add speculative units",
          "status": "fail"
      }

  return {
      "id": id,
      "msg": "Test passed",
      "status": "pass"
  }


def main():
  """
  Entry point for the script.
  """

  result = run_test(INTEGRATION_FOLDER / "single_loop" /
                    "single_loop.c", 0, 10)
  if result["status"] == "pass":
    color_print(result["msg"], TermColors.OKGREEN)
  elif result["status"] == "fail":
    color_print(result["msg"], TermColors.FAIL)
  else:
    color_print(result["msg"], TermColors.WARNING)


if __name__ == "__main__":
  main()
