"""
Script for running Dynamatic speculative integration tests.
"""
import json
import os
import shutil
import subprocess
import sys
import argparse
from pathlib import Path

DYNAMATIC_ROOT = Path(__file__).parent.parent.parent
INTEGRATION_FOLDER = DYNAMATIC_ROOT / "integration-test"

DYNAMATIC_OPT_BIN = DYNAMATIC_ROOT / "build" / "bin" / "dynamatic-opt"
EXPORT_DOT_BIN = DYNAMATIC_ROOT / "build" / "bin" / "export-dot"
EXPORT_RTL_BIN = DYNAMATIC_ROOT / "build" / "bin" / "export-rtl"
SIMULATE_SH = DYNAMATIC_ROOT / "tools" / \
    "dynamatic" / "scripts" / "simulate.sh"

RTL_CONFIG = DYNAMATIC_ROOT / "data" / "rtl-config-vhdl-beta.json"


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


def run_test(c_file: str, spec: bool) -> bool:
    """
    Runs the specified integration test.
    """

    print("Running", c_file, "Speculation Enabled:", spec)

    # Get the c_file directory
    c_file_dir = os.path.dirname(c_file)
    kernel_name = os.path.splitext(os.path.basename(c_file))[0]

    # Get out dir name
    out_dir = os.path.join(c_file_dir, "out")
    # Remove previous out directory
    if os.path.isdir(out_dir):
        shutil.rmtree(out_dir)
    Path(out_dir).mkdir()

    comp_out_dir = os.path.join(out_dir, "comp")
    Path(comp_out_dir).mkdir()

    # Custom compilation flow

    # Start with copying the .cf file to the out_dir
    cf_file_base = os.path.join(c_file_dir, "cf.mlir")
    cf_file = os.path.join(comp_out_dir, "cf.mlir")
    shutil.copy(cf_file_base, cf_file)

    # cf transformations (standard)
    cf_transformed = os.path.join(comp_out_dir, "cf_transformed.mlir")
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
            color_print(
                "Failed to apply standard transformations to cf", TermColors.FAIL)
            return False

    # cf transformations (dynamatic)
    cf_dyn_transformed = os.path.join(comp_out_dir, "cf_dyn_transformed.mlir")
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
            color_print(
                "Failed to apply Dynamatic transformations to cf", TermColors.FAIL)
            return False

    # cf level -> handshake level
    handshake = os.path.join(comp_out_dir, "handshake.mlir")
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
            color_print("Failed to compile cf to handshake", TermColors.FAIL)
            return False

    # handshake transformations
    handshake_transformed = os.path.join(
        comp_out_dir, "handshake_transformed.mlir")
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
            color_print("Failed to apply transformations to handshake",
                        TermColors.FAIL)
            return False

    # Buffer placement (Simple buffer placement)
    handshake_buffered = os.path.join(comp_out_dir, "handshake_buffered.mlir")
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
            color_print("Failed to place simple buffers", TermColors.FAIL)
            return False

    # handshake canonicalization
    handshake_canonicalized = os.path.join(
        comp_out_dir, "handshake_canonicalized.mlir")
    with open(handshake_canonicalized, "w") as f:
        result = subprocess.run([
            DYNAMATIC_OPT_BIN, handshake_buffered,
            "--handshake-canonicalize",
            "--handshake-hoist-ext-instances"
        ],
            stdout=f,
            stderr=sys.stdout
        )
        if result.returncode == 0:
            print("Canonicalized handshake")
        else:
            color_print("Failed to canonicalize Handshake", TermColors.FAIL)
            return False

    handshake_export = os.path.join(comp_out_dir, "handshake_export.mlir")
    if spec:
        # Speculation
        handshake_speculation = os.path.join(
            comp_out_dir, "handshake_speculation.mlir")
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
                color_print("Failed to add speculative units", TermColors.FAIL)
                return False

        buffer_json = os.path.join(c_file_dir, "buffer.json")
        handshake_export = os.path.join(comp_out_dir, "handshake_export.mlir")
        with open(buffer_json, "r") as f:
            buffers = json.load(f)
            buffer_pass_args = []
            for buffer in buffers:
                buffer_pass_args.append(
                    "--handshake-placebuffers-custom=" +
                    f"pred={buffer['pred']} " +
                    f"outid={buffer['outid']} " +
                    f"slots={buffer['slots']} " +
                    f"type={buffer['type']}")
            with open(handshake_export, "w") as f:
                result = subprocess.run([
                    DYNAMATIC_OPT_BIN, handshake_speculation,
                    *buffer_pass_args
                ],
                    stdout=f,
                    stderr=sys.stdout
                )
                if result.returncode == 0:
                    print("Exported Handshake")
                else:
                    color_print("Failed to export Handshake", TermColors.FAIL)
                    return False
    else:
        shutil.copy(handshake_canonicalized, handshake_export)

    # Export dot file
    dot = os.path.join(comp_out_dir, f"{kernel_name}.dot")
    with open(dot, "w") as f:
        result = subprocess.run([
            EXPORT_DOT_BIN, handshake_export,
            "--edge-style=spline", "--label-type=uname"
        ],
            stdout=f,
            stderr=sys.stdout
        )
        if result.returncode == 0:
            print("Created dot file")
        else:
            color_print("Failed to export dot file", TermColors.FAIL)
            return False

    # Convert DOT graph to PNG
    png = os.path.join(comp_out_dir, f"{kernel_name}.png")
    with open(png, "w") as f:
        result = subprocess.run([
            "dot", "-Tpng", dot
        ],
            stdout=f,
            stderr=sys.stdout
        )
        if result.returncode == 0:
            print("Created PNG file")
        else:
            color_print("Failed to create PNG file", TermColors.FAIL)
            return False

    # handshake level -> hw level
    hw = os.path.join(comp_out_dir, "hw.mlir")
    with open(hw, "w") as f:
        result = subprocess.run([
            DYNAMATIC_OPT_BIN, handshake_export,
            "--lower-handshake-to-hw"
        ],
            stdout=f,
            stderr=sys.stdout
        )
        if result.returncode == 0:
            print("Lowered handshake to hw")
        else:
            color_print("Failed to lower handshake to hw", TermColors.FAIL)
            return False

    # Export hdl
    hdl_dir = os.path.join(out_dir, "hdl")

    result = subprocess.run([
        EXPORT_RTL_BIN, hw, hdl_dir, RTL_CONFIG,
        "--dynamatic-path", DYNAMATIC_ROOT, "--hdl", "vhdl"
    ])
    if result.returncode == 0:
        print("Exported hdl")
    else:
        color_print("Failed to export hdl", TermColors.FAIL)
        return False

    # Simulate
    print("Simulator launching")
    result = subprocess.run([
        SIMULATE_SH, DYNAMATIC_ROOT, c_file_dir, out_dir, kernel_name
    ])

    if result.returncode == 0:
        print("Simulation succeeded")
    else:
        color_print("Failed to simulate", TermColors.FAIL)
        return False

    return True


def main():
    """
    Entry point for the script.
    """

    parser = argparse.ArgumentParser(
        description="Run speculation integration test")
    parser.add_argument(
        "test_name", type=str, help="Name of the test to run")
    parser.add_argument(
        "--disable-spec", action="store_false", dest="spec",
        help="Run without speculation (but with custom compilation flow)")

    args = parser.parse_args()
    test_name = args.test_name
    spec = args.spec

    success = run_test(INTEGRATION_FOLDER / test_name /
                       f"{test_name}.c", spec)
    if success:
        color_print("Test passed", TermColors.OKGREEN)


if __name__ == "__main__":
    main()
