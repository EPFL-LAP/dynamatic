"""
Script for running Dynamatic speculative integration tests.
"""
import os
import shutil
import subprocess
import sys
import argparse
import json
from pathlib import Path

DYNAMATIC_ROOT = Path(__file__).parent.parent.parent.parent
INTEGRATION_FOLDER = DYNAMATIC_ROOT / "integration-test"

POLYGEIST_CLANG_BIN = DYNAMATIC_ROOT / "bin" / "cgeist"
LLVM_BINS = DYNAMATIC_ROOT / "polygeist" / "llvm-project" / "build" / "bin"
CLANGXX_BIN = DYNAMATIC_ROOT / "bin" / "clang++"
DYNAMATIC_OPT_BIN = DYNAMATIC_ROOT / "build" / "bin" / "dynamatic-opt"
DYNAMATIC_PROFILER_BIN = DYNAMATIC_ROOT / "bin" / "exp-frequency-profiler"
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


def fail(id, msg):
    return {
        "id": id,
        "msg": msg,
        "status": "fail"
    }


def run_test(c_file, n, variable, copy_cf):
    """
    Runs the specified integration test.
    """

    print(c_file)

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

    cf_dyn_transformed = os.path.join(comp_out_dir, "cf_dyn_transformed.mlir")
    # if copy_cf:
    # Copy cf_dyn_transformed.mlir to comp_out_dir
    cf_file = os.path.join(c_file_dir, "cf_dyn_transformed.mlir")
    shutil.copy(cf_file, cf_dyn_transformed)
    print(f"Copied {cf_file} to {cf_dyn_transformed}")
    # else:
    # Custom compilation flow
    #     clang_file = os.path.join(comp_out_dir, f"clang.ll")
    #     with open(clang_file, "w") as f:
    #         result = subprocess.run([
    #             LLVM_BINS / "clang", "-O0", "-S", "-emit-llvm", c_file,
    #             "-I", DYNAMATIC_ROOT / "include",
    #             "-Xclang", "-ffp-contract=off",
    #             "-o", clang_file
    #         ],
    #             stdout=f,
    #             stderr=sys.stdout
    #         )
    #         if result.returncode == 0:
    #             print("Compiled C file to LLVM IR")
    #         else:
    #             color_print("Failed to compile C file to LLVM IR",
    #                         TermColors.FAIL)
    #             return False

    #     clang_optnone_removed = os.path.join(
    #         comp_out_dir, "clang_optnone_removed.ll")
    #     with open(clang_optnone_removed, "w") as f:
    #         result = subprocess.run([
    #             "sed", "s/optnone//g", clang_file
    #         ],
    #             stdout=f,
    #             stderr=sys.stdout
    #         )
    #         if result.returncode == 0:
    #             print("Removed optnone flgas")
    #         else:
    #             color_print(
    #                 "Failed to remove optnone flags", TermColors.FAIL)
    #             return False

    #     clang_optimized = os.path.join(comp_out_dir, "clang_optimized.ll")
    #     with open(clang_optimized, "w") as f:
    #         result = subprocess.run([
    #             LLVM_BINS / "opt", "-S",
    #             "-passes=mem2reg,instcombine,loop-rotate,consthoist,simplifycfg",
    #             "-strip-debug",
    #             clang_optnone_removed
    #         ],
    #             stdout=f,
    #             stderr=sys.stdout
    #         )
    #         if result.returncode == 0:
    #             print("Optimized LLVM IR")
    #         else:
    #             color_print("Failed to optimize LLVM IR", TermColors.FAIL)
    #             return False

    #     mlir_file = os.path.join(comp_out_dir, "translated.mlir")
    #     with open(mlir_file, "w") as f:
    #         result = subprocess.run([
    #             LLVM_BINS / "mlir-translate",
    #             "--import-llvm",
    #             clang_optimized
    #         ],
    #             stdout=f,
    #             stderr=sys.stdout
    #         )
    #         if result.returncode == 0:
    #             print("Translated LLVM IR to MLIR")
    #         else:
    #             color_print("Failed to translate LLVM IR to MLIR",
    #                         TermColors.FAIL)
    #             return False

    #     dropped_funcs = os.path.join(comp_out_dir, "dropped_funcs.mlir")
    #     with open(dropped_funcs, "w") as f:
    #         result = subprocess.run([
    #             DYNAMATIC_ROOT / "build" / "bin" / "drop-functions", mlir_file,
    #             "--func=main",
    #             "--func=rand",
    #             "--func=srand",
    #             f"-o={dropped_funcs}"
    #         ])
    #         if result.returncode == 0:
    #             print("Dropped functions")
    #         else:
    #             color_print("Failed to drop functions", TermColors.FAIL)
    #             return False

    #     remove_polygeist_attr = os.path.join(
    #         comp_out_dir, "removed_polygeist_attr.mlir")
    #     with open(remove_polygeist_attr, "w") as f:
    #         result = subprocess.run([
    #             DYNAMATIC_OPT_BIN, dropped_funcs,
    #             "--remove-polygeist-attributes",
    #             "--allow-unregistered-dialect"
    #         ],
    #             stdout=f,
    #             stderr=sys.stdout
    #         )
    #         if result.returncode == 0:
    #             print("Removed polygeist attrs")
    #         else:
    #             color_print("Failed to remove polygeist attrs",
    #                         TermColors.FAIL)
    #             return False

    #     cf_file = os.path.join(comp_out_dir, "cf.mlir")
    #     with open(cf_file, "w") as f:
    #         result = subprocess.run([
    #             DYNAMATIC_OPT_BIN, remove_polygeist_attr,
    #             f"--convert-llvm-to-cf=source={c_file} dynamatic-path={DYNAMATIC_ROOT}",
    #             "--remove-polygeist-attributes"
    #         ],
    #             stdout=f,
    #             stderr=sys.stdout
    #         )
    #         if result.returncode == 0:
    #             print("Removed polygeist attrs")
    #         else:
    #             color_print("Failed to remove polygeist attrs",
    #                         TermColors.FAIL)
    #             return False

    #     cf_file_2 = os.path.join(comp_out_dir, "cf_2.mlir")
    #     with open(cf_file_2, "w") as f:
    #         result = subprocess.run([
    #             DYNAMATIC_OPT_BIN, cf_file,
    #             f"--func-set-arg-names=source={c_file}",
    #             "--mark-memory-dependencies",
    #             "--flatten-memref-row-major",
    #             "--mark-memory-interfaces"
    #         ],
    #             stdout=f,
    #             stderr=sys.stdout
    #         )
    #         if result.returncode == 0:
    #             print("Applied standard transformations to cf")
    #         else:
    #             color_print(
    #                 "Failed to apply standard transformations to cf", TermColors.FAIL)
    #             return False

    #     # cf transformations (standard)
    #     cf_transformed = os.path.join(comp_out_dir, "cf_transformed.mlir")
    #     with open(cf_transformed, "w") as f:
    #         result = subprocess.run([
    #             DYNAMATIC_OPT_BIN, cf_file_2,
    #             "--canonicalize", "--cse", "--sccp", "--symbol-dce",
    #             "--control-flow-sink", "--loop-invariant-code-motion", "--canonicalize"],
    #             stdout=f,
    #             stderr=sys.stdout
    #         )
    #         if result.returncode == 0:
    #             print("Applied standard transformations to cf")
    #         else:
    #             color_print(
    #                 "Failed to apply standard transformations to cf", TermColors.FAIL)
    #             return False

    #     # cf transformations (dynamatic)
    #     with open(cf_dyn_transformed, "w") as f:
    #         result = subprocess.run([
    #             DYNAMATIC_OPT_BIN, cf_transformed,
    #             "--arith-reduce-strength=max-adder-depth-mul=1",
    #             "--push-constants",
    #             "--mark-memory-interfaces"
    #         ],
    #             stdout=f,
    #             stderr=sys.stdout
    #         )
    #         if result.returncode == 0:
    #             print("Applied Dynamatic transformations to cf")
    #         else:
    #             return fail(id, "Failed to apply Dynamatic transformations to cf")

    # # cf level -> handshake level
    # handshake = os.path.join(comp_out_dir, "handshake.mlir")
    # with open(handshake, "w") as f:
    #     result = subprocess.run([
    #         DYNAMATIC_OPT_BIN, cf_dyn_transformed,
    #         "--lower-cf-to-handshake"
    #     ],
    #         stdout=f,
    #         stderr=sys.stdout
    #     )
    #     if result.returncode == 0:
    #         print("Compiled cf to handshake")
    #     else:
    #         return fail(id, "Failed to compile cf to handshake")

    # # handshake transformations
    # handshake_transformed = os.path.join(
    #     comp_out_dir, "handshake_transformed.mlir")
    # with open(handshake_transformed, "w") as f:
    #     result = subprocess.run([
    #         DYNAMATIC_OPT_BIN, handshake,
    #         "--handshake-analyze-lsq-usage", "--handshake-replace-memory-interfaces",
    #         "--handshake-minimize-cst-width", "--handshake-optimize-bitwidths",
    #         "--handshake-materialize", "--handshake-infer-basic-blocks",
    #         "--handshake-canonicalize"
    #     ],
    #         stdout=f,
    #         stderr=sys.stdout
    #     )
    #     if result.returncode == 0:
    #         print("Applied transformations to handshake")
    #     else:
    #         return fail(id, "Failed to apply transformations to handshake")

    # # Speculation
    # handshake_speculation = os.path.join(
    #     comp_out_dir, "handshake_speculation.mlir")
    # with open(handshake_speculation, "w") as f:
    #     print(f"n={n}, variable={variable}")
    #     json_path = os.path.join(c_file_dir, "specv2.json")
    #     result = subprocess.run([
    #         DYNAMATIC_OPT_BIN, handshake_transformed,
    #         f"--handshake-speculation-v2=json-path={json_path} n={n} {"variable" if variable else ""}",
    #         "--handshake-materialize",
    #         "--handshake-canonicalize"
    #     ],
    #         stdout=f,
    #         stderr=sys.stdout
    #     )
    #     if result.returncode == 0:
    #         print("Added speculative units")
    #     else:
    #         return fail(id, "Failed to add speculative units")

    shutil.copy(os.path.join(c_file_dir, "handshake_speculation.mlir"),
                handshake_speculation)

    # Buffer placement (fpga20)
    # profiler_bin = os.path.join(comp_out_dir, "profile")
    # result = subprocess.run([
    #     CLANGXX_BIN, c_file,
    #     "-D", "PRINT_PROFILING_INFO",
    #     "-I", str(DYNAMATIC_ROOT / "include"),
    #     "-Wno-deprecated",
    #     "-o", profiler_bin
    # ])
    # if result.returncode == 0:
    #     print("Built kernel for profiling")
    # else:
    #     return fail(id, "Failed to place simple buffers")

    # profiler_inputs = os.path.join(comp_out_dir, "profiler-inputs.txt")
    # with open(profiler_inputs, "w") as f:
    #     result = subprocess.run([profiler_bin],
    #                             stdout=f,
    #                             stderr=sys.stdout
    #                             )
    #     if result.returncode == 0:
    #         print("Ran kernel for profiling")
    #     else:
    #         return fail(id, "Failed to kernel for profiling")

    frequencies = os.path.join(comp_out_dir, "frequencies.csv")
    # with open(frequencies, "w") as f:
    #     result = subprocess.run([
    #         DYNAMATIC_PROFILER_BIN, cf_dyn_transformed,
    #         "--top-level-function=" + kernel_name,
    #         "--input-args-file=" + profiler_inputs,
    #     ],
    #         stdout=f,
    #         stderr=sys.stdout
    #     )
    #     if result.returncode == 0:
    #         print("Profiled cf-level")
    #     else:
    #         return fail(id, "Failed to profile cf-level")
    shutil.copy(os.path.join(c_file_dir, "frequencies.csv"), frequencies)

    # Buffer placement (FPGA20)
    handshake_buffered = os.path.join(comp_out_dir, "handshake_buffered.mlir")
    timing_model = DYNAMATIC_ROOT / "data" / "components-flopoco.json"
    with open(handshake_buffered, "w") as f:
        result = subprocess.run([
            DYNAMATIC_OPT_BIN, handshake_speculation,
            "--handshake-set-buffering-properties=version=fpga20",
            f"--handshake-place-buffers=algorithm=fpga20 frequencies={frequencies} timing-models={timing_model} target-period=20.000 timeout=300 dump-logs"
        ],
            stdout=f,
            stderr=sys.stdout
        )
        if result.returncode == 0:
            print("Placed simple buffers")
        else:
            return fail(id, "Failed to place simple buffers")

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
            return fail(id, "Failed to canonicalize Handshake")

    # buffer_json = os.path.join(c_file_dir, "bufferv2.json")
    handshake_export = os.path.join(comp_out_dir, "handshake_export.mlir")
    shutil.copy(handshake_canonicalized, handshake_export)
    # with open(buffer_json, "r") as f:
    #     buffers = json.load(f)
    #     buffer_pass_args = []
    #     for buffer in buffers:
    #         buffer_pass_args.append(
    #             "--handshake-placebuffers-custom=" +
    #             f"pred={buffer['pred']} " +
    #             f"outid={buffer['outid']} " +
    #             f"slots={buffer['slots']} " +
    #             f"type={buffer['type']}")
    #     with open(handshake_export, "w") as f:
    #         result = subprocess.run([
    #             DYNAMATIC_OPT_BIN, handshake_canonicalized,
    #             *buffer_pass_args
    #         ],
    #             stdout=f,
    #             stderr=sys.stdout
    #         )
    #         if result.returncode == 0:
    #             print("Exported Handshake")
    #         else:
    #             return fail(id, "Failed to export Handshake")

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
            return fail(id, "Failed to export dot file")

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
            return fail(id, "Failed to create PNG file")

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
            return fail(id, "Failed to lower handshake to hw")

    # Export hdl
    hdl_dir = os.path.join(out_dir, "hdl")

    result = subprocess.run([
        EXPORT_RTL_BIN, hw, hdl_dir, RTL_CONFIG,
        "--dynamatic-path", DYNAMATIC_ROOT, "--hdl", "vhdl"
    ])
    if result.returncode == 0:
        print("Exported hdl")
    else:
        return fail(id, "Failed to export hdl")

    # Simulate
    print("Simulator launching")
    result = subprocess.run([
        SIMULATE_SH, DYNAMATIC_ROOT, c_file_dir, out_dir, kernel_name, "", "false"
    ])

    if result.returncode == 0:
        print("Simulation succeeded")
        result = os.path.join(out_dir, "sim/report.txt")
        with open(result, "r") as f:
            report = f.read()
            # Match Latency = \d+ cycles
            latency_match = report.split("Latency = ")
            if len(latency_match) > 1:
                latency = latency_match[1].split(" cycles")[0]
                print(f"Latency: {latency} cycles")
    else:
        return fail(id, "Failed to simulate")

    return {
        "id": id,
        "msg": "Test passed",
        "status": "pass"
    }


def main():
    """
    Entry point for the script.
    """

    parser = argparse.ArgumentParser(
        description="Run speculation integration test")
    parser.add_argument(
        "test_name", type=str, help="Name of the test to run")
    parser.add_argument(
        "--n", type=int, default=3,
        help="Number of iterations to speculate")
    parser.add_argument(
        "--variable", action='store_true',
        help="Run variable speculation")
    parser.add_argument(
        "--copy-cf", action='store_true',
        help="Copy cf_dyn_transformed.mlir")

    args = parser.parse_args()
    test_name = args.test_name
    n = args.n
    variable = args.variable
    copy_cf = args.copy_cf

    result = run_test(INTEGRATION_FOLDER / test_name /
                      f"{test_name}.c", n, variable, copy_cf)
    if result["status"] == "pass":
        color_print(result["msg"], TermColors.OKGREEN)
    elif result["status"] == "fail":
        color_print(result["msg"], TermColors.FAIL)
    else:
        color_print(result["msg"], TermColors.WARNING)


if __name__ == "__main__":
    main()
