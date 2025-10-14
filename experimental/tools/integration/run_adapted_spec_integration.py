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

DYNAMATIC_ROOT = Path(__file__).parent.parent.parent.parent
INTEGRATION_FOLDER = DYNAMATIC_ROOT / "integration-test"

CLANGXX_BIN = DYNAMATIC_ROOT / "bin" / "clang++"
DYNAMATIC_OPT_BIN = DYNAMATIC_ROOT / "build" / "bin" / "dynamatic-opt"
DYNAMATIC_PROFILER_BIN = DYNAMATIC_ROOT / "bin" / "exp-frequency-profiler"
LLVM_BINS = DYNAMATIC_ROOT / "polygeist" / "llvm-project" / "build" / "bin"
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
    parser.add_argument(
        "--transformed-code", type=str, help="If we perform code-level transformation, specify the file name (e.g., <kernel_name>_transformed.c)", default=None)
    parser.add_argument(
        "--out", type=str, help="out dir name (Default: out)", default="out")
    parser.add_argument(
        "--cp", type=str, help="clock period", default="10.000")
    parser.add_argument(
        "--default-value", type=str, help="default speculated value", default="1")
    parser.add_argument(
        "--disable-constant-predictor", action="store_false")
    parser.add_argument(
        "--loop-bottom-passer-disabled", action="store_true")

    args = parser.parse_args()
    test_name = args.test_name
    spec = args.spec

    c_file = INTEGRATION_FOLDER / test_name / f"{test_name}.c"

    print("Running", c_file, "Speculation Enabled:", spec)

    # Get the c_file directory
    c_file_dir = os.path.dirname(c_file)
    kernel_name = os.path.splitext(os.path.basename(c_file))[0]

    if args.transformed_code:
        transformed_code = c_file_dir + f"/{args.transformed_code}"
    else:
        transformed_code = c_file

    # Get out dir name
    out_dir = os.path.join(c_file_dir, args.out)
    # Remove previous out directory
    if os.path.isdir(out_dir):
        shutil.rmtree(out_dir)
    Path(out_dir).mkdir()

    comp_out_dir = os.path.join(out_dir, "comp")
    Path(comp_out_dir).mkdir()

    # Custom compilation flow

    clang_file = os.path.join(comp_out_dir, f"clang.ll")
    with open(clang_file, "w") as f:
        result = subprocess.run([
            LLVM_BINS / "clang", "-O0", "-S", "-emit-llvm", transformed_code,
            "-I", DYNAMATIC_ROOT / "include",
            "-Xclang", "-ffp-contract=off",
            "-o", clang_file
        ],
            stdout=f,
            stderr=sys.stdout
        )
        if result.returncode == 0:
            print("Compiled C file to LLVM IR")
        else:
            color_print("Failed to compile C file to LLVM IR", TermColors.FAIL)
            return False

    clang_optnone_removed = os.path.join(
        comp_out_dir, "clang_optnone_removed.ll")
    with open(clang_optnone_removed, "w") as f:
        result = subprocess.run([
            "sed", "s/optnone//g", clang_file
        ],
            stdout=f,
            stderr=sys.stdout
        )
        if result.returncode == 0:
            print("Removed optnone flgas")
        else:
            color_print(
                "Failed to remove optnone flags", TermColors.FAIL)
            return False

    clang_optimized = os.path.join(comp_out_dir, "clang_optimized.ll")
    with open(clang_optimized, "w") as f:
        result = subprocess.run([
            LLVM_BINS / "opt", "-S",
            "-passes=mem2reg,instcombine,loop-rotate,consthoist,simplifycfg",
            "-strip-debug",
            clang_optnone_removed
        ],
            stdout=f,
            stderr=sys.stdout
        )
        if result.returncode == 0:
            print("Optimized LLVM IR")
        else:
            color_print("Failed to optimize LLVM IR", TermColors.FAIL)
            return False

    mlir_file = os.path.join(comp_out_dir, "translated.mlir")
    with open(mlir_file, "w") as f:
        result = subprocess.run([
            LLVM_BINS / "mlir-translate",
            "--import-llvm",
            clang_optimized
        ],
            stdout=f,
            stderr=sys.stdout
        )
        if result.returncode == 0:
            print("Translated LLVM IR to MLIR")
        else:
            color_print("Failed to translate LLVM IR to MLIR", TermColors.FAIL)
            return False

    remove_polygeist_attr = os.path.join(
        comp_out_dir, "removed_polygeist_attr.mlir")
    with open(remove_polygeist_attr, "w") as f:
        result = subprocess.run([
            DYNAMATIC_OPT_BIN, mlir_file,
            "--remove-polygeist-attributes",
            f"--drop-unlisted-functions=function-names={kernel_name}",
            "--allow-unregistered-dialect"
        ],
            stdout=f,
            stderr=sys.stdout
        )
        if result.returncode == 0:
            print("Removed polygeist attrs")
        else:
            color_print("Failed to remove polygeist attrs", TermColors.FAIL)
            return False

    cf_file = os.path.join(comp_out_dir, "cf.mlir")
    with open(cf_file, "w") as f:
        result = subprocess.run([
            DYNAMATIC_OPT_BIN, remove_polygeist_attr,
            f"--convert-llvm-to-cf=source={transformed_code} dynamatic-path={DYNAMATIC_ROOT}",
            "--remove-polygeist-attributes"
        ],
            stdout=f,
            stderr=sys.stdout
        )
        if result.returncode == 0:
            print("Removed polygeist attrs")
        else:
            color_print("Failed to remove polygeist attrs", TermColors.FAIL)
            return False

    cf_file_2 = os.path.join(comp_out_dir, "cf_2.mlir")
    with open(cf_file_2, "w") as f:
        result = subprocess.run([
            DYNAMATIC_OPT_BIN, cf_file,
            f"--func-set-arg-names=source={transformed_code}",
            "--mark-memory-dependencies",
            "--flatten-memref-row-major",
            "--mark-memory-interfaces"
        ],
            stdout=f,
            stderr=sys.stdout
        )
        if result.returncode == 0:
            print("Applied standard transformations to cf")
        else:
            color_print(
                "Failed to apply standard transformations to cf", TermColors.FAIL)
            return False

    # cf transformations (standard)
    cf_transformed = os.path.join(comp_out_dir, "cf_transformed.mlir")
    with open(cf_transformed, "w") as f:
        result = subprocess.run([
            DYNAMATIC_OPT_BIN, cf_file_2,
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

    gate_binarized = os.path.join(comp_out_dir, "gate_binarized.mlir")
    with open(gate_binarized, "w") as f:
        result = subprocess.run([
            DYNAMATIC_OPT_BIN, cf_dyn_transformed,
            "--cf-gate-binarization"
        ],
            stdout=f,
            stderr=sys.stdout
        )
        if result.returncode == 0:
            print("Applied gate binarization")
        else:
            color_print("Failed to apply gate binarization", TermColors.FAIL)
            return False

    # cf level -> handshake level
    handshake = os.path.join(comp_out_dir, "handshake.mlir")
    with open(handshake, "w") as f:
        result = subprocess.run([
            DYNAMATIC_OPT_BIN, gate_binarized,
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

    # # Buffer placement (Simple buffer placement)
    # handshake_buffered = os.path.join(comp_out_dir, "handshake_buffered.mlir")
    # timing_model = DYNAMATIC_ROOT / "data" / "components-flopoco.json"
    # with open(handshake_buffered, "w") as f:
    #     result = subprocess.run([
    #         DYNAMATIC_OPT_BIN, handshake_transformed,
    #         "--handshake-set-buffering-properties=version=fpga20",
    #         f"--handshake-place-buffers=algorithm=on-merges timing-models={timing_model}"
    #     ],
    #         stdout=f,
    #         stderr=sys.stdout
    #     )
    #     if result.returncode == 0:
    #         print("Placed simple buffers")
    #     else:
    #         color_print("Failed to place simple buffers", TermColors.FAIL)
    #         return False

    # handshake canonicalization
    handshake_canonicalized = os.path.join(
        comp_out_dir, "handshake_canonicalized.mlir")
    with open(handshake_canonicalized, "w") as f:
        result = subprocess.run([
            DYNAMATIC_OPT_BIN, handshake_transformed,
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

    handshake_adapted = os.path.join(
        comp_out_dir, "handshake_adapted.mlir")
    bb_mapping = os.path.join(comp_out_dir, "bb_mapping.csv")
    spec_json_path = os.path.join(c_file_dir, "specv2.json")
    with open(handshake_adapted, "w") as f:
        result = subprocess.run([
            DYNAMATIC_OPT_BIN, handshake_canonicalized,
            f"--handshake-spec-v1-adaptor=bb-mapping={bb_mapping} json-path={spec_json_path}",
            "--handshake-materialize",
        ],
            stdout=f,
            stderr=sys.stdout
        )
        if result.returncode == 0:
            print("Canonicalized handshake")
        else:
            color_print("Failed to canonicalize Handshake", TermColors.FAIL)
            return False

    # handshake_speculation = os.path.join(comp_out_dir, "handshake_speculation.mlir")
    handshake_speculation = os.path.join(
        comp_out_dir, "handshake_speculation.mlir")
    if spec:
        # Speculation
        spec_json = os.path.join(c_file_dir, "spec.json")
        with open(handshake_speculation, "w") as f:
            result = subprocess.run([
                DYNAMATIC_OPT_BIN, handshake_adapted,
                f"--handshake-speculation-adapted=json-path={spec_json}",
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

    else:
        shutil.copy(handshake_canonicalized, handshake_speculation)

    handshake_post_spec_adapted = os.path.join(
        comp_out_dir, "handshake_post_spec_adapted.mlir")
    with open(handshake_post_spec_adapted, "w") as f:
        result = subprocess.run([
            DYNAMATIC_OPT_BIN, handshake_speculation,
            f"--handshake-spec-v1-post-adaptor={'disable-passer-at-exits' if args.loop_bottom_passer_disabled else ''}",
            "--handshake-materialize",
        ],
            stdout=f,
            stderr=sys.stdout
        )
        if result.returncode == 0:
            print("Post-adapted")
        else:
            color_print("Failed to post-adapt", TermColors.FAIL)
            return False

    handshake_post_spec = os.path.join(
        comp_out_dir, "handshake_post_spec.mlir")
    with open(handshake_post_spec, "w") as f:
        result = subprocess.run([
            DYNAMATIC_OPT_BIN, handshake_post_spec_adapted,
            f"--handshake-post-spec-v2",
            "--handshake-materialize",
        ],
            stdout=f,
            stderr=sys.stdout
        )
        if result.returncode == 0:
            print("Post-spec")
        else:
            color_print("Failed to post-spec", TermColors.FAIL)
            return False

    # Buffer placement (fpga20)
    profiler_bin = os.path.join(comp_out_dir, "profile")
    result = subprocess.run([
        CLANGXX_BIN, transformed_code,
        "-D", "PRINT_PROFILING_INFO",
        "-I", str(DYNAMATIC_ROOT / "include"),
        "-Wno-deprecated",
        "-o", profiler_bin
    ])
    if result.returncode == 0:
        print("Built kernel for profiling")
    else:
        print("Failed to place simple buffers")

    profiler_inputs = os.path.join(comp_out_dir, "profiler-inputs.txt")
    with open(profiler_inputs, "w") as f:
        result = subprocess.run([profiler_bin],
                                stdout=f,
                                stderr=sys.stdout
                                )
        if result.returncode == 0:
            print("Ran kernel for profiling")
        else:
            print("Failed to kernel for profiling")

    frequencies = os.path.join(comp_out_dir, "frequencies.csv")
    with open(frequencies, "w") as f:
        result = subprocess.run([
            DYNAMATIC_PROFILER_BIN, gate_binarized,
            "--top-level-function=" + kernel_name,
            "--input-args-file=" + profiler_inputs,
        ],
            stdout=f,
            stderr=sys.stdout
        )
        if result.returncode == 0:
            print("Profiled cf-level")
        else:
            print("Failed to profile cf-level")

    updated_frequencies = os.path.join(
        comp_out_dir, "updated_frequencies.csv")
    with open(updated_frequencies, "w") as f:
        result = subprocess.run([
            "python3", DYNAMATIC_ROOT / "experimental/tools/integration/update_frequencies.py",
            "--frequencies=" + frequencies,
            "--mapping=" + bb_mapping
        ],
            stdout=f,
            stderr=sys.stdout
        )
        if result.returncode == 0:
            print("Updated frequencies.csv")
        else:
            color_print("Failed to update frequencies.csv", TermColors.FAIL)
            return False

    # Buffer placement (FPGA20)
    handshake_buffered = os.path.join(comp_out_dir, "handshake_buffered.mlir")
    timing_model = DYNAMATIC_ROOT / "data" / "components.json"
    with open(handshake_buffered, "w") as f:
        result = subprocess.run([
            DYNAMATIC_OPT_BIN, handshake_post_spec,
            "--handshake-set-buffering-properties=version=fpga20",
            f"--handshake-place-buffers=algorithm=fpga20 frequencies={updated_frequencies} timing-models={timing_model} target-period={args.cp} timeout=300 dump-logs"
        ],
            stdout=f,
            stderr=sys.stdout,
            cwd=comp_out_dir
        )
        if result.returncode == 0:
            print("Placed simple buffers")
        else:
            print("Failed to place simple buffers")

    handshake_spec_post_buffer = os.path.join(
        comp_out_dir, "handshake_spec_post_buffer.mlir")
    with open(handshake_spec_post_buffer, "w") as f:
        result = subprocess.run([
            DYNAMATIC_OPT_BIN, handshake_buffered,
            f"--handshake-spec-post-buffer=default-value={args.default_value} {'constant=false' if not args.disable_constant_predictor else ''}",
            "--handshake-materialize"
        ],
            stdout=f,
            stderr=sys.stdout
        )
        if result.returncode == 0:
            print("Created handshake spec post buffer")
        else:
            print("Failed to create handshake spec post buffer")

    handshake_export = os.path.join(comp_out_dir, "handshake_export.mlir")
    buffer_json = os.path.join(c_file_dir, "buffer.json")
    # exist?
    if os.path.exists(buffer_json):
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
                    DYNAMATIC_OPT_BIN, handshake_spec_post_buffer,
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
        shutil.copy(handshake_spec_post_buffer, handshake_export)

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

    # simulate now needs the vivado path + should we use vivado floating point units
    # this should probably be changed
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
                print("Latency not found in report")
    else:
        color_print("Failed to simulate", TermColors.FAIL)
        return False

    return True


if __name__ == "__main__":
    main()
