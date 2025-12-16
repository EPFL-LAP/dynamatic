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

def bb_pair(s):
    a, b = s.split(",")
    return (a, b)

def main():
    """
    Entry point for the script.
    """

    parser = argparse.ArgumentParser(
        description="Run speculation integration test")
    parser.add_argument(
        "test_name", type=str, help="Name of the test to run")
    parser.add_argument(
        "--one-sided", action='store_true',
        help="Run one-sided speculation")
    parser.add_argument(
        "--emulate-prediction", action='store_true')
    parser.add_argument(
        "--out", type=str, help="out dir name (Default: out)", default="out")
    parser.add_argument(
        "--cp", type=str, help="clock period", required=True)
    parser.add_argument(
        "--use-prof-cache", action='store_true', help="Use profiling cache")
    parser.add_argument(
        "--branch-bb", type=int)
    parser.add_argument(
        "--merge-bb", type=int)
    parser.add_argument(
        "--prioritized-side", type=int)
    parser.add_argument(
        "--steps-until", type=int, default=3)
    parser.add_argument(
        "--disable-spec", action='store_true',
        help="Disable speculation")
    parser.add_argument(
        "--rewrite-a-only", action='store_true',
        help="Use rewrite a only")
    parser.add_argument(
        "--on-merges", action='store_true',
        help="Use naive buffering algorithm")
    parser.add_argument(
        "--factor", type=int,
        help="Unroll factor")

    parser.add_argument(
        "--rewrite-a-bbs",
        action="append",
        type=bb_pair,
        metavar="BB1,BB2",
        help="Apply rewrite A to the given (bb1, bb2) pair. Repeatable."
    )

    parser.add_argument(
        "--skip-merge-bitwidth-opt",
        action="store_true",
        help="Don't optimize merge bitwidths."
    )

    args = parser.parse_args()
    test_name = args.test_name

    c_file = INTEGRATION_FOLDER / test_name / f"{test_name}.c"
    # if result["status"] == "pass":
    #     color_print(result["msg"], TermColors.OKGREEN)
    # elif result["status"] == "fail":
    #     color_print(result["msg"], TermColors.FAIL)
    # else:
    #     color_print(result["msg"], TermColors.WARNING)

    print(c_file)

    # Get the c_file directory
    c_file_dir = os.path.dirname(c_file)
    kernel_name = os.path.splitext(os.path.basename(c_file))[0]

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
            color_print("Failed to compile C file to LLVM IR",
                        TermColors.FAIL)
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
            color_print("Failed to translate LLVM IR to MLIR",
                        TermColors.FAIL)
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
            color_print("Failed to remove polygeist attrs",
                        TermColors.FAIL)
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
            color_print("Failed to remove polygeist attrs",
                        TermColors.FAIL)
            return False

    cf_file_2 = os.path.join(comp_out_dir, "cf_2.mlir")
    with open(cf_file_2, "w") as f:
        result = subprocess.run([
            DYNAMATIC_OPT_BIN, cf_file,
            f"--func-set-arg-names=source={transformed_code}",
            "--mark-memory-dependencies",
            "--flatten-memref-row-major",
            "--force-memory-interface=force-mc"
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
            "--force-memory-interface=force-mc"
        ],
            stdout=f,
            stderr=sys.stdout
        )
        if result.returncode == 0:
            print("Applied Dynamatic transformations to cf")
        else:
            return fail(id, "Failed to apply Dynamatic transformations to cf")

    # gate binarization
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
            return fail(id, "Failed to apply gate binarization")

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
            return fail(id, "Failed to compile cf to handshake")

    skip_merge_bitwidth_opt = f"skip-merges={'true' if args.skip_merge_bitwidth_opt else 'false'}"
    # handshake transformations
    handshake_transformed = os.path.join(
        comp_out_dir, "handshake_transformed.mlir")
    with open(handshake_transformed, "w") as f:
        result = subprocess.run([
            DYNAMATIC_OPT_BIN, handshake,
            "--handshake-analyze-lsq-usage", "--handshake-replace-memory-interfaces",
            "--handshake-minimize-cst-width", f"--handshake-optimize-bitwidths={skip_merge_bitwidth_opt}",
            "--handshake-materialize", "--handshake-infer-basic-blocks",
            # "--handshake-canonicalize"
        ],
            stdout=f,
            stderr=sys.stdout
        )
        if result.returncode == 0:
            print("Applied transformations to handshake")
        else:
            return fail(id, "Failed to apply transformations to handshake")

    # Profiling
    run_prof = True
    frequencies = os.path.join(c_file_dir, "frequencies-cache.csv")
    if args.use_prof_cache:
        if os.path.isfile(frequencies):
            run_prof = False
            print("Using existing profiling cache")
        else:
            run_prof = True
            print("Prof file does not exist; running profiler")

    if run_prof:
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
            return fail(id, "Failed to place simple buffers")

        profiler_inputs = os.path.join(comp_out_dir, "profiler-inputs.txt")
        with open(profiler_inputs, "w") as f:
            result = subprocess.run([profiler_bin],
                                    stdout=f,
                                    stderr=sys.stdout
                                    )
            if result.returncode == 0:
                print("Ran kernel for profiling")
            else:
                return fail(id, "Failed to kernel for profiling")

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
                return fail(id, "Failed to profile cf-level")

    if args.disable_spec:
        updated_frequencies = frequencies
        handshake_speculation = handshake_transformed
    else:
        # apply rewrite sequence with either rewrite c or rewrite b
        if not args.rewrite_a_only:
            handshake_pre_speculation = os.path.join(
                comp_out_dir, "handshake_pre_speculation.mlir")
            with open(handshake_pre_speculation, "w") as f:
                # manually convert to commit-based control flow
                # since the automatic algorithm for this is not finished
                result = subprocess.run([
                    DYNAMATIC_OPT_BIN, handshake_transformed,
                    f"--handshake-pre-spec-v2-gamma=branch-bb={args.branch_bb} merge-bb={args.merge_bb}",
                    "--handshake-materialize",
                ],
                    stdout=f,
                    stderr=sys.stdout
                )
                if result.returncode == 0:
                    print("Added speculative units")
                else:
                    return fail(id, "Failed to add speculative units")

            handshake_speculation = os.path.join(
                comp_out_dir, "handshake_speculation.mlir")
            bb_mapping = os.path.join(comp_out_dir, "bb_mapping.csv")
            with open(handshake_speculation, "w") as f:
                result = subprocess.run([
                    DYNAMATIC_OPT_BIN, handshake_pre_speculation,
                    f"--handshake-spec-v2-gamma=bb-mapping={bb_mapping} branch-bb={args.branch_bb} merge-bb={args.merge_bb} prioritized-side={args.prioritized_side} steps-until={args.steps_until} {"one-sided" if args.one_sided else ""} {"emulate-prediction" if args.emulate_prediction else ""}",
                    "--handshake-materialize",
                ],
                    stdout=f,
                    stderr=sys.stdout
                )
                if result.returncode == 0:
                    print("Added speculative units")
                else:
                    return fail(id, "Failed to add speculative units")

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
                    return fail(id, "Failed to update frequencies.csv")
                
        # only apply rewrite a
        else:
            # if only applying rewrite a, we don't do the "maximizing eager approach"
            # we take a targetted list of where to apply eager execution

            # targeted conversion of branches to suppressors 
            pre_spec_in = handshake_transformed

            for i, (branch_bb, merge_bb) in enumerate(args.rewrite_a_bbs):
                pre_spec_out = os.path.join(
                    comp_out_dir, f"handshake_pre_speculation_{i}.mlir")

                print(f"Running pre-spec {i}")
                with open(pre_spec_out, "w") as f:
                    command = [
                        DYNAMATIC_OPT_BIN, pre_spec_in,
                        f"--handshake-pre-spec-v2-gamma-rewrite-a=branch-bb={branch_bb}",
                        "--handshake-materialize",
                    ]
                    r = subprocess.run(list(command), stdout=f, stderr=sys.stdout)
                    if r.returncode != 0:
                        return fail(id, "Failed pre-speculation")

                pre_spec_in = pre_spec_out  # advance to numbered output

            # targeted movement of suppressors using rewrite a
            post_spec_in = pre_spec_in
            freq_in = frequencies  # initial frequencies

            for i, (branch_bb, merge_bb) in enumerate(args.rewrite_a_bbs):
                post_spec_out = os.path.join(
                    comp_out_dir, f"handshake_speculation_{i}.mlir")
                bb_mapping = os.path.join(comp_out_dir, f"bb_mapping_{i}.csv")

                print(f"Running post-spec {i}")
                with open(post_spec_out, "w") as f:
                    command = [DYNAMATIC_OPT_BIN, post_spec_in,
                        f"--handshake-spec-v2-gamma-rewrite-a=bb-mapping={bb_mapping} branch-bb={branch_bb} merge-bb={merge_bb}",
                        "--handshake-materialize"]
                    r = subprocess.run(list(command), stdout=f, stderr=sys.stdout)
                    if r.returncode != 0:
                        return fail(id, "Failed speculation")

                post_spec_in = post_spec_out

                updated_frequencies = os.path.join(
                    comp_out_dir, f"updated_frequencies_{i}.csv")

                with open(updated_frequencies, "w") as f:
                    r = subprocess.run([
                        "python3",
                        DYNAMATIC_ROOT / "experimental/tools/integration/update_frequencies.py",
                        "--frequencies=" + freq_in,
                        "--mapping=" + bb_mapping
                    ], stdout=f, stderr=sys.stdout)

                freq_in = updated_frequencies  # chain

                if r.returncode != 0:
                    return fail(id, "Failed to update frequencies.csv")
                
            handshake_speculation = post_spec_out


    # Buffer placement (FPGA20)
    handshake_buffered = os.path.join(comp_out_dir, "handshake_buffered.mlir")
    buffer_algorithm = "fpga20"
    if args.on_merges:
        buffer_algorithm = "on-merges"
        
    timing_model = DYNAMATIC_ROOT / "data" / "components.json"
    with open(handshake_buffered, "w") as f:
        result = subprocess.run([
            DYNAMATIC_OPT_BIN, handshake_speculation,
            "--handshake-set-buffering-properties=version=fpga20",
            f"--handshake-place-buffers=algorithm={buffer_algorithm} frequencies={updated_frequencies} timing-models={timing_model} target-period={args.cp} timeout=1000 dump-logs"
        ],
            stdout=f,
            stderr=sys.stdout,
            cwd=comp_out_dir
        )
        if result.returncode == 0:
            print("Placed simple buffers")
        else:
            return fail(id, "Failed to place simple buffers")

    if not args.rewrite_a_only:
        handshake_gamma_post_buffering = os.path.join(
            comp_out_dir, "handshake_gamma_post_buffering.mlir")
        with open(handshake_gamma_post_buffering, "w") as f:
            result = subprocess.run([
                DYNAMATIC_OPT_BIN, handshake_buffered,
                f"--handshake-spec-v2-gamma-post-buffering=prioritized-side={args.prioritized_side}"
            ],
                stdout=f,
                stderr=sys.stdout,
                cwd=comp_out_dir
            )
            if result.returncode == 0:
                print("gamma post buffering")
            else:
                return fail(id, "Failed gamma post buffering")
    else:
        handshake_gamma_post_buffering = handshake_buffered


    # handshake canonicalization
    handshake_canonicalized = os.path.join(
        comp_out_dir, "handshake_canonicalized.mlir")
    with open(handshake_canonicalized, "w") as f:
        result = subprocess.run([
            DYNAMATIC_OPT_BIN, handshake_gamma_post_buffering,
            "--handshake-spec-v2-post-buffering",
            "--handshake-materialize",
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

    handshake_export = os.path.join(comp_out_dir, "handshake_export.mlir")
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


if __name__ == "__main__":
    main()
