"""
Script for running Dynamatic speculative integration tests.
"""
import os
import shutil
import subprocess
import sys
import argparse
import json
import time
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
        "--transformed-code", type=str, help="If we perform code-level transformation, specify the file name (e.g., <kernel_name>_transformed.c)", default=None)
    parser.add_argument(
        "--transformed-cf", type=str, help="cf filename", default=None)
    parser.add_argument(
        "--gate-binarized", type=str, help="gate binarized", default=None)
    parser.add_argument(
        "--out", type=str, help="out dir name (Default: out)", default="out")
    parser.add_argument(
        "--disable-initial-motion", action='store_true',
        help="Disable initial motion of suppressors when loop consists of multiple BBs")
    parser.add_argument(
        "--disable-spec", action='store_true',
        help="Disable speculation")
    parser.add_argument(
        "--cp", type=str, help="clock period", default="10.000")
    parser.add_argument(
        "--use-prof-cache", action='store_true', help="Use profiling cache")
    parser.add_argument(
        "--decide-n", help="Decide n. No generation")
    parser.add_argument(
        "--synth-control", action='store_true')
    parser.add_argument(
        "--exit-eager-eval", action='store_true')
    parser.add_argument(
        "--resolver", action='store_true')
    parser.add_argument(
        "--baseline", action='store_true',
        help="Baseline generation")
    parser.add_argument(
        "--min-buffering", action='store_true')  # TODO: rename to copy-buffers
    parser.add_argument(
        "--on-merges", action='store_true')
    parser.add_argument(
        "--as-copy-src", action='store_true')
    parser.add_argument(
        "--kmp", action='store_true')
    parser.add_argument(
        "--pre_unrolling")
    parser.add_argument(
        "--factor", type=int)

    args = parser.parse_args()
    test_name = args.test_name
    n = args.n
    variable = args.variable
    transformed_code_filename = args.transformed_code

    preUnrolling = args.pre_unrolling

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

    if transformed_code_filename:
        transformed_code = c_file_dir + f"/{transformed_code_filename}"
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
    if args.transformed_cf:
        cf_file = os.path.join(comp_out_dir, "cf.mlir")
        shutil.copy(os.path.join(c_file_dir, args.transformed_cf), cf_file)
    else:
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
            '--force-memory-interface=force-mc=true',
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
            "--force-memory-interface=force-mc=true"
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

    if args.gate_binarized:
        shutil.copy(os.path.join(
            c_file_dir, args.gate_binarized), gate_binarized)

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

    # handshake transformations
    handshake_transformed = os.path.join(
        comp_out_dir, "handshake_transformed.mlir")
    with open(handshake_transformed, "w") as f:
        result = subprocess.run([
            DYNAMATIC_OPT_BIN, handshake,
            "--handshake-analyze-lsq-usage", "--handshake-replace-memory-interfaces",
            "--handshake-minimize-cst-width", "--handshake-optimize-bitwidths",
            "--handshake-materialize", "--handshake-infer-basic-blocks",
            "--handshake-canonicalize"
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

    if args.decide_n is not None:
        original_kernel_name = kernel_name.split("_unrolled")[0]
        decide_n_out = os.path.join(comp_out_dir, "decide_n.txt")
        if args.transformed_code:
            original_transformed = args.transformed_code.split("_unrolled")[
                0] + ".c"
        else:
            original_transformed = f"{original_kernel_name}.c"
        with open(decide_n_out, "w") as f:
            result = subprocess.run([
                "python3", DYNAMATIC_ROOT / "experimental/tools/integration/run_specv2_integration.py",
                original_kernel_name,
                "--out", "out_decide_n_7ns",
                "--cp", "7.00",
                "--resolver",
                "--decide-n", args.decide_n,
                "--only-decide-n",
                "--transformed-code", original_transformed,
            ],
                stdout=f,
                stderr=sys.stdout
            )
            if result.returncode == 0:
                print("Decided n")
            else:
                return fail(id, "Failed to decide n")
        with open(decide_n_out, "r") as f:
            lines = f.readlines()
            n_line = lines[-2]
            n = int(n_line.split(":")[-1].strip())
        print(f"Decided n = {n}")

    spec_json_folder = os.path.join(comp_out_dir, "spec_jsons")
    os.makedirs(spec_json_folder, exist_ok=True)

    if args.disable_spec:
        handshake_post_speculation = handshake_transformed
    else:
        bbs_set = []
        # UPDATE THIS!!!!!
        if "bisection" in kernel_name:
            for i in range(args.factor):
                base = 10*i+2
                bbs_set.append([base, base+1, base+2, base+3])
        elif "single_loop" in kernel_name:
            for i in range(1, args.factor+1):
                bbs_set.append([i])
        elif "kmp" in kernel_name:
            # for i in range(args.factor):
            bbs_set.append([3, 4])
            loop = False
        else:
            print("Kernel not supported for bbs_set generation")
        bbs_set.reverse()
        print(bbs_set)
        prev = handshake_transformed
        passes = []
        for bbs in bbs_set:
            # write specv2.json
            spec_json_path = os.path.join(
                spec_json_folder, f"specv2_{bbs[0]}.json")
            spec_json = {
                "spec-loop-bbs": bbs
            }
            with open(spec_json_path, "w") as spec_f:
                json.dump(spec_json, spec_f, indent=4)
            print(f"Wrote {spec_json_path}")
            passes.extend([
                f"--handshake-pre-spec-v2=json-path={spec_json_path}",
                "--handshake-materialize",
                "--handshake-canonicalize"
            ])

        # Pre-speculation
        handshake_pre_speculation = os.path.join(
            comp_out_dir, f"handshake_pre_speculation_{bbs[0]}.mlir")
        with open(handshake_pre_speculation, "w") as f:
            result = subprocess.run([
                DYNAMATIC_OPT_BIN, prev,
                *passes
            ],
                stdout=f,
                stderr=sys.stdout
            )
            if result.returncode == 0:
                print("Pre-speculation")
            else:
                return fail(id, "Failed on pre-speculation")
        prev = handshake_pre_speculation

        updated_frequencies = frequencies
        if args.baseline:
            handshake_speculation = handshake_pre_speculation
        else:
            prev = handshake_pre_speculation
            prev_updated_frequencies = updated_frequencies
            for bbs in bbs_set:
                spec_json_path = os.path.join(
                    spec_json_folder, f"specv2_{bbs[0]}.json")
                # Speculation
                handshake_speculation = os.path.join(
                    comp_out_dir, f"handshake_speculation_{bbs[0]}.mlir")
                bb_mapping = os.path.join(comp_out_dir, "bb_mapping.csv")
                with open(handshake_speculation, "w") as f:
                    print(f"n={n}, variable={variable}")
                    result = subprocess.run([
                        DYNAMATIC_OPT_BIN, prev,
                        f"--handshake-speculation-v2=json-path={spec_json_path} bb-mapping={bb_mapping} n={n} {"variable" if variable else ""} {"disable-initial-motion" if args.disable_initial_motion else ""} {"exit-eager-eval" if args.exit_eager_eval else ""} {"resolver" if args.resolver else ""}",
                        "--handshake-materialize",
                        "--handshake-canonicalize"
                    ],
                        stdout=f,
                        stderr=sys.stdout
                    )
                    if result.returncode == 0:
                        print("Added speculative units")
                    else:
                        return fail(id, "Failed to add speculative units")

                prev = handshake_speculation

                updated_frequencies = os.path.join(
                    comp_out_dir, f"updated_frequencies_{bbs[0]}.csv")
                with open(updated_frequencies, "w") as f:
                    result = subprocess.run([
                        "python3", DYNAMATIC_ROOT / "experimental/tools/integration/update_frequencies.py",
                        "--frequencies=" + prev_updated_frequencies,
                        "--mapping=" + bb_mapping
                    ],
                        stdout=f,
                        stderr=sys.stdout
                    )
                    if result.returncode == 0:
                        print("Updated frequencies.csv")
                    else:
                        return fail(id, "Failed to update frequencies.csv")
                prev_updated_frequencies = updated_frequencies

        # Post-speculation
        handshake_post_speculation = os.path.join(
            comp_out_dir, "handshake_post_speculation.mlir")
        with open(handshake_post_speculation, "w") as f:
            result = subprocess.run([
                DYNAMATIC_OPT_BIN, handshake_speculation,
                f"--handshake-post-spec-v2",
                "--handshake-materialize",
                "--handshake-canonicalize"
            ],
                stdout=f,
                stderr=sys.stdout
            )
            if result.returncode == 0:
                print("Post-speculation")
            else:
                return fail(id, "Failed on post-speculation")

    # Buffer placement (FPGA20)
    timing_model = DYNAMATIC_ROOT / "data" / "components.json"
    handshake_buffered = os.path.join(comp_out_dir, "handshake_buffered.mlir")
    start = time.time()
    if args.min_buffering:
        if "single_loop" in kernel_name:
            if args.baseline:
                # preUnrolling = f"/home/shundroid/dynamatic/integration-test/{kernel_name}/copy_src_baseline.mlir"
                result = subprocess.run([
                    "python3", DYNAMATIC_ROOT / "experimental" / "tools" /
                    "integration" / "run_specv2_large_integration.py",
                    "single_loop_unrolled_2",
                    "--baseline", "--out", "out_copy_src_baseline", "--factor", "2",
                    "--as-copy-src"
                ],
                    stdout=sys.stdout,
                    stderr=sys.stdout
                )
                if result.returncode == 0:
                    print("Generated copy-src for baseline")
                else:
                    return fail(id, "Failed to generate copy-src for baseline")
                preUnrolling = DYNAMATIC_ROOT / "integration-test" / "single_loop_unrolled_2" / \
                    "out_copy_src_baseline" / "comp" / "handshake_cut_by_buffers.mlir"
            else:
                # preUnrolling = f"/home/shundroid/dynamatic/integration-test/{kernel_name}/copy_src_eager.mlir"
                result = subprocess.run([
                    "python3", DYNAMATIC_ROOT / "experimental" / "tools" /
                    "integration" / "run_specv2_large_integration.py",
                    "single_loop_unrolled_2",
                    "--n", str(n),
                    "--resolver",
                    "--out", "out_copy_src_eager", "--factor", "2",
                    "--as-copy-src"
                ],
                    stdout=sys.stdout,
                    stderr=sys.stdout
                )
                if result.returncode == 0:
                    print("Generated copy-src for baseline")
                else:
                    return fail(id, "Failed to generate copy-src for baseline")
                preUnrolling = DYNAMATIC_ROOT / "integration-test" / "single_loop_unrolled_2" / \
                    "out_copy_src_eager" / "comp" / "handshake_cut_by_buffers.mlir"
        else:
            print("not supported kernel: ", kernel_name)
            return fail(id, "not supported kernel for min-buffering")
        with open(handshake_buffered, "w") as f:
            passes = []
            if "single_loop" in kernel_name:
                for i in range(1, args.factor-1+1):
                    passes.extend([
                        f"--handshake-copy-buffers=pre-unrolling-path={preUnrolling} pre-unrolling-bb=1 post-unrolling-bb={i}",
                    ])
                passes.append(
                    f"--handshake-copy-buffers=pre-unrolling-path={preUnrolling} pre-unrolling-bb=2 post-unrolling-bb={args.factor}")
            elif "bisection" in kernel_name:
                if args.baseline:
                    for i in range(args.factor - 1):
                        for j in range(10):
                            passes.extend([
                                f"--handshake-copy-buffers=pre-unrolling-path={preUnrolling} pre-unrolling-bb={j + 2} post-unrolling-bb={i * 10 + j + 2}",
                            ])
                    last_base = (args.factor - 1) * 10
                    for j in range(10):
                        passes.extend([
                            f"--handshake-copy-buffers=pre-unrolling-path={preUnrolling} pre-unrolling-bb={10 + j + 2} post-unrolling-bb={last_base + j + 2}",
                        ])
                else:
                    for i in range(args.factor - 1):
                        for j in range(7):
                            passes.extend([
                                f"--handshake-copy-buffers=pre-unrolling-path={preUnrolling} pre-unrolling-bb={j + 2} post-unrolling-bb={i * 7 + j + 2}",
                            ])
                    last_base = (args.factor - 1) * 7
                    for j in range(7):
                        passes.extend([
                            f"--handshake-copy-buffers=pre-unrolling-path={preUnrolling} pre-unrolling-bb={7 + j + 2} post-unrolling-bb={last_base + j + 2}",
                        ])
            else:
                print("not supported kernel: ", kernel_name)
                return fail(id, "not supported kernel for min-buffering")
            result = subprocess.run([
                DYNAMATIC_OPT_BIN, handshake_post_speculation,
                *passes
            ],
                stdout=f,
                stderr=sys.stdout,
                cwd=comp_out_dir
            )
            if result.returncode == 0:
                print("Placed simple buffers")
            else:
                return fail(id, "Failed to place simple buffers")
    elif args.on_merges:
        # Buffer placement (on merges)
        with open(handshake_buffered, "w") as f:
            result = subprocess.run([
                DYNAMATIC_OPT_BIN, handshake_post_speculation,
                f"--handshake-place-buffers=algorithm=on-merges frequencies={updated_frequencies} timing-models={timing_model} target-period={args.cp} timeout=7200 dump-logs"
            ],
                stdout=f,
                stderr=sys.stdout,
                cwd=comp_out_dir
            )
            if result.returncode == 0:
                print("Placed on-merges buffers")
            else:
                return fail(id, "Failed to place on-merges buffers")
    else:
        # Buffer placement (FPGA20)
        with open(handshake_buffered, "w") as f:
            result = subprocess.run([
                DYNAMATIC_OPT_BIN, handshake_post_speculation,
                "--handshake-set-buffering-properties=version=fpga20",
                f"--handshake-place-buffers=algorithm=fpga20 frequencies={updated_frequencies} timing-models={timing_model} target-period={args.cp} timeout=7200 dump-logs"
            ],
                stdout=f,
                stderr=sys.stdout,
                cwd=comp_out_dir
            )
            if result.returncode == 0:
                print("Placed simple buffers")
            else:
                return fail(id, "Failed to place simple buffers")
    end = time.time()
    print(f"Buffer placement time: {end - start} seconds")

    if args.as_copy_src:
        if "single_loop" in kernel_name:
            pre_bb = 1
            post_bb = 2
        else:
            print("not supported kernel: ", kernel_name)
            return fail(id, "not supported kernel for copy-src")
        handshake_cut_by_buffers = os.path.join(
            comp_out_dir, "handshake_cut_by_buffers.mlir")
        with open(handshake_cut_by_buffers, "w") as f:
            result = subprocess.run([
                DYNAMATIC_OPT_BIN, handshake_buffered,
                f"--handshake-cut-bbs-by-buffers=pre-bb={pre_bb} post-bb={post_bb}"
            ],
                stdout=f,
                stderr=sys.stdout
            )
            if result.returncode == 0:
                print("Cut by buffers")
            else:
                return fail(id, "Failed to cut by buffers")
        return {
            "id": id,
            "msg": "copy-src generation succeeded",
            "status": "pass"
        }

    # handshake canonicalization
    handshake_canonicalized = os.path.join(
        comp_out_dir, "handshake_canonicalized.mlir")
    with open(handshake_canonicalized, "w") as f:
        result = subprocess.run([
            DYNAMATIC_OPT_BIN, handshake_buffered,
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

    # # Convert DOT graph to PNG
    # png = os.path.join(comp_out_dir, f"{kernel_name}.png")
    # with open(png, "w") as f:
    #     result = subprocess.run([
    #         "dot", "-Tpng", dot
    #     ],
    #         stdout=f,
    #         stderr=sys.stdout
    #     )
    #     if result.returncode == 0:
    #         print("Created PNG file")
    #     else:
    #         return fail(id, "Failed to create PNG file")

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
