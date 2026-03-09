#!/usr/bin/env python3
"""
Dynamatic performance/resource-usage evaluation script.

Builds Dynamatic, then runs the embedded .dyn script for each kernel,
collecting exit codes and logging failures. Supports parallel execution through
the -j flag.
"""

import argparse
import logging
import shutil
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

# ──────────────────────────────────────────────────────────────────────────────
# Kernel list
# ──────────────────────────────────────────────────────────────────────────────
KERNELS = [
    "atax",
    "atax_float",
    "bicg",
    "bicg_float",
    "covariance",
    "gaussian",
    "gemver",
    "gemver_float",
    "get_tanh",
    "histogram",
    "insertion_sort",
    "jacobi_1d_imper",
    "kernel_2mm",
    "kernel_2mm_float",
    "kernel_3mm",
    "kernel_3mm_float",
    "kmp",
    "loop_array",
    "lu",
    "matching",
    "matching_2",
    "matrix_power",
    "pivot",
    "polyn_mult",
    "symm_float",
    "syr2k_float",
    "threshold",
    "triangular",
    "while_loop_1",
]

# ──────────────────────────────────────────────────────────────────────────────
# .dyn script template: {src} is substituted with the kernel source path
# ──────────────────────────────────────────────────────────────────────────────
DYN_SCRIPT = """\
set-src {src}
compile --buffer-algorithm fpga20
write-hdl --hdl vhdl
simulate
synthesize
exit
"""

# ──────────────────────────────────────────────────────────────────────────────
# Paths
# ──────────────────────────────────────────────────────────────────────────────
REPO_ROOT = Path(__file__).resolve().parents[2]


def setup_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)-8s  %(message)s",
        datefmt="%H:%M:%S",
        stream=sys.stdout,
    )


def build() -> None:
    """Run ninja to build Dynamatic."""
    logging.info("Building Dynamatic...")
    result = subprocess.run(
        ["ninja", "-C", "build"],
        cwd=REPO_ROOT,
    )
    if result.returncode != 0:
        logging.error("Build failed with exit code %d. Aborting...", result.returncode)
        sys.exit(result.returncode)
    logging.info("Build succeeded.")


def run_kernel(kernel: str) -> tuple[str, str | None]:
    """
    Run the .dyn script for *kernel*, writing stdout/stderr directly to
    integration-test/{kernel}/out/dynamatic_{out,err}.txt, and return
    (kernel, failure_reason).  failure_reason is None on success.
    """
    logging.info("Running kernel %s...", kernel)
    src = f"integration-test/{kernel}/{kernel}.c"
    script = DYN_SCRIPT.format(src=src)

    out_dir = REPO_ROOT / "integration-test" / kernel / "out"
    # ensure a clean output directory for the kernel
    if out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True)

    out_path = out_dir / "dynamatic_out.txt"
    err_path = out_dir / "dynamatic_err.txt"

    with open(out_path, "w") as out_f, open(err_path, "w") as err_f:
        result = subprocess.run(
            ["bin/dynamatic"],
            input=script,
            text=True,
            stdout=out_f,
            stderr=err_f,
            cwd=REPO_ROOT,
        )

    if result.returncode != 0:
        reason = f"exit code {result.returncode}"
        logging.error("[FAIL] %s (%s)", kernel, reason)
        return kernel, reason

    # Check stdout for FATAL messages
    if "FATAL" in out_path.read_text(errors="replace"):
        reason = "FATAL in stdout"
        logging.error("[FAIL] %s (%s)", kernel, reason)
        return kernel, reason

    # Check simulation report
    report_path = out_dir / "sim" / "report.txt"
    if not report_path.exists():
        reason = "sim/report.txt not found"
        logging.error("[FAIL] %s (%s)", kernel, reason)
        return kernel, reason

    report_text = report_path.read_text(errors="replace")
    if "C and VHDL outputs match" not in report_text:
        reason = 'sim/report.txt missing "C and VHDL outputs match"'
        logging.error("[FAIL] %s (%s)", kernel, reason)
        return kernel, reason

    logging.info("[PASS] %s", kernel)
    return kernel, None


def main() -> None:
    setup_logging()

    parser = argparse.ArgumentParser(
        description="Run Dynamatic evaluation for a list of kernels."
    )
    parser.add_argument(
        "-j",
        "--jobs",
        type=int,
        default=1,
        metavar="JOBS",
        help="Number of kernels to run in parallel (default: 1).",
    )
    parser.add_argument(
        "-o",
        "--output-dir",
        type=Path,
        default=None,
        metavar="DIR",
        help="Copy kernel out/ directories to DIR/{kernel}/out after each run.",
    )
    args = parser.parse_args()

    if args.output_dir is not None:
        if args.output_dir.exists():
            logging.error("Output directory %s already exists. Aborting...", args.output_dir)
            sys.exit(1)
        args.output_dir.mkdir(parents=True)

    build()

    logging.info("Running %d kernel(s) with %d parallel job(s)...", len(KERNELS), args.jobs)
    start_time = time.time()

    passed: list[str] = []
    failed: list[tuple[str, str | None]] = []

    with ThreadPoolExecutor(max_workers=args.jobs) as executor:
        futures = {executor.submit(run_kernel, k): k for k in KERNELS}
        for future in as_completed(futures):
            kernel, failure_reason = future.result()
            if failure_reason is None:
                passed.append(kernel)
            else:
                failed.append((kernel, failure_reason))

            if args.output_dir is not None:
                src = REPO_ROOT / "integration-test" / kernel / "out"
                dst = args.output_dir / kernel / "out"
                shutil.move(src, dst)

    # Summary
    elapsed = int(time.time() - start_time) // 60
    hours = elapsed // 60
    minutes = elapsed % 60
    logging.info("Total time: %dh %02dm.", hours, minutes)

    total = len(KERNELS)
    logging.info(
        "Results: %d/%d passed, %d failed.",
        len(passed),
        total,
        len(failed),
    )
    if failed:
        logging.error(
            "Failed kernels: %s",
            ", ".join(f"{k} ({reason})" for k, reason in sorted(failed)),
        )
        sys.exit(1)


if __name__ == "__main__":
    main()
