#!/usr/bin/env python3
"""
Dynamatic performance/resource-usage evaluation script.

Builds Dynamatic, then runs the embedded .dyn script for each kernel,
collecting exit codes and logging failures. Supports parallel execution through
the -j flag.
"""

import argparse
import json
import logging
import re
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
CLOCK_PERIOD = 5
DYN_SCRIPT = """\
set-src {src}
set-clock-period %d
compile --buffer-algorithm fpga20
write-hdl --hdl vhdl
simulate
synthesize
exit
""" % (CLOCK_PERIOD,)

# ──────────────────────────────────────────────────────────────────────────────
# Paths
# ──────────────────────────────────────────────────────────────────────────────
REPO_ROOT = Path(__file__).resolve().parents[2]


# ──────────────────────────────────────────────────────────────────────────────
# Constants
# ──────────────────────────────────────────────────────────────────────────────
TIMEOUT_DYNAMATIC_SECONDS = 60 * 60
TIMEOUT_SYNTH_LSQ_SECONDS = 30 * 60

# ──────────────────────────────────────────────────────────────────────────────
# Data extraction helpers
# ──────────────────────────────────────────────────────────────────────────────


def parse_sim_report(path: Path):
    """Return (passed, cycle_count) from a simulation report.txt."""
    text = path.read_text()
    passed = "C and VHDL outputs match" in text
    m = re.search(r"Simulation done!\s+Latency\s*=\s*(\d+)\s+cycles", text)
    cycle_count = int(m.group(1)) if m else None
    return cycle_count, passed


def parse_utilization(path: Path):
    """Return (slices, luts, ffs) from a utilization report."""
    slices = luts = ffs = None
    for line in path.read_text().splitlines():
        if m := re.match(r"\|\s*Slice LUTs\s*\|\s*(\d+)", line):
            luts = int(m.group(1))
        elif m := re.match(r"\|\s*Slice Registers\s*\|\s*(\d+)", line):
            ffs = int(m.group(1))
        elif m := re.match(r"\|\s*Slice\s*\|\s*(\d+)", line):
            slices = int(m.group(1))
    return slices, luts, ffs


def parse_timing(path: Path):
    """Return (cp_ns, slack_ns, cp_src, cp_dst) from a timing report."""
    cp_ns = slack_ns = cp_src = cp_dst = None
    for line in path.read_text().splitlines():
        if m := re.match(r"\s*Slack\s*\([^)]*\)\s*:\s*(-?[\d.]+)ns", line):
            slack_ns = float(m.group(1))
        elif m := re.match(r"\s*Data Path Delay:\s*([\d.]+)ns", line):
            cp_ns = float(m.group(1))
        elif m := re.match(r"\s*Source:\s*(\S+)", line):
            cp_src = m.group(1)
        elif m := re.match(r"\s*Destination:\s*(\S+)", line):
            cp_dst = m.group(1)
    return cp_ns, slack_ns, cp_src, cp_dst


def extract_synth_data(synth_dir: Path) -> dict | None:
    """Extract synthesis metrics from a synth directory."""
    util_rpt = synth_dir / "utilization_post_pr.rpt"
    timing_rpt = synth_dir / "timing_post_pr.rpt"

    if not util_rpt.exists() and not timing_rpt.exists():
        return None

    assert util_rpt.exists() and timing_rpt.exists(), \
        "Expected both utilization and timing reports to be present if either is present."

    slices, luts, ffs = parse_utilization(util_rpt)
    resources = {"slice": slices, "lut": luts, "ff": ffs}

    cp_ns, slack_ns, cp_src, cp_dst = parse_timing(timing_rpt)
    timing = {"cp_ns": cp_ns, "slack_ns": slack_ns, "cp_src": cp_src, "cp_dst": cp_dst}

    return {"resources": resources, "timing": timing}


def extract_kernel_data(kernel: str, out_dir: Path) -> dict:
    """Extract metrics from a kernel's out/ directory. Missing files yield None values."""
    data: dict = {}

    sim_report = out_dir / "sim" / "report.txt"
    if sim_report.exists():
        passed, cycle_count = parse_sim_report(sim_report)
        assert passed, "should not get here on failure"
        data["simulation"] = {"cycle_count": cycle_count}
    else:
        data["simulation"] = None

    synth_dir = out_dir / "synth"
    if synth_dir.exists():
        data["synth"] = extract_synth_data(out_dir / "synth")

    # Extract LSQ synth data if present
    lsq_synth_dir = out_dir / "synth_lsq"
    if lsq_synth_dir.exists():
        data["synth_lsq"] = {}
        for lsq_dir in lsq_synth_dir.iterdir():
            if lsq_dir.is_dir():
                lsq_data = extract_synth_data(lsq_dir)
                data["synth_lsq"][lsq_dir.name] = lsq_data
        # Aggregate LSQ data across all LSQs for this kernel
        if data["synth_lsq"]:
            # Sum resources across all LSQs
            total_slices = sum(d["resources"]["slice"] for d in data["synth_lsq"].values())
            total_luts = sum(d["resources"]["lut"] for d in data["synth_lsq"].values())
            total_ffs = sum(d["resources"]["ff"] for d in data["synth_lsq"].values())
            resources = {"slice": total_slices, "lut": total_luts, "ff": total_ffs}
            # Find most critical timing across all LSQs (longest critical path)
            critical_cp_ns = max(d["timing"]["cp_ns"] for d in data["synth_lsq"].values())
            critical_slack_ns = min(d["timing"]["slack_ns"] for d in data["synth_lsq"].values())
            timing = {"cp_ns": critical_cp_ns, "slack_ns": critical_slack_ns}
            data["synth_lsq"]["total"] = {"resources": resources, "timing": timing}

    return data


# ──────────────────────────────────────────────────────────────────────────────
# Evaluation runner
# ──────────────────────────────────────────────────────────────────────────────

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


def run_kernel(kernel: str, no_synth: bool, synth_lsqs: bool) -> tuple[str, str | None]:
    """
    Run the .dyn script for *kernel*, writing stdout/stderr directly to
    integration-test/{kernel}/out/dynamatic_{out,err}.txt, and return
    (kernel, failure_reason).  failure_reason is None on success.
    """
    logging.info("Running kernel %s...", kernel)
    src = f"integration-test/{kernel}/{kernel}.c"
    script = DYN_SCRIPT.format(src=src)
    if no_synth:
        # Remove the "synthesize" command from the script
        script = "\n".join(l for l in script.splitlines() if not l.startswith("synthesize")) + "\n"

    out_dir = REPO_ROOT / "integration-test" / kernel / "out"
    # ensure a clean output directory for the kernel
    if out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True)

    out_path = out_dir / "dynamatic_out.txt"
    err_path = out_dir / "dynamatic_err.txt"

    with open(out_path, "w") as out_f, open(err_path, "w") as err_f:
        try:
            result = subprocess.run(
                ["bin/dynamatic"],
                input=script,
                text=True,
                stdout=out_f,
                stderr=err_f,
                cwd=REPO_ROOT,
                timeout=TIMEOUT_DYNAMATIC_SECONDS,
            )
        except subprocess.TimeoutExpired:
            reason = f"Dynamatic timed out after {TIMEOUT_DYNAMATIC_SECONDS // 60} minutes"
            logging.error("[FAIL] %s (%s)", kernel, reason)
            return kernel, reason

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

    # Run out-of-context synthesis for LSQs
    if synth_lsqs:
        hdl_dir = out_dir / "hdl"
        for lsq_file in hdl_dir.glob("*_lsq*_core.vhd"):
            lsq_top = lsq_file.stem
            lsq_name = re.match(r".*_(lsq\d+)_core", lsq_top).group(1)
            lsq_synth_dir = out_dir / "synth_lsq" / lsq_name
            logging.info("Running LSQ synthesis for %s (kernel %s)...", lsq_name, kernel)

            lsq_synth_dir.mkdir(parents=True)
            lsq_out_path = lsq_synth_dir / "vivado_out.txt"
            lsq_err_path = lsq_synth_dir / "vivado_err.txt"
            with open(lsq_out_path, "w") as out_f, open(lsq_err_path, "w") as err_f:
                try:
                    result = subprocess.run(
                        ["tools/dynamatic/scripts/synthesize.sh",
                         REPO_ROOT,
                         out_dir,
                         lsq_top,
                         f"{CLOCK_PERIOD:.3f}", f"{CLOCK_PERIOD/2:.3f}",
                         lsq_synth_dir],
                        text=True,
                        stdout=out_f,
                        stderr=err_f,
                        cwd=REPO_ROOT,
                        timeout=TIMEOUT_SYNTH_LSQ_SECONDS,
                    )
                except subprocess.TimeoutExpired:
                    reason = f"LSQ synthesis ({lsq_name}) timed out after {TIMEOUT_SYNTH_LSQ_SECONDS // 60} minutes"
                    logging.error("[FAIL] %s (%s)", kernel, reason)
                    return kernel, reason

            if result.returncode != 0:
                reason = f"LSQ synthesis ({lsq_name}): exit code {result.returncode}"
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
        "--no-synth",
        action="store_true",
        help="Skip synthesis (default: False).",
    )
    parser.add_argument(
        "--synth-lsqs",
        action="store_true",
        help="Run out-of-context synthesis for LSQs after simulation (default: False).",
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
        "--json",
        type=Path,
        default=None,
        metavar="PATH",
        help="Write extracted metrics as JSON to PATH after all kernels complete.",
    )
    args = parser.parse_args()

    if args.no_synth and args.synth_lsqs:
        logging.error("Cannot use --synth-lsqs without synthesis. Please remove --synth-lsqs or add --no-synth.")
        sys.exit(1)

    build()

    logging.info("Running %d kernel(s) with %d parallel job(s)...", len(KERNELS), args.jobs)
    start_time = time.time()

    passed: list[str] = []
    failed: list[tuple[str, str | None]] = []

    with ThreadPoolExecutor(max_workers=args.jobs) as executor:
        futures = {executor.submit(run_kernel, k, args.no_synth, args.synth_lsqs): k for k in KERNELS}
        for future in as_completed(futures):
            kernel, failure_reason = future.result()
            if failure_reason is None:
                passed.append(kernel)
            else:
                failed.append((kernel, failure_reason))

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

    if args.json is not None:
        results = {}
        for kernel in KERNELS:
            out_dir = REPO_ROOT / "integration-test" / kernel / "out"
            failure_reason = next((r for k, r in failed if k == kernel), None)
            entry: dict = {"passed": failure_reason is None, "failure_reason": failure_reason}
            if failure_reason is None:
                entry.update(extract_kernel_data(kernel, out_dir))
            results[kernel] = entry
        args.json.write_text(json.dumps(results, indent=2))
        logging.info("Results written to %s.", args.json)

    if failed:
        logging.error(
            "Failed kernels: %s",
            ", ".join(f"{k} ({reason})" for k, reason in sorted(failed)),
        )
        sys.exit(1)


if __name__ == "__main__":
    main()
