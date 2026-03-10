#!/usr/bin/env python3
"""
Dynamatic evaluation data extraction script

Extracts key metrics from evaluation data and writes it to a CSV file.
"""

import argparse
import csv
import re
import sys
from pathlib import Path


def parse_sim_report(path: Path):
    """Return (cycle_count, passed) from a simulation report.txt."""
    text = path.read_text()
    passed = "C and VHDL outputs match" in text
    m = re.search(r"Simulation done!\s+Latency\s*=\s*(\d+)\s+cycles", text)
    cycle_count = int(m.group(1)) if m else None
    return cycle_count, passed


def parse_utilization(path: Path):
    """Return (slices, luts, ffs) from a utilization report."""
    slices = luts = ffs = None
    for line in path.read_text().splitlines():
        # Match table rows like "| Slice LUTs   | 4029 | ..."
        # Use the most-specific patterns first.
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


def main():
    parser = argparse.ArgumentParser(
        description="Extract evaluation metrics from run_evaluation.py output directory."
    )
    parser.add_argument("eval_dir", help="Path to the evaluation output directory")
    args = parser.parse_args()

    eval_dir = Path(args.eval_dir)
    if not eval_dir.is_dir():
        print(f"Error: {eval_dir} is not a directory", file=sys.stderr)
        sys.exit(1)

    columns = [
        "kernel",
        "cycle_count",
        "utilization_slice",
        "utilization_lut",
        "utilization_ff",
        "timing_cp_ns",
        "timing_slack_ns",
        "timing_cp_src",
        "timing_cp_dst",
    ]

    writer = csv.DictWriter(sys.stdout, fieldnames=columns)
    writer.writeheader()

    for kernel_dir in sorted(eval_dir.iterdir()):
        if not kernel_dir.is_dir():
            continue
        kernel = kernel_dir.name

        row: dict = {"kernel": kernel}

        # Simulation report
        sim_report = kernel_dir / "out" / "sim" / "report.txt"
        if not sim_report.exists():
            print(f"Error: {sim_report} not found", file=sys.stderr)
            sys.exit(1)
        cycle_count, passed = parse_sim_report(sim_report)
        if not passed:
            print(f"Error: {kernel}: C and VHDL outputs do not match", file=sys.stderr)
            sys.exit(1)
        row["cycle_count"] = cycle_count

        # Utilization report
        util_rpt = kernel_dir / "out" / "synth" / "utilization_post_pr.rpt"
        if not util_rpt.exists():
            print(f"Error: {util_rpt} not found", file=sys.stderr)
            sys.exit(1)
        slices, luts, ffs = parse_utilization(util_rpt)
        row["utilization_slice"] = slices
        row["utilization_lut"] = luts
        row["utilization_ff"] = ffs

        # Timing report
        timing_rpt = kernel_dir / "out" / "synth" / "timing_post_pr.rpt"
        if not timing_rpt.exists():
            print(f"Error: {timing_rpt} not found", file=sys.stderr)
            sys.exit(1)
        cp_ns, slack_ns, cp_src, cp_dst = parse_timing(timing_rpt)
        row["timing_cp_ns"] = cp_ns
        row["timing_slack_ns"] = slack_ns
        row["timing_cp_src"] = cp_src
        row["timing_cp_dst"] = cp_dst

        writer.writerow(row)


if __name__ == "__main__":
    main()
