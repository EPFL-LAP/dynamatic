"""
Script for running Dynamatic speculative integration tests.
"""
import os
import shutil
import subprocess
import sys
import argparse
import json
import re
from pathlib import Path

DYNAMATIC_ROOT = Path(__file__).parent.parent.parent.parent

parser = argparse.ArgumentParser(
    description="Extraction")
parser.add_argument(
    "test_name", type=str, help="Name of the test")
parser.add_argument(
    "item", type=str, help="Extraction item")

args = parser.parse_args()
test_name = args.test_name
item = args.item

test_dir = DYNAMATIC_ROOT / "integration-test" / test_name
if not test_dir.exists():
    print(f"Test directory {test_dir} does not exist.")
    sys.exit(1)

# enumerate out* dirs
out_dirs = sorted([d for d in test_dir.iterdir()
                  if d.is_dir() and d.name.startswith("out")])

for out_dir in out_dirs:
    out_dir_name = out_dir.name
    if item == "cycles":
        # read sim/report.txt
        report_file = out_dir / "sim" / "report.txt"
        if not report_file.exists():
            print(
                f"Result file {report_file} does not exist for {out_dir_name}.")
            continue

        with open(report_file) as f:
            # find the place "Simulation done! Latency = X cycles"
            # use regex to extract X
            text = f.read()

            # find the place "Simulation done! Latency = X cycles"
            match = re.search(
                r"Simulation done!\s*Latency\s*=\s*(\d+)\s*cycles", text)
            if match:
                latency = int(match.group(1))
                # latencies[out_dir] = latency
                print(f"{out_dir_name}: Latency = {latency} cycles")
            else:
                print(f"No latency info found in {report_file}")
    elif item == "cp":
        # read synth/timing_post_pr.rpt
        timing_file = out_dir / "synth" / "timing_post_pr.rpt"
        if not timing_file.exists():
            print(
                f"Timing file {timing_file} does not exist for {out_dir_name}.")
            continue

        with open(timing_file) as f:
            # find the place "Data Path Delay: X ns"
            # use regex to extract X
            text = f.read()

            match = re.search(r"Data Path Delay:\s*([\d.]+)\s*ns", text)
            if match:
                cp = float(match.group(1))
                print(f"{out_dir_name}: Critical Path = {cp} ns")
            else:
                print(f"No critical path info found in {timing_file}")
