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

args = parser.parse_args()
test_name = args.test_name

test_dir = DYNAMATIC_ROOT / "integration-test" / test_name
if not test_dir.exists():
    print(f"Test directory {test_dir} does not exist.")
    sys.exit(1)

# enumerate out* dirs
out_dirs = sorted([d for d in test_dir.iterdir()
                  if d.is_dir() and d.name.startswith("out")])

for out_dir in out_dirs:
    # read sim/report.txt
    report_file = out_dir / "sim" / "report.txt"
    out_dir_name = out_dir.name
    if not report_file.exists():
        print(f"Result file {report_file} does not exist for {out_dir_name}.")
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
