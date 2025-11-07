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
    "run_name", type=str, help="Name of the run")
parser.add_argument(
    "merge", type=str, help="Name of the merge")
parser.add_argument(
    "prev", type=str, help="Name of the merge prev")
parser.add_argument(
    "prev_out", type=str, help="Name of the merge prev out")
parser.add_argument("--iters", type=int, default=None,
                    help="Number of iterations")


args = parser.parse_args()

proj_dir = DYNAMATIC_ROOT / "integration-test" / args.test_name / args.run_name
visual_dir = proj_dir / "visual"

# mkdir visual_dir
if visual_dir.exists():
    shutil.rmtree(visual_dir)
visual_dir.mkdir(parents=True, exist_ok=True)

LEVEL = "tb/duv_inst/"

print("wlf2log")

f_wlf = proj_dir / "sim" / "HLS_VERIFY" / "vsim.wlf"
f_log = visual_dir / f"{args.test_name}.log"
result = subprocess.run([
    "wlf2log", "-l", LEVEL, "-o", f_log, f_wlf
])

print("log2csv")

f_csv = visual_dir / f"{args.test_name}.csv"
with open(f_csv, "w") as f:
    result = subprocess.run([
        DYNAMATIC_ROOT / "bin" / "log2csv",
        proj_dir / "comp" / "handshake_export.mlir",
        LEVEL, f_log, args.run_name
    ], stdout=f)

prev_interval_cycle = -1
intervals = []
transfer = False
prev_cycle = -1
with open(f_csv) as f:
    lines = f.readlines()
    for line in lines:
        items = line.strip().split(",")
        if len(items) < 6:
            continue
        if not items[0].isdigit():
            continue
        cycle = int(items[0])
        if cycle != prev_cycle:
            if transfer:
                if prev_interval_cycle == -1:
                    raise ValueError("prev_cycle is -1")
                else:
                    intervals.append(prev_cycle - prev_interval_cycle)
                    prev_interval_cycle = prev_cycle
        prev_cycle = cycle
        src_component = items[1].strip()
        dst_component = items[3].strip()
        state = items[5].strip()
        if src_component == args.prev and dst_component == args.merge:
            transfer = state == "transfer"
        if src_component == args.prev_out and dst_component == args.merge and state == "transfer":
            if prev_interval_cycle == -1:
                prev_interval_cycle = cycle
            else:
                print("breaking")
                # break


if args.iters is None:
    iters = len(intervals)
else:
    iters = args.iters
# Average
avg = sum(intervals) / iters

print(intervals)
print(f"Average interval: {avg}")
