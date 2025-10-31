#!/usr/bin/env python3

import shutil
import os
from pathlib import Path
from datetime import datetime
import sys
import re

from common import *
from resize_fifo import *
from compile import *
from synthesis import *
from simulate import *

    
def find_best_target_cp(test_dir: Path, basename: str, log_file: Path):

    # Baseline latency (at 10 ns)
    with open(log_file, "a") as log:
        log.write("\n===== Target CP tuning =====\n")

    simulate_test(test_dir, basename, log_file)
    sim_out = test_dir / SIM_DIR / "report.txt"
    baseline_cycles = parse_cycles(sim_out)

    with open(log_file, "a") as log:
        log.write(f"Baseline latency = {baseline_cycles} cycles at 10.0 ns\n")

    best_cp = 10.0
    best_cycles = baseline_cycles

    cp = 9.5
    while cp >= 3.0:
        with open(log_file, "a") as log:
            log.write(f"\n--- Trying CP = {cp} ns ---\n")

        
        buffer_cp(cp, test_dir, log_file)
        canonicalize_handshake(test_dir, log_file)
        lower_handshake_to_hw(test_dir, log_file)

        # Simulate
        simulate_test(test_dir, basename, log_file)
        cycles = parse_cycles(sim_out)

        with open(log_file, "a") as log:
            log.write(f"→ Latency = {cycles} cycles\n")

        # stop if performance degrades (>1% slower)
        if cycles > baseline_cycles * 1.01:
            with open(log_file, "a") as log:
                log.write("Latency degraded, stopping search.\n")
            break

        best_cp = cp
        best_cycles = cycles
        cp -= 0.5

    with open(log_file, "a") as log:
        log.write(f"\n✅ Best CP: {best_cp} ns (latency = {best_cycles} cycles)\n")

    return best_cp


def find_best_fork_fifo_size(test_dir: Path, basename: str, cp: float, n: int,
    start_fifo_size: int, log_file: Path):
    best_fifo = start_fifo_size
    best_cycles = None

    fifo = start_fifo_size
    while fifo >= 0:
        log(f"Testing fork_fifo_size = {fifo}", log_file)

        # === Compile ===
        rc = compile(test_dir, basename, cp, n, fifo, log_file)
        if rc != 0:
            log(f"Compilation failed for fork_fifo_size={fifo}", log_file)
            break

        # === Simulate ===
        rc = simulate_test(test_dir, basename, log_file)
        if rc != 0:
            log(f"Simulation failed (rc={rc}) for fork_fifo_size={fifo}", log_file)
            break

        # === Parse simulation result ===
        sim_out = test_dir / SIM_DIR / "report.txt"
        cycles = parse_cycles(sim_out)
        if cycles is None:
            log(f"Could not parse cycles for fork_fifo_size={fifo}", log_file)
            break

        log(f"   → {cycles} cycles", log_file)

        if best_cycles is None or cycles <= best_cycles:
            best_cycles = cycles
            best_fifo = fifo
            fifo -= 1
        else:
            log(f"Performance worsened at {fifo} ({cycles} cycles). Keeping best={best_fifo}.", log_file)
            break

    log(f"✅ Best fork_fifo_size = {best_fifo} ({best_cycles} cycles)", log_file)
    return best_fifo



def find_best_timing(test_dir: Path, basename: str, log_file: Path):
    cps = []
    clock_cycles = []
    best_perf = float("inf")

    cp = 10
    while cp >= 3.0:
        log(f"\n=== Testing target CP = {cp} ns ===", log_file)

        buffer_cp(cp, test_dir, log_file)

        simulate_test(test_dir, basename, log_file)
        sim_out = test_dir / SIM_DIR / "report.txt"
        cycle = parse_cycles(sim_out)

        if cycle is None:
            log(f"⚠️ Simulation failed for CP={cp}, skipping...", log_file)
            continue
        clock_cycles.append(cycle)

        cp = find_real_cp(test_dir, basename, cp, log_file)
        cps.append(cp)

        perf = cycle * cp
        best_perf = min(best_perf, perf)
        log(f"Performance metric: {perf:.2f}", log_file)

        # 5️⃣ Stop if performance degrades >5% from best
        if perf > 1.05 * best_perf:
            log("🛑 Stopping early: performance worsened >5%", log_file)
            break
        cp -= 0.5

    best_cp = cps[-1] if cps else 10
    log(f"\n✅ Best CP found: {best_cp:.2f} ns", log_file)
    return best_cp


def run_pipeline(test_name: str, n: int, fifo_start: int, run_root: Path):
    src_dir = TEST_DIR / test_name
    if not src_dir.exists():
        print(f"Source directory not found for test {test_name}")
        return

    # Create test-specific run folder
    test_out_dir = run_root / f"{test_name}-{n}"
    test_out_dir.mkdir(parents=True, exist_ok=True)

    # Copy .c and .h files from the integration-test folder
    for ext in ("*.c", "*.h"):
        for f in src_dir.glob(ext):
            shutil.copy(f, test_out_dir / f.name)

    log_file = test_out_dir / RUN_LOG_NAME
    basename = test_name
    cp = 10

    best_fifo = find_best_fork_fifo_size(test_out_dir, basename, cp, n, fifo_start, log_file)

    # compile again with the best size
    compile(test_out_dir, basename, cp, n, best_fifo, log_file)

    resize_fifo(test_out_dir, basename, best_fifo)

    find_best_timing(test_out_dir, basename, log_file)

    # best_target_cp = find_best_target_cp(test_out_dir, basename, log_file)

    # find_real_cp(test_out_dir, basename, best_target_cp, log_file)


    print(f"✅ Finished {test_name}-{n}")
    

def read_inputs(input_file):
    # Read pairs from the input text file
    inputs = []
    with open(input_file, "r") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            if len(parts) not in (2, 3):
                print(f"Skipping invalid line: {line}")
                continue
            test, n_str = parts[0], parts[1]
            fifo_start = int(parts[2]) if len(parts) == 3 else 10  # default if missing
            try:
                n = int(n_str)
            except ValueError:
                print(f"Invalid n value in line: {line}")
                continue
            inputs.append((test, n, fifo_start))
    return inputs

def main():
    if len(sys.argv) != 2:
        print("Usage: run.py <input_list.txt>")
        sys.exit(1)

    input_file = Path(sys.argv[1])
    if not input_file.exists():
        print(f"Error: file {input_file} not found.")
        sys.exit(1)

    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    run_root = RUNS_BASE / f"run_{timestamp}"
    run_root.mkdir(parents=True, exist_ok=True)

    inputs = read_inputs(input_file)

    print(f"Found {len(inputs)} test pairs to run.")

    for test, n, fifo_start in inputs:
        run_pipeline(test, n, fifo_start, run_root)


if __name__ == "__main__":
    main()
