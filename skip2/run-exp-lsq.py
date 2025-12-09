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
from write_csv import *

    
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


def find_cp_by_adding_slack(test_dir: Path, basename: str, load:int, store:int, log_file: Path):
    vivado_dir = test_dir / VIVADO_DIR
    vivado_dir.mkdir(parents=True, exist_ok=True)

    for current_cp in [4, 6, 8, 10]:
        iter_dir = vivado_dir / f"One_attempt_cp_{current_cp:.2f}"
        iter_dir.mkdir(parents=True, exist_ok=True)

        constraints_path = iter_dir / "constraints.xdc"

        half_period = current_cp / 2

        constraints_content = (
            f"create_clock -name clk -period {current_cp} "
            f"-waveform {{0.000 {half_period}}} [get_ports clk]\n"
        )
        constraints_path.write_text(constraints_content)

        iter_log = iter_dir / "place_and_route.log"

        synth_rc = place_and_route(test_dir, iter_dir, basename, constraints_path, iter_log)

        if synth_rc != 0: 
            log(f"Place and Route failed at CP={current_cp}\n", log_file) 
            return None

        rpt = iter_dir / "timing_post_pr.rpt"
        slack_value = parse_slack_from_report(rpt)
        log(f"[CP {current_cp}] Slack = {slack_value}\n", log_file)

        if slack_value is None:
            log("No slack found in report, stopping.\n", log_file)
            continue

        simulate_test(test_dir, basename, log_file)
        sim_out = test_dir / SIM_DIR / "report.txt"
        cycle = parse_cycles(sim_out)

        if cycle is None:
            log(f" Simulation failed for CP={current_cp}, skipping...", log_file)
            continue

        break



    write_to_csv(benchmark=basename,
            load=load,
            store=store,
            target_cp=current_cp,
            real_cp=current_cp - slack_value,
            cycles=cycle,
            input_file=test_dir / VIVADO_DIR / f"One_attempt_cp_{current_cp:.2f}" / "utilization_post_pr.rpt",
            output_csv=test_dir / "results_one_attempt.csv")

    return current_cp - slack_value



def find_best_timing(test_dir: Path, basename: str, start_cp:float, load: int, store: int, log_file: Path):
    cps = []
    clock_cycles = []
    best_perf = float("inf")

    target_cp = start_cp
    while target_cp >= 3.0:
        log(f"\n=== Testing target CP = {target_cp} ns ===", log_file)

        buffer_cp(target_cp, test_dir, log_file)
        canonicalize_handshake(test_dir, log_file)
        lower_handshake_to_hw(test_dir, log_file)

        simulate_test(test_dir, basename, log_file)
        sim_out = test_dir / SIM_DIR / "report.txt"
        cycle = parse_cycles(sim_out)

        if cycle is None:
            log(f" Simulation failed for CP={target_cp}, skipping...", log_file)
            continue
        clock_cycles.append(cycle)

        real_cp = find_real_cp(test_dir, basename, target_cp - 0.3, log_file)
        cps.append(real_cp)

        perf = cycle * real_cp
        best_perf = min(best_perf, perf)
        log(f"$$$ Target CP: {target_cp} ns", log_file)
        log(f"$$$ Cycles: {cycle}", log_file)
        log(f"$$$ Real CP: {real_cp:.2f} ns", log_file)
        log(f"$$$ Performance metric: {perf:.2f}", log_file)


        # Stop if performance degrades >5% from best
        if perf > 1.2 * best_perf:
            log(" Stopping early: performance worsened >20%", log_file)
            break
        target_cp -= 0.5

    if cps and clock_cycles:
        perfs = [c * r for c, r in zip(clock_cycles, cps)]
        best_idx = perfs.index(min(perfs))

        log("\n===  Best Configuration Found ===", log_file)
        log(f"Target CP : {start_cp - 0.5 * best_idx:.2f} ns", log_file)
        log(f"Real   CP : {cps[best_idx]:.2f} ns", log_file)
        log(f"Cycles    : {clock_cycles[best_idx]}", log_file)
        log(f"Performance metric : {perfs[best_idx]:.2f}", log_file)

        write_to_csv(benchmark=basename,
                      load=load,
                      store=store,
                      target_cp=start_cp - 0.5 * best_idx,
                      real_cp=cps[best_idx],
                      cycles=clock_cycles[best_idx],
                      input_file=test_dir / VIVADO_DIR / f"Timing_based_target_{start_cp - 0.5 * best_idx:.2f}_cp_{cps[best_idx]:.2f}" / "utilization_post_pr.rpt",
                      output_csv=test_dir / "results.csv")
    else:
        log("\n No valid results found during CP tuning.", log_file)


def run_pipeline(test_name: str, load: int, store: int, run_root: Path):
    src_dir = TEST_DIR / test_name
    if not src_dir.exists():
        print(f"Source directory not found for test {test_name}")
        return

    # Create test-specific run folder
    test_out_dir = run_root / f"{test_name}-{load}-{store}"
    test_out_dir.mkdir(parents=True, exist_ok=True)

    # Copy .c and .h files from the integration-test folder
    for ext in ("*.c", "*.h"):
        for f in src_dir.glob(ext):
            shutil.copy(f, test_out_dir / f.name)

    log_file = test_out_dir / RUN_LOG_NAME
    basename = test_name
    cp = 10

    # best_fifo = find_best_fork_fifo_size(test_out_dir, basename, cp, n, fifo_start, log_file)

    # # compile again with the best size
    # compile(test_out_dir, basename, cp, n, best_fifo, log_file)

    compile(test_out_dir, basename, cp, load, store, 10, log_file)

    resize_fifo(test_out_dir, basename, 10)

    achieved = find_cp_by_adding_slack(test_out_dir, basename, load, store, log_file)

    # add 0.5 to achieved and then round it up to the nearest 0.5
    if achieved is not None:
        start_cp = achieved + 0.5
        start_cp = round(start_cp * 2) / 2
        log(f"Starting from {start_cp} ns", log_file)

    find_best_timing(test_out_dir, basename, start_cp, load, store, log_file)


    # best_target_cp = find_best_target_cp(test_out_dir, basename, log_file)

    # find_real_cp(test_out_dir, basename, best_target_cp, log_file)


    print(f" Finished {test_name}-{load}-{store}")
    

def read_inputs(input_file):
    # Read pairs from the input text file
    inputs = []
    with open(input_file, "r") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            if len(parts) != 3:
                print(f"Skipping invalid line: {line}")
                continue
            test, load_str, store_str = parts[0], parts[1], parts[2]
            try:
                load = int(load_str)
                store = int(store_str)
            except ValueError:
                print(f"Invalid n value in line: {line}")
                continue
            inputs.append((test, load, store))
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

    for test, load, store in inputs:
        run_pipeline(test, load, store, run_root)


if __name__ == "__main__":
    main()
