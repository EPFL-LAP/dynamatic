import subprocess
from pathlib import Path
import os
import re
import shutil
import random

from simulate import *
from common import *
from compile import *


def extract_log(buf_name, wlf_file, test_dir):
    """Extract a FIFO signal log from the .wlf simulation dump."""
    os.makedirs("fifo_logs", exist_ok=True)
    log_path = f"fifo_logs/{buf_name}.log"

    cmd = [
        "wlf2log",
        "-l", f"{RTL_BASE_PATH}/{buf_name}/fifo/",
        "-o", log_path,
        wlf_file,
    ]

    log_file = test_dir / RUN_LOG_NAME

    try:
        subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        return log_path
    except subprocess.CalledProcessError as e:
        log(f"[WARN] Could not extract {buf_name}: {e.stderr.decode().strip()}", log_file)
        return None
    

def parse_fifo_depth(logfile_path, fifo_size):
    """
    Parse FIFO occupancy from signal logs.

    Head == Tail but Full == 1 means FIFO is full.
    """
    if logfile_path is None or not os.path.exists(logfile_path):
        return None

    signal_map = {}
    head, tail, full = 0, 0, 0
    max_len = 0

    with open(logfile_path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            line = line.strip()

            # Define signal mapping
            if line.startswith("D "):
                m = re.match(r"D\s+(\S+)\s+(\d+)", line)
                if m:
                    signal_map[m.group(2)] = m.group(1)

            # Update signal values
            elif line.startswith("S "):
                m = re.match(r"S\s+(\d+)\s+\['([0-9UZXWLH-])'\]", line)
                if not m:
                    continue
                sig_id, val = m.groups()
                name = signal_map.get(sig_id, "")

                if "Head" in name and val.isdigit():
                    head = int(val)
                elif "Tail" in name and val.isdigit():
                    tail = int(val)
                elif "Full" in name and val in ("0", "1"):
                    full = int(val)

            # Time tick → compute occupancy
            elif line.startswith("T "):
                fifo_len = (tail - head) % fifo_size
                # Handle full case (head == tail but Full=1)
                if head == tail and full == 1:
                    fifo_len = fifo_size
                max_len = max(max_len, fifo_len)

    return max_len

def remove_buffer(test_dir, buffer_name):
    """Remove a FIFO buffer by substituting its result with its source."""
    handshake_transformed = test_dir / COMP_DIR / "handshake_transformed.mlir"

    lines = []
    with open(handshake_transformed, "r") as f:
        substituted, by = "0", "0"
        for line in f.read().split("\n"):
            m = re.search(
                r'(%\d+)\s*=\s*buffer\s+(%[\d#]+),\s*bufferType\s*=\s*FIFO_BREAK_NONE,'
                r'\s*numSlots\s*=\s*(\d+).*?handshake\.name\s*=\s*"([^"]+)"', line
            )
            if m and m.group(4) == buffer_name:
                substituted, by = m.group(1), m.group(2)
            else:
                lines.append(line)

    joined_lines = "\n".join(lines)
    with open(handshake_transformed, "w") as f:
        f.write(re.sub(
            rf"(?<![0-9a-zA-Z_]){substituted}(?![0-9a-zA-Z_])",
            f"{by}", joined_lines,
        ))


def resize_buffer(test_dir, buffer_name, new_size):
    """Resize FIFO numSlots field in handshake_transformed.mlir."""
    handshake_transformed = test_dir / COMP_DIR / "handshake_transformed.mlir"
    

    with open(handshake_transformed, "r") as f:
        text = f.read()

    pattern = (
        rf'(numSlots\s*=\s*)\d+([^\n]*handshake\.name\s*=\s*"{re.escape(buffer_name)}")'
    )

    new_text = re.sub(pattern, rf'\g<1>{new_size}\g<2>', text)

    with open(handshake_transformed, "w") as f:
        f.write(new_text)


def list_transp_buffers(test_dir, fifo_size) -> list:
    transp_buffers = []
    path = test_dir / COMP_DIR / "handshake_transformed.mlir"
    with open(path, "r") as f:
        for line in f.read().split("\n"):
            m = re.search(
                r'buffer\s+(%[\d#]+),.*numSlots\s*=\s*(\d+).*handshake\.name\s*=\s*"([^"]+)"',
                line,
            )
            if m and int(m.group(2)) == fifo_size:
                transp_buffers.append(m.group(3))
    return transp_buffers


def resize_fifo(test_dir, base_name, fifo_size, rtl_config_name, seed):

    tbuffers = list_transp_buffers(test_dir, fifo_size)


    wlf_file = test_dir / SIM_DIR / "HLS_VERIFY" / "vsim.wlf"
    log_file = test_dir / RUN_LOG_NAME

    simulate_test(test_dir, base_name, rtl_config_name, log_file)
    baseline_cycles = parse_cycles(log_file)
    log(f"Initial BASELINE Cycles: {baseline_cycles}", log_file)

    results = {}

    for buf_name in tbuffers:
        log_path = extract_log(buf_name, wlf_file, test_dir)
        if not log_path:
            continue
        max_len = parse_fifo_depth(log_path, fifo_size)
        if max_len is None:
            continue
        results[buf_name] = max_len
        log(f"{buf_name} ----> {max_len}", log_file)
        log(f"{buf_name:<12} max_len = {max_len}", log_file)

    log("\n=== Applying MLIR modifications (remove) ===", log_file)
    total_slots = 0
    for buf, max_len in sorted(results.items()):
        if max_len == 0:
            log(f"[REMOVE] {buf} (max_len={max_len})", log_file)
            remove_buffer(test_dir, buf)
        else:
            pass


    buffer_cp(10, test_dir, log_file)
    canonicalize_handshake(test_dir, log_file)
    lower_handshake_to_hw(test_dir, log_file)

    # Run BASELINE simulation after initial resizing
    log("\n=== Running BASELINE simulation after initial resize ===", log_file)
    sim_log = test_dir / SIM_DIR / "report.txt"
    try:
        simulate_test(test_dir, base_name, rtl_config_name, sim_log)
        baseline_cycles = parse_cycles(sim_log)
    except Exception:
        log("[FATAL] Baseline simulation failed. Cannot proceed with optimization.", log_file)
        return

    if baseline_cycles is None:
        log("[FATAL] Could not get baseline cycle count. Cannot proceed.", log_file)
        return
        
    log(f"BASELINE Cycles: {baseline_cycles}", log_file)
    
    # Buffer Removal Optimization
    log("\n=== Applying Buffer Removal Optimization ===", log_file)
    
    # Candidates are all buffers that were not removed (max_len > 0)
    candidate_buffers = [buf for buf, max_len in results.items() if max_len > 0]

    if seed is not None:
        log(f"Shuffling candidate buffers with seed {seed}", log_file)
        random.seed(seed)
        random.shuffle(candidate_buffers)
    
    if not candidate_buffers:
        log("No buffers for removal optimization.", log_file)
        return

    # A. Backup the MLIR file state *after* the initial resize
    handshake_transformed = test_dir / COMP_DIR / "handshake_transformed.mlir"
    temp_backup = test_dir / COMP_DIR / "handshake_transformed.mlir.bak"
    shutil.copyfile(handshake_transformed, temp_backup)
    log(f"MLIR file backed up to: {temp_backup}", log_file)
    


    # removed_list = optimize_buffer_removal(
    #     test_dir, 
    #     base_name, 
    #     candidate_buffers, 
    #     baseline_cycles,
    #     fifo_size,
    #     results,
    #     0,
    #     False
    # )
    
    for i in range(fifo_size-1):
        global total_num
        total_num = len(candidate_buffers)

        log(f"\n=== Applying Buffer Resize Optimization to size {i} ===", log_file)
        resized_buffers, _, _ = optimize_buffer_resize(
            test_dir, 
            base_name, 
            candidate_buffers, 
            baseline_cycles,
            rtl_config_name,
            i,
            fifo_size,
            results,
            0,
            False
        )
        candidate_buffers = [buf for buf in candidate_buffers if buf not in resized_buffers]
    
        log(f"\nOptimization complete. Successfully resized {len(resized_buffers)} buffers to {i}", log_file)
        # The current handshake_transformed.mlir holds the final optimized state.
    
    # Cleanup: Remove the temporary backup
    os.remove(temp_backup)



    # tbuffers = list_transp_buffers(test_dir, fifo_size)


    # for buf_name in tbuffers:
    #     log_path = extract_log(buf_name, wlf_file, test_dir)
    #     if not log_path:
    #         continue
    #     max_len = parse_fifo_depth(log_path, fifo_size)
    #     if max_len is None:
    #         continue
    #     results[buf_name] = max_len
    #     log(f"{buf_name:<12} max_len = {max_len}", log_file)

    # log("\n=== Applying MLIR modifications ===", log_file)
    # total_slots = 0
    # for buf, max_len in sorted(results.items()):
    #     if max_len == 0:
    #         log(f"[REMOVE] {buf} (max_len={max_len})", log_file)
    #         remove_buffer(test_dir, buf)
    #     else:
    #         log(f"[RESIZE] {buf} -> {max_len}", log_file)
    #         resize_buffer(test_dir, buf, max_len)
    #         total_slots += max_len
    
    buffer_cp(10, test_dir, log_file)
    canonicalize_handshake(test_dir, log_file)
    lower_handshake_to_hw(test_dir, log_file)


    log("\n=== After buffering ===", log_file)
    sim_log = test_dir / SIM_DIR / "report.txt"
    try:
        simulate_test(test_dir, base_name, rtl_config_name, sim_log)
        baseline_cycles = parse_cycles(sim_log)
    except Exception:
        log("[FATAL] Baseline simulation failed. Cannot proceed with optimization.", log_file)
        return

    if baseline_cycles is None:
        log("[FATAL] Could not get baseline cycle count. Cannot proceed.", log_file)
        return
        
    log(f"BASELINE Cycles: {baseline_cycles}", log_file)
    

def check_performance(test_dir: Path, basename: str, rtl_config_name, baseline_cycles: int) -> bool:

    log_file = test_dir / RUN_LOG_NAME

    buffer_cp(10, test_dir, log_file)
    canonicalize_handshake(test_dir, log_file)
    lower_handshake_to_hw(test_dir, log_file)


    sim_log_path = test_dir / SIM_DIR / "report.txt"
    
    try:
        simulate_test(test_dir, basename, rtl_config_name, sim_log_path)
    except Exception as e:
        log(f"[ERROR] Simulation failed: {e}", log_file)
        return False

    new_cycles = parse_cycles(sim_log_path)

    if new_cycles is None:
        log("[ERROR] Could not parse cycles from simulation output.", log_file)
        return False
    
    if new_cycles >  baseline_cycles + 5:
        log(f"[PERF FAIL] Cycles increased: {baseline_cycles} -> {new_cycles}", log_file)
        return False
    else:
        log(f"[PERF OK] Cycles: {new_cycles} (Baseline: {baseline_cycles})", log_file)
        return True
    

def optimize_buffer_removal(test_dir: Path, basename: str, buffer_list: list, baseline_cycles: int, fifo_size, results, processed_num, skip) -> list:
    global total_num

    removed_buffers = []
    log_file = test_dir / RUN_LOG_NAME
    handshake_transformed = test_dir / COMP_DIR / "handshake_transformed.mlir"
    temp_backup = handshake_transformed.with_suffix(".mlir.bak")

    if not buffer_list:
        return removed_buffers, processed_num, True

    log(f"Processed {processed_num}/{total_num} until now", log_file)
    # Backup current state before trying this batch
    shutil.copyfile(handshake_transformed, temp_backup)

    # Try removing all buffers in this list
    for buf in buffer_list:
        log(f"[TRY REMOVE] {buf}", log_file)
        remove_buffer(test_dir, buf)

    
    # Performance check
    if not skip and check_performance(test_dir, basename, rtl_config_name, baseline_cycles):
        log(f"[OPTIMIZE] Success: Removed {len(buffer_list)} buffers: {buffer_list}", log_file)
        # Keep these removals permanently (overwrite backup)
        shutil.copyfile(handshake_transformed, temp_backup)
        removed_buffers.extend(buffer_list)
        new_processed_num  = processed_num + len(buffer_list)
        is_good = True
    else:
        # Revert and recurse on halves
        shutil.copyfile(temp_backup, handshake_transformed)
        if len(buffer_list) == 1:
            return removed_buffers, processed_num + 1, False

        log(f"[TRIAL FAILED] Performance hurt. Reverting and recursing on {len(buffer_list)} buffers.", log_file)

        midpoint = len(buffer_list) // 2
        first_half = buffer_list[:midpoint]
        second_half = buffer_list[midpoint:]

        rem_buffs_1, p_num_1, is_good_1 = optimize_buffer_removal(test_dir, basename, first_half, baseline_cycles, fifo_size, results, processed_num, False)
        removed_buffers.extend(rem_buffs_1)
        rem_buffs_2, p_num_2, is_good_2 = optimize_buffer_removal(test_dir, basename, second_half, baseline_cycles, fifo_size, results, p_num_1, is_good_1)
        removed_buffers.extend(rem_buffs_2)
        new_processed_num = p_num_2

        is_good = False

    return removed_buffers, new_processed_num, is_good




def optimize_buffer_resize(test_dir: Path, basename: str, buffer_list: list, baseline_cycles: int, rtl_config_name, target_size, fifo_size, results, processed_num, skip) -> list:
    global total_num

    resized_buffers = []
    log_file = test_dir / RUN_LOG_NAME
    handshake_transformed = test_dir / COMP_DIR / "handshake_transformed.mlir"
    temp_backup = handshake_transformed.with_suffix(".mlir.bak")

    if not buffer_list:
        return resized_buffers, processed_num, True

    log(f"Processed {processed_num}/{total_num} until now", log_file)
    # Backup current state before trying this batch
    shutil.copyfile(handshake_transformed, temp_backup)

    # Try removing all buffers in this list
    for buf in buffer_list:
        if target_size > 0:
            log(f"[TRY RESZIZE] {buf} to {target_size}", log_file)
            resize_buffer(test_dir, buf, target_size)
        else:
            log(f"[TRY REMOVE] {buf}", log_file)
            remove_buffer(test_dir, buf)

    
    # Performance check
    if not skip and check_performance(test_dir, basename, rtl_config_name, baseline_cycles):
        log(f"[OPTIMIZE] Success: Resized {len(buffer_list)} buffers to {target_size}: {buffer_list}", log_file)
        # Keep these removals permanently (overwrite backup)
        shutil.copyfile(handshake_transformed, temp_backup)
        resized_buffers.extend(buffer_list)
        new_processed_num  = processed_num + len(buffer_list)
        is_good = True
    else:
        # Revert and recurse on halves
        shutil.copyfile(temp_backup, handshake_transformed)
        if len(buffer_list) == 1:
            return resized_buffers, processed_num + 1, False

        log(f"[TRIAL FAILED] Performance hurt. Reverting and recursing on {len(buffer_list)} buffers.", log_file)

        midpoint = len(buffer_list) // 2
        first_half = buffer_list[:midpoint]
        second_half = buffer_list[midpoint:]

        rem_buffs_1, p_num_1, is_good_1 = optimize_buffer_resize(test_dir, basename, first_half, baseline_cycles, rtl_config_name, target_size, fifo_size, results, processed_num, False)
        resized_buffers.extend(rem_buffs_1)
        rem_buffs_2, p_num_2, is_good_2 = optimize_buffer_resize(test_dir, basename, second_half, baseline_cycles, rtl_config_name, target_size, fifo_size, results, p_num_1, is_good_1)
        resized_buffers.extend(rem_buffs_2)
        new_processed_num = p_num_2

        is_good = False

    return resized_buffers, new_processed_num, is_good