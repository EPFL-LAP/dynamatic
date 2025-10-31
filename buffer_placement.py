"""
===========================================
 Dynamatic Buffer Placment
===========================================

This script automates FIFO buffer optimization for Dynamatic designs by:
  1. Extracting FIFO logs from simulation (.wlf file)
  2. Measuring each FIFO’s maximum occupancy
  3. Iteratively reducing FIFO sizes while monitoring performance
  4. Updating handshake_export.mlir accordingly

Usage:
  python buffer_placement.py <kernel_name> <buffer_size>

Example:
  python3 buffer_placement.py histogram 10
"""

import os
import re
import subprocess
from pathlib import Path
from subprocess import PIPE, run
from sys import argv, stdout

# ====================================================
# CONFIGURATION
# ====================================================

# Default simulation setup
FIFO_SIZE = 10
BASE_PATH = "tb/duv_inst"
DYNAMATIC_PATH = Path("./")
ADDITIONAL_DIR = ""

# ====================================================
# UTILITY FUNCTIONS
# ====================================================

def shell(*args, **kwargs):
    """Run a shell command"""
    stdout.flush()
    return run(*args, **kwargs, check=True)


def extract_log(buf_name, wlf_file):
    """Extract a FIFO signal log from the .wlf simulation dump."""
    os.makedirs("fifo_logs", exist_ok=True)
    log_path = f"fifo_logs/{buf_name}.log"

    cmd = [
        "wlf2log",
        "-l", f"{BASE_PATH}/{buf_name}/fifo/",
        "-o", log_path,
        wlf_file,
    ]

    try:
        subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        return log_path
    except subprocess.CalledProcessError as e:
        print(f"[WARN] Could not extract {buf_name}: {e.stderr.decode().strip()}")
        return None


def parse_fifo_depth(logfile_path, fifo_size, buffer_name):
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


def remove_buffer(dynamatic_path, kernel_name, additional_dir, buffer_name):
    """Remove a FIFO buffer by substituting its result with its source."""
    handshake_export = (
        dynamatic_path / "integration-test" / additional_dir / kernel_name /
        "out" / "comp" / "handshake_export.mlir"
    )

    lines = []
    with open(handshake_export, "r") as f:
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
    with open(handshake_export, "w") as f:
        f.write(re.sub(
            rf"(?<![0-9a-zA-Z_]){substituted}(?![0-9a-zA-Z_])",
            f"{by}", joined_lines,
        ))


def resize_buffer(dynamatic_path, kernel_name, additional_dir, buffer_name, new_size):
    """Resize FIFO numSlots field in handshake_export.mlir."""
    handshake_export = (
        Path(dynamatic_path) / "integration-test" / additional_dir /
        kernel_name / "out" / "comp" / "handshake_export.mlir"
    )

    with open(handshake_export, "r") as f:
        text = f.read()

    pattern = (
        rf'(numSlots\s*=\s*)\d+([^\n]*handshake\.name\s*=\s*"{re.escape(buffer_name)}")'
    )

    new_text = re.sub(pattern, rf'\g<1>{new_size}\g<2>', text)

    with open(handshake_export, "w") as f:
        f.write(new_text)


def generate_hw(dynamatic_path, kernel_name, additional_dir):
    """Regenerate hw.mlir from handshake_export.mlir."""
    handshake_export = (
        dynamatic_path / "integration-test" / additional_dir /
        kernel_name / "out" / "comp" / "handshake_export.mlir"
    )
    hw = (
        dynamatic_path / "integration-test" / additional_dir /
        kernel_name / "out" / "comp" / "hw.mlir"
    )
    with open(hw, "w") as fs:
        capture = shell(
            [
                dynamatic_path / "bin" / "dynamatic-opt",
                handshake_export,
                "--lower-handshake-to-hw",
            ],
            stdout=PIPE,
        )
        fs.write(capture.stdout.decode())


def simulate_design(dynamatic_path, kernel_name, additional_dir) -> int:
    """Run Dynamatic simulation and return latency (clock cycles)."""
    script = (
        f"set-src {dynamatic_path}/integration-test/{additional_dir}/{kernel_name}/{kernel_name}.c; "
        "write-hdl; simulate; exit"
    )
    shell(dynamatic_path / "bin" / "dynamatic", input=str.encode(script))

    report_path = (
        dynamatic_path / "integration-test" / additional_dir /
        kernel_name / "out" / "sim" / "report.txt"
    )
    with open(report_path, "r") as f:
        lines = f.read()
        cycles = re.findall(r"Note: Simulation done! Latency = (\d+)", lines)[-1]
    return int(cycles)


def list_transp_buffers(buffer_size, dynamatic_path, kernel_name, additional_dir) -> list:
    """List transparent FIFO buffers (FIFO_BREAK_NONE, numSlots= buffer_size)."""
    transp_buffers = []
    path = (
        dynamatic_path / "integration-test" / additional_dir /
        kernel_name / "out" / "comp" / "handshake_export.mlir"
    )
    with open(path, "r") as f:
        for line in f.read().split("\n"):
            m = re.search(
                r'buffer\s+(%[\d#]+),.*numSlots\s*=\s*(\d+).*handshake\.name\s*=\s*"([^"]+)"',
                line,
            )
            if m and int(m.group(2)) == buffer_size:
                transp_buffers.append(m.group(3))
    return transp_buffers


# ====================================================
# MAIN PIPELINE
# ====================================================

def main():
    """
    Iteratively reduces FIFO sizes until performance degradation is observed.
    """
    if len(argv) < 5:
        print("Usage: python fifo_optimizer.py <kernel_name> <buffer_size>")
        return

    kernel_name = argv[1]
    buffer_size = int(argv[2])

    wlf_file = f"integration-test/{kernel_name}/out/sim/HLS_VERIFY/vsim.wlf"

    print(f"=== Buffer Placement for {kernel_name} ===")
    print(f"Starting from max_size = {buffer_size}\n")

    tbuffers = list_transp_buffers(DYNAMATIC_PATH, kernel_name, ADDITIONAL_DIR)

    best_latency = float("inf")
    best_size = buffer_size

    # Sweep FIFO size downward until performance hurts
    for max_size_for_iter in range(buffer_size, 0, -1):
        print(f"\n--- Testing FIFO size = {size} ---")
        results = {}

        for buf_name in tbuffers:
            log_path = extract_log(buf_name, wlf_file)
            if not log_path:
                continue
            max_len = parse_fifo_depth(log_path, FIFO_SIZE, buf_name)
            if max_len is None:
                continue
            results[buf_name] = max_len
            print(f"{buf_name:<12} max_len = {max_len}")

        print("\n=== Applying MLIR modifications ===")
        total_slots = 0
        for buf, max_len in sorted(results.items()):
            if max_len == 0:
                print(f"[REMOVE] {buf} (max_len={max_len})")
                remove_buffer(DYNAMATIC_PATH, kernel_name, ADDITIONAL_DIR, buf)
            else:
                new_len = min(max_len, max_size_for_iter)
                print(f"[RESIZE] {buf} -> {new_len}")
                resize_buffer(DYNAMATIC_PATH, kernel_name, ADDITIONAL_DIR, buf, new_len)
                total_slots += new_len

        print(f"\nTotal FIFO slots used: {total_slots}")
        generate_hw(DYNAMATIC_PATH, kernel_name, ADDITIONAL_DIR)
        latency = simulate_design(DYNAMATIC_PATH, kernel_name, ADDITIONAL_DIR)

        print(f"Latency = {latency} cycles")

        if latency > best_latency * 1.05:  # >5% slowdown → stop
            print("\n⚠️ Performance degraded — stopping search.")
            break

        best_latency = latency
        best_size = size

    print(f"\n✅ Optimal FIFO size: {best_size}, latency = {best_latency} cycles")


if __name__ == "__main__":
    main()
