import os
import re
import subprocess
from pathlib import Path
from collections import defaultdict
from subprocess import PIPE, run
from sys import argv, stdout


# ========================
# CONFIGURATION
# ========================
WLF_FILE = "integration-test/histogram/out/sim/HLS_VERIFY/vsim.wlf"
#WLF_FILE = "integration-test/matrix_power/out/sim/HLS_VERIFY/vsim.wlf"
BUFFER_RANGE = range(0, 64)
FIFO_SIZE = 10
BASE_PATH = "tb/duv_inst"

# Project paths
DYNAMATIC_PATH = Path("./")
KERNEL_NAME = "histogram"
# KERNEL_NAME = "matrix_power"
ADDITIONAL_DIR = ""


# ========================
# UTILITY FUNCTIONS
# ========================

def shell(*args, **kwargs):
    stdout.flush()
    return run(*args, **kwargs, check=True)


def extract_log(buf_name):
    """Run wlf2log for a given buffer index."""
    os.makedirs("fifo_logs", exist_ok=True)
    log_path = f"fifo_logs/{buf_name}.log"

    cmd = [
        "wlf2log",
        "-l", f"{BASE_PATH}/{buf_name}/fifo/",
        "-o", log_path,
        WLF_FILE,
    ]

    try:
        subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        return log_path
    except subprocess.CalledProcessError as e:
        print(f"[WARN] Could not extract {buf_name}: {e.stderr.decode().strip()}")
        return None


def parse_fifo_depth(logfile_path, fifo_size, buffer_name):
    """Parse one FIFO log to find its maximum occupancy."""
    if logfile_path is None or not os.path.exists(logfile_path):
        return None

    signal_map = {}
    head, tail, max_len = 0, 0, 0

    with open(logfile_path, "r", encoding="utf-8", errors="ignore") as f:
        new = True
        for line in f:
            line = line.strip()
            if line.startswith("D "):
                m = re.match(r"D\s+(\S+)\s+(\d+)", line)
                if m:
                    signal_map[m.group(2)] = m.group(1)
            elif line.startswith("S "):
                m = re.match(r"S\s+(\d+)\s+\['([0-9UZXWLH-])'\]", line)
                if not m:
                    continue
                sig_id, val = m.group(1), m.group(2)
                name = signal_map.get(sig_id, "")
                if "Head" in name and val.isdigit():
                    head = int(val)
                    if (buffer_name == "buffer20"):
                        # print(f"Head updated to {head}")
                        new = True
                elif "Tail" in name and val.isdigit():
                    tail = int(val)
                    if (buffer_name == "buffer20"):
                        # print(f"Tail updated to {tail}")
                        new = True
            elif line.startswith("T "):
                fifo_len = (tail - head) % fifo_size
                max_len = max(max_len, fifo_len)
                if (buffer_name == "buffer20" and new):
                    print(f"len --> {fifo_len}")
                    new = False

    return max_len


def remove_buffer(dynamatic_path, kernel_name, additional_dir, buffer_name):
    """Your existing remove_buffer() function."""
    handshake_export = (
        dynamatic_path
        / "integration-test"
        / additional_dir
        / kernel_name
        / "out"
        / "comp"
        / "handshake_export.mlir"
    )
    lines = []
    with open(handshake_export, "r") as f:
        substituted, by = "0", "0"
        for line in f.read().split("\n"):
            m = re.search(
                r'(%\d+)\s*=\s*buffer\s+(%[\d#]+),\s*bufferType\s*=\s*FIFO_BREAK_NONE,\s*numSlots\s*=\s*(\d+).*?handshake\.name\s*=\s*"([^"]+)"',
                line,
            )
            if m and m.group(4) == buffer_name:
                substituted = m.group(1)
                by = m.group(2)
            else:
                lines.append(line)

    joined_lines = "\n".join(lines)
    with open(handshake_export, "w") as f:
        f.write(
            re.sub(
                rf"(?<![0-9a-zA-Z_]){substituted}(?![0-9a-zA-Z_])",
                f"{by}",
                joined_lines,
            )
        )


def resize_buffer(dynamatic_path, kernel_name, additional_dir, buffer_name, new_size):
    """Safely resize the FIFO buffer by changing numSlots for the given buffer."""
    handshake_export = (
        Path(dynamatic_path)
        / "integration-test"
        / additional_dir
        / kernel_name
        / "out"
        / "comp"
        / "handshake_export.mlir"
    )

    with open(handshake_export, "r") as f:
        text = f.read()

    pattern = (
        rf'(numSlots\s*=\s*)\d+'
        rf'([^\n]*handshake\.name\s*=\s*"{re.escape(buffer_name)}")'
    )


    new_text = re.sub(pattern, rf'\g<1>{new_size}\g<2>', text)

    
    # print(re.search(pattern, text))


    with open(handshake_export, "w") as f:
        f.write(new_text)



def generate_hw(dynamatic_path, kernel_name, additional_dir):
    handshake_export = (
        dynamatic_path
        / "integration-test"
        / additional_dir
        / kernel_name
        / "out"
        / "comp"
        / "handshake_export.mlir"
    )
    hw = dynamatic_path / "integration-test" / additional_dir/ kernel_name / "out" / "comp" / "hw.mlir"
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
    script = f"set-src {dynamatic_path}/integration-test/{additional_dir}/{kernel_name}/{kernel_name}.c; write-hdl; simulate; exit"
    shell(dynamatic_path / "bin" / "dynamatic", input=str.encode(script))

    with open(
        dynamatic_path
        / "integration-test"
        / additional_dir
        / kernel_name
        / "out"
        / "sim"
        / "report.txt",
        "r",
    ) as f:
        lines = f.read()

        cycles = re.findall(r"Note: Simulation done! Latency = (\d+)", lines)[-1]

    return int(cycles)


def list_transp_buffers(dynamatic_path, kernel_name, additional_dir) -> list:
    transp_buffers = []
    with open(
        dynamatic_path
        / "integration-test"
        / additional_dir
        / kernel_name
        / "out"
        / "comp"
        / "handshake_export.mlir",
        "r",
    ) as f:
        for line in f.read().split("\n"):
            m = re.search(
                r'(%\d+)\s*=\s*buffer\s+(%[\d#]+),\s*bufferType\s*=\s*FIFO_BREAK_NONE,\s*numSlots\s*=\s*(\d+).*?handshake\.name\s*=\s*"([^"]+)"',
                line,
            )

            # Only remove transparent buffers for throughput.
            if m and int(m.group(3)) == 10:
                transp_buffers.append(m.group(4))
    return transp_buffers


def parse_fork_groups(dynamatic_path, kernel_name, additional_dir):
    """
    Parses handshake_export.mlir to map each fork's outputs to their buffer names.
    Returns: dict like { 'fork2': ['buffer7', 'buffer8', 'buffer9'] }
    """
    handshake_export = (
        Path(dynamatic_path)
        / "integration-test"
        / additional_dir
        / kernel_name
        / "out"
        / "comp"
        / "handshake_export.mlir"
    )

    with open(handshake_export, "r") as f:
        text = f.read()

    # 1. Map fork result IDs (%21) to fork names ("fork2")
    fork_map = {
        m.group(1): m.group(2)
        for m in re.finditer(
            r'(%\d+):\d+\s*=\s*fork\s+\[\d+\]\s+%\d+.*?handshake\.name\s*=\s*"([^"]+)"',
            text
        )
    }

    # 2. Map fork -> list of buffers that consume its outputs
    fork_groups = {}

    for m in re.finditer(
        r'buffer\s+(%[\d#]+).*?numSlots\s*=\s*\d+.*?handshake\.name\s*=\s*"([^"]+)"',
        text
    ):
        src, buf_name = m.groups()
        src_id = src.split("#")[0]  # e.g. %21 from %21#1
        if src_id in fork_map:
            fork_name = fork_map[src_id]
            fork_groups.setdefault(fork_name, []).append(buf_name)

    return fork_groups


def get_buffer_sources(dynamatic_path, kernel_name, additional_dir):
    """Parse handshake_export.mlir and return {buffer_name: fork_id} mapping."""
    handshake_export = (
        Path(dynamatic_path)
        / "integration-test"
        / additional_dir
        / kernel_name
        / "out"
        / "comp"
        / "handshake_export.mlir"
    )
    text = handshake_export.read_text()

    pattern = (
        r'(%\d+)\s*=\s*buffer\s+(%[\d#]+),\s*bufferType\s*=\s*\w+,\s*'
        r'numSlots\s*=\s*(\d+).*?handshake\.name\s*=\s*"([^"]+)"'
    )

    fork_map = {}
    for m in re.finditer(pattern, text):
        src = m.group(2)           # e.g. %21#3
        buf_name = m.group(4)      # e.g. buffer9
        fork_id = src.split('#')[0]  # e.g. %21
        fork_map[buf_name] = fork_id
    return fork_map


# ========================
# MAIN PIPELINE
# ========================

def main_old():
    size_to_remove = int(argv[1])
    adjustment = int(argv[2])

    results = {}

    print(f"=== Extracting FIFO info from {WLF_FILE} ===\n")

    tbuffers = list_transp_buffers(DYNAMATIC_PATH, KERNEL_NAME, ADDITIONAL_DIR)
    for buf_name in tbuffers:
        log_path = extract_log(buf_name)
        if not log_path:
            continue
        max_len = parse_fifo_depth(log_path, FIFO_SIZE, buf_name)
        if max_len is None:
            continue
        results[buf_name] = max_len
        print(f"{buf_name:<10} max_len = {max_len}")

    total_num = 0
    # Apply optimization decisions
    print("\n=== Applying MLIR modifications ===")
    for buf, max_len in sorted(results.items()):
        if max_len <= size_to_remove:
            print(f"[REMOVE] {buf} (max_len={max_len})")
            remove_buffer(DYNAMATIC_PATH, KERNEL_NAME, ADDITIONAL_DIR, buf)
        else:
            max_size = 5
            print(f"[RESIZE] {buf} -> min({max_len}, {max_size})  = {min(max_len, max_size)}")
            resize_buffer(DYNAMATIC_PATH, KERNEL_NAME, ADDITIONAL_DIR, buf, min(max_len, max_size))
            total_num += max_len - adjustment

    print("\n✅ Done! handshake_export.mlir updated.")
    print(f"Total FIFO slots used: {total_num}\n")

    generate_hw(DYNAMATIC_PATH, KERNEL_NAME, ADDITIONAL_DIR)
    clock_cycle = simulate_design(DYNAMATIC_PATH, KERNEL_NAME, ADDITIONAL_DIR)
    print("Final Clock Cycles:", clock_cycle)


def main():
    size_to_remove = int(argv[1])
    adjustment = int(argv[2])

    print(f"=== Extracting FIFO info from {WLF_FILE} ===\n")

    # Map buffers to their fork IDs
    fork_map = get_buffer_sources(DYNAMATIC_PATH, KERNEL_NAME, ADDITIONAL_DIR)

    # Collect max FIFO depths
    results = {}
    tbuffers = list_transp_buffers(DYNAMATIC_PATH, KERNEL_NAME, ADDITIONAL_DIR)
    for buf_name in tbuffers:
        log_path = extract_log(buf_name)
        if not log_path:
            continue
        max_len = parse_fifo_depth(log_path, FIFO_SIZE)
        if max_len is None:
            continue
        results[buf_name] = max_len
        print(f"{buf_name:<15} max_len = {max_len}")

    # Group buffers by fork ID
    fork_groups = {}
    for buf, fork_id in fork_map.items():
        if buf in results:
            fork_groups.setdefault(fork_id, []).append((buf, results[buf]))

    print("\n=== Fork-based normalization ===")
    adjusted_results = {}
    for fork_id, group in fork_groups.items():
        min_len = min(length for _, length in group)
        print(f"{fork_id:<10} min_len = {min_len}")
        for buf, length in group:
            # new_len = max(length - min_len + 1, 0)
            new_len = length
            adjusted_results[buf] = new_len
            print(f"  {buf:<15} adjusted_len = {new_len}")

    # Apply resize/remove
    print("\n=== Applying MLIR modifications ===")
    total_num = 0
    for buf, adj_len in sorted(adjusted_results.items()):
        if adj_len <= size_to_remove:
            print(f"[REMOVE] {buf} (adj_len={adj_len})")
            remove_buffer(DYNAMATIC_PATH, KERNEL_NAME, ADDITIONAL_DIR, buf)
        else:
            target_len = max(adj_len - adjustment, 1)
            print(f"[RESIZE] {buf} -> {adj_len} - {adjustment} = {target_len}")
            resize_buffer(DYNAMATIC_PATH, KERNEL_NAME, ADDITIONAL_DIR, buf, target_len)
            total_num += target_len

    print("\n✅ Done! handshake_export.mlir updated.")
    print(f"Total FIFO slots used: {total_num}\n")

    generate_hw(DYNAMATIC_PATH, KERNEL_NAME, ADDITIONAL_DIR)
    clock_cycle = simulate_design(DYNAMATIC_PATH, KERNEL_NAME, ADDITIONAL_DIR)
    print("Final Clock Cycles:", clock_cycle)



if __name__ == "__main__":    
    main_old()


