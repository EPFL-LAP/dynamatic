#!/usr/bin/env python3
import re
import time
from datetime import datetime, timedelta
from pathlib import Path
from subprocess import PIPE, run
from sys import argv, stdout

import runpy
import sys




def shell(*args, **kwargs):
    stdout.flush()
    return run(*args, **kwargs, check=True)


def generate_initial_design(dynamatic_path, kernel_name, additional_dir, l, s, cp):

    print("8787878778")

    old_argv = sys.argv

    # Set sys.argv as if you ran: python change_lsq.py arg1 arg2
    sys.argv = ["change_lsq.py" , str(l), str(s)]

    print("running \n")
    # Run the script
    runpy.run_path("./skip/change_lsq.py")

    # Restore sys.argv
    sys.argv = old_argv


    script = f"set-src {dynamatic_path}/integration-test/{additional_dir}/{kernel_name}/{kernel_name}.c; set-clock-period {cp}; compile --buffer-algorithm fpga20; exit"

    shell(dynamatic_path / "bin" / "dynamatic", input=str.encode(script))


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

        time = re.findall(r"Time: (\d+) ns\s+Iteration:", lines)[-1]

    return int(time)


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
            if m and int(m.group(3)) == 4:
                transp_buffers.append(m.group(4))
    return transp_buffers


def remove_buffer(dynamatic_path, kernel_name, additional_dir, buffer_name):
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
    with open(
        handshake_export,
        "r",
    ) as f:
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


def synthesize(dynamatic_path, kernel_name, additional_dir):
    print("Running synthesis!")
    script = f"set-src {dynamatic_path}/integration-test/{additional_dir}/{kernel_name}/{kernel_name}.c; write-hdl; synthesize; exit"
    shell(dynamatic_path / "bin" / "dynamatic", input=str.encode(script))


def main():
    dynamatic_path = Path(argv[1])
    kernel_name = argv[2]
    l = int(argv[3])
    s = int(argv[4])
    cp = float(argv[5])

    print(kernel_name, l, s, cp)

    additional_dir = ""
    if "memory" in kernel_name:
        additional_dir = "memory/"

    print(kernel_name)
    generate_initial_design(dynamatic_path, kernel_name, additional_dir, l, s, cp)
    tbuffers = list_transp_buffers(dynamatic_path, kernel_name, additional_dir)
    tbuffers_total, tbuffers_done = len(tbuffers), 0
    print("Total number of transparent buffers:", len(tbuffers))
    for b in tbuffers:
        print(b)
    handshake_export = (
        dynamatic_path
        / "integration-test"
        / additional_dir
        / kernel_name
        / "out"
        / "comp"
        / "handshake_export.mlir"
    )
    handshake_export_bak = (
        dynamatic_path
        / "integration-test"
        / additional_dir
        / kernel_name
        / "out"
        / "comp"
        / "handshake_export_export.mlir"
    )
    initial_latency = simulate_design(dynamatic_path, kernel_name, additional_dir)
    print("Initial latency:", initial_latency // 4)
    removed = 0
    for buffer_name in tbuffers:
        shell(["cp", "-f", handshake_export, handshake_export_bak])

        start_time = time.time()

        remove_buffer(dynamatic_path, kernel_name, additional_dir, buffer_name)
        generate_hw(dynamatic_path, kernel_name, additional_dir)
        current_latency = simulate_design(dynamatic_path, kernel_name, additional_dir)
        print("Current latency:", current_latency // 4)
        if current_latency > initial_latency:
            print(
                buffer_name,
                "cannot be removed!",
            )
            shell(["mv", handshake_export_bak, handshake_export])
        else:
            print(
                "Successfully removed",
                buffer_name,
                "without a performance penalty",
            )
            removed += 1
        tbuffers_done += 1
        print(f"Buffers analyzed: {tbuffers_done}/{tbuffers_total}")

        elapsed_time = time.time() - start_time
        total_remaining_time = elapsed_time * (tbuffers_total - tbuffers_done)
        current_time = datetime.now()
        finish_time = current_time + timedelta(seconds=total_remaining_time)
        print(
            f"The simulation before synthesis is expected to finish at {finish_time.strftime('%H:%M:%S')}."
        )

    generate_hw(dynamatic_path, kernel_name, additional_dir)
    current_latency = simulate_design(dynamatic_path, kernel_name, additional_dir)
    print("Final latency:", current_latency // 4)

    # write out the final number of removed buffers
    with open(
        dynamatic_path
        / "integration-test"
        / additional_dir
        / kernel_name
        / "out"
        / "comp"
        / "removed_buffers.txt",
        "w",
    ) as f:
        f.write("Buffers removed: " + str(removed) + "\n" + "Total buffers: " + str(tbuffers_total) + "\n")

    # handshake_transformed = (
    #     dynamatic_path
    #     / "integration-test"
    #     / kernel_name
    #     / "out"
    #     / "comp"
    #     / "handshake_transformed.mlir"
    # )
    # shell(["mv", handshake_export, handshake_transformed])
    # synthesize(dynamatic_path, kernel_name)


if __name__ == "__main__":
    main()