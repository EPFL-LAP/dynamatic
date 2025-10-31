from pathlib import Path
import shutil

from common import *


def compile(test_dir: Path, basename: str, clock: float, n: int, fork_fifo_size: int, log_file: Path):
    temp_script = test_dir / "temp_frontend-script-compile-with-dynamatic.dyn"
    shutil.copy(COMPILE_TEMPLATE, temp_script)

    # Replace placeholders
    text = temp_script.read_text()
    text = text.replace("PATH_TO_C_FILE", str(test_dir / f"{basename}.c"))
    text = text.replace("CLOCK_PERIOD", str(clock))
    text = text.replace("SKIPPABLE_SEQ_N", str(n))
    text = text.replace("FORK_FIFO_SIZE", str(fork_fifo_size))
    temp_script.write_text(text)


    
    cmd = f'{DYNAMATIC_PATH} --run "{temp_script}"'
    return run_cmd(cmd, cwd=None, timeout=TIMEOUT * 60, log_file=log_file)

def buffer_cp(cp, test_dir: Path, log_file: Path):
    comp_dir = test_dir / COMP_DIR 
    handshake_transformed = comp_dir / "handshake_transformed.mlir"
    handshake_buffered = comp_dir / "handshake_buffered.mlir"
    f_frequencies = comp_dir / "frequencies.csv"

    timing_models = Path(DYNAMATIC_PATH).parent.parent / "data" / "components.json"

    # Run buffering directly with dynamatic-opt
    cmd = (
        f'"{DYNAMATIC_OPT_BIN}" "{handshake_transformed}" '
        f'--handshake-mark-fpu-impl="impl=flopoco" '
        f'--handshake-set-buffering-properties="version=fpga20" '
        f'--handshake-place-buffers="algorithm={BUFFERING_ALG} frequencies={f_frequencies} '
        f'timing-models={timing_models} target-period={cp} timeout=300 '
        f'dump-logs blif-files={Path(DYNAMATIC_PATH).parent.parent / "data/aig/"} '
        f'lut-delay=0.55 lut-size=6 acyclic-type" '
        f'> "{handshake_buffered}"'
    )

    rc = run_cmd(cmd, cwd=None, timeout=TIMEOUT * 60, log_file=log_file)
    if rc != 0:
        with open(log_file, "a") as log:
            log.write(f"Buffering failed for CP={cp}\n")

def canonicalize_handshake(test_dir: Path, log_file: Path):
    comp_dir = test_dir / COMP_DIR
    f_buffered = comp_dir / "handshake_buffered.mlir"
    f_export = comp_dir / "handshake_export.mlir"

    cmd = (
        f'"{DYNAMATIC_OPT_BIN}" "{f_buffered}" '
        f'--handshake-canonicalize '
        f'--handshake-hoist-ext-instances '
        f'> "{f_export}"'
    )

    log(f"Running Handshake canonicalization...", log_file)
    rc = run_cmd(cmd, cwd=None, timeout=TIMEOUT * 10, log_file=log_file)
    if rc != 0:
        log("❌ Failed to canonicalize Handshake.", log_file)
        return False
    log("✅ Canonicalized Handshake.", log_file)
    return True


def lower_handshake_to_hw(test_dir: Path, log_file: Path):
    comp_dir = test_dir / COMP_DIR
    f_export = comp_dir / "handshake_export.mlir"
    f_hw = comp_dir / "hw.mlir"

    cmd = (
        f'"{DYNAMATIC_OPT_BIN}" "{f_export}" '
        f'--lower-handshake-to-hw '
        f'> "{f_hw}"'
    )

    log("Lowering Handshake to HW...", log_file)
    rc = run_cmd(cmd, cwd=None, timeout=TIMEOUT * 10, log_file=log_file)
    if rc != 0:
        log("❌ Failed to lower Handshake to HW.", log_file)
        return False
    log("✅ Lowered Handshake to HW.", log_file)
    return True
