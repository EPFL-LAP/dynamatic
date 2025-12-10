import shutil, re
from common import *

def simulate_test(test_dir: Path, basename: str, rtl_config_name, log_file: Path):
    temp_script = test_dir / "temp_frontend-script-simulate.dyn"
    shutil.copy(SIMULATE_TEMPLATE, temp_script)

    # Replace placeholders
    text = temp_script.read_text()
    text = text.replace("PATH_TO_C_FILE", str(test_dir / f"{basename}.c"))
    text = text.replace("RTL_CONFIG_NAME", rtl_config_name)
    temp_script.write_text(text)


    cmd = f'{DYNAMATIC_PATH} --run "{temp_script}"'
    return run_cmd(cmd, cwd=None, timeout=TIMEOUT * 60, log_file=log_file)


def parse_cycles(sim_out: Path) -> int | None:
    if not sim_out.exists():
        return None

    text = sim_out.read_text()
    # Look for "Latency = <number>"
    match = re.search(r"Latency\s*=\s*(\d+)", text)
    if match:
        return int(match.group(1))
    return None