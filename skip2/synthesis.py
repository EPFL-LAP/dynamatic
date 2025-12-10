from pathlib import Path
import shutil

from common import *



def place_and_route(test_dir: Path, iter_dir: Path, basename: str, constraints: Path, iter_log_file: Path):

    tcl_temp = iter_dir / "temp-call-vivado-impl-for-dynamatic.tcl"
    shutil.copy(TCL_TEMPLATE, tcl_temp)

    # Replace TCL placeholders
    text = tcl_temp.read_text()
    text = text.replace("PATH_TO_TEST_FOLDER", str(test_dir))
    text = text.replace("PATH_TO_PLACE_AND_ROUTE", str(iter_dir))
    text = text.replace("PATH_TO_CONSTRAINTS_FILE", str(constraints))
    text = text.replace("TOP_MODULE", basename)
    tcl_temp.write_text(text)

    cmd = f'vivado -notrace -nolog -nojournal -mode tcl -script "{tcl_temp}"'
    log("starting place and route", iter_log_file)

    return run_cmd(cmd, cwd=None, timeout=TIMEOUT * 60, log_file=iter_log_file)






def find_real_cp(test_dir: Path, basename: str, start_cp: float, log_file: Path):
    vivado_dir = test_dir / VIVADO_DIR
    vivado_dir.mkdir(parents=True, exist_ok=True)

    current_cp = start_cp
    max_iterations = 10
    slack_value = None

    for iteration in range(max_iterations):
        half_period = current_cp / 2

        iter_dir = vivado_dir / f"Timing_based_target_{start_cp + 0.3:.2f}_cp_{current_cp:.2f}"
        iter_dir.mkdir(parents=True, exist_ok=True)

        constraints_path = iter_dir / "constraints.xdc"

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
        log(f"[Iteration {iteration}] Slack = {slack_value}\n", log_file)

        if slack_value is None:
            log("No slack found in report, stopping.\n", log_file)
            break

        if slack_value >= 0:
            log(f"Slack >= 0 ({slack_value})\n", log_file)
            return current_cp

        adjust = abs(round(slack_value, 2))
        current_cp = round(current_cp + adjust, 2)
        log(f"Slack < 0 ({slack_value}), increasing CP to {current_cp} and retrying.\n", log_file)

    log(f"Reached max iterations ({max_iterations}) without positive slack.\n", log_file)
    return None


def parse_slack_from_report(rpt_path: Path):
    """Extracts the slack value (float) from a Vivado timing report."""
    if not rpt_path.exists():
        return None
    for line in rpt_path.read_text().splitlines():
        if "Slack" in line:
            try:
                # Example line: "Slack (MET) : 0.123ns"
                parts = line.split()
                for p in parts:
                    if "ns" in p:
                        val = p.replace("ns", "")
                        return float(val)
            except ValueError:
                continue
    return None

