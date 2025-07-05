import os
from multiprocessing import Pool
from utils import VhdlInterfaceInfo, NUM_CORES

def _synth_worker(args):
    synth_tool, tcl_file, log_file = args
    os.system(f"{synth_tool} -mode batch -source {tcl_file} > {log_file}")

def run_synthesis(tcl_files, synth_tool, log_file):
    """
    Run synthesis for the given TCL files using the specified synthesis tool in parallel (if NUM_CORES > 1).
    Args:
        tcl_files (list): List of TCL files to run synthesis on.
        synth_tool (str): Synthesis tool to use (e.g., 'vivado').
    """
    # Run synthesis in parallel using Vivado
    args_list = [(synth_tool, tcl_file, f"{log_file}{i}") for i, tcl_file in enumerate(tcl_files)]
    with Pool(processes=NUM_CORES) as pool:
        pool.map(_synth_worker, args_list)

def write_sdc_constraints(sdc_file, period_ns):
    """
    Write the SDC constraints file with the specified period.
    
    Args:
        sdc_file (str): Path to the SDC file.
        period_ns (float): Period in nanoseconds.
    """
    with open(sdc_file, 'w') as f:
        f.write(f"create_clock -name clk -period {period_ns} -waveform {{0.000 {period_ns/2}}} [get_ports clk]\n")
        f.write("set_property HD.CLK_SRC BUFGCTRL_X0Y0 [get_ports clk]\n")
