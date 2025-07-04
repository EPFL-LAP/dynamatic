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

def write_tcl(top_file, top_entity_name, hdl_files, tcl_file, sdc_file, rpt_timing, vhdl_interface_info):
    """
    Write the TCL file for synthesis based on the top file and HDL files.
    
    Args:
        top_file (str): Path to the top file.
        top_entity_name (str): Name of the top entity.
        hdl_files (list): List of HDL files needed for synthesis.
        tcl_file (str): Path to the output TCL file.
        sdc_file (str): Path to the SDC file for constraints.
        rpt_timing (str): Path to the output timing report file.
        vhdl_interface_info (VhdlInterfaceInfo): VHDL interface information containing generics and ports.
    """
    input_ports = vhdl_interface_info.get_input_ports()
    output_ports = vhdl_interface_info.get_output_ports()
    with open(tcl_file, 'w') as f:
        f.write(f"read_vhdl -vhdl2008 {top_file}\n")
        for hdl_file in hdl_files:
            f.write(f"read_vhdl -vhdl2008 {hdl_file}\n")
        f.write(f"read_xdc {sdc_file}\n")
        f.write("synth_design -top tb -part xc7k160tfbg484-2 -no_iobuf -mode out_of_context\n")
        f.write("opt_design\n")
        f.write("place_design\n")
        f.write("phys_opt_design\n")
        f.write("route_design\n")
        f.write("phys_opt_design\n")
        for iport in input_ports:
            if "clk" in iport or "clock" in iport or "rst" in iport or "reset" in iport:
                continue  # Skip clock and reset ports
            for oport in output_ports:
                f.write(f"report_timing -from [get_ports {iport}] -to [get_ports {oport}] >> {rpt_timing}\n")

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
