#!/usr/bin/env python3

################################################################
# Environment Setup
################################################################

import os
from enum import Enum
import argparse
from pdb import run
import shutil

# Import file templates
from script_templates import *

# Small handwritten util functions
from utils import *


################################################################
# Experiment Enums
################################################################

class DesignFlag(Enum):
    PRE = "pre"
    POST = "post"


class InputFlag(Enum):
    PI = "pi"    # Primary Inputs only
    ALL = "all"


def run_command(command, power_analysis_dir):
    report = os.path.join(power_analysis_dir, "report.txt")
    ret = os.system(command + f" >> {report}")
    return ret == 0

################################################################
# Main Function
################################################################


def main(output_dir, kernel_name, hdl, clock_period, vivado_cmd="vivado", post_synth_netlist=False):
    print("[INFO] Running power estimation")

    # Get the date for generating the scripts
    date = get_date()

    #
    # Step 1: Set up folders and file paths 
    #
    power_analysis_dir = os.path.join(output_dir, "power")
    if os.path.exists(power_analysis_dir):
        shutil.rmtree(power_analysis_dir)

    os.makedirs(power_analysis_dir)
    
    hdl_src_folder = os.path.join(output_dir, "hdl")
    if not os.path.exists(hdl_src_folder):
        print(f"[ERROR] {hdl_src_folder} not found. Please run the 'write-hdl' command")
        return

    # For both VHDL and Verilog
    tb_file_name = f"tb_{kernel_name}.vhd" if hdl == "vhdl" else f"tb_{kernel_name}.v"
    tb_file = os.path.join(output_dir, "sim", "HDL_SRC", tb_file_name)
    if not os.path.exists(tb_file):
        print(f"[ERROR] {tb_file} not found. Please run the 'simulate' command")
        return

    #
    # Step 2: Generate the xdc file for synthesis in Vivado 
    #
    xdc_dict = {
        'tcp': clock_period,
        'halftcp': clock_period / 2
    }

    period_xdc_file = os.path.join(power_analysis_dir, "period.xdc")
    target_file_generation(
        template_file=base_xdc,
        substitute_dict=xdc_dict,
        target_path=period_xdc_file
    )

    #
    # Step 3: Get all files needed for simulation in Modelsim
    #
    # Get all the simulation hdl input files
    # As a list of imports for the simulation.do
    sim_hdl_inputs = [
        f"project addfile {hdl_src_folder}/{f}"
        for f in sorted(os.listdir(hdl_src_folder))
        if f.endswith(".vhd")
    ]
    sim_hdl_inputs.append(f"project addfile {tb_file}")
    
    # Add verilog files if hdl set to verilog
    sim_folder = os.path.join(output_dir, "sim", "HDL_SRC")
    if hdl == "verilog":
        sim_hdl_inputs.extend([
            f"project addfile {sim_folder}/{f}"
            for f in sorted(os.listdir(sim_folder))
            if f.endswith(".v") or f.endswith(".sv")
    ])

    # which file is included varies on pre/post synth workflow
    # so remove the pre-synth file by default
    if hdl != "verilog":
        sim_hdl_inputs.remove(f"project addfile {hdl_src_folder}/{kernel_name}.vhd")
    else:
        sim_hdl_inputs.remove(f"project addfile {sim_folder}/{kernel_name}.v")
    sim_hdl_inputs = "\n".join(sim_hdl_inputs)
    
    
    #
    # Step 4: Get all files needed for power estimation in Vivado
    #
    # Get all the synthesis HDL files
    # As a list of imports for report_power.tcl
    if hdl == "verilog":
        synth_hdl_inputs = "\n" + "\n".join(
            f"read_verilog $HDL_SRC/{f}"
            for f in sorted(os.listdir(hdl_src_folder))
            if f.endswith(".v")) 
    else: 
        synth_hdl_inputs = "\n".join(
            f"read_vhdl -vhdl2008 $HDL_SRC/{f}"
            for f in sorted(os.listdir(hdl_src_folder))
            if f.endswith(".vhd"))

    # We have four configurations for generating the power estimation on post-synthesis
    # netlist from vivado
    # (Case 1) pre_pi:   Generating the SAIF file with behavior simualtion containing only PIs.
    # (Case 2) pre_all:  Generating the SAIF file with behavior simulation containing all ports.
    # (Case 3) post_pi:  Generating the SAIF file with post-synthesis simulation containing only PIs.
    # (Case 4) post_all: Generating the SAIF file with post-synthesis simulaiton containing all ports.

    # Currently only running case 2 and case 4
    design_flag = DesignFlag.PRE if not post_synth_netlist else DesignFlag.POST
    input_flag = InputFlag.ALL

    #
    # Step 5: If doing power estimation on post-synthesis netlist, run synthesis first
    #
    if design_flag == DesignFlag.POST:
        #  Step 1: Run Vivado synthesis flow
        synthesis_dict = {
            'date': date,
            'design': kernel_name,
            'hdlsrc': hdl_src_folder,
            'inputs': synth_hdl_inputs
        }

        synth_script = os.path.join(power_analysis_dir, "synthesis.tcl")

        # Generate the corresponding synthesis.tcl file
        target_file_generation(
            template_file=base_synthesis_tcl,
            substitute_dict=synthesis_dict,
            target_path=synth_script
        )

        print("[INFO] Pre-synthesizing " +
              "to improve switching activity annotation for power estimation")

        # Run the synthesis flow
        synthesis_command = f"cd {power_analysis_dir}; {vivado_cmd} -mode batch -source synthesis.tcl"
        if run_command(synthesis_command, power_analysis_dir):
            print("[INFO] Synthesis succeeded")
        else:
            print("[ERROR] Synthesis failed")
            return

    #
    # Step 6: Generate .do file for Modelsim simulation to generate SAIF file
    #
    if (input_flag == InputFlag.ALL):
        power_flag = "-r -in -inout -out -internal"
    else:
        power_flag = ""

    if (design_flag == DesignFlag.PRE):
        if hdl == "verilog":
            design_src = os.path.join(hdl_src_folder,  f"{kernel_name}.v")
        else:
            design_src = os.path.join(hdl_src_folder,  f"{kernel_name}.vhd")
    else:
        # Get the post-synthesis netlist
        # We generate VHDL netlist for both verilog and vhdl designs
        design_src = os.path.join(power_analysis_dir, f"{kernel_name}_syn.vhd")

    stage = f"{design_flag.value}_{input_flag.value}"
    simulation_dict = {
        'hdlsrc': hdl_src_folder,
        'design': kernel_name,
        'inputs': sim_hdl_inputs,
        'designsrc': design_src,
        'powerflag': power_flag,
        'stage': stage
    }

    verify_folder = os.path.join(output_dir, "sim", "HLS_VERIFY")
    simulation_script = os.path.join(verify_folder, f"{stage}.do")

    # Generate and run the simulation.do file
    target_file_generation(
        template_file=base_simulation_do,
        substitute_dict=simulation_dict,
        target_path=simulation_script
    )
    
    #
    # Step 7: Run simulation to generate SAIF file
    #
    #! A license is required for power analysis in Modelsim
    print("[INFO] Simulating to obtain switching activity information")

    modelsim_command = f"cd {verify_folder}; vsim -c -do {simulation_script}"
    if run_command(modelsim_command, power_analysis_dir):
        print("[INFO] Simulation succeeded")
    else:
        print("[ERROR] Simulation failed")
        return

    #
    # Step 8: Run Power Estimation in Vivado
    #
    power_dict = {
        'date': date,
        'design': kernel_name,
        'hdlsrc': hdl_src_folder,
        'report_folder': power_analysis_dir,
        'inputs': synth_hdl_inputs,
        'saif': os.path.join(verify_folder, f"{stage}.saif"),
    }

    report_power_script = os.path.join(power_analysis_dir, "report_power.tcl")
    target_file_generation(
        template_file=vector_base_report_power_tcl,
        substitute_dict=power_dict,
        target_path=report_power_script)

    print("[INFO] Launching power estimation")

    # Generate and run the report_power tcl script
    report_power_cmd = (
        f"cd {power_analysis_dir};" +
        f"{vivado_cmd} -mode batch -source {report_power_script}"
    )
    run_command(report_power_cmd, power_analysis_dir)
    if run_command(report_power_cmd, power_analysis_dir):
        print("[INFO] Power estimation succeeded")
    else:
        print("[ERROR] Power estimation failed")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--output_dir", required=True, help="Output folder")
    p.add_argument("--kernel_name", required=True, help="Name of kernel under test")
    p.add_argument("--hdl", required=True, help="HDL type (verilog or vhdl)", choices=["verilog", "vhdl"], default="vhdl")
    p.add_argument(
        "--synth",
        choices=["pre", "post"],
        required=False,
        help=(
            "Generate the SAIF file by simulating the pre-synthesis netlist, "
            "or the post-synthesis netlist. Using the post-synthesis netlist "
            "gives higher accuracy, but currently simulation may hang on some kernels."
        ),
        default="pre"
    )
    p.add_argument("--vivado_cmd", type=str, required=False, help="Vivado command", default="vivado")
    p.add_argument("--cp", type=float, required=True, help="Clock period for synthesis")

    args = p.parse_args()
    
    # Select which netlist to use for simulation
    use_post_synth_netlist = (args.synth == "post")

    main(args.output_dir, args.kernel_name, args.hdl, args.cp, args.vivado_cmd, use_post_synth_netlist)