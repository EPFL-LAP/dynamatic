#!/usr/bin/env python3

################################################################
# Environment Setup
################################################################

import os
from enum import Enum
import argparse
from pdb import run
import shutil

from numpy import power

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
    PI = "pi"
    ALL = "all"


def run_command(command, power_analysis_dir):
    report = os.path.join(power_analysis_dir, "report.txt")
    ret = os.system(command + f" >> {report}")
    return ret == 0

################################################################
# Main Function
################################################################


def main(output_dir, kernel_name, clock_period):
    print("[INFO] Running power estimation")

    date = get_date()

    power_analysis_dir = os.path.join(output_dir, "power")

    if os.path.exists(power_analysis_dir):
        shutil.rmtree(power_analysis_dir)

    os.makedirs(power_analysis_dir)

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

    vhdl_src_folder = os.path.join(output_dir, "sim", "HDL_SRC")

    # Get all the input files
    # As a list of imports for simulation.do
    sim_inputs = [
        f"project addfile {vhdl_src_folder}/{f}"
        for f in sorted(os.listdir(vhdl_src_folder))
        if f.endswith(".vhd")
    ]
    # which file is included varies on pre/post synth workflow
    # so remove the pre-synth file by default
    sim_inputs.remove(f"project addfile {vhdl_src_folder}/{kernel_name}.vhd")
    sim_inputs = "\n".join(sim_inputs)

    # Get all the input VHDl files
    # As a list of imports for report_power.tcl
    vhdl_inputs = "\n".join(
        f"read_vhdl -vhdl2008 $VHDL_SRC/{f}"
        for f in sorted(os.listdir(vhdl_src_folder))
        if f.endswith(".vhd")
    )

    # We have four configurations for generating the power estimation on post-synthesis
    # netlist from vivado
    # (Case 1) pre_pi:   Generating the SAIF file with behavior simualtion containing only PIs.
    # (Case 2) pre_all:  Generating the SAIF file with behavior simulation containing all ports.
    # (Case 3) post_pi:  Generating the SAIF file with post-synthesis simulation containing only PIs.
    # (Case 4) post_all: Generating the SAIF file with post-synthesis simulaiton containing all ports.

    # Currently only running pre_all
    design_flag = DesignFlag.PRE
    input_flag = InputFlag.ALL

    # only need to do this synth if doing power estimation
    # on the post-synth vhdl code
    if design_flag == DesignFlag.POST:
        #  Step 1: Run Vivado synthesis flow
        synthesis_dict = {
            'date': date,
            'design': kernel_name,
            'hdlsrc': vhdl_src_folder,
            'inputs': vhdl_inputs
        }

        synth_script = os.path.join(power_analysis_dir, "synthesis.tcl")

        # Generate the corresponding synthesis.tcl file
        target_file_generation(
            template_file=base_synthesis_tcl,
            substitute_dict=synthesis_dict,
            target_path=synth_script
        )

        print("[INFO] Pre-synthesizing" +
              "to improve switching activity annotation for power estimation")

        # Run the synthesis flow
        synthesis_command = f"cd {power_analysis_dir}; vivado -mode batch -source synthesis.tcl"
        if run_command(synthesis_command, power_analysis_dir):
            print("[INFO] Synthesis succeeded")
        else:
            print("[ERROR] Synthesis failed")
            return

    # Step 2: Run Modelsim simulation
    if (input_flag == InputFlag.ALL):
        power_flag = "-r -in -inout -out -internal"
    else:
        power_flag = ""

    if (design_flag == DesignFlag.PRE):
        design_src = os.path.join(vhdl_src_folder,  f"{kernel_name}.vhd")
    else:
        design_src = os.path.join(power_analysis_dir, f"{kernel_name}_syn.vhd")

    stage = f"{design_flag.value}_{input_flag.value}"
    simulation_dict = {
        'hdlsrc': vhdl_src_folder,
        'design': kernel_name,
        'inputs': sim_inputs,
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

    print("[INFO] Simulating to obtain switching activity information")

    modelsim_command = f"cd {verify_folder}; vsim -c -do {simulation_script}"
    if run_command(modelsim_command, power_analysis_dir):
        print("[INFO] Simulation succeeded")
    else:
        print("[ERROR] Simulation failed")
        return

    # Step 3: Run Power Estimation
    power_dict = {
        'date': date,
        'design': kernel_name,
        'hdlsrc': vhdl_src_folder,
        'report_folder': power_analysis_dir,
        'inputs': vhdl_inputs,
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
        f"vivado -mode batch -source {report_power_script}"
    )
    run_command(report_power_cmd, power_analysis_dir)
    if run_command(report_power_cmd, power_analysis_dir):
        print("[INFO] Power estimation succeeded")
    else:
        print("[ERROR] Power estimation failed")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--output_dir", required=True, help="Output folder")
    p.add_argument("--kernel_name", required=True, help="Name of kernel ")
    p.add_argument("--cp", type=float, required=True, help="Clock period for synthesis")

    args = p.parse_args()

    main(args.output_dir, args.kernel_name, args.cp)
