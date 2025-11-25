#!/usr/bin/env python3



################################################################
# Environment Setup
################################################################

import os
from enum import Enum
import argparse

# Import file templates
from script_templates import *

# Small handwritten util functions
from utils import *


################################################################
# Experiment Enums
################################################################

class DesignFlag(Enum):
    PRE = 1
    POST = 2

class InputFlag(Enum):
    PI = 1
    ALL = 2

################################################################
# Main Function
################################################################

def main(output_dir, kernel_name, clock_period):
    date = get_date()



    power_analysis_dir = os.path.join(output_dir, "power")

    synth_dir = os.path.join(power_analysis_dir, "synth_for_power")  

    os.makedirs(synth_dir)

    xdc_dict = {
        'tcp' : clock_period,
        'halftcp' : clock_period / 2
    }
    
    period_xdc_file = os.path.join(synth_dir, "period.xdc")
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
    ## (Case 1) pre_pi:   Generating the SAIF file with behavior simualtion containing only PIs.
    ## (Case 2) pre_all:  Generating the SAIF file with behavior simulation containing all ports.
    ## (Case 3) post_pi:  Generating the SAIF file with post-synthesis simulation containing only PIs.
    ## (Case 4) post_all: Generating the SAIF file with post-synthesis simulaiton containing all ports.

    for design_flag, input_flag in [(DesignFlag.PRE, InputFlag.ALL)]:

        # only need to do this synth if doing power estimation
        # on the post-synth vhdl code
        if design_src == DesignFlag.POST:
            #  Step 1: Run Vivado synthesis flow
            synthesis_dict = {
                'date'   : date,
                'design' : kernel_name,
                'hdlsrc' : vhdl_src_folder,
                'inputs' : vhdl_inputs
            }
            
            synth_script = os.path.join(synth_dir, "synthesis.tcl")

            # Generate the corresponding synthesis.tcl file
            target_file_generation(
                template_file=base_synthesis_tcl, 
                substitute_dict=synthesis_dict, 
                target_path=synth_script
                )
            
            # Run the synthesis flow
            synthesis_command = f"cd {synth_dir}; vivado -mode batch -source synthesis.tcl"
            os.system(synthesis_command)
            
        # Step 2: Run Modelsim simulation
        if (input_flag == "all"):
            power_flag = "-r -in -inout -out -internal"
        else:
            power_flag = ""
            
        if (design_flag == "pre"):
            design_src = os.path.join(vhdl_src_folder,  f"{kernel_name}.vhd")
        else:
            design_src = os.path.join(synth_dir, f"{kernel_name}_syn.vhd")
                
        simulation_dict = {
            'hdlsrc' : vhdl_src_folder,
            'design' : kernel_name,
            'inputs' : sim_inputs,
            'designsrc' : design_src,
            'powerflag' : power_flag,
            'stage' : design_src
        }
        
        verify_folder = os.path.join(output_dir, "sim", "HLS_VERIFY")
        simulation_script = os.path.join(verify_folder, f"{design_flag}.do")

        # Generate and run the simulation.do file
        target_file_generation(
            template_file=base_simulation_do, 
            substitute_dict=simulation_dict, 
            target_path=simulation_script
            )
            
        modelsim_command = f"cd {verify_folder}; vsim -c -do {simulation_script}"
        os.system(modelsim_command) 


        # Step 3: Run Power Estimation

        power_dict = {
            'date' : date,
            'design' : kernel_name,
            'hdlsrc' : vhdl_src_folder,
            'report' : power_analysis_dir,
            'inputs' : vhdl_inputs,
            'saif1'  : os.path.join(verify_folder, "pre_pi.saif"),
            'saif2'  : os.path.join(verify_folder,"pre_all.saif"),
            'saif3'  : os.path.join(verify_folder,"post_pi.saif"),
            'saif4'  : os.path.join(verify_folder,"post_all.saif")
        }
        
        report_power_script = os.path.join(power_analysis_dir, "report_power.tcl")
        target_file_generation(
            template_file=vector_base_report_power_tcl, 
            substitute_dict=power_dict, 
            target_path=report_power_script)
        
        print("Running power estimation for ", kernel_name)

        # Generate and run the report_power tcl script
        report_power_cmd = f"cd {power_analysis_dir}; vivado -mode batch -source {report_power_script}"
        os.system(report_power_cmd)

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--output_dir", required=True, help="Output folder")
    p.add_argument("--kernel_name", required=True, help="Name of kernel ")
    p.add_argument("--cp", type=float, required=True, help="Clock period for synthesis")


    args = p.parse_args()

    main(args.output_dir, args.kernel_name, args.cp)