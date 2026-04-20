#!/usr/bin/env python3
#! For now the flow only supports Verilog backend

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

class DesignStage(Enum):
    SYNTH = "synth"
    IMPL = "impl"


def _normalize_hdl(hdl):
    if hdl == "verilog-beta":
        return "verilog"
    return hdl


def _list_files(folder_path, extensions):
    return sorted(
        f for f in os.listdir(folder_path)
        if f.endswith(tuple(extensions))
    )


def _find_tb_file(sim_src_dir, kernel_name):
    tb_vhd = os.path.join(sim_src_dir, f"tb_{kernel_name}.vhd")
    if os.path.exists(tb_vhd):
        return tb_vhd
    tb_v = os.path.join(sim_src_dir, f"tb_{kernel_name}.v")
    if os.path.exists(tb_v):
        return tb_v
    return ""


def _build_rtl_list(rtl_files):
    return "\n".join(f"  $V_SRC_DIR/{f} \\" for f in rtl_files)


def _build_sim_list(sim_files, tb_file):
    lines = []
    tb_name = os.path.basename(tb_file)
    for fname in sim_files:
        if fname == tb_name:
            continue
        lines.append(f"  $SIM_SRC_DIR/{fname} \\")
    if tb_name:
        lines.append("  $TB_FILE \\")
    return "\n".join(lines)


def run_command(command, power_dir):
    report = os.path.join(power_dir, "report.txt")
    ret = os.system(command + f" >> '{report}'")
    return ret == 0


################################################################
# Main Function
################################################################

def main(output_dir, kernel_name, hdl, clock_period, stage, flatten_hierarchy,
         vivado_cmd="vivado", run_flow=True):
    print("[INFO] Generating power evaluation script")

    # Normalize HDL selection
    # For now, verilog-beta is treated as verilog
    hdl = _normalize_hdl(hdl)

    # Resolve paths
    output_dir = os.path.abspath(output_dir)
    power_dir = os.path.join(output_dir, "power")
    hdl_src_dir = os.path.join(output_dir, "hdl")
    sim_src_dir = os.path.join(output_dir, "sim", "HDL_SRC")
    verify_dir = os.path.join(output_dir, "sim", "HLS_VERIFY")

    if not os.path.exists(hdl_src_dir):
        print(f"[ERROR] {hdl_src_dir} not found. Please run the 'write-hdl' command")
        return

    if not os.path.exists(sim_src_dir):
        print(f"[ERROR] {sim_src_dir} not found. Please run the 'simulate' command")
        return

    check_else_create(power_dir)

    tb_file = _find_tb_file(sim_src_dir, kernel_name)
    if not tb_file:
        print(f"[ERROR] Testbench not found in {sim_src_dir}")
        return

    # Generate the xdc file for synthesis and implementation in Vivado
    xdc_dict = {
        "tcp": clock_period,
        "halftcp": clock_period / 2
    }
    period_xdc_file = os.path.join(power_dir, "period.xdc")
    target_file_generation(
        template_file=base_xdc,
        substitute_dict=xdc_dict,
        target_path=period_xdc_file
    )

    # Collect RTL sources
    rtl_extensions = [".v"] if hdl == "verilog" else [".vhd"]
    rtl_files = _list_files(hdl_src_dir, rtl_extensions)
    if not rtl_files:
        print(f"[ERROR] No RTL sources found in {hdl_src_dir}")
        return

    # Collect simulation sources
    if hdl == "vhdl":
        sim_files = [f for f in _list_files(sim_src_dir, [".vhd"]) if f not in rtl_files]
        design_vhd = f"{kernel_name}.vhd"
        sim_files = [f for f in sim_files if f != design_vhd]
    else:
        sim_files = _list_files(sim_src_dir, [".sv"])
        sim_files.append("tb_join.v")
    sim_files = [f for f in sim_files if f != os.path.basename(tb_file)]

    # Build lists for TCL
    rtl_sources = _build_rtl_list(rtl_files)
    sim_sources = _build_sim_list(sim_files, tb_file)

    # Resolve output paths
    pre_synth_saif = os.path.join(verify_dir, "pre_synth.saif")

    post_synth_fun_saif = os.path.join(verify_dir, "post_synth_fun.saif")
    post_synth_fun_vcd = os.path.join(verify_dir, "post_synth_fun.vcd")
    post_synth_time_saif = os.path.join(verify_dir, "post_synth_time.saif")
    post_synth_time_vcd = os.path.join(verify_dir, "post_synth_time.vcd")

    post_impl_fun_saif = os.path.join(verify_dir, "post_impl_fun.saif")
    post_impl_fun_vcd = os.path.join(verify_dir, "post_impl_fun.vcd")
    post_impl_time_saif = os.path.join(verify_dir, "post_impl_time.saif")
    post_impl_time_vcd = os.path.join(verify_dir, "post_impl_time.vcd")

    post_synth_fun_rpt = os.path.join(power_dir, "post_synth_fun_power.rpt")
    post_synth_time_rpt = os.path.join(power_dir, "post_synth_time_power.rpt")

    post_impl_fun_rpt = os.path.join(power_dir, "post_impl_fun_power.rpt")
    post_impl_time_rpt = os.path.join(power_dir, "post_impl_time_power.rpt")

    flatten_line = ""
    if not flatten_hierarchy:
        flatten_line = (
            "set_property STEPS.SYNTH_DESIGN.ARGS.FLATTEN_HIERARCHY none "
            "[get_runs synth_1]"
        )

    synth_lines = [
        "# =============================================================",
        "# 5. Post-synthesis simulations (functional & timing)",
        "# =============================================================",
    ]
    if flatten_line:
        synth_lines.append(flatten_line)
    synth_lines += [
        "launch_runs synth_1 -jobs 8",
        "wait_on_run synth_1",
        "open_run synth_1",
        "",
        "# --- 5a. Functional netlist sim ---",
        "launch_simulation -mode post-synthesis -type functional",
        "current_scope /tb/duv_inst",
        f"open_saif {post_synth_fun_saif}",
        "log_saif [get_objects -r]",
        f"open_vcd {post_synth_fun_vcd}",
        "log_vcd",
        "",
        "run all",
        "close_saif",
        "close_vcd",
        "close_sim",
        "",
        "# --- 5b. Timing netlist sim ---",
        "launch_simulation -mode post-synthesis -type timing",
        "current_scope /tb/duv_inst",
        f"open_saif {post_synth_time_saif}",
        "log_saif [get_objects -r]",
        f"open_vcd {post_synth_time_vcd}",
        "log_vcd",
        "",
        "run all",
        "close_saif",
        "close_vcd",
        "close_sim",
        "",
        "# =============================================================",
        "# 6. Power reports for post-synthesis simulations",
        "# =============================================================",
        f"read_saif -file {post_synth_fun_saif}",
        f"report_power -file {post_synth_fun_rpt}",
        "reset_switching_activity -all",
        "",
        f"read_saif -file {post_synth_time_saif}",
        f"report_power -file {post_synth_time_rpt} -hierarchical_depth 0 -verbose",
        "reset_switching_activity -all",
    ]
    synth_block = "\n".join(synth_lines) + "\n"

    impl_block = ""
    if stage == DesignStage.IMPL:
        impl_lines = [
            "# =============================================================",
            "# 7. Post-implementation simulations (functional & timing)",
            "# =============================================================",
            "launch_runs impl_1 -to_step route_design -jobs 8",
            "wait_on_run impl_1",
            "",
            "open_run impl_1",
            "",
            "# --- 7a. Functional routed sim ---",
            "launch_simulation -mode post-implementation -type functional",
            "current_scope /tb/duv_inst",
            f"open_saif {post_impl_fun_saif}",
            "log_saif [get_objects -r]",
            f"open_vcd {post_impl_fun_vcd}",
            "log_vcd",
            "",
            "run all",
            "close_saif",
            "close_vcd",
            "close_sim",
            "",
            "# --- 7b. Timing routed sim ---",
            "launch_simulation -mode post-implementation -type timing",
            "current_scope /tb/duv_inst",
            f"open_saif {post_impl_time_saif}",
            "log_saif [get_objects -r]",
            f"open_vcd {post_impl_time_vcd}",
            "log_vcd",
            "",
            "# Only works for Verilog designs for now",
            "run 8000ns",
            "close_saif",
            "close_vcd",
            "close_sim",
            "",
            "# =============================================================",
            "# 8. Power reports for post-implementation simulations",
            "# =============================================================",
            f"read_saif -file {post_impl_fun_saif}",
            f"report_power -file {post_impl_fun_rpt}",
            "reset_switching_activity -all",
            "",
            f"read_saif -file {post_impl_time_saif}",
            f"report_power -file {post_impl_time_rpt}",
        ]
        impl_block = "\n".join(impl_lines) + "\n"

    target_language = "VHDL" if hdl == "vhdl" else "Verilog"
    eval_dict = {
        "date": get_date(),
        "vivado_cmd": vivado_cmd,
        "top_design": kernel_name,
        "tb_top": "tb",
        "v_src_dir": hdl_src_dir,
        "sim_src_dir": sim_src_dir,
        "tb_file": tb_file,
        "xdc_file": period_xdc_file,
        "saif_pre": pre_synth_saif,
        "rtl_sources": rtl_sources,
        "sim_sources": sim_sources,
        "target_language": target_language,
        "synth_block": synth_block,
        "impl_block": impl_block,
    }

    power_extraction_script = os.path.join(power_dir, "power_extraction.tcl")
    target_file_generation(
        template_file=vivado_power_evaluation_tcl,
        substitute_dict=eval_dict,
        target_path=power_extraction_script
    )

    print(f"[INFO] Power evaluation script generated at {power_extraction_script}")

    if run_flow:
        print("[INFO] Launching Vivado power evaluation")
        eval_command = (
            f"cd '{power_dir}'; "
            f"{vivado_cmd} -mode batch -source '{power_extraction_script}'"
        )
        if run_command(eval_command, power_dir):
            print("[INFO] Vivado power evaluation finished")
        else:
            print("[ERROR] Vivado power evaluation failed")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--output_dir", required=True, help="Output folder")
    p.add_argument("--kernel_name", required=True, help="Name of kernel under test")
    p.add_argument(
        "--hdl",
        required=True,
        help="HDL type (verilog, verilog-beta or vhdl)",
        choices=["verilog", "verilog-beta", "vhdl"],
        default="vhdl"
    )
    p.add_argument(
        "--stage",
        dest="stage",
        choices=["synth", "impl"],
        required=False,
        help=(
            "Stage to perform simulation with XSim and vector-based power "
            "evaluation, synthesis or implementation (default: synth)."
        ),
        default="synth"
    )
    p.add_argument(
        "--flatten_hierarchy",
        action="store_true",
        help=(
            "Control hierarchy flattening during synthesis. With 'false' to emit "
            "the FLATTEN_HIERARCHY none property, or 'true' for the fully flattened flow."
        ),
    )
    p.add_argument("--vivado_cmd", type=str, required=False, help="Vivado command", default="vivado")
    p.add_argument("--cp", type=float, required=True, help="Clock period for design implementation")
    p.add_argument(
        "--no-run",
        action="store_true",
        help="Only generate the TCL script without running Vivado."
    )

    args = p.parse_args()

    # Default to synthesis stage if not specified
    stage = args.stage if args.stage is not None else "synth"
    
    # Exit when the hdl is VHDL
    if args.hdl == "vhdl":
        print("[ERROR] VHDL flow is not supported yet for power evaluation. Please use Verilog backend instead.")
        exit(1)

    main(
        args.output_dir,
        args.kernel_name,
        args.hdl,
        args.cp,
        DesignStage(stage),
        args.flatten_hierarchy,
        args.vivado_cmd,
        run_flow=not args.no_run,
    )
