vivado_power_evaluation_tcl = r"""
# Date: %date
# Simulation with XSim + SAIF dump + power extraction
# Command %vivado_cmd -mode batch -source power_extraction.tcl
# =============================================================

# =============================================================
# 0. Setup
# =============================================================
set TOP_DESIGN %top_design
set TB_TOP     %tb_top

set V_SRC_DIR  %v_src_dir
set SIM_SRC_DIR %sim_src_dir
set TB_FILE    %tb_file

set XDC_FILE   %xdc_file

# SAIF output
set SAIF_OUT   %saif_pre


# =============================================================
# 1. Create an in-memory project
# =============================================================
create_project -force -part xc7k160tfbg484-1 pre_synth_sim

# Optional but often helpful for clean compilation behavior
set_property target_language %target_language [current_project]
set_property simulator_language Mixed [current_project]

# =============================================================
# 2. Add RTL sources to sources_1
# =============================================================
add_files -fileset sources_1 [list \
%{rtl_sources}
]

# Constraints (not required for behavioral sim, but harmless if you want periods/IO constraints loaded)
if {[file exists $XDC_FILE]} {
  add_files -fileset constrs_1 $XDC_FILE
}

# Ensure Vivado knows the intended RTL top (helps elaboration)
set_property top $TOP_DESIGN [get_filesets sources_1]

# =============================================================
# 3. Add testbench to sim_1 and set simulation top
# =============================================================
add_files -fileset sim_1 [list \
%{sim_sources}
]
set_property top $TB_TOP [get_filesets sim_1]
set_property top_lib xil_defaultlib [get_filesets sim_1]

update_compile_order -fileset sources_1
update_compile_order -fileset sim_1

# Disable auto simulation limit, allow SAIF for all signals
set_property -name {xsim.simulate.runtime}          -value {0ns}  -objects [get_filesets sim_1]
set_property -name {xsim.simulate.saif_all_signals} -value {true} -objects [get_filesets sim_1]

# =============================================================
# 4. Launch pre-synthesis simulation and dump SAIF
# =============================================================
launch_simulation

# If your DUT instance path is tb/duv_inst:
current_scope /tb/duv_inst

# SAIF dump
open_saif $SAIF_OUT
log_saif [get_objects -r]

# Run the sim
run all

close_saif
close_sim

%{synth_block}
%{impl_block}

# =============================================================
# Exit
# =============================================================
exit
"""
