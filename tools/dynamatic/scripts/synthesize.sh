#!/bin/bash

source "$1"/tools/dynamatic/scripts/utils.sh

# ============================================================================ #
# Variable definitions
# ============================================================================ #

# Script arguments
DYNAMATIC_DIR=$1
OUTPUT_DIR=$2
KERNEL_NAME=$3
FULL_CLOCK=$4
HALF_CLOCK=$5

# Generated directories/files
SYNTH_DIR="$OUTPUT_DIR/synth"
SYNTH_HDL_DIR="$SYNTH_DIR/hdl"
F_REPORT="$SYNTH_DIR/report.txt"
F_SCRIPT="$SYNTH_DIR/synthesize.tcl"
F_PERIOD="$SYNTH_DIR/period_${FULL_CLOCK}.xdc"
F_UTILIZATION_SYN="$SYNTH_DIR/utilization_post_syn.rpt"
F_TIMING_SYN="$SYNTH_DIR/timing_post_syn.rpt"
F_UTILIZATION_PR="$SYNTH_DIR/utilization_post_pr.rpt"
F_TIMING_PR="$SYNTH_DIR/timing_post_pr.rpt"

# Shortcuts
HDL_DIR="$OUTPUT_DIR/hdl"

# ============================================================================ #
# Synthesis flow
# ============================================================================ #

# Reset simulation directory
rm -rf "$SYNTH_DIR" && mkdir -p "$SYNTH_DIR"

# Copy all synthesizable components to specific folder for Vivado
mkdir -p "$SYNTH_HDL_DIR"
cp "$HDL_DIR/"*.vhd "$SYNTH_HDL_DIR" 2> /dev/null
cp "$HDL_DIR/"*.v "$SYNTH_HDL_DIR" 2> /dev/null

# See if we should include any VHDL in the synthesis script
READ_VHDL=""
if ls "$HDL_DIR"/*.vhd 1> /dev/null 2>&1; then
  cp "$HDL_DIR/"*.vhd "$SYNTH_HDL_DIR"
  READ_VHDL="read_vhdl -vhdl2008 [glob $SYNTH_DIR/hdl/*.vhd]"
fi

# See if we should include any Verilog in the synthesis script
READ_VERILOG=""
if ls "$HDL_DIR"/*.v 1> /dev/null 2>&1; then
  cp "$HDL_DIR/"*.v "$SYNTH_HDL_DIR"
  READ_VERILOG="read_verilog [glob $SYNTH_DIR/hdl/*.v]"
fi

# Generate synthesis script
echo -e \
"set_param general.maxThreads 8
$READ_VHDL
$READ_VERILOG
read_xdc "$F_PERIOD"
synth_design -top $KERNEL_NAME -part xc7k160tfbg484-2 -no_iobuf -mode out_of_context
report_utilization > $F_UTILIZATION_SYN
report_timing > $F_TIMING_SYN
opt_design
place_design
phys_opt_design
route_design
phys_opt_design
report_utilization > $F_UTILIZATION_PR
report_timing > $F_TIMING_PR
exit" > "$F_SCRIPT"

echo -e \
"create_clock -name clk -period $FULL_CLOCK -waveform {0.000 $HALF_CLOCK} [get_ports clk]
set_property HD.CLK_SRC BUFGCTRL_X0Y0 [get_ports clk]

#set_input_delay 0 -clock CLK  [all_inputs]
#set_output_delay 0 -clock CLK [all_outputs]" > "$F_PERIOD"

echo_info "Created synthesis scripts"
echo_info "Launching Vivado synthesis"
cd "$SYNTH_DIR"
vivado -mode tcl -source "$F_SCRIPT" > "$F_REPORT"
exit_on_fail "Logic synthesis failed" "Logic synthesis succeeded"
