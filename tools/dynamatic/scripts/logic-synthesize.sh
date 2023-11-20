#!/bin/bash

# ============================================================================ #
# Variable definitions
# ============================================================================ #

# Script variables
LEGACY_DIR=$1
OUTPUT_DIR=$2
KERNEL_NAME=$3

# Generated directories/files
SYNTH_DIR="$OUTPUT_DIR/synth"
HDL_DIR="$SYNTH_DIR/hdl"
F_REPORT="$SYNTH_DIR/report.txt"
F_SCRIPT="$SYNTH_DIR/synthesize.tcl"
F_PERIOD="$SYNTH_DIR/period_4.xdc"
F_UTILIZATION_SYN="$SYNTH_DIR/utilization_post_syn.rpt"
F_TIMING_SYN="$SYNTH_DIR/timing_post_syn.rpt"
F_UTILIZATION_PR="$SYNTH_DIR/utilization_post_pr.rpt"
F_TIMING_PR="$SYNTH_DIR/timing_post_pr.rpt"


# ============================================================================ #
# Helper funtions
# ============================================================================ #

# Prints some information to stdout.
#   $1: the text to print
echo_info() {
    echo "[INFO] $1"
}

# Prints a fatal error message to stdout.
#   $1: the text to print
echo_fatal() {
    echo "[FATAL] $1"
}

# ============================================================================ #
# Simulation flow
# ============================================================================ #

# Reset simulation directory
rm -rf "$SYNTH_DIR" && mkdir -p "$SYNTH_DIR"

# Copy all synthesizable components to specific folder for Vivado
mkdir -p "$HDL_DIR"
cp "$OUTPUT_DIR/$KERNEL_NAME.vhd" "$HDL_DIR"
cp "$LEGACY_DIR"/components/*.vhd "$HDL_DIR"

# See if we should include any LSQ in the synthesis script
READ_VERILOG=""
if ls "$OUTPUT_DIR"/LSQ*.v 1> /dev/null 2>&1; then
  cp "$OUTPUT_DIR/"LSQ*.v "$HDL_DIR"
  READ_VERILOG="read_verilog [glob $SYNTH_DIR/hdl/*.v]"
fi

# Generate synthesis script
echo -e \
"set_param general.maxThreads 8
read_vhdl -vhdl2008 [glob $SYNTH_DIR/hdl/*.vhd]
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
"create_clock -name clk -period 4.000 -waveform {0.000 2.000} [get_ports clk]
set_property HD.CLK_SRC BUFGCTRL_X0Y0 [get_ports clk]

#set_input_delay 0 -clock CLK  [all_inputs]
#set_output_delay 0 -clock CLK [all_outputs]" > "$F_PERIOD"

echo_info "Created synthesization scripts"
echo_info "Launching Vivado synthesis"
vivado -mode tcl -source "$F_SCRIPT" > "$F_REPORT"
RET=$?
rm -rf *.jou *.log .Xil
if [[ $RET -ne 0 ]]; then
  echo_fatal "Logic synthesis failed"
  exit 1
fi
echo_info "Logic synthesis succeeded"

echo_info "All done!"
echo ""
