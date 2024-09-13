#!/bin/bash

source "$1"/tools/dynamatic/scripts/utils.sh

# ============================================================================ #
# Variable definitions
# ============================================================================ #

# Script arguments
DYNAMATIC_DIR=$1
OUTPUT_DIR=$2
KERNEL_NAME=$3
HDL=$4

# Generated directories/files
HDL_DIR="$OUTPUT_DIR/hdl"
COMP_DIR="$OUTPUT_DIR/comp"

# ============================================================================ #
# HDL writing flow
# ============================================================================ #

# Reset output directory
rm -rf "$HDL_DIR" && mkdir -p "$HDL_DIR"

# Set the correct config file
RTL_CONFIG=""
if [ "$HDL" == "vhdl" ]; then
  RTL_CONFIG="$DYNAMATIC_DIR/data/rtl-config-vhdl.json"
elif [ "$HDL" == "verilog" ]; then
  RTL_CONFIG="$DYNAMATIC_DIR/data/rtl-config-verilog.json"
fi

"$DYNAMATIC_DIR/bin/export-rtl" "$COMP_DIR/hw.mlir" "$HDL_DIR" $RTL_CONFIG \
  --dynamatic-path "$DYNAMATIC_DIR" --hdl $HDL
exit_on_fail "Failed to export RTL ($HDL)" "Exported RTL ($HDL)"

echo_info "HDL generation succeeded"
