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

# Shortcuts
COMP_DIR="$OUTPUT_DIR/comp"
LSQ_JAR="$DYNAMATIC_DIR/bin/generators/lsq-generator.jar"

# ============================================================================ #
# HDL writing flow
# ============================================================================ #

# Reset output directory
rm -rf "$HDL_DIR" && mkdir -p "$HDL_DIR"

"$DYNAMATIC_DIR/build/bin/exp-export-rtl" "$COMP_DIR/hw.mlir" "$HDL_DIR" \
  "$DYNAMATIC_DIR/experimental/data/rtl-config.json" \
  --dynamatic-path "$DYNAMATIC_DIR" --hdl $HDL
exit_on_fail "Failed to export RTL ($HDL)" "Exported RTL ($HDL)"

echo_info "HDL generation succeeded"
