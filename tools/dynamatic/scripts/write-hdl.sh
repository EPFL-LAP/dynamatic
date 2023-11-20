#!/bin/bash

source "$1"/tools/dynamatic/scripts/utils.sh

# ============================================================================ #
# Variable definitions
# ============================================================================ #

# Script arguments
DYNAMATIC_DIR=$1
LEGACY_DIR=$2
OUTPUT_DIR=$3
KERNEL_NAME=$4

COMP_DIR="$OUTPUT_DIR/comp"

# ============================================================================ #
# HDL writing flow
# ============================================================================ #

# Append legacy Dynamatic's bin folder to path so that lsq_generate is visible
export PATH="${LEGACY_DIR}/bin:${PATH}"
# Required by the lsq_generate script
export DHLS_INSTALL_DIR="${LEGACY_DIR}/../.."

# Convert DOT graph to VHDL
cd "$COMP_DIR"
"$LEGACY_DIR/dot2vhdl/bin/dot2vhdl" "$COMP_DIR/$KERNEL_NAME" >/dev/null
exit_on_fail "Failed to convert DOT to VHDL" "Converted DOT to VHDL"
echo_info "HDL generation succeeded"
