#!/bin/bash

# ============================================================================ #
# Variable definitions
# ============================================================================ #

# Script variables
LEGACY_DIR=$1
OUTPUT_DIR=$2
KERNEL_NAME=$3

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
# HDL writing flow
# ============================================================================ #

# Append legacy Dynamatic's bin folder to path so that lsq_generate is visible
export PATH="${LEGACY_DIR}/bin:${PATH}"
# Required by the lsq_generate script
export DHLS_INSTALL_DIR="${LEGACY_DIR}/../.."

# Convert DOT graph to VHDL
"$LEGACY_DIR/dot2vhdl/bin/dot2vhdl" "$OUTPUT_DIR/$KERNEL_NAME" >/dev/null
if [[ $? -ne 0 ]]; then
    rm -rf LSQ*
    echo_fatal "Failed to convert DOT to VHDL"
    exit 1
fi
mv LSQ* "$OUTPUT_DIR" 2> /dev/null
rm -f "$OUTPUT_DIR"/*.tcl
echo_info "Converted DOT to VHDL"

echo_info "All done!"
echo ""
