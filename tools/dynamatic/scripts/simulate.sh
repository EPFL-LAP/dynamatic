#!/bin/bash

# ============================================================================ #
# Variable definitions
# ============================================================================ #

# Script variables
DYNAMATIC_DIR=$1
LEGACY_DIR=$2
SRC_DIR=$3
OUTPUT_DIR=$4
KERNEL_NAME=$5

HLS_VERIFIER="$LEGACY_DIR/Regression_test/hls_verifier/HlsVerifier/build/hlsverifier"

# Generated directories/files
SIM_DIR="$OUTPUT_DIR/sim"
C_SRC_DIR="$SIM_DIR/C_SRC"
C_OUT_DIR="$SIM_DIR/C_OUT"
VHDL_SRC_DIR="$SIM_DIR/VHDL_SRC"
VHDL_OUT_DIR="$SIM_DIR/VHDL_OUT"
INPUT_VECTORS_DIR="$SIM_DIR/INPUT_VECTORS"
HLS_VERIFY_DIR="$SIM_DIR/HLS_VERIFY"

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
rm -rf "$SIM_DIR" && mkdir -p "$SIM_DIR"

# Create simulation directories
mkdir -p "$C_SRC_DIR" "$C_OUT_DIR" "$VHDL_SRC_DIR" "$VHDL_OUT_DIR" \
    "$INPUT_VECTORS_DIR" "$HLS_VERIFY_DIR"

# Copy integration headers to sim directory to make it visible by the HLS verifier
cp "$DYNAMATIC_DIR/integration-test/integration_utils.h" "$SIM_DIR"

# Copy VHDL module and VHDL components to dedicated folder
cp "$OUTPUT_DIR/$KERNEL_NAME.vhd" "$VHDL_SRC_DIR"
cp "$OUTPUT_DIR/"LSQ*.v "$VHDL_SRC_DIR" 2> /dev/null
cp "$LEGACY_DIR"/components/*.vhd "$VHDL_SRC_DIR"

# Copy sources to dedicated folder
cp "$SRC_DIR/$KERNEL_NAME.c" "$C_SRC_DIR" 
cp "$SRC_DIR/$KERNEL_NAME.h" "$C_SRC_DIR"

# Simulate and verify design
echo_info "Launching Modelsim simulation"
cd "$HLS_VERIFY_DIR"
"$HLS_VERIFIER" cover -aw32 "../C_SRC/$KERNEL_NAME.c" \
  "../C_SRC/$KERNEL_NAME.c" "$KERNEL_NAME" \
  > "../report.txt"
if [[ $? -ne 0 ]]; then
    echo_fatal "Simulation failed"
    exit 1
fi
echo_info "Simulation succeeded"

echo_info "All done!"
echo ""
