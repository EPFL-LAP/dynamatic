#!/bin/bash

source "$1"/tools/dynamatic/scripts/utils.sh

# ============================================================================ #
# Variable definitions
# ============================================================================ #

# Script arguments
DYNAMATIC_DIR=$1
SRC_DIR=$2
OUTPUT_DIR=$3
KERNEL_NAME=$4

# Generated directories/files
SIM_DIR="$(realpath "$OUTPUT_DIR/sim")"
C_SRC_DIR="$SIM_DIR/C_SRC"
C_OUT_DIR="$SIM_DIR/C_OUT"
COSIM_HDL_SRC_DIR="$SIM_DIR/HDL_SRC"
COSIM_HDL_OUT_DIR="$SIM_DIR/HDL_OUT"
INPUT_VECTORS_DIR="$SIM_DIR/INPUT_VECTORS"
HLS_VERIFY_DIR="$SIM_DIR/HLS_VERIFY"
IO_GEN_BIN="$SIM_DIR/C_SRC/$KERNEL_NAME-io-gen"

# Shortcuts
HDL_DIR="$OUTPUT_DIR/hdl"
CLANGXX_BIN="$DYNAMATIC_DIR/bin/clang++"
HLS_VERIFIER_BIN="$DYNAMATIC_DIR/bin/hls-verifier"
RESOURCE_DIR="$DYNAMATIC_DIR/tools/hls-verifier/resources"

# ============================================================================ #
# Simulation flow
# ============================================================================ #

# Reset simulation directory
rm -rf "$SIM_DIR" && mkdir -p "$SIM_DIR"

# Create simulation directories
mkdir -p "$C_SRC_DIR" "$C_OUT_DIR" "$COSIM_HDL_SRC_DIR" "$COSIM_HDL_OUT_DIR" \
    "$INPUT_VECTORS_DIR" "$HLS_VERIFY_DIR"

# Copy integration header to a C source subdirectory so that its relative path
# is correct with respect to the source file containing the kernel 
DYN_INCLUDE_DIR="$C_SRC_DIR/dynamatic"
mkdir "$DYN_INCLUDE_DIR"
cp "$DYNAMATIC_DIR/include/dynamatic/Integration.h" "$DYN_INCLUDE_DIR"

# Copy VHDL module and VHDL components to dedicated folder
cp "$HDL_DIR/"*.vhd "$COSIM_HDL_SRC_DIR" 2> /dev/null
cp "$HDL_DIR/"*.v "$COSIM_HDL_SRC_DIR" 2> /dev/null

# Copy sources to dedicated folder
cp "$SRC_DIR/$KERNEL_NAME.c" "$C_SRC_DIR" 
# Suppress the error if the header file does not exist (it is optional).
cp "$SRC_DIR/$KERNEL_NAME.h" "$C_SRC_DIR" 2> /dev/null

# Copy TB supplementary files (memory model, etc.)
cp "$RESOURCE_DIR/template_tb_join.vhd" "$COSIM_HDL_SRC_DIR/tb_join.vhd"
cp "$RESOURCE_DIR/template_two_port_RAM.vhd" "$COSIM_HDL_SRC_DIR/two_port_RAM.vhd"
cp "$RESOURCE_DIR/template_single_argument.vhd" "$COSIM_HDL_SRC_DIR/single_argument.vhd"
cp "$RESOURCE_DIR/template_simpackage.vhd" "$COSIM_HDL_SRC_DIR/simpackage.vhd"
cp "$RESOURCE_DIR/modelsim.ini" "$HLS_VERIFY_DIR/modelsim.ini"

# Compile kernel's main function to generate inputs and golden outputs for the
# simulation
"$CLANGXX_BIN" "$SRC_DIR/$KERNEL_NAME.c" -D HLS_VERIFICATION \
  -DHLS_VERIFICATION_PATH="$SIM_DIR" -I "$DYNAMATIC_DIR/include" \
  -Wno-deprecated -o "$IO_GEN_BIN"
exit_on_fail "Failed to build kernel for IO gen." "Built kernel for IO gen." 

# Generate IO
"$IO_GEN_BIN"
exit_on_fail "Failed to run kernel for IO gen." "Ran kernel for IO gen." 

# Simulate and verify design
echo_info "Launching Modelsim simulation"
cd "$HLS_VERIFY_DIR"
"$HLS_VERIFIER_BIN" \
  --sim-path="$SIM_DIR" \
  --kernel-name="$KERNEL_NAME" \
  --handshake-mlir="$OUTPUT_DIR/comp/handshake_export.mlir" \
  > "../report.txt" 2>&1
exit_on_fail "Simulation failed" "Simulation succeeded"
