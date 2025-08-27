#!/bin/bash

# NOTE: this script is temporarily used for evaluating the new frontend. Should
# be removed in the future.

DYNAMATIC=$1

SOURCE="$2"
KERNEL_NAME="$3"
OUTPUT_DIR=$(realpath "$4")
COMP_DIR="$OUTPUT_DIR/comp"
COSIM_DIR=$(realpath "$5")

# Run cosimulation to make sure that the results are ok
rm -rf "$COSIM_DIR/"

mkdir -p "$COSIM_DIR/"{HDL_SRC,HDL_OUT,C_SRC,C_OUT,INPUT_VECTORS,HLS_VERIFY}

mkdir -p "$COSIM_DIR/C_SRC/dynamatic/"

cp "$DYNAMATIC/include/dynamatic/Integration.h" \
  "$COSIM_DIR/C_SRC/dynamatic/"

cp "$SOURCE" "$COSIM_DIR/C_SRC"

cp "$DYNAMATIC/tools/hls-verifier/resources/"*.vhd \
  "$COSIM_DIR/HDL_SRC/"

cp "$DYNAMATIC/tools/hls-verifier/resources/"modelsim.ini \
  "$COSIM_DIR/HLS_VERIFY/modelsim.ini"

cp "$OUTPUT_DIR/hdl/"*.vhd "$COSIM_DIR/HDL_SRC" 2> /dev/null
cp "$OUTPUT_DIR/hdl/"*.v "$COSIM_DIR/HDL_SRC" 2> /dev/null

"$DYNAMATIC/polygeist/llvm-project/build/bin/clang++" \
  "$SOURCE" \
  -D HLS_VERIFICATION \
  -DHLS_VERIFICATION_PATH="$COSIM_DIR" \
  -I "$DYNAMATIC/include" \
  -Wno-deprecated \
  -o "$COSIM_DIR/HLS_VERIFY/output_gen"

"$COSIM_DIR/HLS_VERIFY/output_gen"

cd "$COSIM_DIR/HLS_VERIFY"

"$DYNAMATIC/build/bin/hls-verifier" \
  --sim-path="$COSIM_DIR" \
  --kernel-name="$KERNEL_NAME" \
  --handshake-mlir="$COMP_DIR/handshake_export.mlir" \
  > "../report.txt"
