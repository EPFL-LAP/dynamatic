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

# Convert DOT graph to VHDL
cd "$COMP_DIR"
"$DYNAMATIC_DIR/bin/export-vhdl" $KERNEL_NAME "$COMP_DIR/$KERNEL_NAME.dot" \
  "$COMP_DIR"
exit_on_fail "Failed to convert DOT to VHDL" "Converted DOT to VHDL"

# Generate LSQs
for lsq_config in $COMP_DIR/lsq*_config.json; do
  # Skip non-existent files (the loop will iterate over
  # $COMP_DIR/lsq*_config.json if there are no LSQs)
  if [ ! -e "$lsq_config" ]; then continue; fi

  # Run the LSQ generator
  java -jar -Xmx7G $LEGACY_DIR/chisel_lsq/jar/lsq.jar \
    --target-dir $COMP_DIR --spec-file $lsq_config > /dev/null
  exit_on_fail "Failed to generate LSQs"
done

echo_info "HDL generation succeeded"
