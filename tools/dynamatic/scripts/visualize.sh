#!/bin/bash

source "$1"/tools/dynamatic/scripts/utils.sh

# ============================================================================ #
# Variable definitions
# ============================================================================ #

# Script arguments
DYNAMATIC_DIR=$1
F_DOT=$2
F_WLF=$3
OUTPUT_DIR=$4
KERNEL_NAME=$5

# Generated directories/files
VISUAL_DIR="$OUTPUT_DIR/visual"
F_DOT_POS="$VISUAL_DIR/$KERNEL_NAME.dot"

# Shortcuts
WLF2CSV="$DYNAMATIC_DIR/visual-dataflow/wlf2csv.py"

# ============================================================================ #
# Simulation flow
# ============================================================================ #

# Reset visualization directory
rm -rf "$VISUAL_DIR" && mkdir -p "$VISUAL_DIR"

# Convert the Modelsim waveform to a CSV for the visualizer 
python3 "$WLF2CSV" "$F_DOT" "$F_WLF" "$VISUAL_DIR"
exit_on_fail "Failed to generate channel changes from waveform" "Generated channel changes"

# Generate a version of the DOT with positioning information
dot -Tdot "$F_DOT" > "$F_DOT_POS"
exit_on_fail "Failed to add positioning info. to DOT" "Added positioning info. to DOT"
