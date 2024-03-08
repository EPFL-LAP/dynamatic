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
F_DOT_POS_TMP="$VISUAL_DIR/$KERNEL_NAME.tmp.dot"
F_DOT_POS="$VISUAL_DIR/$KERNEL_NAME.dot"

# Shortcuts
WLF2CSV="$DYNAMATIC_DIR/visual-dataflow/wlf2csv.py"
VISUAL_DATAFLOW_BIN="$DYNAMATIC_DIR/bin/visual-dataflow"

# ============================================================================ #
# Visualization flow
# ============================================================================ #

# Reset visualization directory
rm -rf "$VISUAL_DIR" && mkdir -p "$VISUAL_DIR"

# Convert the Modelsim waveform to a CSV for the visualizer 
python3 "$WLF2CSV" "$F_DOT" "$F_WLF" "$VISUAL_DIR"
exit_on_fail "Failed to generate channel changes from waveform" "Generated channel changes"

# Generate a version of the DOT with positioning information
sed -e 's/splines=spline/splines=ortho/g' "$F_DOT" > "$F_DOT_POS_TMP"
dot -Tdot "$F_DOT_POS_TMP" > "$F_DOT_POS"
exit_on_fail "Failed to add positioning info. to DOT" "Added positioning info. to DOT"
rm "$F_DOT_POS_TMP"

# Launch the dataflow visualizer
"$VISUAL_DATAFLOW_BIN" "--dot=$F_DOT_POS" "--csv=$VISUAL_DIR/sim.csv" \
  2>&1 >/dev/null &
