DYNAMATIC_DIR=.

source "$DYNAMATIC_DIR/tools/dynamatic/scripts/utils.sh"
DYNAMATIC_OPT_BIN="$DYNAMATIC_DIR/bin/dynamatic-opt"

OUT_DIR="experimental/tools/elastic-miter-generator/out"
F_HW="$OUT_DIR/hw.mlir"

COMP_DIR="$OUT_DIR/comp"
F_HANDSHAKE_MITER="$COMP_DIR/handshake_miter.mlir"
DOT="$COMP_DIR/miter.dot"

"bin/export-dot" $F_HANDSHAKE_MITER "--edge-style=spline" > $DOT
dot -Tpng $DOT > $COMP_DIR/visual.png

python3 "../dot2smv/dot2smv" $DOT

python3 experimental/tools/elastic-miter-generator/export-property-svm.py > experimental/tools/elastic-miter-generator/out/comp/add_props.cmd

nuXmv -source $COMP_DIR/prove.cmd