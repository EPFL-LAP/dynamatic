DYNAMATIC_DIR=.

source "$DYNAMATIC_DIR/tools/dynamatic/scripts/utils.sh"
DYNAMATIC_OPT_BIN="$DYNAMATIC_DIR/bin/dynamatic-opt"

OUT_DIR="experimental/tools/elastic-miter-generator/out"
F_HW="$OUT_DIR/hw.mlir"

COMP_DIR="$OUT_DIR/comp"
F_HANDSHAKE_MITER="$COMP_DIR/handshake_miter.mlir"
DOT="$COMP_DIR/miter.dot"

REWRITES="experimental/test/tools/elastic-miter-generator/rewrites"

# TODO remove this
cd build
ninja
exit_on_fail "Failed to build miter module generator"
cd ..

MOD="a"

build/bin/elastic-miter --lhs=$REWRITES/${MOD}_lhs.mlir --rhs=$REWRITES/${MOD}_rhs.mlir -o $COMP_DIR --bufferSlots 1
exit_on_fail "Failed to create miter module"

"bin/export-dot" $F_HANDSHAKE_MITER "--edge-style=spline" > $DOT
exit_on_fail "Failed to convert to dot"
dot -Tpng $DOT > $COMP_DIR/visual.png

python3 "../dot2smv/dot2smv" $DOT
exit_on_fail "Failed to convert to SMV"

# python3 experimental/tools/elastic-miter-generator/export-property-svm.py > experimental/tools/elastic-miter-generator/out/comp/add_props.cmd
# exit_on_fail "Failed to create properties"

python3 experimental/tools/elastic-miter-generator/create_smv_wrapper.py > experimental/tools/elastic-miter-generator/out/comp/main.smv
exit_on_fail "Failed to create SMV main file"

nuXmv -source $COMP_DIR/prove.cmd