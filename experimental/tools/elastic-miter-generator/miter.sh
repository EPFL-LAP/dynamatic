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

build/bin/elastic-miter --lhs=$REWRITES/${MOD}_lhs.mlir --rhs=$REWRITES/${MOD}_rhs.mlir -o $OUT_DIR
exit_on_fail "Failed to create miter module"