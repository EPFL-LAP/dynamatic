DYNAMATIC_DIR=.

source "$DYNAMATIC_DIR/tools/dynamatic/scripts/utils.sh"
DYNAMATIC_OPT_BIN="$DYNAMATIC_DIR/bin/dynamatic-opt"

OUT_DIR="experimental/tools/elastic-miter/out"
F_HW="$OUT_DIR/hw.mlir"

COMP_DIR="$OUT_DIR/comp"
F_HANDSHAKE_MITER="$COMP_DIR/handshake_miter.mlir"
DOT="$COMP_DIR/miter.dot"

REWRITES="experimental/test/tools/elastic-miter/rewrites"

# TODO remove this
cd build
ninja
exit_on_fail "Failed to build miter module generator"
cd ..


build/bin/elastic-miter --lhs=$REWRITES/a_lhs.mlir --rhs=$REWRITES/a_rhs.mlir -o $OUT_DIR
exit_on_fail "(a): Equivalence checking failed"

# build/bin/elastic-miter --lhs=$REWRITES/b_lhs.mlir --rhs=$REWRITES/b_rhs.mlir -o $OUT_DIR --seq_length="0+1=3" --seq_length="0=2" --loop_strict=0,1
# exit_on_fail "(b): Equivalence checking failed"

# build/bin/elastic-miter --lhs=$REWRITES/c_lhs.mlir --rhs=$REWRITES/c_rhs.mlir -o $OUT_DIR --seq_length="0=1"
# exit_on_fail "(c): Equivalence checking failed"

# build/bin/elastic-miter --lhs=$REWRITES/d_lhs.mlir --rhs=$REWRITES/d_rhs.mlir -o $OUT_DIR --loop_strict=0,1
# exit_on_fail "(d): Equivalence checking failed"

# build/bin/elastic-miter --lhs=$REWRITES/e_lhs.mlir --rhs=$REWRITES/e_rhs.mlir -o $OUT_DIR --seq_length="0=1"
# exit_on_fail "(e): Equivalence checking failed"

# build/bin/elastic-miter --lhs=$REWRITES/f_lhs.mlir --rhs=$REWRITES/f_rhs.mlir -o $OUT_DIR
# exit_on_fail "(f): Equivalence checking failed"

# build/bin/elastic-miter --lhs=$REWRITES/g_lhs.mlir --rhs=$REWRITES/g_rhs.mlir -o $OUT_DIR
# exit_on_fail "(g): Equivalence checking failed"

# build/bin/elastic-miter --lhs=$REWRITES/h_lhs.mlir --rhs=$REWRITES/h_rhs.mlir -o $OUT_DIR
# exit_on_fail "(h): Equivalence checking failed"

# build/bin/elastic-miter --lhs=$REWRITES/i_lhs.mlir --rhs=$REWRITES/i_rhs.mlir -o $OUT_DIR
# exit_on_fail "(i): Equivalence checking failed"