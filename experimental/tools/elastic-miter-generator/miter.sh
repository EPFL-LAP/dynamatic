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

# TODO add constraints
build/bin/elastic-miter --lhs=$REWRITES/$a_lhs.mlir --rhs=$REWRITES/$a_rhs.mlir -o $OUT_DIR
exit_on_fail "Equivalence checking failed"
build/bin/elastic-miter --lhs=$REWRITES/$b_lhs.mlir --rhs=$REWRITES/$b_rhs.mlir -o $OUT_DIR
exit_on_fail "Equivalence checking failed"
build/bin/elastic-miter --lhs=$REWRITES/$c_lhs.mlir --rhs=$REWRITES/$c_rhs.mlir -o $OUT_DIR
exit_on_fail "Equivalence checking failed"
build/bin/elastic-miter --lhs=$REWRITES/$d_lhs.mlir --rhs=$REWRITES/$d_rhs.mlir -o $OUT_DIR
exit_on_fail "Equivalence checking failed"
build/bin/elastic-miter --lhs=$REWRITES/$e_lhs.mlir --rhs=$REWRITES/$e_rhs.mlir -o $OUT_DIR
exit_on_fail "Equivalence checking failed"
build/bin/elastic-miter --lhs=$REWRITES/$f_lhs.mlir --rhs=$REWRITES/$f_rhs.mlir -o $OUT_DIR
exit_on_fail "Equivalence checking failed"
build/bin/elastic-miter --lhs=$REWRITES/$g_lhs.mlir --rhs=$REWRITES/$g_rhs.mlir -o $OUT_DIR
exit_on_fail "Equivalence checking failed"
build/bin/elastic-miter --lhs=$REWRITES/$h_lhs.mlir --rhs=$REWRITES/$h_rhs.mlir -o $OUT_DIR
exit_on_fail "Equivalence checking failed"
build/bin/elastic-miter --lhs=$REWRITES/$i_lhs.mlir --rhs=$REWRITES/$i_rhs.mlir -o $OUT_DIR
exit_on_fail "Equivalence checking failed"


# build/bin/elastic-miter --lhs=$REWRITES/${MOD}_lhs.mlir --rhs=$REWRITES/${MOD}_rhs.mlir -o $OUT_DIR --loop_strict=0,1 --seq_length="0+1=0+1" --token_limit=0,0,2
# exit_on_fail "Equivalence checking failed"
# build/bin/elastic-miter --lhs=$REWRITES/${MOD}_lhs.mlir --rhs=$REWRITES/${MOD}_rhs.mlir -o $OUT_DIR --loop=0,1 --loop_strict=0,1 --seq_length="0+1=0+1" --token_limit=0,0,2
# exit_on_fail "Equivalence checking failed"