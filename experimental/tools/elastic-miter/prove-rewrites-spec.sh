SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
DYNAMATIC_DIR="$(cd "$SCRIPT_DIR/../../.." && pwd)"

# ElasticMiter only works in the dynamatic top-level directory
cd $DYNAMATIC_DIR
source tools/dynamatic/scripts/utils.sh

OUT_DIR="experimental/tools/elastic-miter/out"
REWRITES="experimental/tools/elastic-miter/rewrites-spec"

run() {
    NAME=$1
    shift
    CTX="$@"
    ./bin/elastic-miter --lhs=$REWRITES/${NAME}_lhs.mlir --rhs=$REWRITES/${NAME}_rhs.mlir -o $OUT_DIR/${NAME} --cex $CTX
    exit_on_fail "($NAME): Equivalence checking failed"
}

run sup_mux --seq_length="1=2" --seq_length_enhanced="{in:1}={out:0}" --loop_strict=0,2 --disable_ndwire --disable_decoupling --allow_nonacceptance
run ri_fork --allow_nonacceptance --seq_length_enhanced="{out:0}-{out:1}<=1&{out:1}-{out:0}<=1" --disable_ndwire --disable_decoupling
run introduceIdentInterpolator --allow_nonacceptance --disable_ndwire --disable_decoupling
run suppressorInduction --custom-context="$REWRITES/suppressorInduction_ctx.mlir" --allow_nonacceptance --disable_ndwire --disable_decoupling
run introduceResolver --disable_ndwire --disable_decoupling --custom-context="$REWRITES/introduceResolver_ctx.mlir" --allow_nonacceptance --seq_length_enhanced="{in:0}={out:0}"
run sup_and --seq_length="0=1" --allow_nonacceptance --disable_ndwire --disable_decoupling
run sup_fork --allow_nonacceptance
run sup_load --allow_nonacceptance --disable_ndwire --disable_decoupling --seq_length_enhanced="{in:0}={in:1}&{in:0}={out:0}"
run unify_sup --seq_length="0=1" --allow_nonacceptance --disable_ndwire --disable_decoupling
run interpolatorForkSwap --allow_nonacceptance --disable_ndwire --disable_decoupling
run andForkSwap --allow_nonacceptance --disable_ndwire --disable_decoupling
run sup_gamma --allow_nonacceptance --disable_ndwire --disable_decoupling

run mux_add_suppress --disable_ndwire --disable_decoupling --allow_nonacceptance
run mux_to_logic --disable_ndwire --disable_decoupling --allow_nonacceptance --seq_length="0>=2&1>=2" --seq_length_enhanced="{out:0}>={in:2}"
run and_false --disable_ndwire --disable_decoupling --allow_nonacceptance --seq_length_enhanced="{in:0}={out:0}"

# not equivalent
# run sup_source --disable_ndwire --disable_decoupling --allow_nonacceptance --seq_length_enhanced="{in:0}={out:0}"
