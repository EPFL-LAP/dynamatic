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

# run sup_and --seq_length="0=1" --allow_nonacceptance
# run sup_fork --allow_nonacceptance
run sup_mux --seq_length="0=2" --loop=1,2 --allow_nonacceptance
# run unify_sup --seq_length="0=1" --allow_nonacceptance
# run interpolator_ident --allow_nonacceptance
# run interpolator_ind --seq_length="0<=1" --allow_nonacceptance
# run resolver --nd_spec --allow_nonacceptance
