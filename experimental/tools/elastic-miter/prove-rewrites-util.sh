SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
DYNAMATIC_DIR="$(cd "$SCRIPT_DIR/../../.." && pwd)"

# ElasticMiter only works in the dynamatic top-level directory
cd $DYNAMATIC_DIR
source tools/dynamatic/scripts/utils.sh

OUT_DIR="experimental/tools/elastic-miter/out"
REWRITES="experimental/tools/elastic-miter/rewrites-util"

run() {
    NAME=$1
    shift
    CTX="$@"
    ./bin/elastic-miter --lhs=$REWRITES/${NAME}_lhs.mlir --rhs=$REWRITES/${NAME}_rhs.mlir -o $OUT_DIR/${NAME} --cex $CTX
    exit_on_fail "($NAME): Equivalence checking failed"
}

run fork --allow_nonacceptance --seq_length_with_output="{in:0}={out:0}&{in:0}={out:1}"
