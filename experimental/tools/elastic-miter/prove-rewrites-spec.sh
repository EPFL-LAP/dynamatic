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
    start=$(date +%s%3N)
    ./bin/elastic-miter --lhs=$REWRITES/${NAME}_lhs.mlir --rhs=$REWRITES/${NAME}_rhs.mlir -o $OUT_DIR/${NAME} --cex $CTX
    end=$(date +%s%3N)
    execution_time=$((end - start))
    echo "Execution time: ${execution_time} ms"
    exit_on_fail "($NAME): Equivalence checking failed"
}

# Rewrite A (new)
echo "Running Rewrite A"
run sup_add --seq_length="0=1&0=2"
run sup_fork
run sup_and --seq_length="0=1"
run sup_load --seq_length="0=1"

# Rewrite B (new)
echo "Running Rewrite B"
run sup_gamma_mux1 --seq_length="1=2"

# Rewrite C (new)
echo "Running Rewrite C"
run sup_gamma_mux2_mini --seq_length="0=1" --loop=2,0

# Rewrite D (new)
echo "Running Rewrite D"
run sup_mux --seq_length="1=2" --loop=0,2

# Rewrite E (new)
echo "Running Rewrite E"
run mux_to_and --seq_length="0=1"

# Rewrite F (new)
echo "Running Rewrite F"
run unify_sup --seq_length="0=1"

# Rewrite G (new)
echo "Running Rewrite G"
run sup_gamma_new2 --seq_length="0=2&1=2&2=3"

# Rewrite H (new)
echo "Running Rewrite H"
run sup_eager_gamma_mux --seq_length="0=1"
