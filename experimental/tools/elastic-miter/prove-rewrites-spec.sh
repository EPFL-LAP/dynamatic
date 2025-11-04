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
run sup_mul --seq_length="0=1&0=2"
# run sup_fork
# run sup_and --seq_length="0=1"
# run sup_load --seq_length="0=1"

# Rewrite B (new)
# run sup_gamma_mux1 --seq_length="1=2"

# Rewrite C (new)
# run sup_gamma_mux2_mini --seq_length="0=1" --loop=2,0

# Rewrite D (new)
# echo "Running Rewrite D"
# run sup_mux --seq_length="1=2" --loop=0,2 > sup_mux.txt

# Rewrite E (new)
# echo "Running Rewrite E"
# run mux_to_and --seq_length="0=1" > mux_to_and.txt

# Rewrite F (new)
# echo "Running Rewrite F"
# run unify_sup --seq_length="0=1" > unify_sup.txt

# Rewrite G (new)
# echo "Running Rewrite G"
# run sup_gamma_new2 --seq_length="0=2&1=2&2=3" > sup_gamma_new2.txt

# Rewrite H (new)
# echo "Running Rewrite H"
# run sup_eager_gamma_mux --seq_length="0=1" > sup_eager_gamma_mux.txt

# run sup_and --allow_nonacceptance --timing_insensitive --seq_length="0=1"
# run sup_gamma --timing_insensitive

# run sup_sup --timing_insensitive --seq_length="0=1&1=2"
# run general_sup_mux --allow_nonacceptance --timing_insensitive --seq_length="1=3&2=4"
# run extension1 --allow_nonacceptance --timing_insensitive
# run extension2 --allow_nonacceptance --timing_insensitive --seq_length_enhanced="{in:0}<={out:0}"
# run extension3 --allow_nonacceptance --timing_insensitive
# run extension4 --allow_nonacceptance --timing_insensitive
# run extension5 --allow_nonacceptance --timing_insensitive --seq_length_enhanced="{in:0}<={out:0}"

# run general_sup_mumux_copy --allow_nonacceptance --timing_insensitive --seq_length="1=2" --seq_length_enhanced="{in:1}<={out:0}" --loop_strict=0,2
# run general_sup_mumux --timing_insensitive --seq_length="0=2&0=3"


# run sup_mu_mux1 --timing_insensitive --seq_length="2=3"
# run repeating_init --timing_insensitive

# run introduceIdentInterpolator --timing_insensitive
# run interpInduction --timing_insensitive

# run sup_gamma_mux2 --timing_insensitive --seq_length="0=1&0=2"

# run sup_gamma_new --timing_insensitive --allow_nonacceptance --seq_length="0=2&1=2&2=3"

# # Rewrites B
# run sup_mul --allow_nonacceptance --timing_insensitive --seq_length_enhanced="{in:0}={in:1}&{in:0}={in:2}&{in:0}<={out:0}"
# run sup_fork --allow_nonacceptance --timing_insensitive
# run sup_load --allow_nonacceptance --timing_insensitive --seq_length_enhanced="{in:0}={in:1}&{in:0}<={out:0}"

# # Rewrite C
# run simpleInduction --allow_nonacceptance --timing_insensitive

# # Rewrite D

# # Rewrite E
# run muxForkSwap --allow_nonacceptance --timing_insensitive

# run repeating_init --disable_ndwire --disable_decoupling --allow_nonacceptance
# run newInduction --disable_ndwire --disable_decoupling --allow_nonacceptance --custom-context="$REWRITES/newInduction_ctx.mlir" --infinite_tokens

# run ri_fork --allow_nonacceptance --seq_length_enhanced="{out:0}-{out:1}<=1&{out:1}-{out:0}<=1" --disable_ndwire --disable_decoupling
# run introduceIdentInterpolator --allow_nonacceptance --disable_ndwire --disable_decoupling
# run suppressorInduction --custom-context="$REWRITES/suppressorInduction_ctx.mlir" --allow_nonacceptance --disable_ndwire --disable_decoupling
# run introduceResolver --disable_ndwire --disable_decoupling --custom-context="$REWRITES/introduceResolver_ctx.mlir" --allow_nonacceptance --seq_length_enhanced="{in:0}={out:0}"
# run interpolatorForkSwap --allow_nonacceptance --disable_ndwire --disable_decoupling
# run andForkSwap --allow_nonacceptance --disable_ndwire --disable_decoupling
# run sup_gamma --allow_nonacceptance --disable_ndwire --disable_decoupling

# run mux_add_suppress --disable_ndwire --disable_decoupling --allow_nonacceptance
# run mux_to_logic --disable_ndwire --disable_decoupling --allow_nonacceptance --seq_length="0>=2&1>=2" --seq_length_enhanced="{out:0}>={in:2}"
# run and_false --disable_ndwire --disable_decoupling --allow_nonacceptance --seq_length_enhanced="{in:0}={out:0}"

# run gamma_mu_swap --disable_ndwire --disable_decoupling --allow_nonacceptance --loop="4,0" #--seq_length="0=1&1=2&1=3&0=4"
# run mux_br_swap --disable_ndwire --disable_decoupling --allow_nonacceptance

# not equivalent
# run sup_source --disable_ndwire --disable_decoupling --allow_nonacceptance --seq_length_enhanced="{in:0}={out:0}"
