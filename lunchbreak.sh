SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
DYNAMATIC_DIR="$(cd "$SCRIPT_DIR" && pwd)"

cd $DYNAMATIC_DIR

date > start_time.txt

python3 experimental/tools/integration/run_adapted_spec_integration.py sparse_dataspec_transformed --cp 5.00 --transformed-code sparse_dataspec_modified.c --out out_v1_opt_5ns
python3 experimental/tools/integration/run_adapted_spec_integration.py sparse_dataspec_transformed --cp 7.00 --transformed-code sparse_dataspec_modified.c --out out_v1_opt_7ns

python3 experimental/tools/integration/run_gamma_integration.py if_float --branch-bb=1 --merge-bb=4 --steps-until 0 --prioritized-side 0 --out out_baseline_5ns --cp 5.00
python3 experimental/tools/integration/run_adapted_spec_integration.py if_float --default-value=0 --out out_v1_5ns --cp 5.00 --loop-bottom-passer-disabled
python3 experimental/tools/integration/run_gamma_integration.py if_float --branch-bb=1 --merge-bb=4 --prioritized-side 0 --one-sided --out out_0_one_sided_5ns --cp 5.00
python3 experimental/tools/integration/run_gamma_integration.py if_float --branch-bb=1 --merge-bb=4 --prioritized-side 0 --out out_0_two_sided_5ns --use-prof-cache --cp 5.00
python3 experimental/tools/integration/run_gamma_integration.py if_float --branch-bb=1 --merge-bb=4 --prioritized-side 1 --out out_1_two_sided_5ns --use-prof-cache --cp 5.00
python3 experimental/tools/integration/run_gamma_integration.py if_float --branch-bb=1 --merge-bb=4 --prioritized-side 1 --one-sided --out out_1_one_sided_5ns --use-prof-cache --cp 5.00

python3 experimental/tools/integration/run_gamma_integration.py if_float2 --branch-bb=1 --merge-bb=4 --steps-until 0 --prioritized-side 0 --out out_baseline_5ns --cp 5.00
python3 experimental/tools/integration/run_adapted_spec_integration.py if_float2 --default-value=0 --out out_v1_5ns --cp 5.00 --loop-bottom-passer-disabled
python3 experimental/tools/integration/run_gamma_integration.py if_float2 --branch-bb=1 --merge-bb=4 --prioritized-side 0 --one-sided --out out_0_one_sided_5ns --cp 5.00
python3 experimental/tools/integration/run_gamma_integration.py if_float2 --branch-bb=1 --merge-bb=4 --prioritized-side 0 --out out_0_two_sided_5ns --use-prof-cache --cp 5.00
python3 experimental/tools/integration/run_gamma_integration.py if_float2 --branch-bb=1 --merge-bb=4 --prioritized-side 1 --one-sided --out out_1_one_sided_5ns --use-prof-cache --cp 5.00
python3 experimental/tools/integration/run_gamma_integration.py if_float2 --branch-bb=1 --merge-bb=4 --prioritized-side 1 --out out_1_two_sided_5ns --use-prof-cache --cp 5.00

python3 experimental/tools/integration/run_adapted_spec_integration.py if_float --default-value=0 --out out_v1_opt_7ns --cp 5.00 --loop-bottom-passer-disabled
python3 experimental/tools/integration/run_adapted_spec_integration.py if_float2 --default-value=0 --out out_v1_opt_7ns --cp 7.00 --loop-bottom-passer-disabled

timeout --kill-after=10s 900s ./tools/dynamatic/scripts/synthesize.sh $DYNAMATIC_DIR $DYNAMATIC_DIR/integration-test/sparse_dataspec_transformed/out_v1_opt_5ns sparse_dataspec_transformed 5.000 2.500
timeout --kill-after=10s 900s ./tools/dynamatic/scripts/synthesize.sh $DYNAMATIC_DIR $DYNAMATIC_DIR/integration-test/sparse_dataspec_transformed/out_v1_opt_7ns sparse_dataspec_transformed 7.000 3.500

timeout --kill-after=10s 900s ./tools/dynamatic/scripts/synthesize.sh $DYNAMATIC_DIR $DYNAMATIC_DIR/integration-test/if_float/out_baseline_5ns if_float 5.000 2.500
timeout --kill-after=10s 900s ./tools/dynamatic/scripts/synthesize.sh $DYNAMATIC_DIR $DYNAMATIC_DIR/integration-test/if_float/out_v1_5ns if_float 5.000 2.500
timeout --kill-after=10s 900s ./tools/dynamatic/scripts/synthesize.sh $DYNAMATIC_DIR $DYNAMATIC_DIR/integration-test/if_float/out_0_one_sided_5ns if_float 5.000 2.500
timeout --kill-after=10s 900s ./tools/dynamatic/scripts/synthesize.sh $DYNAMATIC_DIR $DYNAMATIC_DIR/integration-test/if_float/out_1_one_sided_5ns if_float 5.000 2.500
timeout --kill-after=10s 900s ./tools/dynamatic/scripts/synthesize.sh $DYNAMATIC_DIR $DYNAMATIC_DIR/integration-test/if_float/out_0_two_sided_5ns if_float 5.000 2.500
timeout --kill-after=10s 900s ./tools/dynamatic/scripts/synthesize.sh $DYNAMATIC_DIR $DYNAMATIC_DIR/integration-test/if_float/out_1_two_sided_5ns if_float 5.000 2.500

timeout --kill-after=10s 900s ./tools/dynamatic/scripts/synthesize.sh $DYNAMATIC_DIR $DYNAMATIC_DIR/integration-test/if_float2/out_baseline_5ns if_float2 5.000 2.500
timeout --kill-after=10s 900s ./tools/dynamatic/scripts/synthesize.sh $DYNAMATIC_DIR $DYNAMATIC_DIR/integration-test/if_float2/out_v1_5ns if_float2 5.000 2.500
timeout --kill-after=10s 900s ./tools/dynamatic/scripts/synthesize.sh $DYNAMATIC_DIR $DYNAMATIC_DIR/integration-test/if_float2/out_0_one_sided_5ns if_float2 5.000 2.500
timeout --kill-after=10s 900s ./tools/dynamatic/scripts/synthesize.sh $DYNAMATIC_DIR $DYNAMATIC_DIR/integration-test/if_float2/out_1_one_sided_5ns if_float2 5.000 2.500
timeout --kill-after=10s 900s ./tools/dynamatic/scripts/synthesize.sh $DYNAMATIC_DIR $DYNAMATIC_DIR/integration-test/if_float2/out_0_two_sided_5ns if_float2 5.000 2.500
timeout --kill-after=10s 900s ./tools/dynamatic/scripts/synthesize.sh $DYNAMATIC_DIR $DYNAMATIC_DIR/integration-test/if_float2/out_1_two_sided_5ns if_float2 5.000 2.500

timeout --kill-after=10s 900s ./tools/dynamatic/scripts/synthesize.sh $DYNAMATIC_DIR $DYNAMATIC_DIR/integration-test/if_float/out_v1_opt_7ns if_float 7.000 3.500
timeout --kill-after=10s 900s ./tools/dynamatic/scripts/synthesize.sh $DYNAMATIC_DIR $DYNAMATIC_DIR/integration-test/if_float2/out_v1_opt_7ns if_float2 7.000 3.500

date > end_time.txt
