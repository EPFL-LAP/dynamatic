SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
DYNAMATIC_DIR="$(cd "$SCRIPT_DIR" && pwd)"

cd $DYNAMATIC_DIR

# ./tools/dynamatic/scripts/synthesize.sh $DYNAMATIC_DIR $DYNAMATIC_DIR/integration-test/single_loop/out_baseline_5ns single_loop 5.000 2.500
# ./tools/dynamatic/scripts/synthesize.sh $DYNAMATIC_DIR $DYNAMATIC_DIR/integration-test/single_loop/out_v1_5ns single_loop 5.000 2.500
# ./tools/dynamatic/scripts/synthesize.sh $DYNAMATIC_DIR $DYNAMATIC_DIR/integration-test/single_loop/out_4_5ns single_loop 5.000 2.500

# python3 experimental/tools/integration/run_specv2_integration.py nested_loop --baseline --out out_baseline_5ns --cp 5.00
# python3 experimental/tools/integration/run_adapted_spec_integration.py nested_loop --out out_v1_5ns --cp 5.00
# python3 experimental/tools/integration/run_specv2_integration.py nested_loop --n 4 --out out_4_5ns --cp 5.00 --resolver

# ./tools/dynamatic/scripts/synthesize.sh $DYNAMATIC_DIR $DYNAMATIC_DIR/integration-test/nested_loop/out_v1_5ns nested_loop 5.000 2.500
# ./tools/dynamatic/scripts/synthesize.sh $DYNAMATIC_DIR $DYNAMATIC_DIR/integration-test/nested_loop/out_baseline_5ns nested_loop 5.000 2.500
# ./tools/dynamatic/scripts/synthesize.sh $DYNAMATIC_DIR $DYNAMATIC_DIR/integration-test/nested_loop/out_4_5ns nested_loop 5.000 2.500

# ./tools/dynamatic/scripts/synthesize.sh $DYNAMATIC_DIR $DYNAMATIC_DIR/integration-test/fixed_log/out_baseline_5ns fixed_log 5.000 2.500
# ./tools/dynamatic/scripts/synthesize.sh $DYNAMATIC_DIR $DYNAMATIC_DIR/integration-test/fixed_log/out_v1_5ns fixed_log 5.000 2.500
# ./tools/dynamatic/scripts/synthesize.sh $DYNAMATIC_DIR $DYNAMATIC_DIR/integration-test/fixed_log/out_3_5ns fixed_log 5.000 2.500

# ./tools/dynamatic/scripts/synthesize.sh $DYNAMATIC_DIR $DYNAMATIC_DIR/integration-test/newton/out_baseline_5ns newton 5.000 2.500
# ./tools/dynamatic/scripts/synthesize.sh $DYNAMATIC_DIR $DYNAMATIC_DIR/integration-test/newton/out_v1_5ns newton 5.000 2.500
# ./tools/dynamatic/scripts/synthesize.sh $DYNAMATIC_DIR $DYNAMATIC_DIR/integration-test/newton/out_1_5ns newton 5.000 2.500

# ./tools/dynamatic/scripts/synthesize.sh $DYNAMATIC_DIR $DYNAMATIC_DIR/integration-test/subdiag_fast/out_baseline_5ns subdiag_fast 5.000 2.500
# ./tools/dynamatic/scripts/synthesize.sh $DYNAMATIC_DIR $DYNAMATIC_DIR/integration-test/subdiag_fast/out_v1_5ns subdiag_fast 5.000 2.500
# ./tools/dynamatic/scripts/synthesize.sh $DYNAMATIC_DIR $DYNAMATIC_DIR/integration-test/subdiag_fast/out_13_5ns subdiag_fast 5.000 2.500

# ./tools/dynamatic/scripts/synthesize.sh $DYNAMATIC_DIR $DYNAMATIC_DIR/integration-test/golden_ratio/out_baseline_5ns golden_ratio 5.000 2.500
# ./tools/dynamatic/scripts/synthesize.sh $DYNAMATIC_DIR $DYNAMATIC_DIR/integration-test/golden_ratio/out_v1_5ns golden_ratio 5.000 2.500
./tools/dynamatic/scripts/synthesize.sh $DYNAMATIC_DIR $DYNAMATIC_DIR/integration-test/golden_ratio/out_1_exit_5ns golden_ratio 5.000 2.500

python3 experimental/tools/integration/run_specv2_integration.py collision_donut --baseline --out out_baseline_5ns --cp 5.00
python3 experimental/tools/integration/run_specv2_integration.py collision_donut --n 5 --out out_5_5ns --cp 5.00 --resolver

./tools/dynamatic/scripts/synthesize.sh $DYNAMATIC_DIR $DYNAMATIC_DIR/integration-test/collision_donut/out_baseline_5ns collision_donut 5.000 2.500
./tools/dynamatic/scripts/synthesize.sh $DYNAMATIC_DIR $DYNAMATIC_DIR/integration-test/collision_donut/out_5_5ns collision_donut 5.000 2.500

./tools/dynamatic/scripts/synthesize.sh $DYNAMATIC_DIR $DYNAMATIC_DIR/integration-test/bisection/out_baseline_5ns bisection 5.000 2.500
./tools/dynamatic/scripts/synthesize.sh $DYNAMATIC_DIR $DYNAMATIC_DIR/integration-test/bisection/out_2_5ns bisection 5.000 2.500

python3 experimental/tools/integration/run_specv2_integration.py sparse_dataspec --baseline --out out_baseline_5ns --cp 5.00
python3 experimental/tools/integration/run_adapted_spec_integration.py sparse_dataspec --out out_v1_5ns --cp 5.00 --default-value 0 --disable-constant-predictor
python3 experimental/tools/integration/run_specv2_integration.py sparse_dataspec --n 8 --out out_8_5ns --cp 5.00 --resolver

./tools/dynamatic/scripts/synthesize.sh $DYNAMATIC_DIR $DYNAMATIC_DIR/integration-test/sparse_dataspec/out_baseline_5ns sparse_dataspec 5.000 2.500
./tools/dynamatic/scripts/synthesize.sh $DYNAMATIC_DIR $DYNAMATIC_DIR/integration-test/sparse_dataspec/out_v1_5ns sparse_dataspec 5.000 2.500
./tools/dynamatic/scripts/synthesize.sh $DYNAMATIC_DIR $DYNAMATIC_DIR/integration-test/sparse_dataspec/out_8_5ns sparse_dataspec 5.000 2.500

./tools/dynamatic/scripts/synthesize.sh $DYNAMATIC_DIR $DYNAMATIC_DIR/integration-test/sparse_dataspec_transformed/out_baseline_5ns sparse_dataspec_transformed 5.000 2.500
./tools/dynamatic/scripts/synthesize.sh $DYNAMATIC_DIR $DYNAMATIC_DIR/integration-test/sparse_dataspec_transformed/out_v1_5ns sparse_dataspec_transformed 5.000 2.500
./tools/dynamatic/scripts/synthesize.sh $DYNAMATIC_DIR $DYNAMATIC_DIR/integration-test/sparse_dataspec_transformed/out_8_5ns sparse_dataspec_transformed 5.000 2.500
