SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
DYNAMATIC_DIR="$(cd "$SCRIPT_DIR" && pwd)"

cd $DYNAMATIC_DIR

python3 experimental/tools/integration/run_specv2_integration.py single_loop --baseline --out out_baseline_7ns --cp 7.00
python3 experimental/tools/integration/run_adapted_spec_integration.py single_loop --out out_v1_7ns --cp 7.00
python3 experimental/tools/integration/run_specv2_integration.py single_loop --n 4 --out out_4_7ns --cp 7.00 --resolver

timeout --kill-after=10s 900s ./tools/dynamatic/scripts/synthesize.sh $DYNAMATIC_DIR $DYNAMATIC_DIR/integration-test/single_loop/out_baseline_7ns single_loop 7.000 3.500
timeout --kill-after=10s 900s ./tools/dynamatic/scripts/synthesize.sh $DYNAMATIC_DIR $DYNAMATIC_DIR/integration-test/single_loop/out_v1_7ns single_loop 7.000 3.500
timeout --kill-after=10s 900s ./tools/dynamatic/scripts/synthesize.sh $DYNAMATIC_DIR $DYNAMATIC_DIR/integration-test/single_loop/out_4_7ns single_loop 7.000 3.500