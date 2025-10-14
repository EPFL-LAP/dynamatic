SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
DYNAMATIC_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"

cd $DYNAMATIC_DIR

# python3 experimental/tools/integration/run_specv2_integration.py fixed_log --n 0 --disable-initial-motion --out out_baseline_merged_5ns --cp 5.00
python3 experimental/tools/integration/run_specv2_integration.py fixed_log --n 0 --disable-initial-motion --out out_baseline_merged_7ns --cp 7.00

./tools/dynamatic/scripts/synthesize.sh $DYNAMATIC_DIR $DYNAMATIC_DIR/integration-test/fixed_log/out_baseline_merged_5ns fixed_log 5.000 2.500
./tools/dynamatic/scripts/synthesize.sh $DYNAMATIC_DIR $DYNAMATIC_DIR/integration-test/fixed_log/out_baseline_merged_7ns fixed_log 7.000 3.500
