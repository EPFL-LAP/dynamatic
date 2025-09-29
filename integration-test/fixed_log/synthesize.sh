SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
DYNAMATIC_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"

cd $DYNAMATIC_DIR

./tools/dynamatic/scripts/synthesize.sh $DYNAMATIC_DIR $DYNAMATIC_DIR/integration-test/fixed_log/out_v1 fixed_log 6.000 3.000

# ./tools/dynamatic/scripts/synthesize.sh $DYNAMATIC_DIR $DYNAMATIC_DIR/integration-test/fixed_log/out_default fixed_log 6.000 3.000
# ./tools/dynamatic/scripts/synthesize.sh $DYNAMATIC_DIR $DYNAMATIC_DIR/integration-test/fixed_log/out_0 fixed_log 6.000 3.000
# # ./tools/dynamatic/scripts/synthesize.sh $DYNAMATIC_DIR $DYNAMATIC_DIR/integration-test/fixed_log/out_1 fixed_log 6.000 3.000
# # ./tools/dynamatic/scripts/synthesize.sh $DYNAMATIC_DIR $DYNAMATIC_DIR/integration-test/fixed_log/out_2 fixed_log 6.000 3.000
# ./tools/dynamatic/scripts/synthesize.sh $DYNAMATIC_DIR $DYNAMATIC_DIR/integration-test/fixed_log/out_3 fixed_log 6.000 3.000
