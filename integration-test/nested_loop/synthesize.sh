SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
DYNAMATIC_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"

cd $DYNAMATIC_DIR

# Spec v1
./tools/dynamatic/scripts/synthesize.sh $DYNAMATIC_DIR $DYNAMATIC_DIR/integration-test/nested_loop/out_v1 nested_loop 4.000 2.000

./tools/dynamatic/scripts/synthesize.sh $DYNAMATIC_DIR $DYNAMATIC_DIR/integration-test/nested_loop/out_0 nested_loop 4.000 2.000
# ./tools/dynamatic/scripts/synthesize.sh $DYNAMATIC_DIR $DYNAMATIC_DIR/integration-test/nested_loop/out_1 nested_loop 4.000 2.000
# ./tools/dynamatic/scripts/synthesize.sh $DYNAMATIC_DIR $DYNAMATIC_DIR/integration-test/nested_loop/out_2 nested_loop 4.000 2.000
# ./tools/dynamatic/scripts/synthesize.sh $DYNAMATIC_DIR $DYNAMATIC_DIR/integration-test/nested_loop/out_3 nested_loop 4.000 2.000
./tools/dynamatic/scripts/synthesize.sh $DYNAMATIC_DIR $DYNAMATIC_DIR/integration-test/nested_loop/out_4 nested_loop 4.000 2.000
# ./tools/dynamatic/scripts/synthesize.sh $DYNAMATIC_DIR $DYNAMATIC_DIR/integration-test/nested_loop/out_5 nested_loop 4.000 2.000
# ./tools/dynamatic/scripts/synthesize.sh $DYNAMATIC_DIR $DYNAMATIC_DIR/integration-test/nested_loop/out_6 nested_loop 4.000 2.000
# ./tools/dynamatic/scripts/synthesize.sh $DYNAMATIC_DIR $DYNAMATIC_DIR/integration-test/nested_loop/out_variable nested_loop 4.000 2.000
