SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
DYNAMATIC_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"

cd $DYNAMATIC_DIR

# # Standard flow
# ./tools/dynamatic/scripts/synthesize.sh $DYNAMATIC_DIR $DYNAMATIC_DIR/integration-test/sparse_dataspec/out_standard sparse_dataspec 4.000 2.000

# # Only code transformation
# ./tools/dynamatic/scripts/synthesize.sh $DYNAMATIC_DIR $DYNAMATIC_DIR/integration-test/sparse_dataspec/out_transformed sparse_dataspec 4.000 2.000

# # Spec v1
# ./tools/dynamatic/scripts/synthesize.sh $DYNAMATIC_DIR $DYNAMATIC_DIR/integration-test/sparse_dataspec/out_v1 sparse_dataspec 4.000 2.000

./tools/dynamatic/scripts/synthesize.sh $DYNAMATIC_DIR $DYNAMATIC_DIR/integration-test/sparse_dataspec/out_baseline sparse_dataspec 8.000 4.000
./tools/dynamatic/scripts/synthesize.sh $DYNAMATIC_DIR $DYNAMATIC_DIR/integration-test/sparse_dataspec/out_v1 sparse_dataspec 8.000 4.000
./tools/dynamatic/scripts/synthesize.sh $DYNAMATIC_DIR $DYNAMATIC_DIR/integration-test/sparse_dataspec/out_8 sparse_dataspec 8.000 4.000
# ./tools/dynamatic/scripts/synthesize.sh $DYNAMATIC_DIR $DYNAMATIC_DIR/integration-test/sparse_dataspec/out_2 sparse_dataspec 4.000 2.000
# ./tools/dynamatic/scripts/synthesize.sh $DYNAMATIC_DIR $DYNAMATIC_DIR/integration-test/sparse_dataspec/out_3 sparse_dataspec 4.000 2.000
# ./tools/dynamatic/scripts/synthesize.sh $DYNAMATIC_DIR $DYNAMATIC_DIR/integration-test/sparse_dataspec/out_4 sparse_dataspec 4.000 2.000
# ./tools/dynamatic/scripts/synthesize.sh $DYNAMATIC_DIR $DYNAMATIC_DIR/integration-test/sparse_dataspec/out_5 sparse_dataspec 4.000 2.000
# ./tools/dynamatic/scripts/synthesize.sh $DYNAMATIC_DIR $DYNAMATIC_DIR/integration-test/sparse_dataspec/out_variable sparse_dataspec 4.000 2.000
