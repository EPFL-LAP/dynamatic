SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
DYNAMATIC_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"

cd $DYNAMATIC_DIR


# ./tools/dynamatic/scripts/synthesize.sh $DYNAMATIC_DIR $DYNAMATIC_DIR/integration-test/newton/out_default newton 8.000 4.000
./tools/dynamatic/scripts/synthesize.sh $DYNAMATIC_DIR $DYNAMATIC_DIR/integration-test/newton/out_0 newton 8.000 4.000
./tools/dynamatic/scripts/synthesize.sh $DYNAMATIC_DIR $DYNAMATIC_DIR/integration-test/newton/out_1 newton 8.000 4.000
# ./tools/dynamatic/scripts/synthesize.sh $DYNAMATIC_DIR $DYNAMATIC_DIR/integration-test/newton/out_2 newton 8.000 4.000
# ./tools/dynamatic/scripts/synthesize.sh $DYNAMATIC_DIR $DYNAMATIC_DIR/integration-test/newton/out_3 newton 8.000 4.000
./tools/dynamatic/scripts/synthesize.sh $DYNAMATIC_DIR $DYNAMATIC_DIR/integration-test/newton/out_v1 newton 8.000 4.000
