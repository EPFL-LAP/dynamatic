SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
DYNAMATIC_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"

cd $DYNAMATIC_DIR

# ./tools/dynamatic/scripts/synthesize.sh $DYNAMATIC_DIR $DYNAMATIC_DIR/integration-test/collision_donut/out_0 collision_donut 5.000 2.500
./tools/dynamatic/scripts/synthesize.sh $DYNAMATIC_DIR $DYNAMATIC_DIR/integration-test/collision_donut/out_5 collision_donut 5.000 2.500
