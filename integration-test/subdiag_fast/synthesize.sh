SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
DYNAMATIC_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"

cd $DYNAMATIC_DIR

./tools/dynamatic/scripts/synthesize.sh $DYNAMATIC_DIR $DYNAMATIC_DIR/integration-test/subdiag_fast/out_standard subdiag_fast 8.000 4.000

# Spec v1
./tools/dynamatic/scripts/synthesize.sh $DYNAMATIC_DIR $DYNAMATIC_DIR/integration-test/subdiag_fast/out_v1 subdiag_fast 8.000 4.000

# ./tools/dynamatic/scripts/synthesize.sh $DYNAMATIC_DIR $DYNAMATIC_DIR/integration-test/subdiag_fast/out_0 subdiag_fast 8.000 4.000
# ./tools/dynamatic/scripts/synthesize.sh $DYNAMATIC_DIR $DYNAMATIC_DIR/integration-test/subdiag_fast/out_1 subdiag_fast 8.000 4.000
# ./tools/dynamatic/scripts/synthesize.sh $DYNAMATIC_DIR $DYNAMATIC_DIR/integration-test/subdiag_fast/out_2 subdiag_fast 8.000 4.000
# ./tools/dynamatic/scripts/synthesize.sh $DYNAMATIC_DIR $DYNAMATIC_DIR/integration-test/subdiag_fast/out_3 subdiag_fast 8.000 4.000
# ./tools/dynamatic/scripts/synthesize.sh $DYNAMATIC_DIR $DYNAMATIC_DIR/integration-test/subdiag_fast/out_4 subdiag_fast 8.000 4.000
# ./tools/dynamatic/scripts/synthesize.sh $DYNAMATIC_DIR $DYNAMATIC_DIR/integration-test/subdiag_fast/out_5 subdiag_fast 8.000 4.000
# ./tools/dynamatic/scripts/synthesize.sh $DYNAMATIC_DIR $DYNAMATIC_DIR/integration-test/subdiag_fast/out_6 subdiag_fast 8.000 4.000
# ./tools/dynamatic/scripts/synthesize.sh $DYNAMATIC_DIR $DYNAMATIC_DIR/integration-test/subdiag_fast/out_7 subdiag_fast 8.000 4.000
# ./tools/dynamatic/scripts/synthesize.sh $DYNAMATIC_DIR $DYNAMATIC_DIR/integration-test/subdiag_fast/out_8 subdiag_fast 8.000 4.000
# ./tools/dynamatic/scripts/synthesize.sh $DYNAMATIC_DIR $DYNAMATIC_DIR/integration-test/subdiag_fast/out_9 subdiag_fast 8.000 4.000
# ./tools/dynamatic/scripts/synthesize.sh $DYNAMATIC_DIR $DYNAMATIC_DIR/integration-test/subdiag_fast/out_10 subdiag_fast 8.000 4.000
# ./tools/dynamatic/scripts/synthesize.sh $DYNAMATIC_DIR $DYNAMATIC_DIR/integration-test/subdiag_fast/out_11 subdiag_fast 8.000 4.000
# ./tools/dynamatic/scripts/synthesize.sh $DYNAMATIC_DIR $DYNAMATIC_DIR/integration-test/subdiag_fast/out_12 subdiag_fast 8.000 4.000
# ./tools/dynamatic/scripts/synthesize.sh $DYNAMATIC_DIR $DYNAMATIC_DIR/integration-test/subdiag_fast/out_13 subdiag_fast 8.000 4.000
# ./tools/dynamatic/scripts/synthesize.sh $DYNAMATIC_DIR $DYNAMATIC_DIR/integration-test/subdiag_fast/out_14 subdiag_fast 8.000 4.000
# ./tools/dynamatic/scripts/synthesize.sh $DYNAMATIC_DIR $DYNAMATIC_DIR/integration-test/subdiag_fast/out_15 subdiag_fast 8.000 4.000
# ./tools/dynamatic/scripts/synthesize.sh $DYNAMATIC_DIR $DYNAMATIC_DIR/integration-test/subdiag_fast/out_16 subdiag_fast 8.000 4.000
# ./tools/dynamatic/scripts/synthesize.sh $DYNAMATIC_DIR $DYNAMATIC_DIR/integration-test/subdiag_fast/out_17 subdiag_fast 8.000 4.000
# ./tools/dynamatic/scripts/synthesize.sh $DYNAMATIC_DIR $DYNAMATIC_DIR/integration-test/subdiag_fast/out_18 subdiag_fast 8.000 4.000
# ./tools/dynamatic/scripts/synthesize.sh $DYNAMATIC_DIR $DYNAMATIC_DIR/integration-test/subdiag_fast/out_19 subdiag_fast 8.000 4.000
# ./tools/dynamatic/scripts/synthesize.sh $DYNAMATIC_DIR $DYNAMATIC_DIR/integration-test/subdiag_fast/out_20 subdiag_fast 8.000 4.000
# ./tools/dynamatic/scripts/synthesize.sh $DYNAMATIC_DIR $DYNAMATIC_DIR/integration-test/subdiag_fast/out_25 subdiag_fast 8.000 4.000
# ./tools/dynamatic/scripts/synthesize.sh $DYNAMATIC_DIR $DYNAMATIC_DIR/integration-test/subdiag_fast/out_30 subdiag_fast 8.000 4.000
# ./tools/dynamatic/scripts/synthesize.sh $DYNAMATIC_DIR $DYNAMATIC_DIR/integration-test/subdiag_fast/out_35 subdiag_fast 8.000 4.000
# ./tools/dynamatic/scripts/synthesize.sh $DYNAMATIC_DIR $DYNAMATIC_DIR/integration-test/subdiag_fast/out_40 subdiag_fast 8.000 4.000
# ./tools/dynamatic/scripts/synthesize.sh $DYNAMATIC_DIR $DYNAMATIC_DIR/integration-test/subdiag_fast/out_45 subdiag_fast 8.000 4.000
# ./tools/dynamatic/scripts/synthesize.sh $DYNAMATIC_DIR $DYNAMATIC_DIR/integration-test/subdiag_fast/out_50 subdiag_fast 8.000 4.000

# ./tools/dynamatic/scripts/synthesize.sh $DYNAMATIC_DIR $DYNAMATIC_DIR/integration-test/subdiag_fast/out_60 subdiag_fast 8.000 4.000
# ./tools/dynamatic/scripts/synthesize.sh $DYNAMATIC_DIR $DYNAMATIC_DIR/integration-test/subdiag_fast/out_70 subdiag_fast 8.000 4.000
# ./tools/dynamatic/scripts/synthesize.sh $DYNAMATIC_DIR $DYNAMATIC_DIR/integration-test/subdiag_fast/out_80 subdiag_fast 8.000 4.000
# ./tools/dynamatic/scripts/synthesize.sh $DYNAMATIC_DIR $DYNAMATIC_DIR/integration-test/subdiag_fast/out_90 subdiag_fast 8.000 4.000
# ./tools/dynamatic/scripts/synthesize.sh $DYNAMATIC_DIR $DYNAMATIC_DIR/integration-test/subdiag_fast/out_100 subdiag_fast 8.000 4.000
