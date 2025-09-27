SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
DYNAMATIC_DIR="$(cd "$SCRIPT_DIR/../../.." && pwd)"

cd $DYNAMATIC_DIR

KERNEL_NAME=$1
OUT_NAME=$2

./tools/dynamatic/scripts/visualize.sh . integration-test/$KERNEL_NAME/$OUT_NAME/comp/$KERNEL_NAME.dot integration-test/$KERNEL_NAME/$OUT_NAME/sim/HLW_VERIFY/vsim.wlf integration-test/$KERNEL_NAME/$OUT_NAME $KERNEL_NAME
