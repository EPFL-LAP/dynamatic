SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
DYNAMATIC_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"

cd $DYNAMATIC_DIR/tools/synth-analysis/

PROJECT_NAME=$1
RUN_NAME=$2

SYNTH_DIR="/home/shundroid/dynamatic/integration-test/$PROJECT_NAME/$RUN_NAME/synth"

sed "s|SYNTH_DIR|$SYNTH_DIR|g" report_usage.tcl > report_usage_with_paths.tcl
/tools/Xilinx/2025.1/Vivado/bin/vivado -mode tcl -source report_usage_with_paths.tcl
mv primitive_counts.txt $SYNTH_DIR
