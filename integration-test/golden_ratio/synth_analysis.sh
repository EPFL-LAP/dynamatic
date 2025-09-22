SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
DYNAMATIC_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"

cd $DYNAMATIC_DIR

./tools/synth-analysis/report_usage.sh golden_ratio out_standard
./tools/synth-analysis/report_usage.sh golden_ratio out_1
