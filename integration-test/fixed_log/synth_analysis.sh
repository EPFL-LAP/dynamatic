SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
DYNAMATIC_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"

cd $DYNAMATIC_DIR

./tools/synth-analysis/report_usage.sh fixed_log out_0
./tools/synth-analysis/report_usage.sh fixed_log out_3
