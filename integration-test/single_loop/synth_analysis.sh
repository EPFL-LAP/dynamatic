SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
DYNAMATIC_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"

cd $DYNAMATIC_DIR

./tools/synth-analysis/report_usage.sh single_loop out_v1
./tools/synth-analysis/report_usage.sh single_loop out_0
./tools/synth-analysis/report_usage.sh single_loop out_1
./tools/synth-analysis/report_usage.sh single_loop out_2
./tools/synth-analysis/report_usage.sh single_loop out_3
./tools/synth-analysis/report_usage.sh single_loop out_4
./tools/synth-analysis/report_usage.sh single_loop out_5
# ./tools/synth-analysis/report_usage.sh single_loop out_variable
