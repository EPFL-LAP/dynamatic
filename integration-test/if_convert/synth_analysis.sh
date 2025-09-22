SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
DYNAMATIC_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"

cd $DYNAMATIC_DIR

./tools/synth-analysis/report_usage.sh if_convert out_standard
./tools/synth-analysis/report_usage.sh if_convert out_transformed
./tools/synth-analysis/report_usage.sh if_convert out_v1
./tools/synth-analysis/report_usage.sh if_convert out_0
./tools/synth-analysis/report_usage.sh if_convert out_1
./tools/synth-analysis/report_usage.sh if_convert out_2
./tools/synth-analysis/report_usage.sh if_convert out_3
./tools/synth-analysis/report_usage.sh if_convert out_4
./tools/synth-analysis/report_usage.sh if_convert out_5
./tools/synth-analysis/report_usage.sh if_convert out_variable
