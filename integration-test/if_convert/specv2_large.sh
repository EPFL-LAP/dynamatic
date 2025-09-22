SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
DYNAMATIC_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"

cd $DYNAMATIC_DIR

python3 experimental/tools/integration/run_specv2_integration.py if_convert --n 7 --out out_7 --transformed-code if_convert_transformed.c
python3 experimental/tools/integration/run_specv2_integration.py if_convert --n 8 --out out_8 --transformed-code if_convert_transformed.c
python3 experimental/tools/integration/run_specv2_integration.py if_convert --n 10 --out out_10 --transformed-code if_convert_transformed.c
python3 experimental/tools/integration/run_specv2_integration.py if_convert --n 11 --out out_11 --transformed-code if_convert_transformed.c
python3 experimental/tools/integration/run_specv2_integration.py if_convert --n 15 --out out_15 --transformed-code if_convert_transformed.c
python3 experimental/tools/integration/run_specv2_integration.py if_convert --n 16 --out out_16 --transformed-code if_convert_transformed.c

./tools/dynamatic/scripts/synthesize.sh $DYNAMATIC_DIR $DYNAMATIC_DIR/integration-test/if_convert/out_7 if_convert 4.000 2.000
./tools/dynamatic/scripts/synthesize.sh $DYNAMATIC_DIR $DYNAMATIC_DIR/integration-test/if_convert/out_8 if_convert 4.000 2.000
./tools/dynamatic/scripts/synthesize.sh $DYNAMATIC_DIR $DYNAMATIC_DIR/integration-test/if_convert/out_10 if_convert 4.000 2.000
./tools/dynamatic/scripts/synthesize.sh $DYNAMATIC_DIR $DYNAMATIC_DIR/integration-test/if_convert/out_11 if_convert 4.000 2.000
./tools/dynamatic/scripts/synthesize.sh $DYNAMATIC_DIR $DYNAMATIC_DIR/integration-test/if_convert/out_15 if_convert 4.000 2.000
./tools/dynamatic/scripts/synthesize.sh $DYNAMATIC_DIR $DYNAMATIC_DIR/integration-test/if_convert/out_16 if_convert 4.000 2.000

./tools/synth-analysis/report_usage.sh if_convert out_7
./tools/synth-analysis/report_usage.sh if_convert out_8
./tools/synth-analysis/report_usage.sh if_convert out_10
./tools/synth-analysis/report_usage.sh if_convert out_11
./tools/synth-analysis/report_usage.sh if_convert out_15
./tools/synth-analysis/report_usage.sh if_convert out_16