SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
DYNAMATIC_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"

cd $DYNAMATIC_DIR

# Standard flow
# python3 experimental/tools/integration/run_specv2_integration.py if_convert --disable-spec --out out_standard --cp 10.00

# Only code transformation
python3 experimental/tools/integration/run_specv2_integration.py if_convert --disable-spec --transformed-cf cf_transformed.mlir --out out_transformed --cp 10.00

# Spec v1
# python3 tools/integration/run_spec_integration.py if_convert --out out_v1 --cp 10.00 --default-value 0

python3 experimental/tools/integration/run_specv2_integration.py if_convert --n 0 --out out_0 --transformed-cf cf_transformed.mlir --cp 10.00
python3 experimental/tools/integration/run_specv2_integration.py if_convert --n 1 --out out_1 --transformed-cf cf_transformed.mlir --cp 10.00
python3 experimental/tools/integration/run_specv2_integration.py if_convert --n 2 --out out_2 --transformed-cf cf_transformed.mlir --cp 10.00
python3 experimental/tools/integration/run_specv2_integration.py if_convert --n 3 --out out_3 --transformed-cf cf_transformed.mlir --cp 10.00
python3 experimental/tools/integration/run_specv2_integration.py if_convert --n 4 --out out_4 --transformed-cf cf_transformed.mlir --cp 10.00
python3 experimental/tools/integration/run_specv2_integration.py if_convert --n 5 --out out_5 --transformed-cf cf_transformed.mlir --cp 10.00
# python3 experimental/tools/integration/run_specv2_integration.py if_convert --variable --out out_variable --transformed-code if_convert_transformed.c --cp 10.00

./integration-test/if_convert/synthesize.sh
