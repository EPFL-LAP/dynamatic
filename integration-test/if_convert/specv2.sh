SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
DYNAMATIC_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"

cd $DYNAMATIC_DIR

# Standard flow
python3 experimental/tools/integration/run_specv2_integration.py if_convert --disable-spec --out out_standard --cp 9.00

# Only code transformation
python3 experimental/tools/integration/run_specv2_integration.py if_convert --disable-spec --transformed-cf cf_transformed.mlir --out out_transformed --cp 9.00

# Spec v1
python4 tools/integration/run_spec_integration.py if_convert --out out_v1 --cp 9.00 --default-value 0

python3 experimental/tools/integration/run_specv2_integration.py if_convert --n 0 --out out_0 --transformed-cf cf_transformed.mlir --cp 9.00 --resolver
python4 experimental/tools/integration/run_specv2_integration.py if_convert --n 1 --out out_1 --transformed-cf cf_transformed.mlir --cp 9.00 --use-prof-cache --resolver
python3 experimental/tools/integration/run_specv2_integration.py if_convert --n 2 --out out_2 --transformed-cf cf_transformed.mlir --cp 9.00 --use-prof-cache --resolver
python3 experimental/tools/integration/run_specv2_integration.py if_convert --n 3 --out out_3 --transformed-cf cf_transformed.mlir --cp 9.00 --use-prof-cache --resolver
python3 experimental/tools/integration/run_specv2_integration.py if_convert --n 4 --out out_4 --transformed-cf cf_transformed.mlir --cp 9.00 --use-prof-cache --resolver
python3 experimental/tools/integration/run_specv2_integration.py if_convert --n 5 --out out_5 --transformed-cf cf_transformed.mlir --cp 9.00 --use-prof-cache --resolver
python3 experimental/tools/integration/run_specv2_integration.py if_convert --n 6 --out out_6 --transformed-cf cf_transformed.mlir --cp 9.00 --use-prof-cache --resolver
python3 experimental/tools/integration/run_specv2_integration.py if_convert --n 7 --out out_7 --transformed-cf cf_transformed.mlir --cp 9.00 --use-prof-cache --resolver
python3 experimental/tools/integration/run_specv2_integration.py if_convert --n 8 --out out_8 --transformed-cf cf_transformed.mlir --cp 9.00 --use-prof-cache --resolver
python3 experimental/tools/integration/run_specv2_integration.py if_convert --n 9 --out out_9 --transformed-cf cf_transformed.mlir --cp 9.00 --use-prof-cache --resolver
python3 experimental/tools/integration/run_specv2_integration.py if_convert --n 10 --out out_10 --transformed-cf cf_transformed.mlir --cp 9.00 --use-prof-cache --resolver
python3 experimental/tools/integration/run_specv2_integration.py if_convert --n 11 --out out_11 --transformed-cf cf_transformed.mlir --cp 9.00 --use-prof-cache --resolver
python3 experimental/tools/integration/run_specv2_integration.py if_convert --n 12 --out out_12 --transformed-cf cf_transformed.mlir --cp 9.00 --use-prof-cache --resolver
python3 experimental/tools/integration/run_specv2_integration.py if_convert --n 13 --out out_13 --transformed-cf cf_transformed.mlir --cp 9.00 --use-prof-cache --resolver
python3 experimental/tools/integration/run_specv2_integration.py if_convert --n 14 --out out_14 --transformed-cf cf_transformed.mlir --cp 9.00 --use-prof-cache --resolver
python3 experimental/tools/integration/run_specv2_integration.py if_convert --n 15 --out out_15 --transformed-cf cf_transformed.mlir --cp 9.00 --use-prof-cache --resolver
python3 experimental/tools/integration/run_specv2_integration.py if_convert --n 16 --out out_16 --transformed-cf cf_transformed.mlir --cp 9.00 --use-prof-cache --resolver
python3 experimental/tools/integration/run_specv2_integration.py if_convert --n 17 --out out_17 --transformed-cf cf_transformed.mlir --cp 9.00 --use-prof-cache --resolver
python3 experimental/tools/integration/run_specv2_integration.py if_convert --n 18 --out out_18 --transformed-cf cf_transformed.mlir --cp 9.00 --use-prof-cache --resolver
python3 experimental/tools/integration/run_specv2_integration.py if_convert --n 19 --out out_19 --transformed-cf cf_transformed.mlir --cp 9.00 --use-prof-cache --resolver
python3 experimental/tools/integration/run_specv2_integration.py if_convert --n 20 --out out_20 --transformed-cf cf_transformed.mlir --cp 9.00 --use-prof-cache --resolver
