SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
DYNAMATIC_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"

cd $DYNAMATIC_DIR

python3 experimental/tools/integration/run_specv2_integration.py bisection --disable-spec --cp 10.00 --out out_standard
python3 experimental/tools/integration/run_specv2_integration.py bisection --gate-binarized cf_transformed.mlir --n 1 --out out_1 --cp 10.00
python3 experimental/tools/integration/run_specv2_integration.py bisection --gate-binarized cf_transformed.mlir --n 2 --out out_2 --cp 10.00
python3 experimental/tools/integration/run_specv2_integration.py bisection --gate-binarized cf_transformed.mlir --n 3 --out out_3 --cp 10.00
# python3 experimental/tools/integration/run_specv2_integration.py single_loop --variable --out out_variable --cp 5.00
